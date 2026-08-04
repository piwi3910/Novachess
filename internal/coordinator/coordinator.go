// Package coordinator drives the training loop.
//
// It issues work units, collects the batches workers produce, and announces a
// dataset once a generation has enough positions. It is the only service that
// holds the loop's state, and deliberately the only one there is exactly one
// of: two coordinators issuing units for the same generation would each count
// half the batches and neither would ever reach its target.
//
// Everything it knows is rebuilt from the bus. There is no database, because a
// stream the coordinator can replay is already a durable record of what
// happened, and a second source of truth is a second thing to be inconsistent
// with the first.
package coordinator

import (
	"context"
	"fmt"
	"log/slog"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
	"github.com/piwi3910/novachess/internal/store"
	"github.com/piwi3910/novachess/internal/train"
)

// Config describes how a generation is run.
type Config struct {
	// Generation is where to start. A restart resumes here.
	Generation int

	// PositionsPerGeneration is how many positions a dataset needs before the
	// trainer is asked to fit a network to it.
	PositionsPerGeneration int

	// GamesPerUnit is how many games one work unit asks for. Small enough that
	// a worker evicted mid-unit loses little, large enough that the per-unit
	// overhead does not dominate.
	GamesPerUnit int

	// NodesPerMove fixes the search effort. A node count and never a time
	// limit: the same unit must produce the same games on any board.
	NodesPerMove int

	// RandomPlies varies the openings.
	RandomPlies int

	// UnitsInFlight is how many units to keep queued. Enough that no worker
	// ever waits for the coordinator, not so many that a generation cannot be
	// abandoned promptly.
	UnitsInFlight int

	// Seed derives every unit's seed, so a whole generation is reproducible
	// from one number.
	Seed uint64

	// NetworkURI is the network workers should play with, empty for the
	// hand-crafted evaluation. Generation zero starts empty; later generations
	// are set from promotions.
	NetworkURI string
}

// DefaultConfig returns settings for a first generation.
func DefaultConfig() Config {
	return Config{
		Generation:             0,
		PositionsPerGeneration: 1_000_000,
		GamesPerUnit:           50,
		NodesPerMove:           20_000,
		RandomPlies:            8,
		UnitsInFlight:          64,
		Seed:                   1,
	}
}

func (c Config) validate() error {
	switch {
	case c.Generation < 0:
		return fmt.Errorf("coordinator: generation %d", c.Generation)
	case c.PositionsPerGeneration <= 0:
		return fmt.Errorf("coordinator: %d positions per generation", c.PositionsPerGeneration)
	case c.GamesPerUnit <= 0:
		return fmt.Errorf("coordinator: %d games per unit", c.GamesPerUnit)
	case c.NodesPerMove <= 0:
		return fmt.Errorf("coordinator: no node budget; self-play must not be time-limited")
	case c.UnitsInFlight <= 0:
		return fmt.Errorf("coordinator: %d units in flight", c.UnitsInFlight)
	}
	return nil
}

// Progress describes a generation in flight.
type Progress struct {
	Generation int
	Positions  int
	Games      int
	Batches    int
	Issued     int
	Complete   bool
}

// Coordinator issues work and assembles datasets.
type Coordinator struct {
	cfg   Config
	bus   bus.Bus
	store store.Store
	log   *slog.Logger

	mu sync.Mutex
	// resolved is every unit that has reported back, whatever came of it.
	//
	// It serves two purposes, and conflating them was a bug. It deduplicates:
	// delivery is at-least-once, so a redelivered unit announces data already
	// counted, and counting it twice would end a generation early with less
	// data than it asked for. And it is what "outstanding" is measured against
	// — a unit whose batch failed verification, or that produced no positions,
	// has finished even though it contributed nothing, and treating it as still
	// in flight would eventually stop the coordinator issuing work at all.
	resolved   map[string]bool
	artifacts  []string
	positions  int
	games      int
	issued     int
	generation int
	done       chan struct{}
	closeOnce  sync.Once

	// nextIndex is the next unit index to consider issuing. It walks forward
	// independently of issued, which counts only units actually published,
	// because a resumed generation must skip indices already covered on disk
	// without renumbering the ones still missing.
	nextIndex int
	// covered marks unit indices satisfied by an artifact found on disk at
	// startup, so topUp knows to step over them rather than reissue and
	// replay work that already succeeded.
	covered map[int]bool
}

// New returns a coordinator.
func New(cfg Config, b bus.Bus, s store.Store, log *slog.Logger) (*Coordinator, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if b == nil || s == nil {
		return nil, fmt.Errorf("coordinator: a bus and a store are both required")
	}
	if log == nil {
		log = slog.New(slog.DiscardHandler)
	}

	return &Coordinator{
		cfg:        cfg,
		bus:        b,
		store:      s,
		log:        log,
		resolved:   make(map[string]bool),
		covered:    make(map[int]bool),
		generation: cfg.Generation,
		done:       make(chan struct{}),
	}, nil
}

// RunGeneration issues work until the generation has enough positions, then
// announces the dataset.
//
// It returns when the target is reached, or when the context ends. The dataset
// announcement is the last thing it does, so a coordinator that dies before
// that point leaves a generation to be resumed rather than a half-announced one
// for the trainer to act on.
func (c *Coordinator) RunGeneration(ctx context.Context) (Progress, error) {
	sub, err := c.bus.Subscribe(ctx, events.SubjectGamesProduced, c.handleBatch)
	if err != nil {
		return c.progress(), fmt.Errorf("coordinator: subscribing to batches: %w", err)
	}
	defer sub.Close()

	// Before issuing anything, find out what an earlier run of this generation
	// already produced. Skipping this would replay every unit from zero on
	// every restart — hours of byte-identical work for a generation that
	// extends across a deploy.
	if err := c.resume(ctx); err != nil {
		return c.progress(), err
	}

	c.log.Info("starting a generation",
		"generation", c.generation,
		"target", c.cfg.PositionsPerGeneration,
		"network", networkName(c.cfg.NetworkURI))

	c.mu.Lock()
	alreadyReached := c.positions >= c.cfg.PositionsPerGeneration
	c.mu.Unlock()

	// The artifacts already on disk are enough on their own. Announce and stop
	// here: issuing work now would ask for positions nobody needs.
	if alreadyReached {
		progress := c.progress()
		if err := c.announce(ctx); err != nil {
			return progress, err
		}
		progress.Complete = true
		return progress, nil
	}

	// Fill the queue before waiting, so workers have something to do the
	// moment they connect.
	if err := c.topUp(ctx); err != nil {
		return c.progress(), err
	}

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return c.progress(), ctx.Err()

		case <-c.done:
			progress := c.progress()
			if err := c.announce(ctx); err != nil {
				return progress, err
			}
			progress.Complete = true
			return progress, nil

		case <-ticker.C:
			// Keep the queue stocked. Units are issued as batches come back
			// rather than all at once, so abandoning a generation does not mean
			// waiting for thousands of already-queued units to drain.
			if err := c.topUp(ctx); err != nil {
				return c.progress(), err
			}
		}
	}
}

// handleBatch counts a batch towards the generation.
func (c *Coordinator) handleBatch(ctx context.Context, env events.Envelope) error {
	var batch events.GameBatch
	if err := bus.Unmarshal(env, &batch); err != nil {
		c.log.Error("discarding an unreadable batch", "error", err, "message", env.ID)
		return nil
	}

	c.mu.Lock()
	if batch.Generation != c.generation {
		// A straggler from a generation already finished. Harmless, and not
		// worth retrying.
		c.mu.Unlock()
		return nil
	}
	if c.resolved[batch.WorkUnitID] {
		c.mu.Unlock()
		c.log.Debug("ignoring a duplicate batch", "unit", batch.WorkUnitID)
		return nil
	}
	// Marked before it is validated, not after. The unit has reported back
	// whatever happens next, and every path below leaves it finished — so
	// recording it here is what keeps the outstanding count honest and stops a
	// duplicate racing through the verification below.
	c.resolved[batch.WorkUnitID] = true
	c.mu.Unlock()

	// A unit whose games all ended in the random opening produces nothing.
	// It announces itself anyway, so that the coordinator learns it is done
	// rather than waiting for an artifact that was never written.
	if batch.Positions == 0 || batch.ArtifactURI == "" {
		c.log.Warn("a unit produced no positions", "unit", batch.WorkUnitID, "worker", batch.WorkerID)
		return nil
	}

	// Confirmed before it is counted. A worker announces a batch it believes it
	// stored; if the artifact is missing or corrupt, counting it would build a
	// dataset around data the trainer cannot read, and the failure would
	// surface much later as a training error nobody could trace back to here.
	if err := store.Verify(ctx, c.store, store.Artifact{
		URI:      batch.ArtifactURI,
		Size:     batch.SizeBytes,
		Checksum: batch.Checksum,
	}); err != nil {
		c.log.Error("rejecting a batch that does not verify",
			"error", err, "unit", batch.WorkUnitID, "worker", batch.WorkerID)
		// Not retried: the artifact will not become correct on redelivery. The
		// unit's positions are simply missing, and because it is already marked
		// resolved the generation issues more work to make up for them.
		return nil
	}

	c.mu.Lock()
	c.artifacts = append(c.artifacts, batch.ArtifactURI)
	c.positions += batch.Positions
	c.games += batch.Games
	positions, batches := c.positions, len(c.artifacts)
	reached := positions >= c.cfg.PositionsPerGeneration
	c.mu.Unlock()

	c.log.Info("counted a batch",
		"unit", batch.WorkUnitID,
		"worker", batch.WorkerID,
		"positions", batch.Positions,
		"total", positions,
		"target", c.cfg.PositionsPerGeneration,
		"nps", int64(batch.NodesPerSecond))

	if reached {
		c.log.Info("generation complete", "positions", positions, "batches", batches)
		c.closeOnce.Do(func() { close(c.done) })
	}
	return nil
}

// topUp issues units until enough are outstanding.
func (c *Coordinator) topUp(ctx context.Context) error {
	// Measured against every unit that has reported back, not against the ones
	// that contributed data. A unit that failed verification or produced no
	// positions is finished; counting it as still in flight would shrink the
	// queue with every such unit until the coordinator stopped issuing work and
	// the generation hung with idle workers. Units resumed from disk are
	// counted the same way here: resume marks them resolved and bumps issued
	// to match, so they contribute nothing outstanding either.
	c.mu.Lock()
	outstanding := c.issued - len(c.resolved)
	needed := c.cfg.UnitsInFlight - outstanding
	c.mu.Unlock()

	issuedThisCall := 0
	for issuedThisCall < needed {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		// Walk forward over unit indices, stepping over any already covered by
		// an artifact found on disk at startup. Skipping does not count against
		// needed: it is the count of new units issued that must reach it, not
		// the count of indices considered.
		c.mu.Lock()
		index := c.nextIndex
		c.nextIndex++
		skip := c.covered[index]
		c.mu.Unlock()

		if skip {
			continue
		}

		unit := c.unit(index)
		if err := c.bus.Publish(ctx, events.SubjectWorkAssign, unit); err != nil {
			// Give the index back, so the next attempt reuses it rather than
			// leaving a gap that renumbers every unit after it.
			c.mu.Lock()
			c.nextIndex--
			c.mu.Unlock()
			return fmt.Errorf("coordinator: issuing work: %w", err)
		}

		c.mu.Lock()
		c.issued++
		c.mu.Unlock()
		issuedThisCall++
	}
	return nil
}

// resume scans the generation's artifact directory for units an earlier run
// already produced, so a restart resumes a generation rather than replaying
// it — reissuing every unit from zero costs hours of byte-identical self-play
// per extension of a long generation.
//
// A file that does not parse as this generation's unit naming is not one of
// ours and is left alone. A file that fails to read as a valid, complete data
// file counts as absent rather than present, and its unit is reissued: a
// truncated artifact left by a worker evicted mid-write is indistinguishable
// from one that was never written, and trusting it would silently drop its
// positions out of the dataset forever.
func (c *Coordinator) resume(ctx context.Context) error {
	uris, err := c.store.List(ctx, fmt.Sprintf("gen%d", c.generation))
	if err != nil {
		return fmt.Errorf("coordinator: scanning for existing artifacts: %w", err)
	}

	var positions, count int
	for _, uri := range uris {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		index, ok := c.unitIndex(uri)
		if !ok {
			continue
		}

		n, err := c.countPositions(ctx, uri)
		if err != nil {
			c.log.Warn("discarding an existing artifact that does not validate",
				"error", err, "artifact", uri)
			continue
		}

		unitID := c.unit(index).ID

		c.mu.Lock()
		if !c.resolved[unitID] {
			c.resolved[unitID] = true
			c.covered[index] = true
			c.artifacts = append(c.artifacts, uri)
			c.positions += n
			// issued is bumped in step with resolved so this unit contributes
			// nothing to topUp's outstanding count: it was never issued this
			// run, and none of the queue's capacity should be spent waiting on
			// it.
			c.issued++
			positions += n
			count++
		}
		c.mu.Unlock()
	}

	if count > 0 {
		c.log.Info("resumed a generation",
			"generation", c.generation,
			"positions", positions,
			"artifacts", count)
	}
	return nil
}

// unitIndex extracts a unit's index from where its artifact would be stored,
// the inverse of the naming unit builds. A name that does not match this
// generation's pattern is ignored rather than guessed at — it did not come
// from this coordinator, or belongs to a different generation's directory.
func (c *Coordinator) unitIndex(uri string) (int, bool) {
	base := path.Base(uri)
	name, ok := strings.CutSuffix(base, train.FileExtension)
	if !ok {
		return 0, false
	}

	prefix := fmt.Sprintf("g%d-u", c.generation)
	digits, ok := strings.CutPrefix(name, prefix)
	if !ok {
		return 0, false
	}

	index, err := strconv.Atoi(digits)
	// The round trip through the same zero-padded width the coordinator uses
	// to build IDs rejects anything that merely looks similar, such as an
	// index with a different number of digits.
	if err != nil || index < 0 || fmt.Sprintf("%06d", index) != digits {
		return 0, false
	}
	return index, true
}

// countPositions reads an artifact end to end and returns how many samples it
// holds, proving along the way that it decodes cleanly. A truncated or
// otherwise corrupt file surfaces as an error here rather than a plausible
// count, so resume treats it as absent instead of trusting a number that
// was never really there.
func (c *Coordinator) countPositions(ctx context.Context, uri string) (int, error) {
	rc, err := c.store.Get(ctx, uri)
	if err != nil {
		return 0, err
	}
	defer rc.Close()

	reader, err := train.NewReader(rc)
	if err != nil {
		return 0, err
	}
	samples, err := reader.ReadAll()
	if err != nil {
		return 0, err
	}
	return len(samples), nil
}

// unit builds the nth work unit of this generation.
//
// The seed is derived from the generation and the index rather than from a
// running counter, so a unit's games depend only on which unit it is. That is
// what lets a whole generation be reproduced from the config alone, and one
// suspect unit be replayed on its own.
func (c *Coordinator) unit(index int) events.WorkUnit {
	return events.WorkUnit{
		ID:           fmt.Sprintf("g%d-u%06d", c.generation, index),
		Generation:   c.generation,
		NetworkURI:   c.cfg.NetworkURI,
		Games:        c.cfg.GamesPerUnit,
		RandomPlies:  c.cfg.RandomPlies,
		NodesPerMove: c.cfg.NodesPerMove,
		Seed:         c.cfg.Seed ^ (uint64(c.generation+1) * 0x9E3779B97F4A7C15) ^ (uint64(index+1) * 0xBF58476D1CE4E5B9),
	}
}

// announce publishes the assembled dataset.
func (c *Coordinator) announce(ctx context.Context) error {
	c.mu.Lock()
	ready := events.DatasetReady{
		Generation:         c.generation,
		ArtifactURIs:       append([]string(nil), c.artifacts...),
		Positions:          c.positions,
		BaselineNetworkURI: c.cfg.NetworkURI,
	}
	c.mu.Unlock()

	if err := c.bus.Publish(ctx, events.SubjectDatasetReady, ready); err != nil {
		return fmt.Errorf("coordinator: announcing the dataset: %w", err)
	}
	c.log.Info("announced a dataset",
		"generation", ready.Generation,
		"positions", ready.Positions,
		"artifacts", len(ready.ArtifactURIs))
	return nil
}

// Progress reports the generation in flight.
func (c *Coordinator) Progress() Progress { return c.progress() }

func (c *Coordinator) progress() Progress {
	c.mu.Lock()
	defer c.mu.Unlock()
	return Progress{
		Generation: c.generation,
		Positions:  c.positions,
		Games:      c.games,
		Batches:    len(c.artifacts),
		Issued:     c.issued,
	}
}

func networkName(uri string) string {
	if uri == "" {
		return "the hand-crafted evaluation"
	}
	return uri
}
