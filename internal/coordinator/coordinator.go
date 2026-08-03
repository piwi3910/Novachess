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
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
	"github.com/piwi3910/novachess/internal/store"
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
	// seen deduplicates batches by work unit. Delivery is at-least-once, so a
	// redelivered unit produces a second announcement of data already counted;
	// counting it twice would end a generation early with less data than it
	// asked for.
	seen       map[string]bool
	artifacts  []string
	positions  int
	games      int
	issued     int
	generation int
	done       chan struct{}
	closeOnce  sync.Once
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
		seen:       make(map[string]bool),
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

	c.log.Info("starting a generation",
		"generation", c.generation,
		"target", c.cfg.PositionsPerGeneration,
		"network", networkName(c.cfg.NetworkURI))

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
	if c.seen[batch.WorkUnitID] {
		c.mu.Unlock()
		c.log.Debug("ignoring a duplicate batch", "unit", batch.WorkUnitID)
		return nil
	}
	c.mu.Unlock()

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
		// unit's positions are simply missing, and the generation issues more
		// work to make up for them.
		return nil
	}

	c.mu.Lock()
	if c.seen[batch.WorkUnitID] {
		// Another delivery won the race while this one was verifying.
		c.mu.Unlock()
		return nil
	}
	c.seen[batch.WorkUnitID] = true
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
	c.mu.Lock()
	outstanding := c.issued - len(c.artifacts)
	needed := c.cfg.UnitsInFlight - outstanding
	c.mu.Unlock()

	for i := 0; i < needed; i++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		c.mu.Lock()
		index := c.issued
		c.issued++
		c.mu.Unlock()

		unit := c.unit(index)
		if err := c.bus.Publish(ctx, events.SubjectWorkAssign, unit); err != nil {
			// Give the number back, so the next attempt reuses it rather than
			// leaving a gap that makes the issued count a lie.
			c.mu.Lock()
			c.issued--
			c.mu.Unlock()
			return fmt.Errorf("coordinator: issuing work: %w", err)
		}
	}
	return nil
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
