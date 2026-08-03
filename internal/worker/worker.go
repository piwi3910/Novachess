// Package worker runs self-play on a cluster node.
//
// A worker takes a unit off the shared queue, plays the games it describes,
// stores the resulting positions, and announces the batch. It is the only
// service that does real work in bulk, and the only one there are many of:
// scaling the pipeline means running more of these.
//
// The ordering inside a unit is the part that matters. The batch is durably
// stored and its announcement acknowledged by the broker *before* the unit is
// acknowledged. Acknowledging earlier would mean a crash in the gap loses the
// games silently — the unit gone from the queue and no artifact to show for it —
// which is the one failure that costs work nobody knows to redo.
package worker

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/eval"
	"github.com/piwi3910/novachess/internal/events"
	"github.com/piwi3910/novachess/internal/nnue"
	"github.com/piwi3910/novachess/internal/store"
	"github.com/piwi3910/novachess/internal/train"
)

// QueueGroup is the group every self-play worker joins, so a unit is played
// once rather than once per pod.
const QueueGroup = "selfplay"

// Config describes one worker.
type Config struct {
	// ID identifies this worker in heartbeats and in the batches it produces.
	// The pod name is the useful value: when one board misbehaves, this is
	// what ties a slow batch back to it.
	ID string

	// NodeName is the machine the worker runs on, reported in heartbeats.
	NodeName string

	// HashMB is the transposition table size for each search.
	HashMB int

	// HeartbeatInterval is how often liveness and throughput are reported.
	// Zero disables heartbeats, which is what tests want.
	HeartbeatInterval time.Duration

	// EngineVersion is stamped on every batch so a dataset can be traced back
	// to the build that produced it.
	EngineVersion string
}

func (c Config) validate() error {
	if c.ID == "" {
		return fmt.Errorf("worker: no ID; batches would not be attributable to a pod")
	}
	if c.HashMB < 1 {
		return fmt.Errorf("worker: hash size %d", c.HashMB)
	}
	return nil
}

// Worker consumes work units and produces batches.
type Worker struct {
	cfg   Config
	bus   bus.Bus
	store store.Store
	log   *slog.Logger

	started time.Time

	mu        sync.Mutex
	current   string
	completed int
	lastNPS   float64

	// networks caches evaluations by URI. A generation is many units against
	// the same network, and fetching and parsing several hundred kilobytes per
	// unit would be pure waste.
	networks map[string]eval.Evaluator
}

// New returns a worker.
func New(cfg Config, b bus.Bus, s store.Store, log *slog.Logger) (*Worker, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if b == nil || s == nil {
		return nil, fmt.Errorf("worker: a bus and a store are both required")
	}
	if log == nil {
		log = slog.New(slog.DiscardHandler)
	}

	return &Worker{
		cfg:      cfg,
		bus:      b,
		store:    s,
		log:      log,
		started:  time.Now(),
		networks: make(map[string]eval.Evaluator),
	}, nil
}

// Run subscribes to the work queue and processes units until the context ends.
func (w *Worker) Run(ctx context.Context) error {
	sub, err := w.bus.QueueSubscribe(ctx, events.SubjectWorkAssign, QueueGroup, w.handle)
	if err != nil {
		return fmt.Errorf("worker: subscribing to work: %w", err)
	}
	defer sub.Close()

	w.log.Info("waiting for work", "worker", w.cfg.ID, "queue", QueueGroup)

	if w.cfg.HeartbeatInterval > 0 {
		go w.heartbeat(ctx)
	}

	<-ctx.Done()
	w.log.Info("shutting down", "worker", w.cfg.ID, "completed", w.completedCount())
	return nil
}

// handle plays one work unit.
//
// Returning an error negatively acknowledges the unit, putting it back for
// another worker. That is the right response to almost any failure here: the
// unit is reproducible, so a different node will produce exactly the same games
// as this one would have.
func (w *Worker) handle(ctx context.Context, env events.Envelope) error {
	var unit events.WorkUnit
	if err := bus.Unmarshal(env, &unit); err != nil {
		// Undecodable work will not decode on retry either. Rejecting it would
		// put it back forever, so it is logged and dropped.
		w.log.Error("discarding an unreadable work unit", "error", err, "message", env.ID)
		return nil
	}

	w.setCurrent(unit.ID)
	defer w.setCurrent("")

	log := w.log.With("unit", unit.ID, "generation", unit.Generation, "games", unit.Games)
	log.Info("starting")

	evaluator, err := w.evaluator(ctx, unit.NetworkURI)
	if err != nil {
		log.Error("could not load the evaluation", "error", err, "network", unit.NetworkURI)
		return err
	}

	samples, stats, err := train.NewGenerator(evaluator, w.cfg.HashMB).Generate(ctx, unit)
	if err != nil {
		if ctx.Err() != nil {
			// Shutting down. The unit goes back so another worker plays it
			// rather than it being lost with this pod.
			log.Info("interrupted; returning the unit to the queue")
		} else {
			log.Error("generation failed", "error", err)
		}
		return err
	}
	// A unit whose games all ended in the random opening is legal but useless.
	// It is still announced, with no artifact and no positions: staying silent
	// would leave the coordinator waiting for a report that never comes, and it
	// counts outstanding work by what has reported back.
	var artifact store.Artifact
	if len(samples) == 0 {
		log.Warn("produced no usable positions", "games", stats.Games)
	} else {
		artifact, err = w.storeBatch(ctx, unit, samples)
		if err != nil {
			log.Error("could not store the batch", "error", err)
			return err
		}
	}

	// Announced only after the artifact is durably stored, and acknowledged by
	// the broker before this returns. A crash between the two replays the unit,
	// which costs time; the reverse order would lose the games entirely.
	batch := events.GameBatch{
		WorkUnitID:     unit.ID,
		WorkerID:       w.cfg.ID,
		Generation:     unit.Generation,
		ArtifactURI:    artifact.URI,
		SizeBytes:      artifact.Size,
		Checksum:       artifact.Checksum,
		Positions:      len(samples),
		Games:          stats.Games,
		NodesPerSecond: stats.NodesPerSecond(),
		Duration:       stats.Duration,
		EngineVersion:  w.cfg.EngineVersion,
	}
	if err := w.bus.Publish(ctx, events.SubjectGamesProduced, batch); err != nil {
		log.Error("could not announce the batch", "error", err, "artifact", artifact.URI)
		return err
	}

	w.recordCompletion(stats.NodesPerSecond())
	log.Info("done",
		"positions", len(samples),
		"games", stats.Games,
		"nps", int64(stats.NodesPerSecond()),
		"artifact", artifact.URI)
	return nil
}

// storeBatch encodes the samples and writes them.
func (w *Worker) storeBatch(ctx context.Context, unit events.WorkUnit, samples []train.Sample) (store.Artifact, error) {
	var buf bytes.Buffer

	writer, err := train.NewWriter(&buf)
	if err != nil {
		return store.Artifact{}, err
	}
	if err := writer.WriteAll(samples); err != nil {
		return store.Artifact{}, err
	}

	// The key carries the generation and the unit, so a redelivered unit
	// overwrites its own artifact rather than adding a second one. That is safe
	// precisely because generation is deterministic: the replay produces
	// byte-identical data.
	key := fmt.Sprintf("gen%d/%s%s", unit.Generation, unit.ID, train.FileExtension)
	return w.store.Put(ctx, key, &buf)
}

// evaluator returns the evaluation a unit asks for, fetching and caching a
// network when one is named.
//
// An empty URI means the hand-crafted evaluation, which is how generation zero
// bootstraps: there is no network yet, and something has to play the first
// games.
func (w *Worker) evaluator(ctx context.Context, uri string) (eval.Evaluator, error) {
	if uri == "" {
		return eval.NewHCE(), nil
	}

	w.mu.Lock()
	cached, ok := w.networks[uri]
	w.mu.Unlock()
	if ok {
		return cached, nil
	}

	rc, err := w.store.Get(ctx, uri)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", uri, err)
	}
	defer rc.Close()

	network, err := nnue.ReadFrom(rc)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", uri, err)
	}
	evaluator, err := nnue.NewEvaluator(network)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", uri, err)
	}

	w.mu.Lock()
	w.networks[uri] = evaluator
	w.mu.Unlock()

	w.log.Info("loaded a network", "uri", uri)
	return evaluator, nil
}

// heartbeat reports liveness and throughput until the context ends.
//
// Throughput per worker is the practical way to spot a board that is throttling
// or has lost its fan. Inferring it from batch durations would work only for
// workers that finished something recently, which is exactly not the case for
// the board that has slowed to a crawl.
func (w *Worker) heartbeat(ctx context.Context) {
	ticker := time.NewTicker(w.cfg.HeartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			w.mu.Lock()
			beat := events.WorkerHeartbeat{
				WorkerID:          w.cfg.ID,
				NodeName:          w.cfg.NodeName,
				CurrentWorkUnitID: w.current,
				NodesPerSecond:    w.lastNPS,
				GamesCompleted:    w.completed,
				Uptime:            time.Since(w.started),
				EngineVersion:     w.cfg.EngineVersion,
			}
			w.mu.Unlock()

			// A failed heartbeat is not worth interrupting work for. It is
			// fire-and-forget by design; the next one is a few seconds away.
			if err := w.bus.Publish(ctx, events.SubjectWorkerHeartbeat, beat); err != nil && ctx.Err() == nil {
				w.log.Debug("heartbeat failed", "error", err)
			}
		}
	}
}

func (w *Worker) setCurrent(id string) {
	w.mu.Lock()
	w.current = id
	w.mu.Unlock()
}

func (w *Worker) recordCompletion(nps float64) {
	w.mu.Lock()
	w.completed++
	w.lastNPS = nps
	w.mu.Unlock()
}

func (w *Worker) completedCount() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.completed
}
