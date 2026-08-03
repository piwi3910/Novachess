// Package pipeline holds the end-to-end test of the training loop.
//
// Every component is covered on its own elsewhere. What this adds is the
// seams: a real broker between the coordinator and the workers, a shared
// volume between the workers and the trainer, and the conventions each side
// assumes the other follows. Those are the joints where two correct components
// disagree, and they are also exactly what a deployment exercises for the first
// time on the cluster rather than here.
package pipeline

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	natsserver "github.com/nats-io/nats-server/v2/server"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/coordinator"
	"github.com/piwi3910/novachess/internal/events"
	"github.com/piwi3910/novachess/internal/store"
	"github.com/piwi3910/novachess/internal/train"
	"github.com/piwi3910/novachess/internal/worker"
)

func runBroker(t *testing.T) string {
	t.Helper()

	server, err := natsserver.NewServer(&natsserver.Options{
		Host:      "127.0.0.1",
		Port:      -1,
		JetStream: true,
		StoreDir:  t.TempDir(),
		NoLog:     true,
		NoSigs:    true,
	})
	if err != nil {
		t.Fatalf("starting a broker: %v", err)
	}
	go server.Start()

	if !server.ReadyForConnections(10 * time.Second) {
		server.Shutdown()
		t.Fatal("the broker did not become ready")
	}
	t.Cleanup(server.Shutdown)

	return server.ClientURL()
}

func connect(t *testing.T, url, name, durable string) *bus.NATS {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	b, err := bus.Connect(ctx, bus.Config{URLs: []string{url}, ClientName: name, Durable: durable})
	if err != nil {
		t.Fatalf("connecting %s: %v", name, err)
	}
	t.Cleanup(func() { b.Close() })
	return b
}

// TestGenerationZeroEndToEnd runs the loop the cluster will run on its first
// day: a coordinator issuing work over a real broker, several workers playing
// it, and a dataset assembled from what they stored.
//
// Generation zero is the interesting one because nothing has to exist first.
// There is no network, so the workers bootstrap from the hand-crafted
// evaluation — which is the only reason the loop can start at all.
func TestGenerationZeroEndToEnd(t *testing.T) {
	url := runBroker(t)

	// One shared volume, as the cluster has: workers write to it and the
	// coordinator reads from it. On Kubernetes this is one ReadWriteMany claim
	// mounted at the same path in every pod.
	shared := t.TempDir()

	newStore := func() store.Store {
		s, err := store.NewFS(shared)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { s.Close() })
		return s
	}

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	// Three workers, each with its own connection, as three pods would have.
	const workers = 3
	var running sync.WaitGroup
	for i := 0; i < workers; i++ {
		name := fmt.Sprintf("worker-%d", i)

		w, err := worker.New(worker.Config{
			ID:            name,
			NodeName:      fmt.Sprintf("board-%d", i),
			HashMB:        1,
			EngineVersion: "test",
		}, connect(t, url, name, ""), newStore(), nil)
		if err != nil {
			t.Fatal(err)
		}

		running.Add(1)
		go func() {
			defer running.Done()
			if err := w.Run(ctx); err != nil && ctx.Err() == nil {
				t.Errorf("%s: %v", name, err)
			}
		}()
	}

	cfg := coordinator.DefaultConfig()
	cfg.Generation = 0
	cfg.PositionsPerGeneration = 400
	cfg.GamesPerUnit = 2
	cfg.NodesPerMove = 600
	cfg.RandomPlies = 6
	cfg.UnitsInFlight = 6
	cfg.Seed = 0xC0FFEE
	// Empty: generation zero has no network to play with, and bootstraps from
	// the hand-crafted evaluation.
	cfg.NetworkURI = ""

	coordBus := connect(t, url, "coordinator", "coordinator")
	c, err := coordinator.New(cfg, coordBus, newStore(), nil)
	if err != nil {
		t.Fatal(err)
	}

	// Watch for the dataset the trainer would consume.
	var mu sync.Mutex
	var announced []events.DatasetReady
	if _, err := coordBus.Subscribe(ctx, events.SubjectDatasetReady, func(_ context.Context, env events.Envelope) error {
		var ready events.DatasetReady
		if err := bus.Unmarshal(env, &ready); err != nil {
			return err
		}
		mu.Lock()
		announced = append(announced, ready)
		mu.Unlock()
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	progress, err := c.RunGeneration(ctx)
	if err != nil {
		t.Fatalf("the generation did not finish: %v (progress %+v)", err, progress)
	}
	cancel()
	running.Wait()

	if !progress.Complete {
		t.Fatalf("the generation stopped short: %+v", progress)
	}
	if progress.Positions < cfg.PositionsPerGeneration {
		t.Errorf("%d positions, the target was %d", progress.Positions, cfg.PositionsPerGeneration)
	}

	// The dataset must have been announced, and must describe what was counted.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(announced)
		mu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(announced) == 0 {
		t.Fatal("no dataset was announced; the trainer would never start")
	}
	ready := announced[0]

	if ready.Generation != 0 {
		t.Errorf("dataset is for generation %d, want 0", ready.Generation)
	}
	if len(ready.ArtifactURIs) == 0 {
		t.Fatal("the dataset lists no artifacts")
	}

	// Every artifact it names must be readable as training data, by a process
	// that did not write it. This is the seam a shared volume exists to close.
	//
	// A fresh context: the run's has been cancelled to stop the workers, and
	// reading the results is not part of the run.
	readCtx := context.Background()
	reader := newStore()
	total := 0
	for _, uri := range ready.ArtifactURIs {
		rc, err := reader.Get(readCtx, uri)
		if err != nil {
			t.Fatalf("the trainer could not open %s: %v", uri, err)
		}
		r, err := train.NewReader(rc)
		if err != nil {
			rc.Close()
			t.Fatalf("%s is not training data: %v", uri, err)
		}
		samples, err := r.ReadAll()
		rc.Close()
		if err != nil {
			t.Fatalf("reading %s: %v", uri, err)
		}
		total += len(samples)
	}

	if total != ready.Positions {
		t.Errorf("the artifacts hold %d positions, the dataset claims %d", total, ready.Positions)
	}

	t.Logf("%d workers produced %d positions in %d batches from %d units; dataset lists %d artifacts",
		workers, progress.Positions, progress.Batches, progress.Issued, len(ready.ArtifactURIs))
}

// TestWorkSurvivesAWorkerDying is the property that makes an eviction a delay
// rather than a loss. A worker that takes a unit and disappears must have it
// redelivered, and the generation must still complete.
func TestWorkSurvivesAWorkerDying(t *testing.T) {
	url := runBroker(t)
	shared := t.TempDir()

	newStore := func() store.Store {
		s, err := store.NewFS(shared)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { s.Close() })
		return s
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	// A worker that fails the first unit it sees, as one being evicted would.
	var failures int
	var mu sync.Mutex
	unreliable := connect(t, url, "unreliable", "")
	if _, err := unreliable.QueueSubscribe(ctx, events.SubjectWorkAssign, worker.QueueGroup,
		func(context.Context, events.Envelope) error {
			mu.Lock()
			defer mu.Unlock()
			if failures == 0 {
				failures++
				return fmt.Errorf("the pod went away")
			}
			// Afterwards it declines to take more, leaving the real worker to
			// finish the generation.
			return fmt.Errorf("still unwell")
		}); err != nil {
		t.Fatal(err)
	}

	w, err := worker.New(worker.Config{
		ID: "worker-good", HashMB: 1, EngineVersion: "test",
	}, connect(t, url, "worker-good", ""), newStore(), nil)
	if err != nil {
		t.Fatal(err)
	}
	go w.Run(ctx)

	cfg := coordinator.DefaultConfig()
	cfg.PositionsPerGeneration = 150
	cfg.GamesPerUnit = 2
	cfg.NodesPerMove = 600
	cfg.RandomPlies = 6
	cfg.UnitsInFlight = 4
	cfg.Seed = 99

	c, err := coordinator.New(cfg, connect(t, url, "coordinator", "coordinator"), newStore(), nil)
	if err != nil {
		t.Fatal(err)
	}

	progress, err := c.RunGeneration(ctx)
	if err != nil {
		t.Fatalf("the generation did not survive a failing worker: %v (progress %+v)", err, progress)
	}
	if !progress.Complete {
		t.Errorf("the generation stopped short: %+v", progress)
	}

	mu.Lock()
	defer mu.Unlock()
	if failures == 0 {
		t.Error("the unreliable worker never took a unit; the test proved nothing")
	}
	t.Logf("completed with %d positions despite a worker rejecting its unit", progress.Positions)
}
