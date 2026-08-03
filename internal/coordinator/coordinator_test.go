package coordinator

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
	"github.com/piwi3910/novachess/internal/store"
)

func testStore(t *testing.T) *store.FS {
	t.Helper()
	s, err := store.NewFS(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func testConfig() Config {
	c := DefaultConfig()
	c.PositionsPerGeneration = 300
	c.GamesPerUnit = 5
	c.NodesPerMove = 1000
	c.UnitsInFlight = 4
	c.Seed = 0xC0FFEE
	return c
}

// putBatch writes an artifact and returns the announcement describing it, the
// way a worker would.
func putBatch(t *testing.T, s store.Store, unitID string, generation, positions int) events.GameBatch {
	t.Helper()

	artifact, err := s.Put(context.Background(),
		fmt.Sprintf("gen%d/%s.novadata", generation, unitID),
		strings.NewReader(strings.Repeat("x", 64)+unitID))
	if err != nil {
		t.Fatal(err)
	}

	return events.GameBatch{
		WorkUnitID:  unitID,
		WorkerID:    "worker-1",
		Generation:  generation,
		ArtifactURI: artifact.URI,
		SizeBytes:   artifact.Size,
		Checksum:    artifact.Checksum,
		Positions:   positions,
		Games:       5,
	}
}

func collect(t *testing.T, b bus.Bus, subject string) func() []events.Envelope {
	t.Helper()

	var mu sync.Mutex
	var seen []events.Envelope

	if _, err := b.Subscribe(context.Background(), subject, func(_ context.Context, env events.Envelope) error {
		mu.Lock()
		seen = append(seen, env)
		mu.Unlock()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return func() []events.Envelope {
		mu.Lock()
		defer mu.Unlock()
		return append([]events.Envelope(nil), seen...)
	}
}

// TestCoordinatorIssuesWorkAndAssemblesADataset is the loop's outer turn: work
// goes out, batches come back, and once there are enough positions a dataset is
// announced for the trainer.
func TestCoordinatorIssuesWorkAndAssemblesADataset(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	units := collect(t, b, events.SubjectWorkAssign)
	datasets := collect(t, b, events.SubjectDatasetReady)

	c, err := New(testConfig(), b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Feed batches in from another goroutine, as workers would.
	go func() {
		for i := 0; ; i++ {
			batch := putBatch(t, s, fmt.Sprintf("g0-u%06d", i), 0, 100)
			if err := b.Publish(context.Background(), events.SubjectGamesProduced, batch); err != nil {
				return
			}
			if c.Progress().Positions >= testConfig().PositionsPerGeneration {
				return
			}
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	progress, err := c.RunGeneration(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if !progress.Complete {
		t.Error("the generation did not complete")
	}
	if progress.Positions < testConfig().PositionsPerGeneration {
		t.Errorf("finished with %d positions, the target was %d",
			progress.Positions, testConfig().PositionsPerGeneration)
	}

	if len(units()) == 0 {
		t.Error("no work was issued")
	}

	announced := datasets()
	if len(announced) != 1 {
		t.Fatalf("%d datasets announced, want 1", len(announced))
	}

	var ready events.DatasetReady
	if err := bus.Unmarshal(announced[0], &ready); err != nil {
		t.Fatal(err)
	}
	if ready.Positions != progress.Positions {
		t.Errorf("the dataset claims %d positions, the generation counted %d", ready.Positions, progress.Positions)
	}
	if len(ready.ArtifactURIs) != progress.Batches {
		t.Errorf("the dataset lists %d artifacts, %d batches were counted", len(ready.ArtifactURIs), progress.Batches)
	}
	t.Logf("%d positions across %d batches from %d units issued",
		progress.Positions, progress.Batches, progress.Issued)
}

// TestCoordinatorIgnoresDuplicateBatches is what makes at-least-once delivery
// safe here. A redelivered unit announces data already counted; counting it
// again would end the generation early with less data than it asked for, and
// the shortfall would be invisible.
func TestCoordinatorIgnoresDuplicateBatches(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	c, err := New(testConfig(), b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	batch := putBatch(t, s, "g0-u000000", 0, 100)
	env, err := bus.NewEnvelope(events.SubjectGamesProduced, "test", batch)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		if err := c.handleBatch(context.Background(), env); err != nil {
			t.Fatal(err)
		}
	}

	if got := c.Progress(); got.Positions != 100 || got.Batches != 1 {
		t.Errorf("three deliveries of one batch counted %d positions across %d batches, want 100 across 1",
			got.Positions, got.Batches)
	}
}

// TestCoordinatorRejectsUnverifiableBatches is the integrity gate. A worker
// announces a batch it believes it stored; if the artifact is missing or
// corrupt, counting it would build a dataset around data the trainer cannot
// read, and the failure would surface much later as an error nobody could trace
// back here.
func TestCoordinatorRejectsUnverifiableBatches(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	c, err := New(testConfig(), b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name  string
		batch func() events.GameBatch
	}{
		{
			"the artifact does not exist",
			func() events.GameBatch {
				batch := putBatch(t, s, "g0-missing", 0, 100)
				if err := s.Delete(context.Background(), batch.ArtifactURI); err != nil {
					t.Fatal(err)
				}
				return batch
			},
		},
		{
			"the checksum does not match",
			func() events.GameBatch {
				batch := putBatch(t, s, "g0-corrupt", 0, 100)
				batch.Checksum = strings.Repeat("0", 64)
				return batch
			},
		},
		{
			"the size does not match",
			func() events.GameBatch {
				batch := putBatch(t, s, "g0-short", 0, 100)
				batch.SizeBytes = 999999
				return batch
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := c.Progress()

			env, err := bus.NewEnvelope(events.SubjectGamesProduced, "test", tc.batch())
			if err != nil {
				t.Fatal(err)
			}
			// Acknowledged rather than retried: the artifact will not become
			// correct on redelivery.
			if err := c.handleBatch(context.Background(), env); err != nil {
				t.Errorf("an unverifiable batch was rejected for redelivery: %v", err)
			}

			if after := c.Progress(); after.Positions != before.Positions {
				t.Errorf("an unverifiable batch was counted: %d positions became %d",
					before.Positions, after.Positions)
			}
		})
	}
}

// TestCoordinatorIgnoresOtherGenerations checks that a straggler from a
// finished generation does not inflate the current one, which would mix data
// generated by two different networks into one dataset.
func TestCoordinatorIgnoresOtherGenerations(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	cfg := testConfig()
	cfg.Generation = 3

	c, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	stale := putBatch(t, s, "g2-u000000", 2, 100)
	env, err := bus.NewEnvelope(events.SubjectGamesProduced, "test", stale)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.handleBatch(context.Background(), env); err != nil {
		t.Fatal(err)
	}

	if got := c.Progress(); got.Positions != 0 {
		t.Errorf("a batch from generation 2 added %d positions to generation 3", got.Positions)
	}
}

// TestUnitsAreReproducibleAndDistinct covers the seed derivation. A generation
// has to be reproducible from its config alone, and every unit within it has to
// differ — otherwise the cluster plays the same games many times over.
func TestUnitsAreReproducibleAndDistinct(t *testing.T) {
	b, s := bus.NewMemory("test"), testStore(t)

	build := func() *Coordinator {
		c, err := New(testConfig(), b, s, nil)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}

	first, second := build(), build()

	seeds := make(map[uint64]bool)
	ids := make(map[string]bool)

	for i := 0; i < 200; i++ {
		a, bUnit := first.unit(i), second.unit(i)
		if a.ID != bUnit.ID || a.Seed != bUnit.Seed || a.Generation != bUnit.Generation ||
			a.Games != bUnit.Games || a.NodesPerMove != bUnit.NodesPerMove || a.NetworkURI != bUnit.NetworkURI {
			t.Fatalf("unit %d differs between coordinators: %+v vs %+v", i, a, bUnit)
		}
		if seeds[a.Seed] {
			t.Errorf("unit %d reuses seed %d", i, a.Seed)
		}
		if ids[a.ID] {
			t.Errorf("unit %d reuses ID %q", i, a.ID)
		}
		seeds[a.Seed], ids[a.ID] = true, true
	}

	// A different generation must not repeat the previous one's games.
	cfg := testConfig()
	cfg.Generation = 1
	next, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 200; i++ {
		if seeds[next.unit(i).Seed] {
			t.Errorf("generation 1 unit %d reuses a seed from generation 0", i)
		}
	}
}

// TestUnitsCarryTheNetworkAndNodeBudget checks the two fields a unit must never
// be missing: which network to play with, and a node count rather than a time
// limit.
func TestUnitsCarryTheNetworkAndNodeBudget(t *testing.T) {
	cfg := testConfig()
	cfg.NetworkURI = "file:///data/networks/gen3.nnue"

	c, err := New(cfg, bus.NewMemory("test"), testStore(t), nil)
	if err != nil {
		t.Fatal(err)
	}

	unit := c.unit(0)
	if unit.NetworkURI != cfg.NetworkURI {
		t.Errorf("unit names network %q, want %q", unit.NetworkURI, cfg.NetworkURI)
	}
	if unit.NodesPerMove != cfg.NodesPerMove {
		t.Errorf("unit has %d nodes per move, want %d", unit.NodesPerMove, cfg.NodesPerMove)
	}
	if unit.Games != cfg.GamesPerUnit {
		t.Errorf("unit asks for %d games, want %d", unit.Games, cfg.GamesPerUnit)
	}
}

// TestCoordinatorKeepsTheQueueStocked checks that work is issued ahead of
// demand, so a worker never waits on the coordinator — but not all at once,
// since a generation that is abandoned should not leave thousands of queued
// units to drain.
func TestCoordinatorKeepsTheQueueStocked(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	units := collect(t, b, events.SubjectWorkAssign)

	cfg := testConfig()
	cfg.UnitsInFlight = 6

	c, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := c.topUp(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := len(units()); got != cfg.UnitsInFlight {
		t.Errorf("%d units queued, want %d", got, cfg.UnitsInFlight)
	}

	// Topping up again with nothing completed must not queue more.
	if err := c.topUp(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := len(units()); got != cfg.UnitsInFlight {
		t.Errorf("%d units queued after a second top-up, want %d", got, cfg.UnitsInFlight)
	}

	// One batch comes back; one more unit should go out.
	env, err := bus.NewEnvelope(events.SubjectGamesProduced, "test", putBatch(t, s, "g0-u000000", 0, 10))
	if err != nil {
		t.Fatal(err)
	}
	if err := c.handleBatch(context.Background(), env); err != nil {
		t.Fatal(err)
	}
	if err := c.topUp(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got, want := len(units()), cfg.UnitsInFlight+1; got != want {
		t.Errorf("%d units queued after one completed, want %d", got, want)
	}
}

func TestCoordinatorRejectsBadConfig(t *testing.T) {
	b, s := bus.NewMemory("test"), testStore(t)

	for _, tc := range []struct {
		name   string
		mutate func(*Config)
	}{
		{"negative generation", func(c *Config) { c.Generation = -1 }},
		{"no target", func(c *Config) { c.PositionsPerGeneration = 0 }},
		{"no games per unit", func(c *Config) { c.GamesPerUnit = 0 }},
		{"no node budget", func(c *Config) { c.NodesPerMove = 0 }},
		{"nothing in flight", func(c *Config) { c.UnitsInFlight = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig()
			tc.mutate(&cfg)
			if _, err := New(cfg, b, s, nil); err == nil {
				t.Error("accepted a configuration it should have refused")
			}
		})
	}

	if _, err := New(testConfig(), nil, s, nil); err == nil {
		t.Error("accepted a nil bus")
	}
	if _, err := New(testConfig(), b, nil, nil); err == nil {
		t.Error("accepted a nil store")
	}
}

func TestCoordinatorHonoursContextCancellation(t *testing.T) {
	c, err := New(testConfig(), bus.NewMemory("test"), testStore(t), nil)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := c.RunGeneration(ctx); err == nil {
		t.Error("the generation ignored a cancelled context")
	}
}

// TestGenerationDoesNotStallOnRejectedBatches is the accounting bug this test
// was written to catch.
//
// Outstanding work was computed as issued minus counted batches. A unit whose
// batch fails verification is never counted, so it stayed "outstanding"
// forever, and after UnitsInFlight such units the coordinator stopped issuing
// work entirely. The generation would then hang with idle workers, no error
// anywhere, and a target it could never reach.
func TestGenerationDoesNotStallOnRejectedBatches(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	units := collect(t, b, events.SubjectWorkAssign)

	cfg := testConfig()
	cfg.UnitsInFlight = 3

	c, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := c.topUp(context.Background()); err != nil {
		t.Fatal(err)
	}
	issuedFirst := len(units())

	// More rejected batches than the in-flight limit, so a stall would be
	// certain rather than possible.
	for i := 0; i < cfg.UnitsInFlight+2; i++ {
		batch := putBatch(t, s, fmt.Sprintf("g0-bad%d", i), 0, 100)
		// Corrupt beyond repair: the artifact is gone.
		if err := s.Delete(context.Background(), batch.ArtifactURI); err != nil {
			t.Fatal(err)
		}

		env, err := bus.NewEnvelope(events.SubjectGamesProduced, "test", batch)
		if err != nil {
			t.Fatal(err)
		}
		if err := c.handleBatch(context.Background(), env); err != nil {
			t.Fatal(err)
		}
	}

	if err := c.topUp(context.Background()); err != nil {
		t.Fatal(err)
	}

	if got := len(units()); got <= issuedFirst {
		t.Errorf("after %d rejected batches the coordinator issued no more work (%d units total); the generation would hang",
			cfg.UnitsInFlight+2, got)
	}
}

// TestGenerationDoesNotStallOnEmptyUnits covers the other way a unit finishes
// without contributing. A unit whose games all ended in the random opening
// produces no positions; if that left nothing on the bus, the coordinator would
// wait for an announcement that never came.
func TestGenerationDoesNotStallOnEmptyUnits(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	units := collect(t, b, events.SubjectWorkAssign)

	cfg := testConfig()
	cfg.UnitsInFlight = 3

	c, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.topUp(context.Background()); err != nil {
		t.Fatal(err)
	}
	issuedFirst := len(units())

	for i := 0; i < cfg.UnitsInFlight+2; i++ {
		empty := events.GameBatch{
			WorkUnitID: fmt.Sprintf("g0-empty%d", i),
			WorkerID:   "worker-1",
			Generation: 0,
			Positions:  0,
		}
		env, err := bus.NewEnvelope(events.SubjectGamesProduced, "test", empty)
		if err != nil {
			t.Fatal(err)
		}
		if err := c.handleBatch(context.Background(), env); err != nil {
			t.Fatal(err)
		}
	}

	if err := c.topUp(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := len(units()); got <= issuedFirst {
		t.Errorf("after %d empty units the coordinator issued no more work (%d units total)",
			cfg.UnitsInFlight+2, got)
	}
}
