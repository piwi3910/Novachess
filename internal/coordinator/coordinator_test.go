package coordinator

import (
	"bytes"
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/events"
	"github.com/piwi3910/novachess/internal/store"
	"github.com/piwi3910/novachess/internal/train"
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

	batch, err := putBatchErr(s, unitID, generation, positions)
	if err != nil {
		t.Fatal(err)
	}
	return batch
}

// putBatchErr is putBatch's error-returning twin, for use from goroutines
// other than the test's own: t.Fatal/t.FailNow must only ever be called on
// the goroutine running the test function, so code running on a spawned
// goroutine reports failure by returning an error instead and lets the test
// goroutine call t.Fatal after joining.
func putBatchErr(s store.Store, unitID string, generation, positions int) (events.GameBatch, error) {
	artifact, err := s.Put(context.Background(),
		fmt.Sprintf("gen%d/%s.novadata", generation, unitID),
		strings.NewReader(strings.Repeat("x", 64)+unitID))
	if err != nil {
		return events.GameBatch{}, err
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
	}, nil
}

// publishBatches runs on its own goroutine, publishing synthetic batches
// until the generation reaches its target or the context is cancelled. It
// never calls anything on *testing.T: t.Fatal/t.FailNow are only safe on the
// goroutine that runs the test function, so any error is instead sent back
// on the returned channel for the test goroutine to report after joining via
// the returned WaitGroup.
func publishBatches(ctx context.Context, wg *sync.WaitGroup, b bus.Bus, s store.Store, c *Coordinator, idFor func(i int) string, generation, positions, target int) <-chan error {
	errc := make(chan error, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(errc)
		for i := 0; ; i++ {
			select {
			case <-ctx.Done():
				return
			default:
			}

			batch, err := putBatchErr(s, idFor(i), generation, positions)
			if err != nil {
				errc <- err
				return
			}
			if err := b.Publish(context.Background(), events.SubjectGamesProduced, batch); err != nil {
				return
			}
			if c.Progress().Positions >= target {
				return
			}
		}
	}()
	return errc
}

// putValidArtifact writes a real .novadata artifact with n samples, the way
// a worker's own write actually looks on disk. putBatch's fixture data is not
// train-format — it exists only to exercise store.Verify's checksum — and
// resume needs something train.NewReader will genuinely decode.
func putValidArtifact(t *testing.T, s store.Store, generation int, unitID string, n int) string {
	t.Helper()

	var buf bytes.Buffer
	w, err := train.NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}

	pos := chess.NewPosition()
	for i := 0; i < n; i++ {
		if err := w.Write(train.Sample{Position: pos, Score: 10, Result: train.Drawn, Ply: uint16(i)}); err != nil {
			t.Fatal(err)
		}
	}

	key := fmt.Sprintf("gen%d/%s%s", generation, unitID, train.FileExtension)
	artifact, err := s.Put(context.Background(), key, &buf)
	if err != nil {
		t.Fatal(err)
	}
	return artifact.URI
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

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Feed batches in from another goroutine, as workers would. The
	// WaitGroup is joined below, and the error channel drained, before this
	// test can return — a goroutine that outlived the test previously called
	// t.Fatal after completion and panicked the whole binary.
	var wg sync.WaitGroup
	errc := publishBatches(ctx, &wg, b, s, c, func(i int) string {
		return fmt.Sprintf("g0-u%06d", i)
	}, 0, 100, testConfig().PositionsPerGeneration)

	progress, err := c.RunGeneration(ctx)
	if err != nil {
		t.Fatal(err)
	}

	cancel()
	wg.Wait()
	if err := <-errc; err != nil {
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

// TestCoordinatorResumesFromExistingArtifacts is the replay tax this task
// exists to remove: a restart must not reissue work whose artifact is
// already on disk. Units 0..2 are pre-populated as a previous run would have
// left them; a fresh coordinator must count their positions without ever
// putting their IDs back on the bus.
func TestCoordinatorResumesFromExistingArtifacts(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)
	cfg := testConfig()

	// A throwaway coordinator only to compute the canonical IDs a fresh run
	// would assign to indices 0..2 — the same derivation resume must match.
	ids, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	const preUnits = 3
	const perUnit = 60
	preIDs := make(map[string]bool, preUnits)
	for i := 0; i < preUnits; i++ {
		id := ids.unit(i).ID
		preIDs[id] = true
		putValidArtifact(t, s, cfg.Generation, id, perUnit)
	}

	units := collect(t, b, events.SubjectWorkAssign)
	datasets := collect(t, b, events.SubjectDatasetReady)

	c, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Feed batches in from another goroutine, as workers would, using IDs
	// that cannot collide with the pre-existing ones. The WaitGroup is
	// joined below, and the error channel drained, before this test can
	// return — a goroutine that outlived the test previously called t.Fatal
	// after completion and panicked the whole binary.
	var wg sync.WaitGroup
	errc := publishBatches(ctx, &wg, b, s, c, func(i int) string {
		return fmt.Sprintf("g%d-live%06d", cfg.Generation, i+100)
	}, cfg.Generation, 50, cfg.PositionsPerGeneration)

	progress, err := c.RunGeneration(ctx)
	if err != nil {
		t.Fatal(err)
	}

	cancel()
	wg.Wait()
	if err := <-errc; err != nil {
		t.Fatal(err)
	}

	if !progress.Complete {
		t.Error("the resumed generation did not complete")
	}
	if progress.Positions < cfg.PositionsPerGeneration {
		t.Errorf("finished with %d positions, the target was %d", progress.Positions, cfg.PositionsPerGeneration)
	}
	if progress.Positions < preUnits*perUnit {
		t.Errorf("the resumed total of %d positions does not include the %d pre-existing ones",
			progress.Positions, preUnits*perUnit)
	}

	for _, env := range units() {
		var u events.WorkUnit
		if err := bus.Unmarshal(env, &u); err != nil {
			t.Fatal(err)
		}
		if preIDs[u.ID] {
			t.Errorf("unit %s was reissued despite its artifact already being on disk", u.ID)
		}
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
}

// TestCoordinatorReissuesATruncatedArtifact is the safety valve on resume: a
// file that cannot be read back in full must not be trusted, because a
// truncated artifact left by an evicted worker looks, at a glance, exactly
// like a real one.
func TestCoordinatorReissuesATruncatedArtifact(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)
	cfg := testConfig()

	ids, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}
	unitID := ids.unit(0).ID

	var buf bytes.Buffer
	w, err := train.NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Write(train.Sample{Position: chess.NewPosition(), Score: 1, Result: train.Drawn}); err != nil {
		t.Fatal(err)
	}
	truncated := buf.Bytes()[:buf.Len()-5] // cut off mid-record

	key := fmt.Sprintf("gen%d/%s%s", cfg.Generation, unitID, train.FileExtension)
	if _, err := s.Put(context.Background(), key, bytes.NewReader(truncated)); err != nil {
		t.Fatal(err)
	}

	units := collect(t, b, events.SubjectWorkAssign)

	c, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := c.resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := c.Progress(); got.Positions != 0 || got.Batches != 0 {
		t.Errorf("a truncated artifact counted %d positions across %d batches, want 0 across 0",
			got.Positions, got.Batches)
	}

	if err := c.topUp(context.Background()); err != nil {
		t.Fatal(err)
	}

	found := false
	for _, env := range units() {
		var u events.WorkUnit
		if err := bus.Unmarshal(env, &u); err != nil {
			t.Fatal(err)
		}
		if u.ID == unitID {
			found = true
		}
	}
	if !found {
		t.Errorf("the truncated unit %s was not reissued", unitID)
	}
}

// TestCoordinatorAnnouncesWithoutIssuingWhenAlreadyMet covers the case where
// a restart finds a generation that had, in fact, already finished: the
// dataset announces immediately and nothing new goes out.
func TestCoordinatorAnnouncesWithoutIssuingWhenAlreadyMet(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)
	cfg := testConfig()

	ids, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	perUnit := cfg.PositionsPerGeneration / 3
	for i := 0; i < 3; i++ {
		putValidArtifact(t, s, cfg.Generation, ids.unit(i).ID, perUnit)
	}

	units := collect(t, b, events.SubjectWorkAssign)
	datasets := collect(t, b, events.SubjectDatasetReady)

	c, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	progress, err := c.RunGeneration(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if !progress.Complete {
		t.Error("a generation already met by existing artifacts did not complete")
	}
	if progress.Positions < cfg.PositionsPerGeneration {
		t.Errorf("finished with %d positions, the target was %d", progress.Positions, cfg.PositionsPerGeneration)
	}
	if got := len(units()); got != 0 {
		t.Errorf("%d units were issued when existing artifacts already met the target", got)
	}

	announced := datasets()
	if len(announced) != 1 {
		t.Fatalf("%d datasets announced, want 1", len(announced))
	}
}

// TestCoordinatorIgnoresLiveBatchesForResumedUnits keeps the at-least-once
// dedup honest across a restart: a redelivered batch for a unit resume
// already counted from disk must be ignored exactly like any other
// duplicate, not counted a second time.
func TestCoordinatorIgnoresLiveBatchesForResumedUnits(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)
	cfg := testConfig()

	ids, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}
	unitID := ids.unit(0).ID
	putValidArtifact(t, s, cfg.Generation, unitID, 100)

	c, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.resume(context.Background()); err != nil {
		t.Fatal(err)
	}

	before := c.Progress()

	// A redelivery of the same unit, as at-least-once delivery guarantees can
	// produce for a message sent before the restart.
	batch := putBatch(t, s, unitID, cfg.Generation, 100)
	env, err := bus.NewEnvelope(events.SubjectGamesProduced, "test", batch)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.handleBatch(context.Background(), env); err != nil {
		t.Fatal(err)
	}

	after := c.Progress()
	if after.Positions != before.Positions || after.Batches != before.Batches {
		t.Errorf("a live batch for a resumed unit was counted again: %d positions/%d batches became %d/%d",
			before.Positions, before.Batches, after.Positions, after.Batches)
	}
}

// TestCoordinatorResumeScanIsConcurrentAndDeterministic covers the point of
// this change: resume must scan a large artifact set concurrently, count
// every one of them correctly, and still land on the exact same resumed
// totals and the same (sorted) artifact list every time, regardless of which
// goroutine in the worker pool happens to finish first. A resume that raced
// its own bookkeeping would show up here as a flaky total or a shuffled
// ArtifactURIs list, not as a crash.
func TestCoordinatorResumeScanIsConcurrentAndDeterministic(t *testing.T) {
	const numArtifacts = 200
	const perUnit = 3

	b := bus.NewMemory("test")
	s := testStore(t)
	cfg := testConfig()

	ids, err := New(cfg, b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	wantURIs := make([]string, numArtifacts)
	for i := 0; i < numArtifacts; i++ {
		wantURIs[i] = putValidArtifact(t, s, cfg.Generation, ids.unit(i).ID, perUnit)
	}
	sort.Strings(wantURIs)

	// A couple of files that do not match this generation's unit naming, to
	// confirm the concurrent scan still leaves them alone.
	if _, err := s.Put(context.Background(),
		fmt.Sprintf("gen%d/not-a-unit.txt", cfg.Generation),
		strings.NewReader("not a training artifact")); err != nil {
		t.Fatal(err)
	}

	runResume := func() (int, int, []string) {
		t.Helper()
		c, err := New(cfg, b, s, nil)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := c.resume(ctx); err != nil {
			t.Fatal(err)
		}
		progress := c.Progress()
		// Deliberately not re-sorted here: resume must already hand back a
		// deterministic, sorted list on its own, independent of which
		// goroutine in the scan's worker pool happened to finish first.
		artifacts := append([]string(nil), c.artifacts...)
		return progress.Positions, progress.Batches, artifacts
	}

	wantPositions := numArtifacts * perUnit

	for run := 0; run < 3; run++ {
		positions, batches, artifacts := runResume()
		if positions != wantPositions {
			t.Errorf("run %d: resumed %d positions, want %d", run, positions, wantPositions)
		}
		if batches != numArtifacts {
			t.Errorf("run %d: resumed %d artifacts, want %d", run, batches, numArtifacts)
		}
		if len(artifacts) != len(wantURIs) {
			t.Fatalf("run %d: resumed %d artifact URIs, want %d", run, len(artifacts), len(wantURIs))
		}
		for i := range artifacts {
			if artifacts[i] != wantURIs[i] {
				t.Errorf("run %d: artifact %d is %q, want %q", run, i, artifacts[i], wantURIs[i])
			}
		}
	}
}
