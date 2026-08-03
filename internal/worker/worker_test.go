package worker

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
	"github.com/piwi3910/novachess/internal/nnue"
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
	return Config{
		ID:            "worker-1",
		NodeName:      "board-3",
		HashMB:        1,
		EngineVersion: "test",
	}
}

func testUnit() events.WorkUnit {
	return events.WorkUnit{
		ID:           "g0-u000001",
		Generation:   0,
		Games:        2,
		RandomPlies:  6,
		NodesPerMove: 800,
		Seed:         0xC0FFEE,
	}
}

// collect subscribes to a subject and returns the envelopes it sees.
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

// TestWorkerPlaysAUnitAndAnnouncesIt is the whole job in one test: a unit
// arrives, games are played, the batch is stored, and the announcement
// describes what was actually written.
func TestWorkerPlaysAUnitAndAnnouncesIt(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	w, err := New(testConfig(), b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	batches := collect(t, b, events.SubjectGamesProduced)

	env, err := bus.NewEnvelope(events.SubjectWorkAssign, "test", testUnit())
	if err != nil {
		t.Fatal(err)
	}
	if err := w.handle(context.Background(), env); err != nil {
		t.Fatal(err)
	}

	announced := batches()
	if len(announced) != 1 {
		t.Fatalf("%d batches announced, want 1", len(announced))
	}

	var batch events.GameBatch
	if err := bus.Unmarshal(announced[0], &batch); err != nil {
		t.Fatal(err)
	}

	if batch.WorkUnitID != testUnit().ID {
		t.Errorf("batch names unit %q, want %q", batch.WorkUnitID, testUnit().ID)
	}
	if batch.WorkerID != "worker-1" {
		t.Errorf("batch names worker %q", batch.WorkerID)
	}
	if batch.Positions == 0 {
		t.Error("the batch reports no positions")
	}
	if batch.NodesPerSecond == 0 {
		t.Error("the batch reports no throughput; a throttled board would be invisible")
	}

	// The announcement must describe the artifact that exists, not what the
	// worker meant to write. A coordinator verifies against exactly these
	// numbers before counting the batch.
	if err := store.Verify(context.Background(), s, store.Artifact{
		URI:      batch.ArtifactURI,
		Size:     batch.SizeBytes,
		Checksum: batch.Checksum,
	}); err != nil {
		t.Errorf("the announced batch does not match the stored artifact: %v", err)
	}

	// And the artifact must be readable as training data.
	rc, err := s.Get(context.Background(), batch.ArtifactURI)
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()

	r, err := train.NewReader(rc)
	if err != nil {
		t.Fatal(err)
	}
	samples, err := r.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != batch.Positions {
		t.Errorf("the artifact holds %d positions, the announcement claims %d", len(samples), batch.Positions)
	}
}

// TestWorkerStoresBeforeAnnouncing pins the ordering that makes an eviction
// survivable. If the announcement came first, a crash in the gap would leave
// the coordinator counting a batch that does not exist.
func TestWorkerStoresBeforeAnnouncing(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	w, err := New(testConfig(), b, s, nil)
	if err != nil {
		t.Fatal(err)
	}

	// The memory bus delivers synchronously inside Publish, so a subscriber
	// sees the announcement at the moment it is made — which is exactly when
	// the artifact must already be readable.
	var checked bool
	if _, err := b.Subscribe(context.Background(), events.SubjectGamesProduced, func(ctx context.Context, env events.Envelope) error {
		var batch events.GameBatch
		if err := bus.Unmarshal(env, &batch); err != nil {
			return err
		}
		checked = true
		if _, err := s.Stat(ctx, batch.ArtifactURI); err != nil {
			t.Errorf("the batch was announced before its artifact existed: %v", err)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	env, err := bus.NewEnvelope(events.SubjectWorkAssign, "test", testUnit())
	if err != nil {
		t.Fatal(err)
	}
	if err := w.handle(context.Background(), env); err != nil {
		t.Fatal(err)
	}
	if !checked {
		t.Fatal("no announcement was seen")
	}
}

// TestWorkerReturnsTheUnitWhenStorageFails checks that a failure to store
// rejects the unit rather than acknowledging it. The unit is reproducible, so
// another worker will produce exactly the same games; silently dropping it
// would leave the generation short with nothing recording why.
func TestWorkerReturnsTheUnitWhenStorageFails(t *testing.T) {
	b := bus.NewMemory("test")

	w, err := New(testConfig(), b, refusingStore{}, nil)
	if err != nil {
		t.Fatal(err)
	}

	batches := collect(t, b, events.SubjectGamesProduced)

	env, err := bus.NewEnvelope(events.SubjectWorkAssign, "test", testUnit())
	if err != nil {
		t.Fatal(err)
	}
	if err := w.handle(context.Background(), env); err == nil {
		t.Error("a unit whose batch could not be stored was acknowledged")
	}
	if got := batches(); len(got) != 0 {
		t.Errorf("%d batches were announced despite storage failing", len(got))
	}
}

// errWriteFailed stands in for a full or unmounted volume.
var errWriteFailed = errors.New("no space left on device")

// refusingStore fails every write.
type refusingStore struct{ store.Store }

func (refusingStore) Put(context.Context, string, io.Reader) (store.Artifact, error) {
	return store.Artifact{}, errWriteFailed
}

// TestWorkerDropsUndecodableWork checks that a message which will never parse
// is acknowledged rather than rejected. Rejecting it would put it back forever
// and block the queue behind it.
func TestWorkerDropsUndecodableWork(t *testing.T) {
	b := bus.NewMemory("test")
	w, err := New(testConfig(), b, testStore(t), nil)
	if err != nil {
		t.Fatal(err)
	}

	env := events.Envelope{
		ID:            "broken",
		Type:          events.SubjectWorkAssign,
		SchemaVersion: events.SchemaVersion,
		Payload:       []byte("not json"),
	}
	if err := w.handle(context.Background(), env); err != nil {
		t.Errorf("an unreadable unit was rejected rather than dropped: %v", err)
	}
}

// TestWorkerLoadsAndCachesANetwork covers the path later generations take, and
// the caching that keeps it from re-reading several hundred kilobytes per unit.
func TestWorkerLoadsAndCachesANetwork(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	var buf bytes.Buffer
	if _, err := nnue.NewZero().WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	artifact, err := s.Put(ctx, "networks/gen1.nnue", &buf)
	if err != nil {
		t.Fatal(err)
	}

	w, err := New(testConfig(), bus.NewMemory("test"), s, nil)
	if err != nil {
		t.Fatal(err)
	}

	first, err := w.evaluator(ctx, artifact.URI)
	if err != nil {
		t.Fatal(err)
	}
	second, err := w.evaluator(ctx, artifact.URI)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Error("the network was loaded twice; a generation is many units against one network")
	}

	// An empty URI is generation zero bootstrapping from the hand-crafted
	// evaluation, which is how the loop starts with no network in existence.
	hce, err := w.evaluator(ctx, "")
	if err != nil {
		t.Fatal(err)
	}
	if hce == nil {
		t.Error("no evaluation was returned for an empty network URI")
	}
}

func TestWorkerRejectsABadNetwork(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	artifact, err := s.Put(ctx, "networks/junk.nnue", strings.NewReader("not a network"))
	if err != nil {
		t.Fatal(err)
	}

	w, err := New(testConfig(), bus.NewMemory("test"), s, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.evaluator(ctx, artifact.URI); err == nil {
		t.Error("a network that is not a network was accepted")
	}
}

func TestWorkerRejectsBadConfig(t *testing.T) {
	b, s := bus.NewMemory("test"), testStore(t)

	for _, tc := range []struct {
		name   string
		mutate func(*Config)
	}{
		{"no ID", func(c *Config) { c.ID = "" }},
		{"no hash", func(c *Config) { c.HashMB = 0 }},
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

// TestWorkerHeartbeats checks that liveness and throughput are reported, since
// a board that has slowed to a crawl is otherwise invisible until it stops
// finishing units entirely.
func TestWorkerHeartbeats(t *testing.T) {
	b := bus.NewMemory("test")

	cfg := testConfig()
	cfg.HeartbeatInterval = 10 * time.Millisecond

	w, err := New(cfg, b, testStore(t), nil)
	if err != nil {
		t.Fatal(err)
	}

	beats := collect(t, b, events.SubjectWorkerHeartbeat)

	ctx, cancel := context.WithCancel(context.Background())
	go w.heartbeat(ctx)

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) && len(beats()) == 0 {
		time.Sleep(2 * time.Millisecond)
	}
	cancel()

	got := beats()
	if len(got) == 0 {
		t.Fatal("no heartbeat was published")
	}

	var beat events.WorkerHeartbeat
	if err := bus.Unmarshal(got[0], &beat); err != nil {
		t.Fatal(err)
	}
	if beat.WorkerID != cfg.ID || beat.NodeName != cfg.NodeName {
		t.Errorf("heartbeat identifies %q on %q, want %q on %q",
			beat.WorkerID, beat.NodeName, cfg.ID, cfg.NodeName)
	}
	if beat.Uptime <= 0 {
		t.Error("heartbeat reports no uptime")
	}
}

// TestWorkerIsIdempotentAcrossRedelivery checks the property at-least-once
// delivery requires. A redelivered unit produces byte-identical data, so
// replaying it overwrites its own artifact rather than adding a second, and the
// coordinator deduplicates the announcement.
func TestWorkerIsIdempotentAcrossRedelivery(t *testing.T) {
	b := bus.NewMemory("test")
	s := testStore(t)

	w, err := New(testConfig(), b, s, nil)
	if err != nil {
		t.Fatal(err)
	}
	batches := collect(t, b, events.SubjectGamesProduced)

	env, err := bus.NewEnvelope(events.SubjectWorkAssign, "test", testUnit())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if err := w.handle(context.Background(), env); err != nil {
			t.Fatal(err)
		}
	}

	got := batches()
	if len(got) != 2 {
		t.Fatalf("%d announcements from two deliveries, want 2", len(got))
	}

	var first, second events.GameBatch
	if err := bus.Unmarshal(got[0], &first); err != nil {
		t.Fatal(err)
	}
	if err := bus.Unmarshal(got[1], &second); err != nil {
		t.Fatal(err)
	}

	if first.ArtifactURI != second.ArtifactURI {
		t.Errorf("a replayed unit wrote a second artifact: %q then %q", first.ArtifactURI, second.ArtifactURI)
	}
	if first.Checksum != second.Checksum {
		t.Errorf("a replayed unit produced different data: %s then %s", first.Checksum, second.Checksum)
	}
	if first.Positions != second.Positions {
		t.Errorf("a replayed unit produced %d then %d positions", first.Positions, second.Positions)
	}
}

// TestWorkerAnnouncesEvenAnEmptyUnit covers the unit that produces nothing.
//
// Staying silent would be the obvious thing and is wrong: the coordinator
// measures outstanding work by what has reported back, so a silent unit counts
// as still in flight forever. Enough of them and the coordinator stops issuing
// work and the generation hangs with idle workers.
func TestWorkerAnnouncesEvenAnEmptyUnit(t *testing.T) {
	b := bus.NewMemory("test")

	w, err := New(testConfig(), b, testStore(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	batches := collect(t, b, events.SubjectGamesProduced)

	// Random plies deep enough that the opening finishes the game before the
	// engine ever moves, so nothing usable is produced.
	unit := testUnit()
	unit.Games = 1
	unit.RandomPlies = 400

	env, err := bus.NewEnvelope(events.SubjectWorkAssign, "test", unit)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.handle(context.Background(), env); err != nil {
		t.Fatal(err)
	}

	got := batches()
	if len(got) != 1 {
		t.Fatalf("%d announcements from a unit that produced nothing, want 1", len(got))
	}

	var batch events.GameBatch
	if err := bus.Unmarshal(got[0], &batch); err != nil {
		t.Fatal(err)
	}
	if batch.WorkUnitID != unit.ID {
		t.Errorf("the announcement names unit %q, want %q", batch.WorkUnitID, unit.ID)
	}
	if batch.Positions != 0 {
		t.Errorf("an empty unit reported %d positions", batch.Positions)
	}
	if batch.ArtifactURI != "" {
		t.Errorf("an empty unit named an artifact %q", batch.ArtifactURI)
	}
}
