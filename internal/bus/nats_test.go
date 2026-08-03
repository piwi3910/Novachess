package bus

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	natsserver "github.com/nats-io/nats-server/v2/server"
	"github.com/nats-io/nats.go/jetstream"

	"github.com/piwi3910/novachess/internal/events"
)

// runBroker starts a real NATS server with JetStream, in process.
//
// Testing against the real broker rather than a mock is the point. Everything
// this implementation gets wrong is in the parts a mock would have to
// reimplement: whether a queue group really gives a unit to one member,
// whether a negative acknowledgement really redelivers, whether a durable
// consumer really resumes where it left off. A fake that answered those
// questions would be asserting its own behaviour.
func runBroker(t *testing.T) string {
	t.Helper()

	opts := &natsserver.Options{
		Host:      "127.0.0.1",
		Port:      -1, // any free port
		JetStream: true,
		StoreDir:  t.TempDir(),
		NoLog:     true,
		NoSigs:    true,
	}

	server, err := natsserver.NewServer(opts)
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

func connect(t *testing.T, url, name string) *NATS {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	b, err := Connect(ctx, Config{
		URLs:        []string{url},
		ClientName:  name,
		MaxInFlight: 4,
	})
	if err != nil {
		t.Fatalf("connecting: %v", err)
	}
	t.Cleanup(func() { b.Close() })
	return b
}

// waitFor polls until a condition holds, so tests assert on effects rather
// than on sleeps. A fixed sleep would be both slower and less reliable: this
// container's timings vary by more than fourfold run to run.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestNATSPublishAndSubscribe(t *testing.T) {
	b := connect(t, runBroker(t), "test")
	ctx := context.Background()

	var got atomic.Int32
	var mu sync.Mutex
	var received events.WorkUnit

	sub, err := b.Subscribe(ctx, events.SubjectWorkAssign, func(_ context.Context, env events.Envelope) error {
		var unit events.WorkUnit
		if err := Unmarshal(env, &unit); err != nil {
			return err
		}
		mu.Lock()
		received = unit
		mu.Unlock()
		got.Add(1)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Close()

	want := events.WorkUnit{ID: "unit-1", Generation: 3, Games: 10, NodesPerMove: 5000, Seed: 42}
	if err := b.Publish(ctx, events.SubjectWorkAssign, want); err != nil {
		t.Fatal(err)
	}

	waitFor(t, "the message to arrive", func() bool { return got.Load() == 1 })

	mu.Lock()
	defer mu.Unlock()
	if received.ID != want.ID || received.Generation != want.Generation || received.Seed != want.Seed {
		t.Errorf("received %+v, want %+v", received, want)
	}
}

// TestNATSQueueGroupDeliversEachUnitOnce is the property work distribution
// rests on. Every self-play worker joins one group, and a unit played twice is
// a board's time wasted on data the pipeline already has.
func TestNATSQueueGroupDeliversEachUnitOnce(t *testing.T) {
	url := runBroker(t)
	b := connect(t, url, "publisher")
	ctx := context.Background()

	const workers = 4
	const units = 12

	var total atomic.Int32
	seen := make([]atomic.Int32, workers)

	for i := 0; i < workers; i++ {
		worker := i
		sub, err := b.QueueSubscribe(ctx, events.SubjectWorkAssign, "selfplay", func(context.Context, events.Envelope) error {
			seen[worker].Add(1)
			total.Add(1)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		defer sub.Close()
	}

	for i := 0; i < units; i++ {
		if err := b.Publish(ctx, events.SubjectWorkAssign, events.WorkUnit{ID: fmt.Sprintf("unit-%d", i)}); err != nil {
			t.Fatal(err)
		}
	}

	waitFor(t, "every unit to be delivered", func() bool { return total.Load() >= units })

	// Give any duplicate a chance to show up before concluding there is none.
	time.Sleep(300 * time.Millisecond)
	if got := total.Load(); got != units {
		t.Errorf("%d units were published but %d deliveries happened", units, got)
	}

	counts := make([]int32, workers)
	for i := range seen {
		counts[i] = seen[i].Load()
	}
	t.Logf("%d units spread across workers as %v", units, counts)
}

// TestNATSFanOutReachesEverySubscriber is the other half. A promoted network
// must reach every worker and the bot, not one of them.
func TestNATSFanOutReachesEverySubscriber(t *testing.T) {
	b := connect(t, runBroker(t), "test")
	ctx := context.Background()

	const subscribers = 3
	var counts [subscribers]atomic.Int32

	for i := 0; i < subscribers; i++ {
		which := i
		sub, err := b.Subscribe(ctx, events.SubjectNetPromoted, func(context.Context, events.Envelope) error {
			counts[which].Add(1)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		defer sub.Close()
	}

	if err := b.Publish(ctx, events.SubjectNetPromoted, events.NetworkVerdict{Generation: 2, Promoted: true}); err != nil {
		t.Fatal(err)
	}

	waitFor(t, "every subscriber to see the message", func() bool {
		for i := range counts {
			if counts[i].Load() == 0 {
				return false
			}
		}
		return true
	})
}

// TestNATSRedeliversRejectedWork is the guarantee that makes an evicted pod
// survivable. A handler returning an error must put the unit back for someone
// else, which is the at-least-once behaviour every handler is required to
// tolerate.
func TestNATSRedeliversRejectedWork(t *testing.T) {
	b := connect(t, runBroker(t), "test")
	ctx := context.Background()

	var attempts atomic.Int32
	var succeeded atomic.Bool

	sub, err := b.QueueSubscribe(ctx, events.SubjectWorkAssign, "selfplay", func(context.Context, events.Envelope) error {
		// Fail the first delivery, as a worker dying part way through would.
		if attempts.Add(1) == 1 {
			return fmt.Errorf("the worker went away")
		}
		succeeded.Store(true)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Close()

	if err := b.Publish(ctx, events.SubjectWorkAssign, events.WorkUnit{ID: "redelivered"}); err != nil {
		t.Fatal(err)
	}

	waitFor(t, "the rejected unit to be redelivered", func() bool { return succeeded.Load() })
	if got := attempts.Load(); got < 2 {
		t.Errorf("the unit was delivered %d times; a rejection must lead to another attempt", got)
	}
}

// TestNATSWorkSurvivesTheSubscriberStarting checks that a unit published
// before any worker exists is still waiting when one arrives. Without this the
// coordinator would have to be started after every worker, and a rolling
// restart would silently drop the units in flight.
func TestNATSWorkSurvivesTheSubscriberStarting(t *testing.T) {
	b := connect(t, runBroker(t), "test")
	ctx := context.Background()

	if err := b.Publish(ctx, events.SubjectWorkAssign, events.WorkUnit{ID: "published-first"}); err != nil {
		t.Fatal(err)
	}

	var got atomic.Int32
	sub, err := b.QueueSubscribe(ctx, events.SubjectWorkAssign, "selfplay", func(context.Context, events.Envelope) error {
		got.Add(1)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Close()

	waitFor(t, "the stored unit to be delivered", func() bool { return got.Load() == 1 })
}

// TestNATSDurableConsumerResumes checks that a worker restarting does not
// replay work its group already finished, which is what the durable name is
// for. The bus reconnects rather than the process, but the consumer state
// lives on the server either way.
func TestNATSDurableConsumerResumes(t *testing.T) {
	url := runBroker(t)
	ctx := context.Background()

	first := connect(t, url, "worker-1")

	var firstCount atomic.Int32
	sub, err := first.QueueSubscribe(ctx, events.SubjectWorkAssign, "selfplay", func(context.Context, events.Envelope) error {
		firstCount.Add(1)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := first.Publish(ctx, events.SubjectWorkAssign, events.WorkUnit{ID: "before-restart"}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the first unit", func() bool { return firstCount.Load() == 1 })

	// The worker goes away, its unit already acknowledged.
	sub.Close()
	first.Close()

	second := connect(t, url, "worker-2")
	var secondCount atomic.Int32
	sub2, err := second.QueueSubscribe(ctx, events.SubjectWorkAssign, "selfplay", func(context.Context, events.Envelope) error {
		secondCount.Add(1)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sub2.Close()

	if err := second.Publish(ctx, events.SubjectWorkAssign, events.WorkUnit{ID: "after-restart"}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the second unit", func() bool { return secondCount.Load() == 1 })

	// Long enough for a replay of the finished unit to show up if it were
	// going to.
	time.Sleep(300 * time.Millisecond)
	if got := secondCount.Load(); got != 1 {
		t.Errorf("the restarted worker saw %d units; it should not replay work the group finished", got)
	}
}

// TestNATSHeartbeatsAreNotStored checks the routing decision. Heartbeats are
// worth nothing a second later, and storing them would fill the stream with
// the one message type nobody ever replays.
func TestNATSHeartbeatsAreNotStored(t *testing.T) {
	b := connect(t, runBroker(t), "test")
	ctx := context.Background()

	if IsDurable(events.SubjectWorkerHeartbeat) {
		t.Fatal("heartbeats are marked durable")
	}
	for _, subject := range DurableSubjects {
		if !IsDurable(subject) {
			t.Errorf("%s is in DurableSubjects but IsDurable says otherwise", subject)
		}
	}

	var got atomic.Int32
	sub, err := b.Subscribe(ctx, events.SubjectWorkerHeartbeat, func(context.Context, events.Envelope) error {
		got.Add(1)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Close()

	if err := b.Publish(ctx, events.SubjectWorkerHeartbeat, events.WorkerHeartbeat{WorkerID: "board-3", NodesPerSecond: 1e6}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the heartbeat to arrive", func() bool { return got.Load() == 1 })

	// Nothing reached either stream.
	for name, stream := range map[string]jetstream.Stream{"work": b.work, "events": b.facts} {
		info, err := stream.Info(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if info.State.Msgs != 0 {
			t.Errorf("the %s stream holds %d messages after only a heartbeat was published", name, info.State.Msgs)
		}
	}
}

// TestNATSPublishIsDurableBeforeItReturns is what a worker depends on when it
// announces a batch and then acknowledges the unit that produced it. If the
// announcement were still in flight, a broker restart in between would lose
// the games entirely.
func TestNATSPublishIsDurableBeforeItReturns(t *testing.T) {
	b := connect(t, runBroker(t), "test")
	ctx := context.Background()

	if err := b.Publish(ctx, events.SubjectGamesProduced, events.GameBatch{WorkUnitID: "unit-9", Positions: 500}); err != nil {
		t.Fatal(err)
	}

	// No subscriber, no waiting: the message must already be in the stream.
	info, err := b.facts.Info(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if info.State.Msgs != 1 {
		t.Errorf("the stream holds %d messages immediately after Publish returned, want 1", info.State.Msgs)
	}
}

func TestNATSSubscriptionCloseStopsDelivery(t *testing.T) {
	b := connect(t, runBroker(t), "test")
	ctx := context.Background()

	var got atomic.Int32
	sub, err := b.Subscribe(ctx, events.SubjectNetPromoted, func(context.Context, events.Envelope) error {
		got.Add(1)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := b.Publish(ctx, events.SubjectNetPromoted, events.NetworkVerdict{Generation: 1}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the first message", func() bool { return got.Load() == 1 })

	if err := sub.Close(); err != nil {
		t.Fatal(err)
	}
	// Closing twice is how a deferred Close behaves alongside an explicit one.
	if err := sub.Close(); err != nil {
		t.Errorf("closing a subscription twice returned %v", err)
	}

	if err := b.Publish(ctx, events.SubjectNetPromoted, events.NetworkVerdict{Generation: 2}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(300 * time.Millisecond)
	if got.Load() != 1 {
		t.Errorf("a closed subscription received %d messages", got.Load())
	}
}

func TestNATSOperationsAfterCloseFail(t *testing.T) {
	b := connect(t, runBroker(t), "test")
	ctx := context.Background()

	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	// Closing twice is safe, so shutdown paths can be unconditional.
	if err := b.Close(); err != nil {
		t.Errorf("closing twice returned %v", err)
	}

	if err := b.Publish(ctx, events.SubjectWorkAssign, events.WorkUnit{}); err != ErrClosed {
		t.Errorf("publishing after close gave %v, want ErrClosed", err)
	}
	if _, err := b.Subscribe(ctx, events.SubjectWorkAssign, func(context.Context, events.Envelope) error { return nil }); err != ErrClosed {
		t.Errorf("subscribing after close gave %v, want ErrClosed", err)
	}
}

func TestNATSRejectsBadArguments(t *testing.T) {
	b := connect(t, runBroker(t), "test")
	ctx := context.Background()

	if _, err := b.QueueSubscribe(ctx, events.SubjectWorkAssign, "", func(context.Context, events.Envelope) error { return nil }); err == nil {
		t.Error("a queue subscription with no group name was accepted")
	}
	if _, err := b.Subscribe(ctx, events.SubjectWorkAssign, nil); err == nil {
		t.Error("a subscription with no handler was accepted")
	}
}

func TestConnectRejectsNoServers(t *testing.T) {
	if _, err := Connect(context.Background(), Config{}); err == nil {
		t.Error("Connect accepted a config with no URLs")
	}
}

// TestNATSMatchesMemoryEnvelopes checks that both implementations put the same
// thing on the wire. The in-memory bus is what most tests run against, so a
// divergence here would mean those tests validate a format production never
// sees.
func TestNATSMatchesMemoryEnvelopes(t *testing.T) {
	b := connect(t, runBroker(t), "conformance")
	ctx := context.Background()

	unit := events.WorkUnit{ID: "unit-5", Generation: 7, Games: 3, NodesPerMove: 1000, Seed: 99}

	var fromNATS events.Envelope
	ready := make(chan struct{})
	sub, err := b.Subscribe(ctx, events.SubjectWorkAssign, func(_ context.Context, env events.Envelope) error {
		fromNATS = env
		close(ready)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Close()

	if err := b.Publish(ctx, events.SubjectWorkAssign, unit); err != nil {
		t.Fatal(err)
	}
	select {
	case <-ready:
	case <-time.After(20 * time.Second):
		t.Fatal("no message arrived")
	}

	memory := NewMemory("conformance")
	var fromMemory events.Envelope
	if _, err := memory.Subscribe(ctx, events.SubjectWorkAssign, func(_ context.Context, env events.Envelope) error {
		fromMemory = env
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := memory.Publish(ctx, events.SubjectWorkAssign, unit); err != nil {
		t.Fatal(err)
	}

	if fromNATS.Type != fromMemory.Type {
		t.Errorf("type differs: %q vs %q", fromNATS.Type, fromMemory.Type)
	}
	if fromNATS.SchemaVersion != fromMemory.SchemaVersion {
		t.Errorf("schema version differs: %d vs %d", fromNATS.SchemaVersion, fromMemory.SchemaVersion)
	}
	if fromNATS.Producer != fromMemory.Producer {
		t.Errorf("producer differs: %q vs %q", fromNATS.Producer, fromMemory.Producer)
	}
	if string(fromNATS.Payload) != string(fromMemory.Payload) {
		t.Errorf("payload differs:\n nats: %s\n mem:  %s", fromNATS.Payload, fromMemory.Payload)
	}

	// Both must decode back to the value that was published.
	var decoded events.WorkUnit
	if err := Unmarshal(fromNATS, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.ID != unit.ID || decoded.Generation != unit.Generation ||
		decoded.Games != unit.Games || decoded.NodesPerMove != unit.NodesPerMove ||
		decoded.Seed != unit.Seed {
		t.Errorf("decoded %+v, want %+v", decoded, unit)
	}
}

// TestBothBusesSatisfyTheInterface is a compile-time check that neither
// implementation has drifted from the contract services depend on.
func TestBothBusesSatisfyTheInterface(t *testing.T) {
	var _ Bus = (*Memory)(nil)
	var _ Bus = (*NATS)(nil)
}
