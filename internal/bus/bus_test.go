package bus

import (
	"context"
	"testing"

	"github.com/piwi3910/novachess/internal/events"
)

// TestQueueSubscribeDeliversToOneMember is the property the whole work
// distribution scheme rests on: a work unit published once must be played once,
// not once per worker pod. If this broke, every worker in the cluster would
// generate identical games and the training data would collapse to a fraction
// of the diversity it should have — while every dashboard still showed full
// throughput.
func TestQueueSubscribeDeliversToOneMember(t *testing.T) {
	ctx := context.Background()
	b := NewMemory("test")
	defer b.Close()

	const workers = 4
	received := make([]int, workers)

	for i := 0; i < workers; i++ {
		i := i
		_, err := b.QueueSubscribe(ctx, events.SubjectWorkAssign, "selfplay",
			func(context.Context, events.Envelope) error {
				received[i]++
				return nil
			})
		if err != nil {
			t.Fatalf("QueueSubscribe: %v", err)
		}
	}

	const units = 12
	for i := 0; i < units; i++ {
		if err := b.Publish(ctx, events.SubjectWorkAssign, events.WorkUnit{ID: "u"}); err != nil {
			t.Fatalf("Publish: %v", err)
		}
	}

	total := 0
	for _, n := range received {
		total += n
	}
	if total != units {
		t.Errorf("group received %d messages in total, want %d (each unit exactly once)", total, units)
	}
	// Every worker should have been given some share; a group that always
	// picked the same member would still total correctly.
	for i, n := range received {
		if n == 0 {
			t.Errorf("worker %d received nothing; work is not being spread", i)
		}
	}
}

// TestSubscribeFansOutToEveryone is the complementary property. A promoted
// network must reach every worker and the bot, so this subject must not behave
// like a queue group.
func TestSubscribeFansOutToEveryone(t *testing.T) {
	ctx := context.Background()
	b := NewMemory("test")
	defer b.Close()

	const consumers = 3
	seen := 0

	for i := 0; i < consumers; i++ {
		_, err := b.Subscribe(ctx, events.SubjectNetPromoted,
			func(context.Context, events.Envelope) error {
				seen++
				return nil
			})
		if err != nil {
			t.Fatalf("Subscribe: %v", err)
		}
	}

	if err := b.Publish(ctx, events.SubjectNetPromoted, events.NetworkVerdict{Promoted: true}); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	if seen != consumers {
		t.Errorf("%d consumers saw the promotion, want %d", seen, consumers)
	}
}

func TestEnvelopeRoundTrip(t *testing.T) {
	ctx := context.Background()
	b := NewMemory("coordinator-0")
	defer b.Close()

	want := events.WorkUnit{
		ID:           "unit-42",
		Generation:   7,
		Games:        128,
		NodesPerMove: 5000,
		RandomPlies:  8,
		Seed:         0xDEADBEEF,
		OpeningFENs:  []string{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
	}

	var got events.WorkUnit
	var env events.Envelope
	_, err := b.Subscribe(ctx, events.SubjectWorkAssign, func(_ context.Context, e events.Envelope) error {
		env = e
		return Unmarshal(e, &got)
	})
	if err != nil {
		t.Fatalf("Subscribe: %v", err)
	}

	if err := b.Publish(ctx, events.SubjectWorkAssign, want); err != nil {
		t.Fatalf("Publish: %v", err)
	}

	if env.Type != events.SubjectWorkAssign {
		t.Errorf("envelope type = %q, want %q", env.Type, events.SubjectWorkAssign)
	}
	if env.Producer != "coordinator-0" {
		t.Errorf("envelope producer = %q, want %q", env.Producer, "coordinator-0")
	}
	if env.SchemaVersion != events.SchemaVersion {
		t.Errorf("envelope schema version = %d, want %d", env.SchemaVersion, events.SchemaVersion)
	}
	if env.ID == "" {
		t.Error("envelope has no ID; at-least-once consumers cannot deduplicate")
	}

	if got.ID != want.ID || got.Generation != want.Generation ||
		got.NodesPerMove != want.NodesPerMove || got.Seed != want.Seed ||
		len(got.OpeningFENs) != len(want.OpeningFENs) {
		t.Errorf("round trip: got %+v, want %+v", got, want)
	}
}

// TestUnmarshalRejectsNewerSchema checks that a consumer refuses a message it
// may not fully understand. During a rolling update, producers and consumers
// run different builds; decoding a newer message on a best-effort basis would
// silently drop unknown fields, which for a work unit means generating the
// wrong games while reporting success.
func TestUnmarshalRejectsNewerSchema(t *testing.T) {
	env := events.Envelope{
		ID:            "abc",
		Type:          events.SubjectWorkAssign,
		SchemaVersion: events.SchemaVersion + 1,
		Payload:       []byte(`{"id":"unit-1"}`),
	}
	var unit events.WorkUnit
	if err := Unmarshal(env, &unit); err == nil {
		t.Error("Unmarshal accepted a newer schema version, want error")
	}
}

func TestSubscriptionCloseStopsDelivery(t *testing.T) {
	ctx := context.Background()
	b := NewMemory("test")
	defer b.Close()

	count := 0
	sub, err := b.Subscribe(ctx, events.SubjectGamesProduced, func(context.Context, events.Envelope) error {
		count++
		return nil
	})
	if err != nil {
		t.Fatalf("Subscribe: %v", err)
	}

	if err := b.Publish(ctx, events.SubjectGamesProduced, events.GameBatch{}); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	if err := sub.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := b.Publish(ctx, events.SubjectGamesProduced, events.GameBatch{}); err != nil {
		t.Fatalf("Publish: %v", err)
	}

	if count != 1 {
		t.Errorf("handler ran %d times, want 1 (delivery should stop after Close)", count)
	}
}

func TestPublishAfterCloseFails(t *testing.T) {
	b := NewMemory("test")
	if err := b.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := b.Publish(context.Background(), events.SubjectWorkAssign, events.WorkUnit{}); err != ErrClosed {
		t.Errorf("Publish after Close = %v, want ErrClosed", err)
	}
}
