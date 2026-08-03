package bus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"

	"github.com/piwi3910/novachess/internal/events"
)

// WorkSubjects carry work to be done exactly once.
//
// These get work-queue retention: a message is removed once a consumer
// acknowledges it, because a played work unit is of no further interest and
// keeping every one would fill the disk with completed work.
var WorkSubjects = []string{
	events.SubjectWorkAssign,
}

// EventSubjects carry facts, which more than one service may care about.
//
// A promoted network has to reach every worker and the bot, so these cannot use
// work-queue retention — a work-queue stream permits only one consumer per
// subject, which is precisely the fan-out this needs. They are kept by age
// instead, long enough to survive a rolling restart and no longer.
var EventSubjects = []string{
	events.SubjectGamesProduced,
	events.SubjectDatasetReady,
	events.SubjectNetCandidate,
	events.SubjectNetPromoted,
	events.SubjectNetRejected,
}

// DurableSubjects is every subject the broker stores.
//
// Everything carrying work or a decision is durable: losing a work assignment
// costs a generation's games, and losing a promotion means the cluster keeps
// playing with a network the gatekeeper already replaced. Heartbeats are
// deliberately absent — they are worth nothing a second later, and storing them
// would fill the stream with the one message type nobody ever replays.
var DurableSubjects = append(append([]string{}, WorkSubjects...), EventSubjects...)

// EventRetention is how long fact messages are kept. Long enough that a service
// restarting during a rollout does not miss a promotion; short enough that the
// stream does not become an archive nothing reads.
const EventRetention = 7 * 24 * time.Hour

// DefaultStream is the stream name prefix used when Config.Stream is empty.
// The two streams are named from it, so several pipelines can share a broker.
const DefaultStream = "NOVA"

// NATS is a Bus backed by a NATS server.
//
// Subjects are routed by durability rather than by a single setting for the
// whole connection, because the pipeline genuinely needs both. Work assignment
// must survive a broker restart and be redelivered when a worker dies holding
// it; a heartbeat must not be stored at all. Deciding per subject means a
// service does not have to hold two buses and remember which is which.
type NATS struct {
	conn *nats.Conn
	js   jetstream.JetStream

	// Two streams, because work and facts are not the same kind of message.
	// work has work-queue retention, so a unit vanishes once someone
	// acknowledges it; facts is kept by age and may have any number of
	// consumers, which is what a promotion announcement needs.
	work   jetstream.Stream
	facts  jetstream.Stream
	closed bool

	producer string

	// ackWait bounds how long a consumer may hold a message before the server
	// assumes the consumer died and redelivers it. A self-play work unit is
	// minutes of games, so the default of thirty seconds would redeliver every
	// unit repeatedly while the first worker was still playing it.
	ackWait time.Duration

	// durable names this service's consumer for fan-out subscriptions, so its
	// position survives a restart. Empty means ephemeral. Ignored on
	// fan-out subscriptions when replayAll is set.
	durable string

	// replayAll makes fan-out subscriptions use an ephemeral, DeliverAll
	// consumer on every connection instead of a durable one that resumes
	// past its acked position. See Config.ReplayAll.
	replayAll bool

	// maxInFlight caps unacknowledged messages per consumer — which for a queue
	// group means across the whole group, since its members share one consumer.
	maxInFlight int

	mu   sync.Mutex
	subs []*natsSub
}

// DialTimeout bounds the initial connection attempt. A service that cannot
// reach the bus should fail its readiness probe rather than hang.
const DialTimeout = 10 * time.Second

// DefaultAckWait is how long a consumer may hold a work unit before the server
// concludes it died. Self-play units are minutes of games, so this is generous
// by design: redelivering a unit that is still being played wastes a whole
// worker's time on a duplicate.
const DefaultAckWait = 30 * time.Minute

// DefaultMaxInFlight is the ceiling on unacknowledged messages.
//
// This is a group-wide limit, not a per-worker one: the members of a queue
// group share a single consumer, and the server enforces MaxAckPending against
// that consumer. Setting it to the number of units a worker should hold — one —
// throttles the entire cluster to one unit at a time, so adding boards does
// nothing at all. The failure is silent; throughput simply never improves.
//
// So it is set well above any plausible worker count, and the "one unit at a
// time" property is enforced where it actually belongs: on how many messages
// each subscriber buffers.
const DefaultMaxInFlight = 512

// prefetch is how many messages a single subscriber buffers.
//
// One, so a worker takes a unit only when it is ready to play it. The client
// default is 500, which would let the first worker to connect pull hundreds of
// units into memory while its peers sat idle — units that are minutes of games
// each, so the imbalance would last for hours.
const prefetch = 1

// Connect opens a bus against a NATS server and ensures the stream exists.
func Connect(ctx context.Context, cfg Config) (*NATS, error) {
	if len(cfg.URLs) == 0 {
		return nil, fmt.Errorf("bus: no server URLs")
	}

	name := cfg.ClientName
	if name == "" {
		name = "novachess"
	}

	conn, err := nats.Connect(strings.Join(cfg.URLs, ","),
		nats.Name(name),
		nats.Timeout(DialTimeout),
		// Reconnect forever. A pipeline service outliving a broker restart is
		// the normal case on a cluster that reschedules pods, and a worker that
		// gave up would need an operator to notice and restart it.
		nats.MaxReconnects(-1),
		nats.ReconnectWait(time.Second),
		nats.RetryOnFailedConnect(true),
	)
	if err != nil {
		return nil, fmt.Errorf("bus: connecting to %v: %w", cfg.URLs, err)
	}

	js, err := jetstream.New(conn)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("bus: opening JetStream: %w", err)
	}

	prefix := cfg.Stream
	if prefix == "" {
		prefix = DefaultStream
	}

	// Work assignment is a queue: a unit is played once and is then of no
	// further interest, so it is removed on acknowledgement. Keeping every unit
	// forever would fill the disk with completed work.
	work, err := js.CreateOrUpdateStream(ctx, jetstream.StreamConfig{
		Name:      prefix + "_WORK",
		Subjects:  WorkSubjects,
		Retention: jetstream.WorkQueuePolicy,
		Storage:   jetstream.FileStorage,
		Discard:   jetstream.DiscardOld,
	})
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("bus: creating the work stream: %w", err)
	}

	// Facts are kept by age rather than consumed. This is not a preference: a
	// work-queue stream allows only one consumer per subject, and a promoted
	// network has to reach every worker and the bot at once.
	facts, err := js.CreateOrUpdateStream(ctx, jetstream.StreamConfig{
		Name:      prefix + "_EVENTS",
		Subjects:  EventSubjects,
		Retention: jetstream.LimitsPolicy,
		MaxAge:    EventRetention,
		Storage:   jetstream.FileStorage,
		Discard:   jetstream.DiscardOld,
	})
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("bus: creating the events stream: %w", err)
	}

	inFlight := cfg.MaxInFlight
	if inFlight <= 0 {
		inFlight = DefaultMaxInFlight
	}

	return &NATS{
		conn:        conn,
		js:          js,
		work:        work,
		facts:       facts,
		producer:    name,
		ackWait:     DefaultAckWait,
		durable:     cfg.Durable,
		replayAll:   cfg.ReplayAll,
		maxInFlight: inFlight,
	}, nil
}

// IsDurable reports whether a subject is stored by the broker.
func IsDurable(subject string) bool {
	for _, s := range DurableSubjects {
		if s == subject {
			return true
		}
	}
	return false
}

// isWork reports whether a subject carries work rather than a fact.
func isWork(subject string) bool {
	for _, s := range WorkSubjects {
		if s == subject {
			return true
		}
	}
	return false
}

// streamFor returns the stream carrying a subject.
func (n *NATS) streamFor(subject string) jetstream.Stream {
	if isWork(subject) {
		return n.work
	}
	return n.facts
}

// Publish sends a payload on a subject.
//
// For a durable subject this waits for the server's acknowledgement, so the
// message is on disk before Publish returns. That is what the interface
// promises and what a self-play worker depends on: it must know its batch
// announcement survived before it acknowledges the work unit that produced it,
// or a broker restart in between loses the games entirely.
func (n *NATS) Publish(ctx context.Context, subject string, payload any) error {
	n.mu.Lock()
	closed := n.closed
	n.mu.Unlock()
	if closed {
		return ErrClosed
	}

	env, err := NewEnvelope(subject, n.producer, payload)
	if err != nil {
		return err
	}
	body, err := json.Marshal(env)
	if err != nil {
		return fmt.Errorf("bus: marshal envelope for %s: %w", subject, err)
	}

	if IsDurable(subject) {
		if _, err := n.js.Publish(ctx, subject, body); err != nil {
			return fmt.Errorf("bus: publishing to %s: %w", subject, err)
		}
		return nil
	}

	if err := n.conn.Publish(subject, body); err != nil {
		return fmt.Errorf("bus: publishing to %s: %w", subject, err)
	}
	// Core NATS has no acknowledgement, so the most that can be promised is
	// that the message left this process. Flushing keeps the ordering guarantee
	// a caller expects when it publishes and then waits for a reply.
	//
	// A deadline is imposed when the caller gave none, because the client
	// refuses to flush against an open-ended context — and an unbounded flush
	// is not what a caller wants anyway when the broker has gone away.
	flushCtx := ctx
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		flushCtx, cancel = context.WithTimeout(ctx, DialTimeout)
		defer cancel()
	}
	if err := n.conn.FlushWithContext(flushCtx); err != nil {
		return fmt.Errorf("bus: flushing %s: %w", subject, err)
	}
	return nil
}

// Subscribe delivers every message on a subject to the handler.
//
// On a durable subject each subscriber gets its own consumer, so every one of
// them sees every message — which is what a fan-out event like net.promoted
// needs, since each worker and the bot must all pick up the new network.
func (n *NATS) Subscribe(ctx context.Context, subject string, h Handler) (Subscription, error) {
	return n.subscribe(ctx, subject, "", h)
}

// QueueSubscribe delivers each message to exactly one member of the group.
//
// On a durable subject the group shares one consumer, named after it. That is
// the work-distribution primitive: every self-play worker joins the same group,
// the server hands each unit to one of them, and a worker that dies without
// acknowledging has its unit redelivered to another.
func (n *NATS) QueueSubscribe(ctx context.Context, subject, queue string, h Handler) (Subscription, error) {
	if queue == "" {
		return nil, fmt.Errorf("bus: queue subscription on %s needs a group name", subject)
	}
	return n.subscribe(ctx, subject, queue, h)
}

func (n *NATS) subscribe(ctx context.Context, subject, queue string, h Handler) (Subscription, error) {
	n.mu.Lock()
	if n.closed {
		n.mu.Unlock()
		return nil, ErrClosed
	}
	n.mu.Unlock()

	if h == nil {
		return nil, fmt.Errorf("bus: subscription to %s has no handler", subject)
	}

	sub := &natsSub{bus: n}

	if !IsDurable(subject) {
		var err error
		if queue == "" {
			sub.core, err = n.conn.Subscribe(subject, n.coreHandler(ctx, h))
		} else {
			sub.core, err = n.conn.QueueSubscribe(subject, queue, n.coreHandler(ctx, h))
		}
		if err != nil {
			return nil, fmt.Errorf("bus: subscribing to %s: %w", subject, err)
		}
		if err := n.track(sub); err != nil {
			sub.stop()
			return nil, err
		}
		return sub, nil
	}

	cfg := jetstream.ConsumerConfig{
		FilterSubject: subject,
		AckPolicy:     jetstream.AckExplicitPolicy,
		AckWait:       n.ackWait,
		MaxAckPending: n.maxInFlight,
	}
	switch {
	case queue != "":
		// A named durable consumer shared by the group. Its position in the
		// stream survives restarts, so a worker coming back does not replay
		// units the group already finished. Untouched by replayAll: that
		// option is for fan-out subscriptions, and giving it to a queue
		// group would mean every restart replaying already-played work.
		cfg.Durable = consumerName(queue)
	case n.replayAll:
		// Ephemeral, and told explicitly to start from the beginning of
		// what the stream still retains rather than the server's default
		// of "from now on". Takes priority over n.durable: a caller that
		// asked to replay everything wants exactly that on every
		// reconnect, not a position that survives across them.
		cfg.DeliverPolicy = jetstream.DeliverAllPolicy
	case n.durable != "":
		// A fan-out consumer belonging to this service alone, named so that it
		// resumes rather than starting from the present. The subject is part of
		// the name because one service may subscribe to several, and a single
		// consumer cannot filter for all of them.
		cfg.Durable = consumerName(n.durable + "-" + subject)
	}

	consumer, err := n.streamFor(subject).CreateOrUpdateConsumer(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("bus: creating a consumer for %s: %w", subject, err)
	}

	consume, err := consumer.Consume(n.streamHandler(ctx, h), jetstream.PullMaxMessages(prefetch))
	if err != nil {
		return nil, fmt.Errorf("bus: consuming %s: %w", subject, err)
	}
	sub.consume = consume

	if err := n.track(sub); err != nil {
		sub.stop()
		return nil, err
	}
	return sub, nil
}

// consumerName turns a queue group into a name the server will accept. Durable
// names may not contain dots, which subjects are full of.
func consumerName(queue string) string {
	return strings.NewReplacer(".", "-", " ", "-", "*", "-", ">", "-").Replace(queue)
}

// coreHandler adapts a handler to a core NATS subscription.
//
// There is no acknowledgement to give, so a handler that fails has nowhere to
// report it: core NATS is fire-and-forget and the message is already gone.
// That is exactly why work never travels this way, and why only heartbeats do.
func (n *NATS) coreHandler(ctx context.Context, h Handler) nats.MsgHandler {
	return func(msg *nats.Msg) {
		var env events.Envelope
		if err := json.Unmarshal(msg.Data, &env); err != nil {
			return
		}
		_ = h(ctx, env)
	}
}

// streamHandler adapts a handler to a JetStream consumer, translating its
// return into an acknowledgement.
func (n *NATS) streamHandler(ctx context.Context, h Handler) jetstream.MessageHandler {
	return func(msg jetstream.Msg) {
		var env events.Envelope
		if err := json.Unmarshal(msg.Data(), &env); err != nil {
			// Undecodable bytes will not decode on the next attempt either, so
			// redelivering forever would block the queue on one bad message.
			// Terminate takes it out of the stream instead.
			_ = msg.Term()
			return
		}

		if err := h(ctx, env); err != nil {
			// Negative acknowledgement, so another worker gets it. This is the
			// at-least-once behaviour handlers are required to tolerate.
			_ = msg.Nak()
			return
		}
		_ = msg.Ack()
	}
}

// track registers a subscription, refusing if the bus closed while it was
// being set up.
//
// Subscribing releases the lock to make network calls, so a Close can land in
// the middle. Without this check the new subscription would be created after
// Close captured the list, leaving it delivering messages to a handler whose
// service believes it has shut down.
func (n *NATS) track(sub *natsSub) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.closed {
		return ErrClosed
	}
	n.subs = append(n.subs, sub)
	return nil
}

// Close stops every subscription and closes the connection.
func (n *NATS) Close() error {
	n.mu.Lock()
	if n.closed {
		n.mu.Unlock()
		return nil
	}
	n.closed = true
	subs := n.subs
	n.subs = nil
	n.mu.Unlock()

	var errs []error
	for _, sub := range subs {
		if err := sub.stop(); err != nil {
			errs = append(errs, err)
		}
	}

	// Drain rather than close, so messages already accepted for publication
	// reach the server before the socket goes away.
	if err := n.conn.Drain(); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

// Conn exposes the underlying connection, for health checks and for the
// monitoring endpoints that make a misbehaving board identifiable.
func (n *NATS) Conn() *nats.Conn { return n.conn }

// natsSub is one active subscription, of either kind.
type natsSub struct {
	bus     *NATS
	core    *nats.Subscription
	consume jetstream.ConsumeContext

	once sync.Once
	err  error
}

// Close stops delivery.
func (s *natsSub) Close() error {
	s.stop()

	s.bus.mu.Lock()
	defer s.bus.mu.Unlock()
	for i, other := range s.bus.subs {
		if other == s {
			s.bus.subs = append(s.bus.subs[:i], s.bus.subs[i+1:]...)
			break
		}
	}
	return s.err
}

func (s *natsSub) stop() error {
	s.once.Do(func() {
		if s.consume != nil {
			s.consume.Stop()
		}
		if s.core != nil {
			// Unsubscribe rather than drain: the caller asked for delivery to
			// stop, and draining would keep handing messages to a handler that
			// may be about to go away with its dependencies.
			s.err = s.core.Unsubscribe()
		}
	})
	return s.err
}
