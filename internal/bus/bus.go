// Package bus abstracts the message bus the pipeline services coordinate over.
//
// The interface is deliberately narrow and free of broker types. NATS is the
// intended implementation — a single static Go binary with first-class arm64
// support, which suits a cluster of ARM boards far better than a JVM broker —
// but services depend on this interface rather than on NATS, so tests can run
// against an in-memory bus with no broker at all.
//
// Only what the pipeline needs is modelled here. There is no request/reply, no
// key-value store, no object store: those exist in NATS, and adding them to
// this interface would couple every service to features two of them use.
package bus

import (
	"context"
	"errors"

	"github.com/piwi3910/novachess/internal/events"
)

// ErrClosed is returned by operations on a closed Bus.
var ErrClosed = errors.New("bus: closed")

// Handler processes a delivered message.
//
// Returning nil acknowledges the message and it will not be redelivered.
// Returning an error negatively acknowledges it, and the bus redelivers it —
// possibly to a different consumer. Handlers must therefore be safe to run
// twice: delivery is at-least-once, and a worker that dies after finishing its
// work but before acknowledging it will see that work again.
type Handler func(ctx context.Context, env events.Envelope) error

// Subscription is an active subscription. Closing it stops delivery; messages
// already in flight are still delivered to the handler.
type Subscription interface {
	Close() error
}

// Bus publishes and consumes pipeline messages.
type Bus interface {
	// Publish sends a payload on a subject. The payload is marshalled and
	// wrapped in an events.Envelope by the implementation.
	//
	// Publish returns once the bus has durably accepted the message, not once
	// a consumer has handled it. That distinction matters for self-play
	// workers: a worker must know its batch announcement survived before it
	// acknowledges the work unit that produced it.
	Publish(ctx context.Context, subject string, payload any) error

	// Subscribe delivers every message on a subject to the handler. Each
	// subscriber receives its own copy, which is what fan-out events like
	// net.promoted need — every worker and the bot must all see it.
	Subscribe(ctx context.Context, subject string, h Handler) (Subscription, error)

	// QueueSubscribe delivers each message to exactly one member of the named
	// queue group. This is the work-distribution primitive: every self-play
	// worker joins the same group on nova.work.assign, so a unit is played
	// once rather than once per pod.
	//
	// Adding and removing subscribers rebalances automatically, which is what
	// lets the cluster be scaled by changing a replica count.
	QueueSubscribe(ctx context.Context, subject, queue string, h Handler) (Subscription, error)

	// Close shuts down the connection and all subscriptions.
	Close() error
}

// Config describes how to reach the bus.
type Config struct {
	// URLs are the broker addresses. More than one gives failover.
	URLs []string

	// ClientName identifies this service instance in broker monitoring. Worth
	// setting to the pod name: when one board misbehaves, this is what ties a
	// connection back to it.
	ClientName string

	// Stream is the name prefix for the JetStream streams. Two are created from
	// it, one for work and one for facts; see the note on DurableSubjects for
	// why they cannot be the same stream. Empty selects a default prefix.
	//
	// Whether a given subject is stored is decided per subject rather than by
	// this field: work and decisions are durable, heartbeats are not. A single
	// switch would force the pipeline to choose between losing work
	// assignments and archiving heartbeats forever.
	Stream string

	// Durable names this service's consumer for fan-out subscriptions, so its
	// position survives a restart and it does not miss, say, a promotion that
	// happened while it was being rescheduled. It must be unique per service:
	// two services sharing a name would share a consumer and each see only
	// half the messages.
	//
	// Empty means an ephemeral consumer, which starts from the present. That is
	// the right choice for a service that only cares what happens from now on.
	//
	// Queue subscriptions ignore this and derive their durable name from the
	// group, since sharing is the entire point there.
	//
	// Ignored on fan-out subscriptions when ReplayAll is set; see ReplayAll.
	Durable string

	// ReplayAll makes fan-out subscriptions (Subscribe, not QueueSubscribe)
	// use a fresh ephemeral consumer with DeliverAll on every connection,
	// instead of a durable consumer that resumes past its acked position.
	//
	// This is for a service whose state from the stream lives in memory
	// only, such as the dashboard's generation progress: after a restart, a
	// durable consumer would come back showing nothing until new messages
	// arrived, even though the stream still holds the whole retention
	// window. Replaying it all and deduping on the receiving end (by work
	// unit, by generation, by whatever key the caller's records use)
	// rebuilds the same state a durable consumer would have kept, every
	// time, at the cost of reprocessing the window on every reconnect.
	//
	// Leave this false for a service that only cares what happens from now
	// on, or whose state is durable on its own side (a worker's played work
	// units, for instance) — those want Durable instead.
	ReplayAll bool

	// MaxInFlight bounds unacknowledged messages.
	//
	// This is a ceiling per consumer, and the members of a queue group share
	// one — so for work assignment it bounds the whole cluster, not each
	// worker. Setting it to the number of units a worker should hold at once
	// throttles every worker to that number between them, and adding boards
	// then changes nothing. Leave it at the default unless the cluster is
	// larger than that default allows.
	//
	// The "one unit at a time per worker" property is not this field's job: it
	// comes from how many messages a subscriber buffers, which the
	// implementation fixes at one.
	MaxInFlight int
}
