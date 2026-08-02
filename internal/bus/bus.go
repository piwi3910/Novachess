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

	// Stream is the JetStream stream backing durable subjects. Empty selects
	// core NATS, which is fire-and-forget with no redelivery — acceptable for
	// heartbeats, never for work assignment.
	Stream string

	// Durable names the consumer so that its position in the stream survives
	// restarts. Without it, a restarting service either replays everything or
	// resumes from the present, and neither is what a work queue wants.
	Durable string

	// MaxInFlight bounds unacknowledged messages per consumer. For self-play
	// workers this should be 1: a worker holding several unstarted units while
	// idle peers wait is exactly the imbalance the queue group exists to
	// prevent.
	MaxInFlight int
}
