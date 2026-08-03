package bus

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/events"
)

// Memory is an in-process Bus for tests.
//
// It delivers synchronously inside Publish, which makes tests deterministic:
// once Publish returns, every handler has run. A real broker delivers
// asynchronously, so a test that depends on this ordering is testing the fake
// rather than the system — assert on handler effects, not on timing.
//
// Redelivery is not simulated. Handlers returning an error are recorded in
// DeliveryErrors rather than retried, because a synchronous retry loop would
// deadlock the publisher. Tests that need to exercise at-least-once semantics
// should call Publish twice.
type Memory struct {
	mu       sync.RWMutex
	subs     map[string][]*memorySub
	queues   map[string]map[string][]*memorySub
	nextInQ  map[string]int
	closed   bool
	producer string

	// DeliveryErrors collects errors returned by handlers, so a test can
	// assert that a handler rejected a message it should have.
	DeliveryErrors []error
}

type memorySub struct {
	bus     *Memory
	subject string
	queue   string
	handler Handler
	closed  bool
}

// NewMemory returns an in-process bus. producer is recorded in the envelope of
// every published message.
func NewMemory(producer string) *Memory {
	return &Memory{
		subs:     make(map[string][]*memorySub),
		queues:   make(map[string]map[string][]*memorySub),
		nextInQ:  make(map[string]int),
		producer: producer,
	}
}

func (m *Memory) Publish(ctx context.Context, subject string, payload any) error {
	m.mu.RLock()
	if m.closed {
		m.mu.RUnlock()
		return ErrClosed
	}

	env, err := NewEnvelope(subject, m.producer, payload)
	if err != nil {
		m.mu.RUnlock()
		return err
	}

	// Snapshot the targets before releasing the lock, so a handler that
	// subscribes or unsubscribes does not mutate the slice being ranged over.
	targets := append([]*memorySub(nil), m.subs[subject]...)
	for queue, members := range m.queues[subject] {
		if len(members) == 0 {
			continue
		}
		// Round-robin within the group, approximating the load spreading a
		// real queue group performs.
		key := subject + "\x00" + queue
		idx := m.nextInQ[key] % len(members)
		targets = append(targets, members[idx])
		m.nextInQ[key] = idx + 1
	}
	m.mu.RUnlock()

	for _, sub := range targets {
		if err := sub.handler(ctx, env); err != nil {
			m.mu.Lock()
			m.DeliveryErrors = append(m.DeliveryErrors, fmt.Errorf("%s: %w", subject, err))
			m.mu.Unlock()
		}
	}
	return nil
}

func (m *Memory) Subscribe(ctx context.Context, subject string, h Handler) (Subscription, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}
	sub := &memorySub{bus: m, subject: subject, handler: h}
	m.subs[subject] = append(m.subs[subject], sub)
	return sub, nil
}

func (m *Memory) QueueSubscribe(ctx context.Context, subject, queue string, h Handler) (Subscription, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}
	sub := &memorySub{bus: m, subject: subject, queue: queue, handler: h}
	if m.queues[subject] == nil {
		m.queues[subject] = make(map[string][]*memorySub)
	}
	m.queues[subject][queue] = append(m.queues[subject][queue], sub)
	return sub, nil
}

func (m *Memory) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	m.subs = make(map[string][]*memorySub)
	m.queues = make(map[string]map[string][]*memorySub)
	return nil
}

func (s *memorySub) Close() error {
	s.bus.mu.Lock()
	defer s.bus.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true

	if s.queue == "" {
		s.bus.subs[s.subject] = removeSub(s.bus.subs[s.subject], s)
		return nil
	}
	if groups := s.bus.queues[s.subject]; groups != nil {
		groups[s.queue] = removeSub(groups[s.queue], s)
	}
	return nil
}

func removeSub(list []*memorySub, target *memorySub) []*memorySub {
	for i, s := range list {
		if s == target {
			return append(list[:i:i], list[i+1:]...)
		}
	}
	return list
}

// NewEnvelope marshals a payload and wraps it for transport. Implementations
// share this so that the envelope written by the in-memory bus is byte-identical
// to the one a broker would carry — otherwise tests would validate a format
// production never sees.
func NewEnvelope(subject, producer string, payload any) (events.Envelope, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return events.Envelope{}, fmt.Errorf("bus: marshal payload for %s: %w", subject, err)
	}
	id, err := newMessageID()
	if err != nil {
		return events.Envelope{}, err
	}
	return events.Envelope{
		ID:            id,
		Type:          subject,
		SchemaVersion: events.SchemaVersion,
		Producer:      producer,
		PublishedAt:   time.Now().UTC(),
		Payload:       body,
	}, nil
}

// Unmarshal decodes an envelope's payload into v. It checks the schema version
// first: a consumer that silently decodes a newer message would drop fields it
// does not know about, which during a rolling update means quietly discarding
// work rather than failing visibly.
func Unmarshal(env events.Envelope, v any) error {
	if env.SchemaVersion > events.SchemaVersion {
		return fmt.Errorf("bus: message %s has schema version %d, this build understands %d",
			env.ID, env.SchemaVersion, events.SchemaVersion)
	}
	if err := json.Unmarshal(env.Payload, v); err != nil {
		return fmt.Errorf("bus: unmarshal %s payload: %w", env.Type, err)
	}
	return nil
}

func newMessageID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("bus: generate message id: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}
