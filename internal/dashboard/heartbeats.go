package dashboard

import (
	"context"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
)

// WatchHeartbeats feeds worker heartbeats into the state. Heartbeats travel
// over core NATS, not JetStream, so there is no history to replay: the boards
// view starts empty and fills within one heartbeat interval.
func WatchHeartbeats(ctx context.Context, b bus.Bus, s *State) error {
	_, err := b.Subscribe(ctx, events.SubjectWorkerHeartbeat, func(ctx context.Context, env events.Envelope) error {
		var hb events.WorkerHeartbeat
		if err := bus.Unmarshal(env, &hb); err != nil {
			return err
		}
		s.NoteHeartbeat(hb, time.Now())
		return nil
	})
	return err
}
