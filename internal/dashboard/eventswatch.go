package dashboard

import (
	"context"
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
)

// WatchEvents follows the retained event stream: batch production feeds the
// generation progress view, dataset announcements and network verdicts feed
// the history. The bus subscribes with ReplayAll, so a fresh ephemeral
// consumer replays everything the stream's 7-day window still holds on every
// connection - including after a dashboard restart - rather than resuming
// from an acked position the way a durable consumer would. Progress survives
// a reschedule because it dedups by WorkUnitID as it replays; history
// survives it because each stream-derived record is appended only if an
// equivalent one is not already there (see History.AppendIfAbsent) - so
// replaying the same window twice, whether from two restarts or from two
// deliveries of one event, costs nothing.
func WatchEvents(ctx context.Context, b bus.Bus, s *State, h *History) error {
	var mu sync.Mutex
	seen := make(map[string]struct{}) // WorkUnitID dedup: delivery is at-least-once
	var progress GenProgress

	if _, err := b.Subscribe(ctx, events.SubjectGamesProduced, func(ctx context.Context, env events.Envelope) error {
		var batch events.GameBatch
		if err := bus.Unmarshal(env, &batch); err != nil {
			return err
		}
		mu.Lock()
		if batch.Generation != progress.Generation {
			// A new generation resets the running counters; history keeps the old.
			progress = GenProgress{Generation: batch.Generation}
			seen = make(map[string]struct{})
		}
		if _, dup := seen[batch.WorkUnitID]; !dup {
			seen[batch.WorkUnitID] = struct{}{}
			progress.Batches++
			progress.Positions += batch.Positions
			progress.UpdatedAt = time.Now()
			p := progress
			mu.Unlock()
			s.NoteProgress(p)
			return nil
		}
		mu.Unlock()
		return nil
	}); err != nil {
		return err
	}

	if _, err := b.Subscribe(ctx, events.SubjectDatasetReady, func(ctx context.Context, env events.Envelope) error {
		var ds events.DatasetReady
		if err := bus.Unmarshal(env, &ds); err != nil {
			return err
		}
		return h.AppendIfAbsent(
			Record{Type: "dataset", Generation: ds.Generation, Positions: ds.Positions, At: time.Now()},
			func(r Record) bool { return r.Type == "dataset" && r.Generation == ds.Generation },
		)
	}); err != nil {
		return err
	}

	verdict := func(promoted bool) bus.Handler {
		return func(ctx context.Context, env events.Envelope) error {
			var v events.NetworkVerdict
			if err := bus.Unmarshal(env, &v); err != nil {
				return err
			}
			rec := Record{Type: "gate", Generation: v.Generation, Promoted: promoted,
				Wins: v.Wins, Losses: v.Losses, Draws: v.Draws, EloDelta: v.EloDelta, LOS: v.LOS,
				Reason: v.Reason, NetworkURI: v.ArtifactURI, At: time.Now()}
			return h.AppendIfAbsent(rec, func(r Record) bool {
				return r.Type == "gate" && r.Generation == v.Generation && r.NetworkURI == v.ArtifactURI
			})
		}
	}
	if _, err := b.Subscribe(ctx, events.SubjectNetPromoted, verdict(true)); err != nil {
		return err
	}
	if _, err := b.Subscribe(ctx, events.SubjectNetRejected, verdict(false)); err != nil {
		return err
	}
	return nil
}
