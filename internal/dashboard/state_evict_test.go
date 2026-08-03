package dashboard

import (
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/events"
)

func TestBoardsEvictedAfterLongSilence(t *testing.T) {
	s := NewState()
	now := time.Now()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "ghost"}, now.Add(-EvictAfter-time.Minute))
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "sick"}, now.Add(-StaleAfter-time.Minute))
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "live"}, now)

	snap := s.Snapshot(now)
	if len(snap.Boards) != 2 {
		t.Fatalf("got %d boards, want 2: a board silent past EvictAfter is a dead pod, not a sick one", len(snap.Boards))
	}
	for _, b := range snap.Boards {
		switch b.WorkerID {
		case "sick":
			if !b.Stale {
				t.Fatal("a board past StaleAfter but before EvictAfter must show stale")
			}
		case "live":
			if b.Stale {
				t.Fatal("a fresh board must not be stale")
			}
		case "ghost":
			t.Fatal("evicted board still present in snapshot")
		}
	}
}

func TestTouchWakesSubscribers(t *testing.T) {
	s := NewState()
	ch, cancel := s.Subscribe()
	defer cancel()
	s.Touch()
	select {
	case <-ch:
	case <-time.After(time.Second):
		t.Fatal("Touch did not notify subscribers")
	}
}
