package dashboard

import (
	"context"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
)

func TestHeartbeatsPopulateBoards(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	b := bus.NewMemory("test") // match the constructor used in bus_test.go
	s := NewState()
	if err := WatchHeartbeats(ctx, b, s); err != nil {
		t.Fatal(err)
	}
	hb := events.WorkerHeartbeat{
		WorkerID: "w-1", NodeName: "master-11",
		CurrentWorkUnitID: "g0-u000009", NodesPerSecond: 1_150_000,
		GamesCompleted: 12, EngineVersion: "b524b12",
	}
	if err := b.Publish(ctx, events.SubjectWorkerHeartbeat, hb); err != nil {
		t.Fatal(err)
	}
	// The memory bus delivers asynchronously; poll for the invariant.
	deadline := time.Now().Add(2 * time.Second)
	for {
		snap := s.Snapshot(time.Now())
		if len(snap.Boards) == 1 {
			got := snap.Boards[0]
			if got.WorkerID != "w-1" || got.NodeName != "master-11" || got.Stale {
				t.Fatalf("bad board state: %+v", got)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("heartbeat never reached the state")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestBoardsGoStale(t *testing.T) {
	s := NewState()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "w-1"}, time.Now().Add(-time.Minute))
	snap := s.Snapshot(time.Now())
	if !snap.Boards[0].Stale {
		t.Fatal("a board silent for a minute must be stale")
	}
}
