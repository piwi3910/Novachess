package dashboard

import (
	"context"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/events"
)

func TestNoteMetricsJoinsByPodName(t *testing.T) {
	s := NewState()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "pod-a"}, time.Now())
	s.NoteMetrics(map[string]PodMetrics{
		"pod-a":    {CPUMillicores: 970, MemoryBytes: 200 << 20},
		"stranger": {CPUMillicores: 5, MemoryBytes: 1 << 20}, // not a board: dropped
	})
	snap := s.Snapshot(time.Now())
	if len(snap.Boards) != 1 {
		t.Fatalf("metrics for unknown pods must not create boards; got %d", len(snap.Boards))
	}
	b := snap.Boards[0]
	if b.CPUMillicores != 970 || b.MemoryBytes != 200<<20 {
		t.Fatalf("metrics not joined: %+v", b)
	}
}

func TestBoardWithoutSampleOmitsMetrics(t *testing.T) {
	s := NewState()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "pod-a"}, time.Now())
	b := s.Snapshot(time.Now()).Boards[0]
	if b.CPUMillicores != 0 || b.MemoryBytes != 0 {
		t.Fatal("no sample yet must mean zero values (omitted from JSON)")
	}
}

func TestHeartbeatPreservesMetricsAcrossInterleaving(t *testing.T) {
	s := NewState()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "pod-a"}, time.Now())
	s.NoteMetrics(map[string]PodMetrics{
		"pod-a": {CPUMillicores: 970, MemoryBytes: 200 << 20},
	})
	// A later heartbeat (the ordinary steady-state interleaving of the two
	// ~15s cadences) must not zero out the sample the poller already joined.
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "pod-a"}, time.Now())
	b := s.Snapshot(time.Now()).Boards[0]
	if b.CPUMillicores != 970 || b.MemoryBytes != 200<<20 {
		t.Fatalf("heartbeat wiped metrics carried from a prior sample: %+v", b)
	}
}

func TestMetricsPollerFeedsState(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	fc := newFakeCluster()
	fc.metrics = map[string]PodMetrics{
		"pod-a": {CPUMillicores: 500, MemoryBytes: 100 << 20},
	}

	s := NewState()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "pod-a"}, time.Now())
	h := NewHistory("")
	ct := NewController(fc, h, t.TempDir(), 4)

	srv := NewServer(ctx, s, h, ct, fc,
		WithMetricsInterval(10*time.Millisecond),
		WithFollowerIntervals(10*time.Millisecond, 100*time.Millisecond, 10*time.Millisecond, 100*time.Millisecond),
	)
	_ = srv

	waitFor(t, func() bool {
		b := s.Snapshot(time.Now()).Boards[0]
		return b.CPUMillicores == 500 && b.MemoryBytes == 100<<20
	})

	// The poller must survive errors and resume once they clear.
	fc.setMetrics(map[string]PodMetrics{
		"pod-a": {CPUMillicores: 750, MemoryBytes: 150 << 20},
	}, context.DeadlineExceeded)
	time.Sleep(30 * time.Millisecond)
	fc.setMetrics(map[string]PodMetrics{
		"pod-a": {CPUMillicores: 750, MemoryBytes: 150 << 20},
	}, nil)

	waitFor(t, func() bool {
		b := s.Snapshot(time.Now()).Boards[0]
		return b.CPUMillicores == 750 && b.MemoryBytes == 150<<20
	})
}
