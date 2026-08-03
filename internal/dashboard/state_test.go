package dashboard

import (
	"testing"
	"time"
)

// TestSnapshotReportsTargetBeforeAnyBatch is the property Finding 1 fixes: the
// UI's progress bar and ETA need a target even before the first batch of a
// generation has been counted, otherwise they can never render at startup or
// right after a generation rollover.
func TestSnapshotReportsTargetBeforeAnyBatch(t *testing.T) {
	s := NewState()
	s.SetTarget(1_000_000)

	got := s.Snapshot(time.Now()).Generation
	if got.Target != 1_000_000 {
		t.Fatalf("Target = %d, want 1000000 (no batches counted yet)", got.Target)
	}
	if got.Positions != 0 || got.Batches != 0 {
		t.Fatalf("expected zero progress alongside the target, got %+v", got)
	}
}

// TestSnapshotTargetSurvivesGenerationRollover checks that the target is not
// lost when NoteProgress replaces GenProgress wholesale on a generation
// change (see WatchEvents), since SetTarget and NoteProgress are independent
// state.
func TestSnapshotTargetSurvivesGenerationRollover(t *testing.T) {
	s := NewState()
	s.SetTarget(500_000)
	s.NoteProgress(GenProgress{Generation: 1, Positions: 4000, Batches: 1})

	got := s.Snapshot(time.Now()).Generation
	if got.Target != 500_000 {
		t.Fatalf("Target = %d, want 500000 to survive a progress reset", got.Target)
	}
	if got.Generation != 1 || got.Positions != 4000 {
		t.Fatalf("expected progress fields to still reflect NoteProgress, got %+v", got)
	}
}
