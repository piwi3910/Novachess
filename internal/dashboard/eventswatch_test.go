package dashboard

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
)

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("condition never held")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestGamesProducedAccumulateProgress(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	b := bus.NewMemory("test")
	s := NewState()
	h := NewHistory(filepath.Join(t.TempDir(), "history.jsonl"))
	if err := WatchEvents(ctx, b, s, h); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		err := b.Publish(ctx, events.SubjectGamesProduced, events.GameBatch{
			WorkUnitID: fmt.Sprintf("g0-u%06d", i), Generation: 0, Positions: 4000, Games: 50,
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	waitFor(t, func() bool {
		p := s.Snapshot(time.Now()).Generation
		return p.Batches == 3 && p.Positions == 12000 && p.Generation == 0
	})
}

func TestDuplicateBatchesCountOnce(t *testing.T) {
	// Publish the same WorkUnitID twice; batches and positions must count it
	// once. (At-least-once delivery is the transport's contract; dedup is
	// ours.)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	b := bus.NewMemory("test")
	s := NewState()
	h := NewHistory(filepath.Join(t.TempDir(), "history.jsonl"))
	if err := WatchEvents(ctx, b, s, h); err != nil {
		t.Fatal(err)
	}
	batch := events.GameBatch{WorkUnitID: "g0-u000001", Generation: 0, Positions: 4000, Games: 50}
	if err := b.Publish(ctx, events.SubjectGamesProduced, batch); err != nil {
		t.Fatal(err)
	}
	if err := b.Publish(ctx, events.SubjectGamesProduced, batch); err != nil {
		t.Fatal(err)
	}
	waitFor(t, func() bool {
		p := s.Snapshot(time.Now()).Generation
		return p.Batches == 1 && p.Positions == 4000
	})
}

// TestReplayingBatchesAcrossGenerationsEndsCorrect checks that the progress
// dedup/reset logic in WatchEvents still lands on the right state after an
// ordered replay spanning a generation change - exactly what a ReplayAll
// reconnect delivers: every gen0 batch, in order, followed by every gen1
// batch.
func TestReplayingBatchesAcrossGenerationsEndsCorrect(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	b := bus.NewMemory("test")
	s := NewState()
	h := NewHistory(filepath.Join(t.TempDir(), "history.jsonl"))
	if err := WatchEvents(ctx, b, s, h); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		if err := b.Publish(ctx, events.SubjectGamesProduced, events.GameBatch{
			WorkUnitID: fmt.Sprintf("g0-u%06d", i), Generation: 0, Positions: 4000, Games: 50,
		}); err != nil {
			t.Fatal(err)
		}
	}
	waitFor(t, func() bool {
		p := s.Snapshot(time.Now()).Generation
		return p.Generation == 0 && p.Batches == 3
	})

	for i := 0; i < 2; i++ {
		if err := b.Publish(ctx, events.SubjectGamesProduced, events.GameBatch{
			WorkUnitID: fmt.Sprintf("g1-u%06d", i), Generation: 1, Positions: 4000, Games: 50,
		}); err != nil {
			t.Fatal(err)
		}
	}

	waitFor(t, func() bool {
		p := s.Snapshot(time.Now()).Generation
		return p.Generation == 1 && p.Batches == 2 && p.Positions == 8000
	})

	// The gen0 batches must not still be counted in gen1's totals.
	p := s.Snapshot(time.Now()).Generation
	if p.Batches != 2 || p.Positions != 8000 {
		t.Fatalf("expected gen1 progress to hold only gen1's 2 batches / 8000 positions, got %+v", p)
	}
}

// TestDuplicateDatasetEventAppendsHistoryOnce checks the history-side dedup:
// the same dataset-ready event delivered twice (at-least-once, or replayed
// twice by a ReplayAll reconnect) must append one record, not two.
func TestDuplicateDatasetEventAppendsHistoryOnce(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	b := bus.NewMemory("test")
	s := NewState()
	h := NewHistory(filepath.Join(t.TempDir(), "history.jsonl"))
	if err := WatchEvents(ctx, b, s, h); err != nil {
		t.Fatal(err)
	}

	ds := events.DatasetReady{Generation: 4, Positions: 1004378}
	if err := b.Publish(ctx, events.SubjectDatasetReady, ds); err != nil {
		t.Fatal(err)
	}
	if err := b.Publish(ctx, events.SubjectDatasetReady, ds); err != nil {
		t.Fatal(err)
	}

	waitFor(t, func() bool {
		count := 0
		for _, r := range h.Records() {
			if r.Type == "dataset" && r.Generation == 4 {
				count++
			}
		}
		return count == 1
	})

	count := 0
	for _, r := range h.Records() {
		if r.Type == "dataset" && r.Generation == 4 {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected exactly one dataset record for generation 4, got %d", count)
	}
}

// TestRestartDoesNotDuplicateHistory simulates a dashboard restart: a History
// already holding a dataset record for a generation, then a fresh
// WatchEvents call over it seeing the same event again (what a ReplayAll
// reconnect would redeliver from the stream's retained window). The record
// must not be duplicated.
func TestRestartDoesNotDuplicateHistory(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	path := filepath.Join(t.TempDir(), "history.jsonl")
	h := NewHistory(path)
	ds := events.DatasetReady{Generation: 7, Positions: 999000}
	if err := h.Append(Record{Type: "dataset", Generation: ds.Generation, Positions: ds.Positions, At: time.Now()}); err != nil {
		t.Fatalf("seed history: %v", err)
	}

	b := bus.NewMemory("test")
	s := NewState()
	if err := WatchEvents(ctx, b, s, h); err != nil {
		t.Fatal(err)
	}
	if err := b.Publish(ctx, events.SubjectDatasetReady, ds); err != nil {
		t.Fatal(err)
	}

	// No waitFor condition to observe here beyond "stays at one" - the
	// memory bus delivers synchronously, so by the time Publish returns the
	// handler has already run (or not run at all, which is also a pass).
	count := 0
	for _, r := range h.Records() {
		if r.Type == "dataset" && r.Generation == 7 {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected the restart-shaped replay to append nothing, got %d dataset records for generation 7", count)
	}
}

func TestDatasetReadyRecordsHistory(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	b := bus.NewMemory("test")
	s := NewState()
	h := NewHistory(filepath.Join(t.TempDir(), "history.jsonl"))
	if err := WatchEvents(ctx, b, s, h); err != nil {
		t.Fatal(err)
	}
	if err := b.Publish(ctx, events.SubjectDatasetReady, events.DatasetReady{
		Generation: 0, Positions: 1004378,
	}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, func() bool {
		for _, r := range h.Records() {
			if r.Type == "dataset" && r.Generation == 0 {
				return true
			}
		}
		return false
	})
}
