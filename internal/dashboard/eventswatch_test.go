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
