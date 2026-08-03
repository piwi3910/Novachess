package dashboard

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestHistoryRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.jsonl")
	h := NewHistory(path)
	if err := h.Append(Record{Type: "dataset", Generation: 0, Positions: 100, At: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if err := h.Append(Record{Type: "gate", Generation: 0, Promoted: true, At: time.Now()}); err != nil {
		t.Fatal(err)
	}
	reloaded := NewHistory(path)
	if got := len(reloaded.Records()); got != 2 {
		t.Fatalf("reloaded %d records, want 2", got)
	}
	if !reloaded.HasPromotion(0) {
		t.Fatal("promotion for gen 0 lost on reload")
	}
	if reloaded.HasPromotion(1) {
		t.Fatal("phantom promotion for gen 1")
	}
}

func TestHistorySurvivesTruncatedLastLine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.jsonl")
	h := NewHistory(path)
	_ = h.Append(Record{Type: "dataset", Generation: 0, At: time.Now()})
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString(`{"type":"gate","gener`) // a crash mid-write
	f.Close()
	reloaded := NewHistory(path)
	if got := len(reloaded.Records()); got != 1 {
		t.Fatalf("truncated line must be dropped, not fatal; got %d records", got)
	}
	// And the store must still be appendable afterwards.
	if err := reloaded.Append(Record{Type: "gate", Generation: 0, At: time.Now()}); err != nil {
		t.Fatal(err)
	}
}
