package dashboard

import (
	"bufio"
	"encoding/json"
	"os"
	"sync"
	"time"
)

// Record is one fact in the training history: a dataset completed, a
// training run finished, a gate decided, a network was promoted. The
// history file is the long-term memory; the 7-day event stream only
// backfills it.
type Record struct {
	Type       string    `json:"type"`
	Generation int       `json:"generation"`
	Positions  int       `json:"positions,omitempty"`
	Promoted   bool      `json:"promoted,omitempty"`
	Wins       int       `json:"wins,omitempty"`
	Losses     int       `json:"losses,omitempty"`
	Draws      int       `json:"draws,omitempty"`
	EloDelta   float64   `json:"elo_delta,omitempty"`
	LOS        float64   `json:"los,omitempty"`
	Reason     string    `json:"reason,omitempty"`
	NetworkURI string    `json:"network_uri,omitempty"`
	JobName    string    `json:"job_name,omitempty"`
	FinalLoss  float64   `json:"final_loss,omitempty"`
	At         time.Time `json:"at"`
}

type History struct {
	mu      sync.Mutex
	path    string
	records []Record
}

// NewHistory loads what it can from path. A missing file is an empty
// history; a truncated final line (a crash mid-append) is dropped rather
// than fatal, because everything before it is still good.
func NewHistory(path string) *History {
	h := &History{path: path}
	f, err := os.Open(path)
	if err != nil {
		return h
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r Record
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			continue // torn or corrupt line: skip, keep the rest
		}
		h.records = append(h.records, r)
	}
	return h
}

func (h *History) Append(r Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	line, err := json.Marshal(r)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(h.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(append(line, '\n')); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	h.records = append(h.records, r)
	return nil
}

func (h *History) Records() []Record {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]Record, len(h.records))
	copy(out, h.records)
	return out
}

func (h *History) HasPromotion(generation int) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, r := range h.records {
		if r.Generation == generation && (r.Type == "promotion" || (r.Type == "gate" && r.Promoted)) {
			return true
		}
	}
	return false
}
