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
	return h.appendLocked(r)
}

// AppendIfAbsent appends r unless a record already satisfies matches, and is
// what makes replaying the retained event stream on every restart safe: the
// same dataset-ready or gate event arriving twice appends once. The check
// and the append happen under the same lock so two near-simultaneous
// deliveries of the same event cannot both pass the check before either
// appends.
func (h *History) AppendIfAbsent(r Record, matches func(Record) bool) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, existing := range h.records {
		if matches(existing) {
			return nil
		}
	}
	return h.appendLocked(r)
}

func (h *History) appendLocked(r Record) error {
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

// HasDataset reports whether the history holds a "dataset" record for
// generation - i.e. self-play for that generation has already produced its
// full dataset. Used to refuse restarting self-play on a generation that is
// already done, since a restarted coordinator re-runs its generation from
// scratch.
func (h *History) HasDataset(generation int) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, r := range h.records {
		if r.Generation == generation && r.Type == "dataset" {
			return true
		}
	}
	return false
}
