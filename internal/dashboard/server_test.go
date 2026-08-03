package dashboard

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/events"
)

func TestStateEndpoint(t *testing.T) {
	s := NewState()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "w1", NodeName: "n1"}, time.Now())
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	ct := NewController(fc, h, dataDir, 4)
	srv := NewServer(s, h, ct, fc)

	req := httptest.NewRequest(http.MethodGet, "/api/state", nil)
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var got struct {
		Boards   []Board `json:"boards"`
		Selfplay struct {
			Workers     any `json:"workers"`
			Coordinator any `json:"coordinator"`
		} `json:"selfplay"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v, body=%s", err, w.Body.String())
	}
	if len(got.Boards) != 1 {
		t.Fatalf("expected 1 board, got %d", len(got.Boards))
	}
}

func TestStateEndpointBoardsEmptyNotNull(t *testing.T) {
	s := NewState()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	ct := NewController(fc, h, dataDir, 4)
	srv := NewServer(s, h, ct, fc)

	req := httptest.NewRequest(http.MethodGet, "/api/state", nil)
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, req)

	if !strings.Contains(w.Body.String(), `"boards":[]`) {
		t.Fatalf("expected boards:[] in body, got %s", w.Body.String())
	}
	if strings.Contains(w.Body.String(), `"boards":null`) {
		t.Fatalf("boards must not be null: %s", w.Body.String())
	}
}

func TestControlEndpointsCallController(t *testing.T) {
	s := NewState()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	ct := NewController(fc, h, dataDir, 4)
	srv := NewServer(s, h, ct, fc)

	req := httptest.NewRequest(http.MethodPost, "/api/selfplay/pause", nil)
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("pause status = %d, body = %s", w.Code, w.Body.String())
	}
	if got := fc.replicas[DeployWorkers]; got != 0 {
		t.Fatalf("expected workers scaled to 0, got %d", got)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/generation/advance", strings.NewReader(`{"to":1}`))
	w = httptest.NewRecorder()
	srv.ServeHTTP(w, req)
	if w.Code != http.StatusConflict {
		t.Fatalf("advance status = %d, body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "promotion") {
		t.Fatalf("expected body to mention promotion, got %s", w.Body.String())
	}
}

func TestSSESendsStateOnChange(t *testing.T) {
	s := NewState()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	ct := NewController(fc, h, dataDir, 4)
	srv := NewServer(s, h, ct, fc)

	ts := httptest.NewServer(srv)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/api/stream", nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET /api/stream: %v", err)
	}
	defer resp.Body.Close()

	go func() {
		time.Sleep(50 * time.Millisecond)
		s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "w1", NodeName: "n1"}, time.Now())
	}()

	buf := make([]byte, 4096)
	var acc strings.Builder
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			acc.Write(buf[:n])
		}
		if strings.Contains(acc.String(), "data:") {
			lines := strings.Split(acc.String(), "\n")
			for _, l := range lines {
				if strings.HasPrefix(l, "data:") {
					payload := strings.TrimSpace(strings.TrimPrefix(l, "data:"))
					var snap Snapshot
					if jerr := json.Unmarshal([]byte(payload), &snap); jerr == nil {
						return
					}
				}
			}
		}
		if err != nil {
			t.Fatalf("read: %v", err)
		}
	}
	t.Fatal("timed out waiting for SSE data")
}

func TestSPAServedAtRoot(t *testing.T) {
	s := NewState()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	ct := NewController(fc, h, dataDir, 4)
	srv := NewServer(s, h, ct, fc)

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.Contains(ct, "text/html") {
		t.Fatalf("content-type = %q, want text/html", ct)
	}
}

func TestParseFinalLoss(t *testing.T) {
	l, ok := parseFinalLoss("final training loss 0.014205, validation loss 0.015112")
	if !ok || l != 0.014205 {
		t.Fatalf("got %v %v", l, ok)
	}
	if _, ok := parseFinalLoss("epoch  3  loss 0.02"); ok {
		t.Fatal("epoch lines are not the final loss")
	}
}

func TestParseResult(t *testing.T) {
	log := []byte("testing a against b\nRESULT {\"verdict\":\"promoted\",\"wins\":40,\"losses\":20,\"draws\":40,\"elo_delta\":35.1,\"los\":0.99,\"reason\":\"H1 accepted\"}\n")
	rec, ok := parseResult(log)
	if !ok || !rec.Promoted || rec.Wins != 40 {
		t.Fatalf("got %+v %v", rec, ok)
	}
}
