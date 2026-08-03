package dashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
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
	srv := NewServer(context.Background(), s, h, ct, fc)

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
	srv := NewServer(context.Background(), s, h, ct, fc)

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
	srv := NewServer(context.Background(), s, h, ct, fc)

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
	srv := NewServer(context.Background(), s, h, ct, fc)

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
	srv := NewServer(context.Background(), s, h, ct, fc)

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

func TestFollowTrainerRetriesUntilLogsOpen(t *testing.T) {
	s := NewState()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	fc.streamLogsErrs = []error{
		fmt.Errorf("dashboard: no pods for job train-gen3-1 yet"),
		fmt.Errorf("dashboard: no pods for job train-gen3-1 yet"),
		nil, // third attempt succeeds
	}
	fc.streamLogsData = []byte("epoch 1 loss 0.5\nfinal training loss 0.014205, validation loss 0.015112\n")
	ct := NewController(fc, h, dataDir, 4)
	srv := NewServer(context.Background(), s, h, ct, fc,
		WithFollowerIntervals(5*time.Millisecond, time.Second, 5*time.Millisecond, time.Second))

	logCh := srv.trainLog.Subscribe()
	defer srv.trainLog.Unsubscribe(logCh)

	srv.followTrainer(context.Background(), "train-gen3-1", 3)

	if fc.streamLogsCalls != 3 {
		t.Fatalf("expected 3 StreamJobLogs calls (2 failures + 1 success), got %d", fc.streamLogsCalls)
	}
	recs := h.Records()
	if len(recs) != 1 || recs[0].Type != "training" || recs[0].FinalLoss != 0.014205 || recs[0].Generation != 3 {
		t.Fatalf("expected one training record with FinalLoss 0.014205 gen 3, got %+v", recs)
	}

	select {
	case line := <-logCh:
		if line == "" {
			t.Fatal("expected a non-empty trainlog line")
		}
	default:
		t.Fatal("expected at least one trainlog line to have been published")
	}
}

func TestFollowTrainerGivesUpAfterCap(t *testing.T) {
	s := NewState()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	alwaysFails := make([]error, 0, 100)
	for i := 0; i < 100; i++ {
		alwaysFails = append(alwaysFails, fmt.Errorf("dashboard: no pods for job train-gen3-1 yet"))
	}
	fc.streamLogsErrs = alwaysFails
	ct := NewController(fc, h, dataDir, 4)

	var logBuf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logBuf, nil))
	srv := NewServer(context.Background(), s, h, ct, fc,
		WithLogger(logger),
		WithFollowerIntervals(2*time.Millisecond, 8*time.Millisecond, 5*time.Millisecond, time.Second))

	srv.followTrainer(context.Background(), "train-gen3-1", 3)

	if len(h.Records()) != 0 {
		t.Fatalf("expected no history record when the log never opens, got %+v", h.Records())
	}
	if !strings.Contains(logBuf.String(), "trainer log stream never opened") {
		t.Fatalf("expected give-up log message, got %q", logBuf.String())
	}
	if fc.streamLogsCalls < 2 {
		t.Fatalf("expected multiple retries before giving up, got %d calls", fc.streamLogsCalls)
	}
}

func TestAwaitGateAppendsGateAndPromotionRecords(t *testing.T) {
	s := NewState()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	fc.jobResponses = []JobState{
		{Active: true},
		{Active: true},
		{Active: false, Succeeded: true},
	}
	fc.streamLogsData = []byte("testing a against b\n" +
		"RESULT {\"verdict\":\"promoted\",\"wins\":40,\"losses\":20,\"draws\":40,\"elo_delta\":35.1,\"los\":0.99,\"reason\":\"H1 accepted\"}\n")
	ct := NewController(fc, h, dataDir, 4)
	srv := NewServer(context.Background(), s, h, ct, fc,
		WithFollowerIntervals(time.Second, time.Second, 5*time.Millisecond, time.Second))

	srv.awaitGate(context.Background(), "gate-gen2-1", 2)

	if fc.jobCalls != 3 {
		t.Fatalf("expected 3 Job polls before terminal, got %d", fc.jobCalls)
	}
	recs := h.Records()
	if len(recs) != 2 {
		t.Fatalf("expected gate + promotion records, got %+v", recs)
	}
	if recs[0].Type != "gate" || !recs[0].Promoted || recs[0].Wins != 40 || recs[0].Generation != 2 {
		t.Fatalf("unexpected gate record: %+v", recs[0])
	}
	if recs[1].Type != "promotion" || recs[1].Generation != 2 {
		t.Fatalf("unexpected promotion record: %+v", recs[1])
	}
}

func TestAwaitGateNoPromotionWhenNotPromoted(t *testing.T) {
	s := NewState()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	fc.jobResponses = []JobState{{Active: false, Succeeded: true}}
	fc.streamLogsData = []byte("RESULT {\"verdict\":\"rejected\",\"wins\":10,\"losses\":40,\"draws\":10,\"elo_delta\":-20.0,\"los\":0.01,\"reason\":\"H0 accepted\"}\n")
	ct := NewController(fc, h, dataDir, 4)
	srv := NewServer(context.Background(), s, h, ct, fc,
		WithFollowerIntervals(time.Second, time.Second, 5*time.Millisecond, time.Second))

	srv.awaitGate(context.Background(), "gate-gen2-1", 2)

	recs := h.Records()
	if len(recs) != 1 {
		t.Fatalf("expected only a gate record when not promoted, got %+v", recs)
	}
	if recs[0].Type != "gate" || recs[0].Promoted {
		t.Fatalf("unexpected record: %+v", recs[0])
	}
}

func TestParseResult(t *testing.T) {
	log := []byte("testing a against b\nRESULT {\"verdict\":\"promoted\",\"wins\":40,\"losses\":20,\"draws\":40,\"elo_delta\":35.1,\"los\":0.99,\"reason\":\"H1 accepted\"}\n")
	rec, ok := parseResult(log)
	if !ok || !rec.Promoted || rec.Wins != 40 {
		t.Fatalf("got %+v %v", rec, ok)
	}
}
