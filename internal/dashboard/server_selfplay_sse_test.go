package dashboard

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The pause/resume buttons are only believable if the counts they change
// reach every open page. That means the SSE state event must carry the
// selfplay replica counts, and a successful control action must push a
// fresh frame even when no heartbeats are flowing (paused workers send
// none).
func TestSSEStateCarriesSelfplayAndControlActionsPushIt(t *testing.T) {
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
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	go func() {
		time.Sleep(100 * time.Millisecond)
		r, perr := http.Post(ts.URL+"/api/selfplay/pause", "application/json", nil)
		if perr == nil {
			r.Body.Close()
		}
	}()

	// The connect-time frame arrives immediately; the pause must produce a
	// later frame whose selfplay worker count is 0, without any heartbeat.
	buf := make([]byte, 4096)
	var acc strings.Builder
	deadline := time.Now().Add(4 * time.Second)
	for time.Now().Before(deadline) {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			acc.Write(buf[:n])
		}
		frames := strings.Split(acc.String(), "\n\n")
		for _, f := range frames[1:] { // skip the connect-time frame
			if strings.Contains(f, "event: state") && strings.Contains(f, `"selfplay"`) && strings.Contains(f, `"workers":0`) {
				return
			}
		}
		if rerr != nil {
			break
		}
	}
	t.Fatalf("no post-pause SSE state frame with selfplay workers 0; got: %s", acc.String())
}
