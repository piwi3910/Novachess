package dashboard

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	webdist "github.com/piwi3910/novachess/web/dashboard"
)

// Server is the dashboard's HTTP surface: read-only snapshots and history,
// a server-sent-events stream of state and live training logs, and the
// control endpoints that map onto the Controller's guarded actions.
type Server struct {
	state   *State
	history *History
	ctl     *Controller
	// cluster reads current worker/coordinator scale for /api/state;
	// injected so tests use the fake cluster.
	cluster Cluster
	// trainLog fans a running trainer job's log lines out to every SSE
	// subscriber as trainlog events. One broadcaster lives for the life of
	// the server; only one trainer job is expected to run at a time, but
	// nothing here assumes that - it is simply a pub/sub of lines.
	trainLog *trainLogBroadcast
}

// NewServer builds the dashboard's HTTP handler.
func NewServer(s *State, h *History, ctl *Controller, cluster Cluster) http.Handler {
	srv := &Server{state: s, history: h, ctl: ctl, cluster: cluster, trainLog: newTrainLogBroadcast()}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/state", srv.getState)
	mux.HandleFunc("GET /api/history", srv.getHistory)
	mux.HandleFunc("GET /api/stream", srv.stream)
	mux.HandleFunc("POST /api/selfplay/{action}", srv.selfplay)
	mux.HandleFunc("POST /api/trainer/start", srv.trainerStart)
	mux.HandleFunc("POST /api/gatekeeper/start", srv.gatekeeperStart)
	mux.HandleFunc("POST /api/generation/advance", srv.advance)
	dist, _ := fs.Sub(webdist.Dist, "dist")
	mux.Handle("/", http.FileServerFS(dist))
	return mux
}

type selfplayInfo struct {
	Workers     any `json:"workers"`
	Coordinator any `json:"coordinator"`
}

func (s *Server) getState(w http.ResponseWriter, r *http.Request) {
	snap := s.state.Snapshot(time.Now())
	out := struct {
		Snapshot
		Selfplay selfplayInfo `json:"selfplay"`
	}{Snapshot: snap}

	ctx := r.Context()
	if n, err := s.cluster.Replicas(ctx, DeployWorkers); err != nil {
		out.Selfplay.Workers = err.Error()
	} else {
		out.Selfplay.Workers = n
	}
	if n, err := s.cluster.Replicas(ctx, DeployCoordinator); err != nil {
		out.Selfplay.Coordinator = err.Error()
	} else {
		out.Selfplay.Coordinator = n
	}

	writeJSON(w, http.StatusOK, out)
}

func (s *Server) getHistory(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.history.Records())
}

func (s *Server) stream(w http.ResponseWriter, r *http.Request) {
	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	ch, cancel := s.state.Subscribe()
	defer cancel()
	logCh := s.trainLog.Subscribe()
	defer s.trainLog.Unsubscribe(logCh)
	send := func() {
		snap := s.state.Snapshot(time.Now())
		data, _ := json.Marshal(snap)
		w.Write([]byte("event: state\ndata: "))
		w.Write(data)
		w.Write([]byte("\n\n"))
		fl.Flush()
	}
	sendLog := func(line string) {
		data, _ := json.Marshal(map[string]string{"line": line})
		w.Write([]byte("event: trainlog\ndata: "))
		w.Write(data)
		w.Write([]byte("\n\n"))
		fl.Flush()
	}
	send()
	throttle := time.NewTicker(time.Second)
	defer throttle.Stop()
	dirty := false
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ch:
			dirty = true
		case line, ok := <-logCh:
			if ok {
				sendLog(line)
			}
		case <-throttle.C:
			if dirty {
				dirty = false
				send()
			}
		}
	}
}

// trainLogBroadcast fans lines from a running trainer job out to every SSE
// subscriber currently connected. It is a small pub/sub kept independent of
// state.go's watcher mechanism because trainlog events are job-scoped, not
// state-scoped, and have no meaningful "current value" to send on connect.
type trainLogBroadcast struct {
	subscribe   chan chan string
	unsubscribe chan chan string
	publish     chan string
	done        chan struct{}
}

func newTrainLogBroadcast() *trainLogBroadcast {
	b := &trainLogBroadcast{
		subscribe:   make(chan chan string),
		unsubscribe: make(chan chan string),
		publish:     make(chan string),
		done:        make(chan struct{}),
	}
	go b.run()
	return b
}

func (b *trainLogBroadcast) run() {
	subs := make(map[chan string]struct{})
	for {
		select {
		case ch := <-b.subscribe:
			subs[ch] = struct{}{}
		case ch := <-b.unsubscribe:
			delete(subs, ch)
			close(ch)
		case line := <-b.publish:
			for ch := range subs {
				select {
				case ch <- line:
				default:
				}
			}
		case <-b.done:
			for ch := range subs {
				close(ch)
			}
			return
		}
	}
}

func (b *trainLogBroadcast) Subscribe() chan string {
	ch := make(chan string, 16)
	b.subscribe <- ch
	return ch
}

func (b *trainLogBroadcast) Unsubscribe(ch chan string) {
	select {
	case b.unsubscribe <- ch:
	case <-b.done:
	}
}

func (b *trainLogBroadcast) Publish(line string) {
	select {
	case b.publish <- line:
	case <-b.done:
	}
}

func (b *trainLogBroadcast) Close() {
	close(b.done)
}

func (s *Server) selfplay(w http.ResponseWriter, r *http.Request) {
	action := r.PathValue("action")
	ctx := r.Context()
	var err error
	switch action {
	case "pause":
		err = s.ctl.PauseSelfplay(ctx)
	case "resume":
		err = s.ctl.ResumeSelfplay(ctx)
	case "stop":
		err = s.ctl.StopSelfplay(ctx)
	case "start":
		err = s.ctl.StartSelfplay(ctx)
	default:
		writeError(w, http.StatusNotFound, fmt.Errorf("dashboard: unknown selfplay action %q", action))
		return
	}
	if err != nil {
		writeControlError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{})
}

func (s *Server) trainerStart(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Generation int  `json:"generation"`
		Force      bool `json:"force"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	job, err := s.ctl.StartTrainer(r.Context(), body.Generation, body.Force)
	if err != nil {
		writeControlError(w, err)
		return
	}
	go s.followTrainer(context.Background(), job, body.Generation)
	writeJSON(w, http.StatusOK, map[string]string{"job": job})
}

func (s *Server) gatekeeperStart(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Generation int `json:"generation"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	job, err := s.ctl.StartGatekeeper(r.Context(), body.Generation)
	if err != nil {
		writeControlError(w, err)
		return
	}
	go s.awaitGate(context.Background(), job, body.Generation)
	writeJSON(w, http.StatusOK, map[string]string{"job": job})
}

func (s *Server) advance(w http.ResponseWriter, r *http.Request) {
	var body struct {
		To int `json:"to"`
	}
	if !decodeJSON(w, r, &body) {
		return
	}
	if err := s.ctl.AdvanceGeneration(r.Context(), body.To); err != nil {
		writeControlError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{})
}

// followTrainer streams the trainer job's log, forwarding each line to SSE
// subscribers as a trainlog event and, on the final-loss line, appending a
// training history record. It runs for the lifetime of the job's logs, so
// it is started with a background context rather than the request's.
func (s *Server) followTrainer(ctx context.Context, jobName string, generation int) {
	rc, err := s.ctl.cluster.StreamJobLogs(ctx, jobName)
	if err != nil || rc == nil {
		return
	}
	defer rc.Close()
	sc := bufio.NewScanner(rc)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		s.trainLog.Publish(line)
		if loss, ok := parseFinalLoss(line); ok {
			_ = s.history.Append(Record{
				Type:       "training",
				Generation: generation,
				FinalLoss:  loss,
				JobName:    jobName,
				At:         time.Now(),
			})
		}
	}
}

// awaitGate polls the gatekeeper job until it is no longer active, reads its
// log once, and turns the last RESULT line into a gate history record (and
// a promotion record, when promoted).
func (s *Server) awaitGate(ctx context.Context, jobName string, generation int) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		js, err := s.ctl.cluster.Job(ctx, jobName)
		if err == nil && !js.Active && (js.Succeeded || js.Failed) {
			break
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
	rc, err := s.ctl.cluster.StreamJobLogs(ctx, jobName)
	if err != nil || rc == nil {
		return
	}
	log, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return
	}
	rec, ok := parseResult(log)
	if !ok {
		return
	}
	rec.Generation = generation
	rec.JobName = jobName
	rec.At = time.Now()
	_ = s.history.Append(rec)
	if rec.Promoted {
		promo := rec
		promo.Type = "promotion"
		_ = s.history.Append(promo)
	}
}

var finalLossRe = regexp.MustCompile(`^final training loss ([0-9.eE+-]+), validation loss ([0-9.eE+-]+)`)

// parseFinalLoss extracts the training loss from the trainer's one final
// summary line, distinguishing it from the many per-epoch progress lines
// that share the word "loss".
func parseFinalLoss(line string) (float64, bool) {
	m := finalLossRe.FindStringSubmatch(line)
	if m == nil {
		return 0, false
	}
	v, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// parseResult finds the last "RESULT {...}" line in a gatekeeper log and
// decodes it into a gate Record. Later RESULT lines win, in case the
// gatekeeper logs more than one for any reason.
func parseResult(log []byte) (Record, bool) {
	sc := bufio.NewScanner(bytes.NewReader(log))
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	var last string
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "RESULT ") {
			last = strings.TrimPrefix(line, "RESULT ")
		}
	}
	if last == "" {
		return Record{}, false
	}
	var payload struct {
		Verdict  string  `json:"verdict"`
		Wins     int     `json:"wins"`
		Losses   int     `json:"losses"`
		Draws    int     `json:"draws"`
		EloDelta float64 `json:"elo_delta"`
		LOS      float64 `json:"los"`
		Reason   string  `json:"reason"`
	}
	if err := json.Unmarshal([]byte(last), &payload); err != nil {
		return Record{}, false
	}
	return Record{
		Type:     "gate",
		Promoted: payload.Verdict == "promoted",
		Wins:     payload.Wins,
		Losses:   payload.Losses,
		Draws:    payload.Draws,
		EloDelta: payload.EloDelta,
		LOS:      payload.LOS,
		Reason:   payload.Reason,
	}, true
}

func decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if r.Body == nil {
		return true
	}
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil && err != io.EOF {
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	return true
}

func writeControlError(w http.ResponseWriter, err error) {
	var refused *RefusedError
	if errors.As(err, &refused) {
		writeError(w, http.StatusConflict, err)
		return
	}
	writeError(w, http.StatusInternalServerError, err)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
