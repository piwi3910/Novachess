# Training Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An in-cluster web dashboard showing the 8 boards, generation progress, live training, and per-generation history, with guarded controls for the whole training loop.

**Architecture:** New `cmd/dashboard` binary (collectors over NATS + Kubernetes, JSONL history on the shared PVC, REST+SSE API, server-side guardrails) with a React SPA embedded via `embed.FS`. Spec: `docs/superpowers/specs/2026-08-03-training-dashboard-design.md`.

**Tech Stack:** Go 1.26, `internal/bus` (NATS/JetStream, memory bus for tests), client-go (+ fake clientset for tests), React 19 + Vite + TypeScript + vitest.

## Global Constraints

- Before ANY push: `gofmt -l .` silent, `go vet ./...` clean, `go test -race ./...` passing, `GOOS=linux GOARCH=arm64 go build ./...` succeeding (CLAUDE.md).
- Never change search thread counts, node limits, or seeds; never add time-based search limits.
- Coordinator replicas may only ever be 0 or 1. Never change its `Recreate` strategy.
- The two JetStream streams stay separate; the dashboard NEVER consumes `NOVA_WORK` or publishes to any `nova.*` subject.
- No cgo. The runtime image stays `distroless/static`.
- Tests assert invariants, not exact scores/node counts.
- Commit messages: sentence-case imperative summary, body explains why, trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Module path is `github.com/piwi3910/novachess`.

## File Structure

| Path | Responsibility |
|---|---|
| `internal/train/trainer.go` (modify) | Add per-epoch `Progress` callback to `Params` |
| `cmd/trainer/main.go` (modify) | Print epoch loss live; drop the post-hoc loop |
| `cmd/gatekeeper/main.go` (modify) | Print one machine-readable `RESULT {json}` line |
| `internal/dashboard/state.go` | Shared in-memory state + JSON snapshot types |
| `internal/dashboard/heartbeats.go` | Core-NATS heartbeat collector |
| `internal/dashboard/eventswatch.go` | Durable NOVA_EVENTS collector (progress, datasets, verdicts) |
| `internal/dashboard/history.go` | Append-only JSONL history on the PVC |
| `internal/dashboard/cluster.go` | `Cluster` interface + client-go implementation |
| `internal/dashboard/control.go` | Control actions with guardrails |
| `internal/dashboard/server.go` | HTTP API + SSE + embedded SPA |
| `cmd/dashboard/main.go` | Wiring, env config |
| `web/dashboard/` | React SPA (Vite+TS), `embed.go` exposing `dist` |
| `Dockerfile` (modify) | Node build stage, embed dist, assert 7th binary |
| `deploy/dashboard.yaml` | SA, Role, RoleBinding, Deployment, LoadBalancer Service |
| `deploy/kustomization.yaml` (modify) | Add dashboard.yaml |
| `.github/workflows/ci.yml` (modify) | Frontend build+test job |
| `HANDOVER.md` (modify) | Document the dashboard |

---

### Task 1: Trainer live epoch progress

**Files:**
- Modify: `internal/train/trainer.go` (Params struct ~line 14, epoch loop ~line 311)
- Modify: `cmd/trainer/main.go` (~line 173 post-hoc epoch loop)
- Test: `internal/train/trainer_test.go`

**Interfaces:**
- Produces: `train.Params.Progress func(epoch int, loss float64)` — called once per completed epoch, epoch is 1-based, loss equals the value later found in `Result.EpochLoss[epoch-1]`. Nil-safe.

- [ ] **Step 1: Write the failing test** (add to `internal/train/trainer_test.go`, reuse whatever sample-fixture helper existing tests use — look at an existing `Fit` test in that file and copy its setup):

```go
func TestProgressReportsEveryEpoch(t *testing.T) {
	// Reuse the existing test's sample setup; only Params differ.
	var calls []float64
	var epochs []int
	p := smallTestParams(t) // whatever helper/inline setup the existing Fit test uses
	p.Epochs = 3
	p.Progress = func(epoch int, loss float64) {
		epochs = append(epochs, epoch)
		calls = append(calls, loss)
	}
	res := mustFit(t, p) // same invocation style as the existing Fit test

	if len(calls) != p.Epochs {
		t.Fatalf("Progress called %d times, want %d", len(calls), p.Epochs)
	}
	for i, e := range epochs {
		if e != i+1 {
			t.Fatalf("epoch %d reported as %d; want 1-based ascending", i+1, e)
		}
	}
	for i := range calls {
		if calls[i] != res.EpochLoss[i] {
			t.Fatalf("epoch %d: Progress loss %v != Result.EpochLoss %v", i+1, calls[i], res.EpochLoss[i])
		}
	}
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `go test ./internal/train/ -run TestProgressReportsEveryEpoch -v`
Expected: FAIL — `p.Progress undefined`.

- [ ] **Step 3: Implement.** In `Params` add (with a doc comment in the file's voice):

```go
	// Progress, when non-nil, is called after each epoch with the 1-based
	// epoch number and that epoch's mean training loss — the same value
	// that ends up in Result.EpochLoss. It exists so a long training run
	// can be watched while it happens rather than only summarized after.
	Progress func(epoch int, loss float64)
```

In the epoch loop (after `epochLoss` for the epoch is final, before the next iteration):

```go
		if t.params.Progress != nil {
			t.params.Progress(epoch+1, epochLoss)
		}
```

(If the loop stores the mean into a slice at the end of each epoch, place the call right after that store and pass the stored value — Progress and `EpochLoss` must report identical numbers.)

Do NOT add Progress to `Validate()` checks — nil is valid.

- [ ] **Step 4: Run the test**

Run: `go test ./internal/train/ -run TestProgress -v` — PASS. Then `go test -race ./internal/train/` — PASS.

- [ ] **Step 5: Wire cmd/trainer.** In `cmd/trainer/main.go`, set on the params before Fit:

```go
	params.Progress = func(epoch int, loss float64) {
		fmt.Fprintf(os.Stderr, "  epoch %2d  loss %.6f\n", epoch, loss)
	}
```

and DELETE the post-run loop at ~line 176-178 that prints the same lines from `Result.EpochLoss` (they would now print twice). Keep the `final training loss` line.

- [ ] **Step 6: Full gates and commit**

Run: `gofmt -l . && go vet ./... && go test -race ./... && GOOS=linux GOARCH=arm64 go build ./...`

```bash
git add internal/train/trainer.go internal/train/trainer_test.go cmd/trainer/main.go
git commit -m "Report training loss per epoch as it happens

The trainer printed epoch losses only after Fit returned, so a
multi-hour run was silent until the end. A Progress callback on
train.Params now reports each epoch live; cmd/trainer prints the same
lines it used to, just during the run instead of after it. This is what
the dashboard's live loss chart tails.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Gatekeeper machine-readable result line

**Files:**
- Modify: `cmd/gatekeeper/main.go` (after the `report` prints, ~line 105)
- Test: none in-repo for main() (repo pattern: cmd/ is thin); the parsing side is tested in Task 6.

**Interfaces:**
- Produces: final stderr line `RESULT {"verdict":"promoted|rejected|inconclusive","wins":N,"losses":N,"draws":N,"games":N,"elo_delta":F,"elo_margin":F,"los":F,"reason":"..."}` — single line, JSON after the `RESULT ` prefix. The dashboard greps the last `RESULT ` line of the job log.

- [ ] **Step 1: Implement.** After the existing `LOS %.1f%%` print, before the exit switch:

```go
	verdictName := map[gate.Verdict]string{
		gate.Accept: "promoted",
		gate.Reject: "rejected",
	}[report.Verdict]
	if verdictName == "" {
		verdictName = "inconclusive"
	}
	summary, _ := json.Marshal(map[string]any{
		"verdict":   verdictName,
		"wins":      report.Wins,
		"losses":    report.Losses,
		"draws":     report.Draws,
		"games":     report.Games,
		"elo_delta": report.EloDelta,
		"elo_margin": report.EloMargin,
		"los":       report.LOS,
		"reason":    report.Reason,
	})
	// One machine-readable line so a dashboard or script can read the
	// outcome from the log without re-deriving it from the exit code.
	fmt.Fprintf(os.Stderr, "RESULT %s\n", summary)
```

Add `"encoding/json"` to imports. NOTE: check `gate.Report` for the exact games field (`Games` is referenced in `explain` at gate.go:157) — if the field is named differently, match it.

- [ ] **Step 2: Gates and commit**

Run: `gofmt -l . && go vet ./... && go test -race ./... && GOOS=linux GOARCH=arm64 go build ./...`

```bash
git add cmd/gatekeeper/main.go
git commit -m "Print a machine-readable RESULT line from the gatekeeper

The exit code is the verdict but carries none of the evidence. A final
RESULT line with the SPRT record, Elo estimate and reason lets the
dashboard record match history without parsing prose.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Dashboard state + heartbeat collector

**Files:**
- Create: `internal/dashboard/state.go`
- Create: `internal/dashboard/heartbeats.go`
- Test: `internal/dashboard/heartbeats_test.go`

**Interfaces:**
- Consumes: `bus.Bus` (`internal/bus`), `events.WorkerHeartbeat`, `events.SubjectWorkerHeartbeat`. Tests use the memory bus — find its constructor in `internal/bus/memory.go` (grep `func New`) and use it exactly as `internal/bus/bus_test.go` does.
- Produces:
  - `dashboard.State` with `NoteHeartbeat(hb events.WorkerHeartbeat, at time.Time)`, `Snapshot(now time.Time) Snapshot`, `Subscribe() (ch <-chan struct{}, cancel func())` (change notification for SSE).
  - `dashboard.Snapshot{Boards []Board, Generation GenProgress, ...}` — JSON tags all lower_snake.
  - `dashboard.Board{WorkerID, NodeName, CurrentUnit, EngineVersion string, NodesPerSecond float64, GamesCompleted int, LastSeen time.Time, Stale bool}`.
  - `dashboard.WatchHeartbeats(ctx context.Context, b bus.Bus, s *State) error`.

- [ ] **Step 1: Write the failing test**

```go
package dashboard

import (
	"context"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
)

func TestHeartbeatsPopulateBoards(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	b := bus.NewMemory() // match the constructor name used in bus_test.go
	s := NewState()
	if err := WatchHeartbeats(ctx, b, s); err != nil {
		t.Fatal(err)
	}
	hb := events.WorkerHeartbeat{
		WorkerID: "w-1", NodeName: "master-11",
		CurrentWorkUnitID: "g0-u000009", NodesPerSecond: 1_150_000,
		GamesCompleted: 12, EngineVersion: "b524b12",
	}
	if err := b.Publish(ctx, events.SubjectWorkerHeartbeat, hb); err != nil {
		t.Fatal(err)
	}
	// The memory bus delivers asynchronously; poll for the invariant.
	deadline := time.Now().Add(2 * time.Second)
	for {
		snap := s.Snapshot(time.Now())
		if len(snap.Boards) == 1 {
			got := snap.Boards[0]
			if got.WorkerID != "w-1" || got.NodeName != "master-11" || got.Stale {
				t.Fatalf("bad board state: %+v", got)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("heartbeat never reached the state")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestBoardsGoStale(t *testing.T) {
	s := NewState()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "w-1"}, time.Now().Add(-time.Minute))
	snap := s.Snapshot(time.Now())
	if !snap.Boards[0].Stale {
		t.Fatal("a board silent for a minute must be stale")
	}
}
```

- [ ] **Step 2: Run to verify failure** — `go test ./internal/dashboard/ -v` → FAIL (package does not exist).

- [ ] **Step 3: Implement `state.go`:**

```go
// Package dashboard collects the training loop's state and serves it to a
// browser, with guarded controls for the steps an operator does by hand.
package dashboard

import (
	"sort"
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/events"
)

// StaleAfter is how long a board can be silent before it is flagged. Workers
// heartbeat every 15 seconds; three misses means the pod or the board is gone.
const StaleAfter = 45 * time.Second

type Board struct {
	WorkerID       string    `json:"worker_id"`
	NodeName       string    `json:"node_name"`
	CurrentUnit    string    `json:"current_unit,omitempty"`
	EngineVersion  string    `json:"engine_version"`
	NodesPerSecond float64   `json:"nodes_per_second"`
	GamesCompleted int       `json:"games_completed"`
	LastSeen       time.Time `json:"last_seen"`
	Stale          bool      `json:"stale"`
}

type GenProgress struct {
	Generation int       `json:"generation"`
	Positions  int       `json:"positions"`
	Batches    int       `json:"batches"`
	Target     int       `json:"target,omitempty"`
	UpdatedAt  time.Time `json:"updated_at"`
}

type Snapshot struct {
	Boards     []Board     `json:"boards"`
	Generation GenProgress `json:"generation"`
}

type State struct {
	mu       sync.Mutex
	boards   map[string]Board
	progress GenProgress
	watchers map[chan struct{}]struct{}
}

func NewState() *State {
	return &State{
		boards:   make(map[string]Board),
		watchers: make(map[chan struct{}]struct{}),
	}
}

func (s *State) NoteHeartbeat(hb events.WorkerHeartbeat, at time.Time) {
	s.mu.Lock()
	s.boards[hb.WorkerID] = Board{
		WorkerID:       hb.WorkerID,
		NodeName:       hb.NodeName,
		CurrentUnit:    hb.CurrentWorkUnitID,
		EngineVersion:  hb.EngineVersion,
		NodesPerSecond: hb.NodesPerSecond,
		GamesCompleted: hb.GamesCompleted,
		LastSeen:       at,
	}
	s.mu.Unlock()
	s.notify()
}

func (s *State) NoteProgress(p GenProgress) {
	s.mu.Lock()
	s.progress = p
	s.mu.Unlock()
	s.notify()
}

func (s *State) Snapshot(now time.Time) Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := Snapshot{Generation: s.progress}
	for _, b := range s.boards {
		b.Stale = now.Sub(b.LastSeen) > StaleAfter
		out.Boards = append(out.Boards, b)
	}
	sort.Slice(out.Boards, func(i, j int) bool { return out.Boards[i].WorkerID < out.Boards[j].WorkerID })
	return out
}

// Subscribe returns a channel that receives a tick whenever state changes,
// and a cancel that must be called when the subscriber goes away.
func (s *State) Subscribe() (<-chan struct{}, func()) {
	ch := make(chan struct{}, 1)
	s.mu.Lock()
	s.watchers[ch] = struct{}{}
	s.mu.Unlock()
	return ch, func() {
		s.mu.Lock()
		delete(s.watchers, ch)
		s.mu.Unlock()
	}
}

func (s *State) notify() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for ch := range s.watchers {
		select {
		case ch <- struct{}{}:
		default: // a slow subscriber keeps its pending tick; never block here
		}
	}
}
```

`heartbeats.go`:

```go
package dashboard

import (
	"context"
	"encoding/json"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
)

// WatchHeartbeats feeds worker heartbeats into the state. Heartbeats travel
// over core NATS, not JetStream, so there is no history to replay: the boards
// view starts empty and fills within one heartbeat interval.
func WatchHeartbeats(ctx context.Context, b bus.Bus, s *State) error {
	_, err := b.Subscribe(ctx, events.SubjectWorkerHeartbeat, func(ctx context.Context, env events.Envelope) error {
		var hb events.WorkerHeartbeat
		if err := json.Unmarshal(env.Payload, &hb); err != nil {
			return err
		}
		s.NoteHeartbeat(hb, time.Now())
		return nil
	})
	return err
}
```

NOTE: check `events.Envelope`'s payload field name/type (events.go ~line 56) and how existing subscribers (e.g. the coordinator's handler, or `internal/worker`) decode it — mirror that exactly; if there is an `env.Decode(&hb)` style helper, use it instead of raw `json.Unmarshal`.

- [ ] **Step 4: Run the tests** — `go test -race ./internal/dashboard/ -v` → PASS.

- [ ] **Step 5: Gates and commit**

```bash
git add internal/dashboard/
git commit -m "Add the dashboard state and heartbeat collector

Last-seen state per board from the core-NATS heartbeats, with a 45s
staleness flag (three missed beats) and a change-notification hook the
SSE layer will fan out from.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Events collector (generation progress, datasets, verdicts)

**Files:**
- Create: `internal/dashboard/eventswatch.go`
- Test: `internal/dashboard/eventswatch_test.go`

**Interfaces:**
- Consumes: `bus.Bus.Subscribe` on `events.SubjectGamesProduced`, `events.SubjectDatasetReady`, `events.SubjectNetPromoted`, `events.SubjectNetRejected`; `*State.NoteProgress`; `*History` (Task 5 — write this task against the History interface below and run its tests after Task 5 lands, or stub with the real type if implementing in order).
- Produces: `dashboard.WatchEvents(ctx, b bus.Bus, s *State, h *History) error`.

Implementation note: in production the bus is `bus.Connect` with `Config.Durable = "dashboard"`, which makes these subscriptions durable JetStream consumers that resume and replay the retained window after a restart — that replay plus the history file is the crash-recovery story. The memory bus in tests delivers only live messages, which is fine: the tests assert handler behavior, not broker retention.

- [ ] **Step 1: Failing test**

```go
func TestGamesProducedAccumulateProgress(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	b := bus.NewMemory()
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
	// Publish the same WorkUnitID twice; batches and positions must count it once.
	// (At-least-once delivery is the transport's contract; dedup is ours.)
	// ...same harness; publish two batches with WorkUnitID "g0-u000001",
	// then waitFor Batches == 1 && Positions == 4000.
}

func TestDatasetReadyRecordsHistory(t *testing.T) {
	// Publish events.DatasetReady{Generation: 0, Positions: 1004378}.
	// waitFor: h.Records() contains a record with Type=="dataset", Generation==0.
}
```

With a shared helper in the test file:

```go
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
```

- [ ] **Step 2: Run to verify failure** — FAIL (`WatchEvents` undefined; History lands in Task 5 — if implementing tasks in order, write Task 5's `history.go` type first or land Tasks 4+5 together; the commit boundary below assumes 4 then 5 compile together).

- [ ] **Step 3: Implement `eventswatch.go`:**

```go
package dashboard

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/events"
)

// WatchEvents follows the durable event stream: batch production feeds the
// generation progress view, dataset announcements and network verdicts feed
// the history. With a durable consumer name the broker replays the retained
// window after a restart, so progress survives a dashboard reschedule.
func WatchEvents(ctx context.Context, b bus.Bus, s *State, h *History) error {
	var mu sync.Mutex
	seen := make(map[string]struct{}) // WorkUnitID dedup: delivery is at-least-once
	var progress GenProgress

	if _, err := b.Subscribe(ctx, events.SubjectGamesProduced, func(ctx context.Context, env events.Envelope) error {
		var batch events.GameBatch
		if err := json.Unmarshal(env.Payload, &batch); err != nil {
			return err
		}
		mu.Lock()
		if batch.Generation != progress.Generation {
			// A new generation resets the running counters; history keeps the old.
			progress = GenProgress{Generation: batch.Generation}
			seen = make(map[string]struct{})
		}
		if _, dup := seen[batch.WorkUnitID]; !dup {
			seen[batch.WorkUnitID] = struct{}{}
			progress.Batches++
			progress.Positions += batch.Positions
			progress.UpdatedAt = time.Now()
			p := progress
			mu.Unlock()
			s.NoteProgress(p)
			return nil
		}
		mu.Unlock()
		return nil
	}); err != nil {
		return err
	}

	if _, err := b.Subscribe(ctx, events.SubjectDatasetReady, func(ctx context.Context, env events.Envelope) error {
		var ds events.DatasetReady
		if err := json.Unmarshal(env.Payload, &ds); err != nil {
			return err
		}
		return h.Append(Record{Type: "dataset", Generation: ds.Generation, Positions: ds.Positions, At: time.Now()})
	}); err != nil {
		return err
	}

	verdict := func(promoted bool) bus.Handler {
		return func(ctx context.Context, env events.Envelope) error {
			var v events.NetworkVerdict
			if err := json.Unmarshal(env.Payload, &v); err != nil {
				return err
			}
			rec := Record{Type: "gate", Generation: v.Generation, Promoted: promoted,
				Wins: v.Wins, Losses: v.Losses, Draws: v.Draws, EloDelta: v.EloDelta, LOS: v.LOS,
				Reason: v.Reason, NetworkURI: v.ArtifactURI, At: time.Now()}
			return h.Append(rec)
		}
	}
	if _, err := b.Subscribe(ctx, events.SubjectNetPromoted, verdict(true)); err != nil {
		return err
	}
	if _, err := b.Subscribe(ctx, events.SubjectNetRejected, verdict(false)); err != nil {
		return err
	}
	return nil
}
```

(Nothing publishes `nova.net.*` today — the dashboard itself appends gate records from job results in Task 7. These subscriptions cost nothing and mean the pipeline can start publishing verdicts later without touching the dashboard. Same envelope-decoding NOTE as Task 3 applies.)

- [ ] **Step 4: Run tests** — after Task 5's `history.go` exists: `go test -race ./internal/dashboard/ -v` → PASS.

- [ ] **Step 5: Commit** (with Task 5, see below).

---

### Task 5: History store

**Files:**
- Create: `internal/dashboard/history.go`
- Test: `internal/dashboard/history_test.go`

**Interfaces:**
- Produces:
  - `dashboard.Record{Type string, Generation int, Positions int, Promoted bool, Wins, Losses, Draws int, EloDelta, LOS float64, Reason, NetworkURI, JobName string, FinalLoss float64, At time.Time}` — JSON tags lower_snake, `omitempty` on the optional numerics' siblings (`reason`, `network_uri`, `job_name`). `Type` is one of `"dataset" | "training" | "gate" | "promotion"`.
  - `dashboard.NewHistory(path string) *History`
  - `(*History).Append(Record) error` — appends one JSON line, fsyncs.
  - `(*History).Records() []Record` — in-memory copy, load-on-construct.
  - `(*History).HasPromotion(generation int) bool`

- [ ] **Step 1: Failing tests**

```go
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
```

- [ ] **Step 2: Verify failure**, **Step 3: Implement:**

```go
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
```

- [ ] **Step 4: Run** `go test -race ./internal/dashboard/ -v` — Tasks 4 and 5 tests all PASS.

- [ ] **Step 5: Gates and commit Tasks 4+5 together**

```bash
git add internal/dashboard/
git commit -m "Count generation progress from the event stream and keep history

A durable consumer on NOVA_EVENTS feeds live progress (deduplicated by
work unit, since delivery is at-least-once) and appends dataset and
verdict facts to an append-only JSONL history on the shared volume. The
stream's 7-day window backfills after a restart; the file is the
long-term memory. A torn final line from a crash mid-append is dropped
on load rather than fatal.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Kubernetes cluster wrapper

**Files:**
- Create: `internal/dashboard/cluster.go`
- Test: `internal/dashboard/cluster_test.go`
- Modify: `go.mod` (add `k8s.io/client-go`, `k8s.io/api`, `k8s.io/apimachinery` — pick the release matching client-go v0.34.x for the v1.34 cluster: `go get k8s.io/client-go@v0.34.1`)

**Interfaces:**
- Produces:

```go
type JobState struct {
	Name      string     `json:"name"`
	Active    bool       `json:"active"`
	Succeeded bool       `json:"succeeded"`
	Failed    bool       `json:"failed"`
	ExitCode  *int32     `json:"exit_code,omitempty"` // gatekeeper: the verdict
	StartedAt *time.Time `json:"started_at,omitempty"`
}

type Cluster interface {
	Replicas(ctx context.Context, deployment string) (int32, error)
	Scale(ctx context.Context, deployment string, replicas int32) error
	SetCoordinatorGeneration(ctx context.Context, generation int, networkURI string) error
	CreateJobFromCron(ctx context.Context, cronName, jobName string, command []string, args []string) error
	Job(ctx context.Context, name string) (JobState, error)
	StreamJobLogs(ctx context.Context, jobName string) (io.ReadCloser, error)
}
```

  plus `NewCluster(namespace string) (Cluster, error)` (in-cluster config) and `newClusterWith(cs kubernetes.Interface, namespace string) *k8sCluster` (unexported seam the tests use with the fake clientset).
- Deployment names as constants: `DeployWorkers = "novachess-worker"`, `DeployCoordinator = "novachess-coordinator"`; cron names `CronTrainer = "novachess-trainer"`, `CronGatekeeper = "novachess-gatekeeper"`.

- [ ] **Step 1: Failing tests** (fake clientset):

```go
package dashboard

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func int32p(v int32) *int32 { return &v }

func testDeployment(name string, replicas int32) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "novachess"},
		Spec: appsv1.DeploymentSpec{
			Replicas: int32p(replicas),
			Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
				Name: "coordinator",
				Env: []corev1.EnvVar{
					{Name: "NOVA_GENERATION", Value: "0"},
					{Name: "NOVA_NETWORK_URI", Value: ""},
				},
			}}}},
		},
	}
}

func TestScaleAndReplicas(t *testing.T) {
	cs := fake.NewSimpleClientset(testDeployment(DeployWorkers, 8))
	c := newClusterWith(cs, "novachess")
	if err := c.Scale(context.Background(), DeployWorkers, 0); err != nil {
		t.Fatal(err)
	}
	n, err := c.Replicas(context.Background(), DeployWorkers)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("replicas = %d after scaling to 0", n)
	}
}

func TestSetCoordinatorGenerationPatchesOnlyItsEnv(t *testing.T) {
	cs := fake.NewSimpleClientset(testDeployment(DeployCoordinator, 0))
	c := newClusterWith(cs, "novachess")
	if err := c.SetCoordinatorGeneration(context.Background(), 1, "file:///data/networks/gen0.nnue"); err != nil {
		t.Fatal(err)
	}
	d, _ := cs.AppsV1().Deployments("novachess").Get(context.Background(), DeployCoordinator, metav1.GetOptions{})
	env := map[string]string{}
	for _, e := range d.Spec.Template.Spec.Containers[0].Env {
		env[e.Name] = e.Value
	}
	if env["NOVA_GENERATION"] != "1" || env["NOVA_NETWORK_URI"] != "file:///data/networks/gen0.nnue" {
		t.Fatalf("env not advanced: %v", env)
	}
}

func TestCreateJobFromCronCopiesTemplateAndOverridesCommand(t *testing.T) {
	cron := &batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{Name: CronTrainer, Namespace: "novachess"},
		Spec: batchv1.CronJobSpec{JobTemplate: batchv1.JobTemplateSpec{
			Spec: batchv1.JobSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				RestartPolicy: corev1.RestartPolicyNever,
				Containers:    []corev1.Container{{Name: "trainer", Image: "novachess:dev", Command: []string{"/usr/local/bin/trainer", "-out", "/data/networks/gen0.nnue"}, Args: []string{"/data/gen0/"}}},
			}}},
		}},
	}
	cs := fake.NewSimpleClientset(cron)
	c := newClusterWith(cs, "novachess")
	err := c.CreateJobFromCron(context.Background(), CronTrainer, "train-gen1",
		[]string{"/usr/local/bin/trainer", "-out", "/data/networks/gen1.nnue"}, []string{"/data/gen1/"})
	if err != nil {
		t.Fatal(err)
	}
	j, err := cs.BatchV1().Jobs("novachess").Get(context.Background(), "train-gen1", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got := j.Spec.Template.Spec.Containers[0]
	if got.Command[2] != "/data/networks/gen1.nnue" || got.Args[0] != "/data/gen1/" {
		t.Fatalf("template not overridden for the generation: %v %v", got.Command, got.Args)
	}
	if got.Image != "novachess:dev" {
		t.Fatal("image must come from the cron template, not be invented")
	}
}
```

- [ ] **Step 2: Verify failure**, **Step 3: Implement `cluster.go`:**

```go
package dashboard

import (
	"context"
	"fmt"
	"io"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

const (
	DeployWorkers     = "novachess-worker"
	DeployCoordinator = "novachess-coordinator"
	CronTrainer       = "novachess-trainer"
	CronGatekeeper    = "novachess-gatekeeper"
)

type k8sCluster struct {
	cs        kubernetes.Interface
	namespace string
}

func NewCluster(namespace string) (Cluster, error) {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("dashboard: not running in a cluster: %w", err)
	}
	cs, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}
	return newClusterWith(cs, namespace), nil
}

func newClusterWith(cs kubernetes.Interface, namespace string) *k8sCluster {
	return &k8sCluster{cs: cs, namespace: namespace}
}

func (c *k8sCluster) Replicas(ctx context.Context, deployment string) (int32, error) {
	s, err := c.cs.AppsV1().Deployments(c.namespace).GetScale(ctx, deployment, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}
	return s.Spec.Replicas, nil
}

func (c *k8sCluster) Scale(ctx context.Context, deployment string, replicas int32) error {
	s, err := c.cs.AppsV1().Deployments(c.namespace).GetScale(ctx, deployment, metav1.GetOptions{})
	if err != nil {
		return err
	}
	s.Spec.Replicas = replicas
	_, err = c.cs.AppsV1().Deployments(c.namespace).UpdateScale(ctx, deployment, s, metav1.UpdateOptions{})
	return err
}

// SetCoordinatorGeneration rewrites exactly NOVA_GENERATION and
// NOVA_NETWORK_URI on the coordinator, touching nothing else. Search
// configuration is deliberately unreachable from here.
func (c *k8sCluster) SetCoordinatorGeneration(ctx context.Context, generation int, networkURI string) error {
	deps := c.cs.AppsV1().Deployments(c.namespace)
	d, err := deps.Get(ctx, DeployCoordinator, metav1.GetOptions{})
	if err != nil {
		return err
	}
	env := d.Spec.Template.Spec.Containers[0].Env
	for i, e := range env {
		switch e.Name {
		case "NOVA_GENERATION":
			env[i].Value = fmt.Sprintf("%d", generation)
		case "NOVA_NETWORK_URI":
			env[i].Value = networkURI
		}
	}
	d.Spec.Template.Spec.Containers[0].Env = env
	_, err = deps.Update(ctx, d, metav1.UpdateOptions{})
	return err
}

func (c *k8sCluster) CreateJobFromCron(ctx context.Context, cronName, jobName string, command, args []string) error {
	cron, err := c.cs.BatchV1().CronJobs(c.namespace).Get(ctx, cronName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	spec := *cron.Spec.JobTemplate.Spec.DeepCopy()
	if len(spec.Template.Spec.Containers) != 1 {
		return fmt.Errorf("dashboard: cron %s has %d containers, expected 1", cronName, len(spec.Template.Spec.Containers))
	}
	if command != nil {
		spec.Template.Spec.Containers[0].Command = command
	}
	if args != nil {
		spec.Template.Spec.Containers[0].Args = args
	}
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: jobName, Namespace: c.namespace,
			Labels: map[string]string{"app": "novachess-" + cronName, "novachess.dev/launched-by": "dashboard"}},
		Spec: spec,
	}
	_, err = c.cs.BatchV1().Jobs(c.namespace).Create(ctx, job, metav1.CreateOptions{})
	return err
}

func (c *k8sCluster) Job(ctx context.Context, name string) (JobState, error) {
	j, err := c.cs.BatchV1().Jobs(c.namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return JobState{}, err
	}
	out := JobState{Name: name, Active: j.Status.Active > 0, Succeeded: j.Status.Succeeded > 0, Failed: j.Status.Failed > 0}
	if j.Status.StartTime != nil {
		t := j.Status.StartTime.Time
		out.StartedAt = &t
	}
	// The gatekeeper's exit code is the verdict; dig it out of the pod.
	pods, err := c.cs.CoreV1().Pods(c.namespace).List(ctx, metav1.ListOptions{LabelSelector: "job-name=" + name})
	if err == nil {
		for _, p := range pods.Items {
			for _, st := range p.Status.ContainerStatuses {
				if st.State.Terminated != nil {
					code := st.State.Terminated.ExitCode
					out.ExitCode = &code
				}
			}
		}
	}
	return out, nil
}

func (c *k8sCluster) StreamJobLogs(ctx context.Context, jobName string) (io.ReadCloser, error) {
	pods, err := c.cs.CoreV1().Pods(c.namespace).List(ctx, metav1.ListOptions{LabelSelector: "job-name=" + jobName})
	if err != nil {
		return nil, err
	}
	if len(pods.Items) == 0 {
		return nil, fmt.Errorf("dashboard: no pods for job %s yet", jobName)
	}
	req := c.cs.CoreV1().Pods(c.namespace).GetLogs(pods.Items[0].Name, &corev1.PodLogOptions{Follow: true})
	return req.Stream(ctx)
}
```

(Also put the `JobState` and `Cluster` declarations from the Interfaces block at the top of this file, and the unused-import check will force `time` — it is used by `JobState.StartedAt`.)

- [ ] **Step 4: Run** `go test -race ./internal/dashboard/ -v` → PASS. Note: `fake.NewSimpleClientset` may be deprecated in favor of `fake.NewClientset` in v0.34 — use whichever `go vet` accepts cleanly.

- [ ] **Step 5: Gates** (arm64 build now compiles client-go — expect it to just work, it is pure Go) **and commit**

```bash
git add go.mod go.sum internal/dashboard/cluster.go internal/dashboard/cluster_test.go
git commit -m "Add the dashboard's Kubernetes wrapper

Scaling, coordinator generation advancement, launching Jobs from the
suspended CronJob templates with per-generation paths, job status with
the gatekeeper's exit-code verdict, and log streaming - behind a small
interface so the control layer tests against the fake clientset.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Control actions with guardrails

**Files:**
- Create: `internal/dashboard/control.go`
- Test: `internal/dashboard/control_test.go`

**Interfaces:**
- Consumes: `Cluster` (Task 6), `*History` (Task 5), `*State` (Task 3).
- Produces:

```go
type Controller struct { /* Cluster, *History, dataDir string, workerReplicas int32 */ }
func NewController(c Cluster, h *History, dataDir string, workerReplicas int32) *Controller
func (ct *Controller) PauseSelfplay(ctx) error        // workers -> 0
func (ct *Controller) ResumeSelfplay(ctx) error       // workers -> workerReplicas
func (ct *Controller) StopSelfplay(ctx) error         // workers -> 0, coordinator -> 0
func (ct *Controller) StartSelfplay(ctx) error        // coordinator -> 1, workers -> workerReplicas
func (ct *Controller) StartTrainer(ctx, generation int, force bool) (jobName string, err error)
func (ct *Controller) StartGatekeeper(ctx, generation int) (jobName string, err error)
func (ct *Controller) AdvanceGeneration(ctx, toGeneration int) error
```

Semantics the tests pin down:
- Every coordinator scale call goes through one private `scaleCoordinator(ctx, n)` that returns an error for any n outside {0,1} — the clamp is structural, not per-caller.
- `StartTrainer(gen)`: refuses with a descriptive error when the coordinator has replicas > 0 and `force` is false. Job name `fmt.Sprintf("train-gen%d-%d", gen, time.Now().Unix())` (unique across retries), command `["/usr/local/bin/trainer", "-out", fmt.Sprintf("/data/networks/gen%d.nnue", gen)]`, args `[fmt.Sprintf("/data/gen%d/", gen)]`.
- `StartGatekeeper(gen)`: candidate must exist on disk (`os.Stat(filepath.Join(dataDir, "networks", fmt.Sprintf("gen%d.nnue", gen)))`) else error. Command `["/usr/local/bin/gatekeeper", "-candidate", "/data/networks/gen<gen>.nnue"]` plus `"-incumbent", "/data/networks/gen<gen-1>.nnue"` only when gen > 0 AND the incumbent file exists (gen 0 gates against the hand-crafted evaluation with no flag).
- `AdvanceGeneration(to)`: requires `History.HasPromotion(to-1)` AND the network file `gen<to-1>.nnue` exists in `dataDir/networks/`, else error; then `SetCoordinatorGeneration(to, "file:///data/networks/gen<to-1>.nnue")` and `scaleCoordinator(1)`.

- [ ] **Step 1: Failing tests.** Write a `fakeCluster` struct in the test file implementing `Cluster` with recorded calls (do NOT use the k8s fake here — this layer's tests care about which methods were called with what, and that coordinator scaling can never exceed 1):

```go
type fakeCluster struct {
	scales     map[string][]int32
	coordEnvTo []string
	jobs       []string
	replicas   map[string]int32
}

func (f *fakeCluster) Replicas(_ context.Context, d string) (int32, error) { return f.replicas[d], nil }
func (f *fakeCluster) Scale(_ context.Context, d string, n int32) error {
	f.scales[d] = append(f.scales[d], n)
	f.replicas[d] = n
	return nil
}
// ... SetCoordinatorGeneration records, CreateJobFromCron appends jobName to f.jobs,
// Job/StreamJobLogs return zero values.

func TestCoordinatorCanNeverExceedOne(t *testing.T) {
	// Walk every public control method; afterwards assert that every recorded
	// scale of DeployCoordinator is 0 or 1.
}

func TestAdvanceRefusesWithoutPromotion(t *testing.T) {
	// Empty history => AdvanceGeneration(1) errors and no cluster call was made.
}

func TestAdvanceAfterPromotion(t *testing.T) {
	// dataDir := t.TempDir(); write dataDir/networks/gen0.nnue (any bytes);
	// history with a promoted gate record for gen 0.
	// AdvanceGeneration(1) => SetCoordinatorGeneration(1, "file:///data/networks/gen0.nnue")
	// recorded, then coordinator scaled to 1.
}

func TestTrainerRefusedWhileCoordinatorRuns(t *testing.T) {
	// replicas[DeployCoordinator]=1 => StartTrainer(0, false) errors mentioning
	// the coordinator; StartTrainer(0, true) succeeds.
}

func TestGen0GatekeeperHasNoIncumbentFlag(t *testing.T) {
	// Record the command passed to CreateJobFromCron; for gen 0 it must not
	// contain "-incumbent".
}
```

- [ ] **Step 2: Verify failure**, **Step 3: Implement `control.go`** exactly per the semantics above. The one subtlety, spelled out:

```go
// scaleCoordinator is the only path that touches the coordinator's replica
// count. Two coordinators would each count half the batches and neither
// would reach its target — a deadlock with no error anywhere — so the clamp
// lives here structurally rather than in each caller's judgment.
func (ct *Controller) scaleCoordinator(ctx context.Context, n int32) error {
	if n != 0 && n != 1 {
		return fmt.Errorf("dashboard: coordinator replicas must be 0 or 1, got %d", n)
	}
	return ct.cluster.Scale(ctx, DeployCoordinator, n)
}
```

- [ ] **Step 4: Run** `go test -race ./internal/dashboard/ -v` → PASS.
- [ ] **Step 5: Gates and commit**

```bash
git add internal/dashboard/control.go internal/dashboard/control_test.go
git commit -m "Add loop controls with the repo's correctness rules enforced

Pause/resume/stop/start self-play, trainer and gatekeeper launch with
per-generation paths, and generation advancement that refuses to run
without a recorded promotion and the network file on disk. The
coordinator's replica count is structurally clamped to zero or one.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: HTTP server, SSE, and the dashboard binary

**Files:**
- Create: `internal/dashboard/server.go`
- Create: `web/dashboard/embed.go`, `web/dashboard/dist/index.html` (committed placeholder), `web/dashboard/.gitignore`
- Create: `cmd/dashboard/main.go`
- Test: `internal/dashboard/server_test.go`

**Interfaces:**
- Consumes: everything above.
- Produces HTTP API (all JSON):
  - `GET /api/state` → `Snapshot` + `{"selfplay":{"workers":N,"coordinator":N}}` + active job states
  - `GET /api/history` → `[]Record`
  - `GET /api/stream` → SSE; events: `state` (full snapshot JSON, sent on change, throttled to ≥1s apart), `trainlog` (`{"line":"..."}` for each trainer log line while a launched trainer job runs)
  - `POST /api/selfplay/{pause|resume|stop|start}` → 200 `{}` or `{"error":...}` with 409/500
  - `POST /api/trainer/start` body `{"generation":0,"force":false}` → `{"job":"train-gen0-...."}`
  - `POST /api/gatekeeper/start` body `{"generation":0}` → `{"job":"gate-gen0-..."}`
  - `POST /api/generation/advance` body `{"to":1}`
- `web/dashboard/embed.go`:

```go
// Package webdist embeds the built dashboard SPA. The committed
// dist/index.html is a placeholder so `go build ./...` works without Node;
// the Docker build overwrites it with the real Vite output.
package webdist

import "embed"

//go:embed all:dist
var Dist embed.FS
```

- `web/dashboard/.gitignore`:

```
dist/*
!dist/index.html
node_modules/
```

- Placeholder `web/dashboard/dist/index.html`:

```html
<!doctype html><meta charset="utf-8"><title>Novachess</title>
<p>Dashboard UI was not built into this binary. Build the image with the
Node stage (the normal Docker/CI build) rather than a bare go build.</p>
```

- [ ] **Step 1: Failing tests** — exercise the server over `httptest` with the fakes already written:

```go
func TestStateEndpoint(t *testing.T) {
	// NewServer(state, history, controller, replicasFn) -> http.Handler
	// seed state with one heartbeat; GET /api/state; decode; assert one board.
}

func TestControlEndpointsCallController(t *testing.T) {
	// POST /api/selfplay/pause -> fakeCluster records workers scale to 0.
	// POST /api/generation/advance with empty history -> 409 and body contains "promotion".
}

func TestSSESendsStateOnChange(t *testing.T) {
	// httptest.NewServer; GET /api/stream with a context; call
	// state.NoteHeartbeat; read until a line starting with "data:" arrives;
	// assert it parses as a Snapshot. Guard with a 5s timeout.
}

func TestSPAServedAtRoot(t *testing.T) {
	// GET / returns 200 and text/html (the embedded placeholder).
}
```

- [ ] **Step 2: Verify failure**, **Step 3: Implement `server.go`.** Shape:

```go
package dashboard

import (
	"encoding/json"
	"io/fs"
	"net/http"
	"time"

	webdist "github.com/piwi3910/novachess/web/dashboard"
)

type Server struct {
	state   *State
	history *History
	ctl     *Controller
	// replicas reads current worker/coordinator scale for /api/state;
	// injected so tests use the fake cluster.
	cluster Cluster
}

func NewServer(s *State, h *History, ctl *Controller, cluster Cluster) http.Handler {
	srv := &Server{state: s, history: h, ctl: ctl, cluster: cluster}
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
```

SSE handler essentials (flush per event, honor client disconnect, ≥1s coalescing):

```go
func (s *Server) stream(w http.ResponseWriter, r *http.Request) {
	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	ch, cancel := s.state.Subscribe()
	defer cancel()
	send := func() {
		snap := s.state.Snapshot(time.Now())
		data, _ := json.Marshal(snap)
		w.Write([]byte("event: state\ndata: "))
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
		case <-throttle.C:
			if dirty {
				dirty = false
				send()
			}
		}
	}
}
```

Control handlers translate controller errors: guardrail refusals → 409, everything else → 500, JSON `{"error": "..."}`. Trainer start also spawns a goroutine that `StreamJobLogs` for the new job, scans lines, forwards each to SSE subscribers as a `trainlog` event, and on the `final training loss %f, validation loss %f` line appends a `training` history record (`FinalLoss`, `JobName`, `Generation`). Gatekeeper start spawns a goroutine polling `Job()` until terminal, then reads the log once (non-follow), finds the last `RESULT ` line, `json.Unmarshal`s it, and appends a `gate` record (and a `promotion` record when `verdict=="promoted"`). Keep both goroutines' logic in small named methods (`followTrainer`, `awaitGate`) so they stay readable; parsing helpers `parseFinalLoss(line string) (float64, bool)` and `parseResult(log []byte) (Record, bool)` are plain functions with their own table tests in `server_test.go`:

```go
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
```

`cmd/dashboard/main.go` (mirror cmd/coordinator's env/flag/slog style):

```go
// Command dashboard serves the training dashboard: the boards, the
// generation, live training, history, and the operator controls.
package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/dashboard"
)

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	urls := strings.Split(env("NOVA_BUS_URLS", "nats://nova-nats:4222"), ",")
	dataDir := env("NOVA_DATA_DIR", "/data")
	namespace := env("NOVA_NAMESPACE", "novachess")
	addr := env("NOVA_HTTP_ADDR", ":8080")
	replicas, err := strconv.Atoi(env("NOVA_WORKER_REPLICAS", "8"))
	if err != nil || replicas < 0 {
		log.Error("bad NOVA_WORKER_REPLICAS")
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	connectCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	b, err := bus.Connect(connectCtx, bus.Config{
		URLs:       urls,
		ClientName: env("HOSTNAME", "dashboard"),
		Durable:    "dashboard",
	})
	cancel()
	if err != nil {
		log.Error("connecting to the bus", "error", err)
		os.Exit(1)
	}
	defer b.Close()

	cluster, err := dashboard.NewCluster(namespace)
	if err != nil {
		log.Error("kubernetes client", "error", err)
		os.Exit(1)
	}

	state := dashboard.NewState()
	history := dashboard.NewHistory(filepath.Join(dataDir, "dashboard", "history.jsonl"))
	ctl := dashboard.NewController(cluster, history, dataDir, int32(replicas))

	if err := dashboard.WatchHeartbeats(ctx, b, state); err != nil {
		log.Error("heartbeat watch", "error", err)
		os.Exit(1)
	}
	if err := dashboard.WatchEvents(ctx, b, state, history); err != nil {
		log.Error("event watch", "error", err)
		os.Exit(1)
	}

	srv := &http.Server{Addr: addr, Handler: dashboard.NewServer(state, history, ctl, cluster)}
	go func() { <-ctx.Done(); srv.Shutdown(context.Background()) }()
	log.Info("starting", "addr", addr, "namespace", namespace)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Error("serving", "error", err)
		os.Exit(1)
	}
}
```

(`os.MkdirAll(filepath.Join(dataDir, "dashboard"), 0o755)` before `NewHistory` — the directory does not exist on first run.)

- [ ] **Step 4: Run** `go test -race ./internal/dashboard/ -v` and the full suite → PASS.
- [ ] **Step 5: Gates and commit**

```bash
git add internal/dashboard/server.go internal/dashboard/server_test.go web/dashboard/ cmd/dashboard/
git commit -m "Serve the dashboard API, SSE stream and embedded SPA

REST for snapshots and history, server-sent events coalesced to one
update a second, control endpoints mapping guardrail refusals to 409,
and job followers that turn trainer logs into live loss events and
gatekeeper results into history records. The SPA is embedded; a
committed placeholder keeps bare go builds working without Node.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: React SPA

**Files:**
- Create: `web/dashboard/package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/api.ts`, `src/types.ts`, `src/lib.ts`, `src/lib.test.ts`, `src/components/Boards.tsx`, `src/components/Generation.tsx`, `src/components/Training.tsx`, `src/App.css`

**Interfaces:**
- Consumes: the HTTP API of Task 8 verbatim (`/api/state`, `/api/history`, `/api/stream` SSE events `state` and `trainlog`, the POST control endpoints).
- Produces: `npm run build` → `dist/` consumed by `embed.go`; `npm test` (vitest) green.

- [ ] **Step 1: Scaffold.** `package.json` (pin what matters, no extra libraries — charts are hand-rolled SVG):

```json
{
  "name": "novachess-dashboard",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "test": "vitest run"
  },
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.6.0",
    "vite": "^6.0.0",
    "vitest": "^2.1.0"
  }
}
```

`vite.config.ts`:

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8080" } }, // local dev against a port-forwarded dashboard
});
```

`src/types.ts` mirrors the Go JSON exactly (lower_snake keys):

```ts
export interface Board {
  worker_id: string;
  node_name: string;
  current_unit?: string;
  engine_version: string;
  nodes_per_second: number;
  games_completed: number;
  last_seen: string;
  stale: boolean;
}
export interface GenProgress {
  generation: number;
  positions: number;
  batches: number;
  target?: number;
  updated_at: string;
}
export interface Snapshot {
  boards: Board[];
  generation: GenProgress;
  selfplay?: { workers: number; coordinator: number };
}
export interface HistoryRecord {
  type: "dataset" | "training" | "gate" | "promotion";
  generation: number;
  positions?: number;
  promoted?: boolean;
  wins?: number; losses?: number; draws?: number;
  elo_delta?: number; los?: number;
  reason?: string; network_uri?: string; job_name?: string;
  final_loss?: number;
  at: string;
}
```

- [ ] **Step 2: Pure logic + failing tests first.** `src/lib.ts`:

```ts
// Estimated completion from the observed rate over the sampled window.
export function eta(positions: number, target: number, ratePerSec: number): number | null {
  if (!target || ratePerSec <= 0 || positions >= target) return null;
  return (target - positions) / ratePerSec;
}

// Rate from a rolling window of (time, positions) samples.
export function rate(samples: Array<[number, number]>): number {
  if (samples.length < 2) return 0;
  const [t0, p0] = samples[0];
  const [t1, p1] = samples[samples.length - 1];
  if (t1 <= t0) return 0;
  return ((p1 - p0) / (t1 - t0)) * 1000;
}

export function parseEpochLine(line: string): { epoch: number; loss: number } | null {
  const m = line.match(/epoch\s+(\d+)\s+loss\s+([\d.]+)/);
  return m ? { epoch: Number(m[1]), loss: Number(m[2]) } : null;
}

export function fmtNps(nps: number): string {
  return nps >= 1e6 ? (nps / 1e6).toFixed(2) + "M" : Math.round(nps / 1e3) + "k";
}
```

`src/lib.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { eta, parseEpochLine, rate } from "./lib";

describe("eta", () => {
  it("is null when done or rate unknown", () => {
    expect(eta(10, 10, 5)).toBeNull();
    expect(eta(0, 10, 0)).toBeNull();
    expect(eta(0, 0, 5)).toBeNull();
  });
  it("divides remaining by rate", () => {
    expect(eta(400, 1000, 60)).toBe(10);
  });
});

describe("rate", () => {
  it("computes positions per second across the window", () => {
    expect(rate([[0, 0], [10_000, 5000]])).toBe(500);
  });
  it("is 0 with one sample", () => {
    expect(rate([[0, 0]])).toBe(0);
  });
});

describe("parseEpochLine", () => {
  it("parses the trainer's live epoch line", () => {
    expect(parseEpochLine("  epoch  3  loss 0.014205")).toEqual({ epoch: 3, loss: 0.014205 });
  });
  it("rejects other lines", () => {
    expect(parseEpochLine("final training loss 0.01, validation loss 0.02")).toBeNull();
  });
});
```

Run `npm test` in `web/dashboard/` — write `lib.ts` stubs first if practicing strict red/green, then fill.

- [ ] **Step 3: `src/api.ts`** — one place that talks to the server:

```ts
import type { HistoryRecord, Snapshot } from "./types";

export async function getState(): Promise<Snapshot> {
  const r = await fetch("/api/state");
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getHistory(): Promise<HistoryRecord[]> {
  const r = await fetch("/api/history");
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function post(path: string, body?: unknown): Promise<unknown> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error ?? r.statusText);
  return data;
}

export function stream(onState: (s: Snapshot) => void, onTrainLog: (line: string) => void): () => void {
  const es = new EventSource("/api/stream");
  es.addEventListener("state", (e) => onState(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("trainlog", (e) => onTrainLog(JSON.parse((e as MessageEvent).data).line));
  // EventSource reconnects on its own; nothing to do on error but log.
  return () => es.close();
}
```

- [ ] **Step 4: Components.** Keep them lean; all styling in `App.css` (dark background, monospace numbers, red highlight for stale boards). `App.tsx` holds the state (snapshot, history, trainer log lines, rolling `[time, positions]` samples for the rate) and renders the three areas; components are presentational:
  - `Boards.tsx`: grid of cards; each shows `node_name`, `fmtNps(nodes_per_second) + " nps"`, `current_unit ?? "idle"`, `engine_version`; card gets class `stale` when `stale`. Keep an in-component map of the last ~30 nps values per worker for a 100×24 SVG polyline sparkline.
  - `Generation.tsx`: progress bar (`positions/target` when target known), `positions.toLocaleString()`, positions/hour from `rate(samples)*3600`, `eta(...)` formatted as `~Xh Ym`; four buttons Pause/Resume/Stop/Start calling `post("/api/selfplay/...")`, disabled while a request is in flight, errors shown in a dismissible banner.
  - `Training.tsx`: when trainer log lines exist, an SVG line chart of `parseEpochLine` points (x epochetc, y loss, axes labeled); the raw last ~10 log lines beneath. Below: history table (one row per generation: positions from `dataset`, final_loss from `training`, verdict cell from `gate` showing `+W -L =D`, elo_delta and `promoted ✓ / rejected ✗`), and the action buttons: "Run trainer" (`post("/api/trainer/start", {generation})`), "Run gatekeeper", "Advance to gen N+1" (only rendered when the latest gate record for the current generation has `promoted: true`).
- [ ] **Step 5: Build + test**

Run: `cd web/dashboard && npm install && npm test && npm run build`
Expected: vitest green; `dist/` produced.
Then from repo root: `go build ./...` still green (embed picks up the real dist locally now).

- [ ] **Step 6: Commit** (note: `package-lock.json` IS committed; `node_modules` and built `dist/*` besides the placeholder are ignored by Task 8's `.gitignore` — verify with `git status`):

```bash
git add web/dashboard/
git commit -m "Add the dashboard SPA

React + Vite, no chart or state libraries: boards with nps sparklines
and staleness, generation progress with measured rate and ETA, live
training loss from the SSE trainlog events, and the generation history
with its controls. Pure logic (rate, ETA, epoch-line parsing) is under
vitest.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---### Task 10: Dockerfile and CI

**Files:**
- Modify: `Dockerfile`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Dockerfile.** Add a web stage BEFORE the Go build stage and copy its output in; extend the binary assertion:

```dockerfile
# The SPA builds first so the Go stage can embed it. package.json is copied
# alone first for layer caching, same reasoning as go.mod below.
FROM --platform=$BUILDPLATFORM node:22 AS web
WORKDIR /web
COPY web/dashboard/package.json web/dashboard/package-lock.json ./
RUN npm ci
COPY web/dashboard/ ./
RUN npm run build
```

In the existing `build` stage, after `COPY . .`, add:

```dockerfile
COPY --from=web /web/dist/ web/dashboard/dist/
```

Extend the assertion line with `test -x /out/dashboard`.

- [ ] **Step 2: CI.** In `.github/workflows/ci.yml` add a job alongside the others:

```yaml
  frontend:
    name: Dashboard frontend
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7

      - uses: actions/setup-node@v4
        with:
          node-version: "22"
          cache: npm
          cache-dependency-path: web/dashboard/package-lock.json

      - name: Install
        run: npm ci
        working-directory: web/dashboard

      - name: Test
        run: npm test
        working-directory: web/dashboard

      # The build is the artifact the Go binary embeds; a frontend that
      # tests green but does not build would still break the image job.
      - name: Build
        run: npm run build
        working-directory: web/dashboard
```

- [ ] **Step 3: Verify the image builds** — push (after full local Go gates) and confirm the CI `Build and push image` job passes with the new stages; check its log for the dashboard binary assertion. Do NOT build locally with Docker (standing instruction: images build in CI).

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .github/workflows/ci.yml
git commit -m "Build the dashboard SPA and binary into the image

A Node stage builds the frontend before the Go stage embeds it, CI
gains a frontend test+build job, and the image asserts the seventh
binary exists like the other six.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Deployment manifests

**Files:**
- Create: `deploy/dashboard.yaml`
- Modify: `deploy/kustomization.yaml` (add `- dashboard.yaml` to resources)

- [ ] **Step 1: Write `deploy/dashboard.yaml`** — follow the comment voice of the other manifests:

```yaml
# The training dashboard: read-mostly, plus the operator controls that were
# previously kubectl commands and YAML edits.
apiVersion: v1
kind: ServiceAccount
metadata:
  name: novachess-dashboard
  namespace: novachess
---
# Everything the dashboard may do, and nothing else. Notably absent: any
# write on pods, any access outside the namespace, and any verb on the
# worker Deployment beyond scaling. The coordinator env patch is what
# advances a generation.
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: novachess-dashboard
  namespace: novachess
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch", "update", "patch"]
  - apiGroups: ["apps"]
    resources: ["deployments/scale"]
    verbs: ["get", "update", "patch"]
  - apiGroups: ["batch"]
    resources: ["cronjobs"]
    verbs: ["get"]
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["get", "list", "watch", "create"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: novachess-dashboard
  namespace: novachess
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: novachess-dashboard
subjects:
  - kind: ServiceAccount
    name: novachess-dashboard
    namespace: novachess
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novachess-dashboard
  namespace: novachess
  labels:
    app: novachess-dashboard
spec:
  replicas: 1
  selector:
    matchLabels:
      app: novachess-dashboard
  template:
    metadata:
      labels:
        app: novachess-dashboard
    spec:
      serviceAccountName: novachess-dashboard
      containers:
        - name: dashboard
          image: novachess:dev
          command: ["/usr/local/bin/dashboard"]
          env:
            - name: NOVA_BUS_URLS
              value: "nats://nova-nats:4222"
            - name: NOVA_DATA_DIR
              value: "/data"
            - name: NOVA_NAMESPACE
              value: "novachess"
            - name: NOVA_WORKER_REPLICAS
              value: "8"
          ports:
            - containerPort: 8080
              name: http
          volumeMounts:
            - name: data
              mountPath: /data
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              memory: 512Mi
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: novachess-data
---
# LoadBalancer so the dashboard has a stable LAN address. No auth by
# decision: the LAN is the trust boundary here.
apiVersion: v1
kind: Service
metadata:
  name: novachess-dashboard
  namespace: novachess
  labels:
    app: novachess-dashboard
spec:
  type: LoadBalancer
  selector:
    app: novachess-dashboard
  ports:
    - name: http
      port: 80
      targetPort: http
```

- [ ] **Step 2: Validate** — `kubectl apply -k deploy/ --dry-run=server` against the cluster (context `kw`).

- [ ] **Step 3: Commit**

```bash
git add deploy/dashboard.yaml deploy/kustomization.yaml
git commit -m "Deploy the dashboard

One replica, least-privilege Role (scale, coordinator env, job creation
from the cron templates, logs), the shared volume for history, and a
LoadBalancer address on the LAN.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Handover documentation + ship

**Files:**
- Modify: `HANDOVER.md`

- [ ] **Step 1: Document.** Add a short section after section 5 ("Running a generation, by hand") titled "5.5 Or from the dashboard": the Service address (`kubectl -n novachess get svc novachess-dashboard`), what it shows, and that its controls enforce the same rules (coordinator 0/1, promotion-gated advance). Update section 7's heartbeat note to mention the dashboard is the normal way to watch them now (the `nats sub` command stays as the raw fallback). Update section 8: item 4 (monitoring) partially addressed — note what the dashboard covers and that Prometheus metrics remain open; items 2 and 3 mitigated via the dashboard's operator flow (coordinator parking is one click; per-generation job paths are computed) — reword rather than delete, since the underlying automation gap remains.

- [ ] **Step 2: Full gates one last time**

Run: `gofmt -l . && go vet ./... && go test -race ./... && GOOS=linux GOARCH=arm64 go build ./...` and `cd web/dashboard && npm test && npm run build`.

- [ ] **Step 3: Commit, push, PR** — per CLAUDE.md: draft PR against `main`, wait for ALL checks (including the new frontend job and the image job — check logs, not just ticks), address review comments, mark ready, **merge commit**, restart branch from main, clean up.

- [ ] **Step 4: Deploy and verify** — after merge: pin `deploy/kustomization.yaml` `newTag` to the merged commit's image tag (follow-up commit, same flow), `kubectl --context kw apply -k deploy/`, then verify: pod Running, `GET /api/state` from the LoadBalancer IP shows 8 boards within 30s, history file created at `/data/dashboard/history.jsonl`, SSE stream updates in a browser, and a Pause→Resume round-trip scales workers 8→0→8 (confirm units are redelivered afterward — `counted a batch` lines resume).

---

## Self-review notes

- Spec coverage: boards/live view (T3, T9), progress (T4), history (T5), live loss (T1, T8, T9), gate verdicts (T2, T8), controls+guardrails (T6, T7, T8), RBAC/deploy (T11), CI/image (T10), handover (T12). Gatekeeper log-parse assumption from the spec resolved by T2 making the line machine-readable.
- The spec's "trainer/start warns unless force" is implemented as 409-with-force flag (T7/T8); the UI sends force only after a confirm dialog — noted here so the UI implementer adds `window.confirm` before retrying with force.
- Type consistency: `Record`, `Snapshot`, `Board`, `GenProgress`, `JobState`, `Cluster` defined once (T3/T5/T6) and consumed by name in T7-T9; TS `types.ts` mirrors the JSON tags exactly.
