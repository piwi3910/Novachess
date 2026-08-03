# Worker-Card Metrics + Start Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CPU/memory mini-graphs on each worker card (metrics-server-fed), and a Start guardrail that refuses to restart self-play for a generation whose dataset is already complete.

**Architecture:** Spec: `docs/superpowers/specs/2026-08-03-worker-card-metrics-design.md`. A `Metrics` method on the existing `Cluster` interface (metrics client alongside client-go), a 15s poller feeding `State.NoteMetrics`, two new omitempty Board fields riding the existing SSE frames, two SVG sparklines per card. The guardrail mirrors the trainer's force pattern: `StartSelfplay` gains a `force` flag and refuses with `RefusedError` when history holds a dataset record for the coordinator's configured generation.

**Tech Stack:** Go 1.26, k8s.io/metrics v0.34.x, existing React/Vite SPA.

## Global Constraints

- Before ANY push: `gofmt -l .` silent, `go vet ./...` clean, `go test -race ./...` passing, `GOOS=linux GOARCH=arm64 go build ./...` succeeding; `cd web/dashboard && npm test && npm run build` green with the dist placeholder restored (check `git status`).
- Coordinator replicas only ever 0/1 via the existing `scaleCoordinator` chokepoint — the guardrail must not add a second scaling path.
- The dashboard never consumes `NOVA_WORK` or publishes to `nova.*`. No cgo. Tests assert invariants.
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

| Path | Responsibility |
|---|---|
| `internal/dashboard/cluster.go` (modify) | `PodMetrics` type, `Metrics` + `CoordinatorGeneration` on `Cluster` + impl |
| `internal/dashboard/state.go` (modify) | Board fields, `NoteMetrics` join |
| `internal/dashboard/server.go` (modify) | metrics poller goroutine; `selfplay` handler force plumbing |
| `internal/dashboard/control.go` (modify) | `StartSelfplay(ctx, force)` guardrail |
| `deploy/dashboard.yaml` (modify) | metrics.k8s.io RBAC rule |
| `web/dashboard/src/` (modify) | Board type, card sparklines, `metricScale`/`fmtMillicores`/`fmtBytes`, Start confirm flow |
| `go.mod` (modify) | `k8s.io/metrics@v0.34.1` |

---

### Task 1: Metrics collection (Cluster → poller → State → snapshot)

**Files:**
- Modify: `internal/dashboard/cluster.go`, `internal/dashboard/state.go`, `internal/dashboard/server.go`, `cmd/dashboard/main.go` (only if wiring needs it — the poller belongs to the Server like the followers)
- Test: `internal/dashboard/metrics_test.go`

**Interfaces:**
- Produces:

```go
type PodMetrics struct {
	CPUMillicores int64 `json:"cpu_millicores"`
	MemoryBytes   int64 `json:"memory_bytes"`
}
// On Cluster:
Metrics(ctx context.Context) (map[string]PodMetrics, error) // keyed by pod name, namespace-wide
// On State:
NoteMetrics(m map[string]PodMetrics)
// Board gains:
CPUMillicores int64 `json:"cpu_millicores,omitempty"`
MemoryBytes   int64 `json:"memory_bytes,omitempty"`
```

- [ ] **Step 1: Failing tests** in `internal/dashboard/metrics_test.go`:

```go
package dashboard

import (
	"context"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/events"
)

func TestNoteMetricsJoinsByPodName(t *testing.T) {
	s := NewState()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "pod-a"}, time.Now())
	s.NoteMetrics(map[string]PodMetrics{
		"pod-a":    {CPUMillicores: 970, MemoryBytes: 200 << 20},
		"stranger": {CPUMillicores: 5, MemoryBytes: 1 << 20}, // not a board: dropped
	})
	snap := s.Snapshot(time.Now())
	if len(snap.Boards) != 1 {
		t.Fatalf("metrics for unknown pods must not create boards; got %d", len(snap.Boards))
	}
	b := snap.Boards[0]
	if b.CPUMillicores != 970 || b.MemoryBytes != 200<<20 {
		t.Fatalf("metrics not joined: %+v", b)
	}
}

func TestBoardWithoutSampleOmitsMetrics(t *testing.T) {
	s := NewState()
	s.NoteHeartbeat(events.WorkerHeartbeat{WorkerID: "pod-a"}, time.Now())
	b := s.Snapshot(time.Now()).Boards[0]
	if b.CPUMillicores != 0 || b.MemoryBytes != 0 {
		t.Fatal("no sample yet must mean zero values (omitted from JSON)")
	}
}

func TestMetricsPollerFeedsState(t *testing.T) {
	// fakeCluster gains a `metrics map[string]PodMetrics` field and a
	// `metricsErr error`; Metrics() returns them. Use WithMetricsInterval
	// (new ServerOption mirroring WithFollowerIntervals) at ~10ms, build a
	// Server, then waitFor the board to carry the fake's numbers.
	// Also: set metricsErr for a few ticks then clear it — the poller must
	// survive errors and resume.
}
```

- [ ] **Step 2: Verify failure** — `go test ./internal/dashboard/ -run 'TestNoteMetrics|TestBoardWithout|TestMetricsPoller'` fails to build.
- [ ] **Step 3: Implement.**
  - `go get k8s.io/metrics@v0.34.1 && go mod tidy`.
  - cluster.go: add `PodMetrics`, extend `Cluster`, keep a `metricsClient versioned.Interface` (`k8s.io/metrics/pkg/client/clientset/versioned`) on `k8sCluster`, built in `NewCluster` from the same rest config. `Metrics` lists `PodMetricses(namespace)` and sums per pod across containers: `m.Containers[i].Usage.Cpu().MilliValue()` and `.Memory().Value()`. The unexported constructor `newClusterWith` keeps its signature for existing tests; the metrics client rides only the real implementation (fake path goes through the interface).
  - state.go: add the two Board fields; `NoteMetrics` locks, iterates the map, updates only pods already present in `s.boards`, then notifies once (not per pod).
  - server.go: `pollMetrics(ctx)` goroutine started in `NewServer` next to the follower wiring: ticker at `s.metricsInterval` (default 15s, new `WithMetricsInterval` ServerOption for tests), calls `s.cluster.Metrics`, on success `s.state.NoteMetrics`; on error logs once per failure streak (a `failing bool` latch — log on first failure and on recovery, not per tick).
- [ ] **Step 4: Run** the three tests + full package under `-race` → PASS.
- [ ] **Step 5: Gates and commit**

```bash
git add go.mod go.sum internal/dashboard/
git commit -m "Collect per-pod CPU and memory for the board cards

metrics-server already runs on the cluster; a 15-second poll (its own
resolution) joins pod usage onto the boards by pod name and rides the
SSE frames that already flow. Errors latch one log line per failure
streak - a missing metrics-server degrades to cards without graphs,
never to a broken dashboard.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Start guardrail

**Files:**
- Modify: `internal/dashboard/cluster.go` (add `CoordinatorGeneration`), `internal/dashboard/control.go`, `internal/dashboard/server.go`
- Test: extend `internal/dashboard/control_test.go`, `internal/dashboard/server_test.go`

**Interfaces:**
- `Cluster` gains `CoordinatorGeneration(ctx context.Context) (int, error)` — reads the coordinator Deployment's `NOVA_GENERATION` env (strconv; missing env = 0, matching the binary's default).
- `Controller.StartSelfplay(ctx context.Context, force bool) error` — refuses with `&RefusedError{...}` when `!force` and the history holds a `dataset` record for the coordinator's configured generation: a restarted coordinator re-runs its generation from scratch, so starting self-play on a completed generation redoes finished work.
- `POST /api/selfplay/start` accepts optional JSON body `{"force":bool}` (absent body = false). Other selfplay actions unchanged.
- Frontend: on a 409 from start containing "already complete", `window.confirm("Generation N's dataset is already complete - starting will redo finished work. Start anyway?")` then retry with force (mirror of the trainer flow in Training.tsx).

- [ ] **Step 1: Failing tests.**

```go
func TestStartRefusedWhenGenerationComplete(t *testing.T) {
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	_ = h.Append(Record{Type: "dataset", Generation: 0, Positions: 1000000, At: time.Now()})
	fc := newFakeCluster() // coordinator env NOVA_GENERATION=0 in its template
	ct := NewController(fc, h, dataDir, 8)
	err := ct.StartSelfplay(context.Background(), false)
	var re *RefusedError
	if !errors.As(err, &re) {
		t.Fatalf("want RefusedError, got %v", err)
	}
	if len(fc.calls) != 0 {
		t.Fatal("a refused start must not touch the cluster")
	}
	if err := ct.StartSelfplay(context.Background(), true); err != nil {
		t.Fatalf("force must override: %v", err)
	}
}

func TestStartAllowedWhenGenerationOpen(t *testing.T) {
	// empty history -> StartSelfplay(ctx, false) succeeds and scales
	// coordinator to 1 and workers to 8 (assert via fc.calls as the
	// existing selfplay tests do).
}
```

Server test: `POST /api/selfplay/start` with dataset record present → 409 whose body mentions "complete"; with `{"force":true}` → 200.

- [ ] **Step 2: Verify failure.** Note `TestCoordinatorCanNeverExceedOne` walks every control method — it will need `StartSelfplay(ctx, true)` (or a history without the dataset record) so the walk still exercises the scaling path; adjust it deliberately, not incidentally.
- [ ] **Step 3: Implement** per the interfaces above. `CoordinatorGeneration` on fakeCluster reads the same env the fake's deployment template carries. In the selfplay handler, decode an optional body only for the `start` action (`decodeJSON` on empty body must not error — read the existing helper; if it requires a body, branch: only decode when `r.ContentLength > 0`).
- [ ] **Step 4: Frontend confirm flow** in Generation.tsx: `run("start")` catches the 409, matches /complete/i, confirms, retries `post("/api/selfplay/start", {force:true})`. The notice for a forced start reads "Started — redoing generation N".
- [ ] **Step 5: Run** all Go + vitest suites → PASS.
- [ ] **Step 6: Gates and commit**

```bash
git add internal/dashboard/ web/dashboard/src/
git commit -m "Refuse to restart self-play for a completed generation

A restarted coordinator re-runs its configured generation from scratch,
so Start on a generation whose dataset is already announced redoes
finished work - exactly what happened after generation zero completed.
Start now refuses with a 409 unless forced, mirroring the trainer's
coordinator warning, and the UI confirms before forcing.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Card graphs, RBAC, ship

**Files:**
- Modify: `web/dashboard/src/types.ts`, `src/lib.ts`, `src/lib.test.ts`, `src/components/Boards.tsx`, `src/App.css`, `deploy/dashboard.yaml`

- [ ] **Step 1: Failing vitest** for the pure helpers in `src/lib.test.ts`:

```ts
describe("metricScale", () => {
  it("clamps to [0,1]", () => {
    expect(metricScale(500, 1000)).toBe(0.5);
    expect(metricScale(2000, 1000)).toBe(1);
    expect(metricScale(-5, 1000)).toBe(0);
    expect(metricScale(5, 0)).toBe(0);
  });
});

describe("fmtMillicores", () => {
  it("prints millicores under a core and cores above", () => {
    expect(fmtMillicores(970)).toBe("970m");
    expect(fmtMillicores(1500)).toBe("1.50");
  });
});

describe("fmtBytes", () => {
  it("prints Mi and Gi", () => {
    expect(fmtBytes(212 * 1024 * 1024)).toBe("212Mi");
    expect(fmtBytes(3 * 1024 * 1024 * 1024)).toBe("3.00Gi");
  });
});
```

- [ ] **Step 2: Implement helpers** in lib.ts:

```ts
// Fraction of a resource ceiling, clamped so a spike over the request or a
// zero ceiling can never break the bar geometry.
export function metricScale(value: number, max: number): number {
  if (max <= 0 || value <= 0) return 0;
  return Math.min(1, value / max);
}

export function fmtMillicores(m: number): string {
  return m < 1000 ? `${Math.round(m)}m` : (m / 1000).toFixed(2);
}

export function fmtBytes(b: number): string {
  const mi = b / (1024 * 1024);
  return mi < 1024 ? `${Math.round(mi)}Mi` : `${(mi / 1024).toFixed(2)}Gi`;
}
```

- [ ] **Step 3: Cards.** types.ts Board gains `cpu_millicores?: number; memory_bytes?: number;`. Boards.tsx: extend the existing per-worker rolling-window ref (same ~30-sample pattern as nps) with cpu and mem series; under the nps sparkline render two labeled rows, each a small sparkline plus the current value: `cpu 970m`, `mem 212Mi`. Scale cpu sparkline y-axis against 1000m and mem against 512Mi (`metricScale`); render nothing (row absent) while the field is undefined. Styling in App.css consistent with the card's existing rows.
- [ ] **Step 4: RBAC.** In deploy/dashboard.yaml's Role:

```yaml
  - apiGroups: ["metrics.k8s.io"]
    resources: ["pods"]
    verbs: ["get", "list"]
```

with a one-line comment in the manifest's voice ("read-only pod usage for the board cards"). Validate: `kubectl --context kw apply -k deploy/ --dry-run=server`.
- [ ] **Step 5: Full gates, commit**

```bash
git add web/dashboard/src/ deploy/dashboard.yaml
git commit -m "Draw CPU and memory on the worker cards

Two sparklines per card under nodes/sec, scaled against the 1-core
request and the 512Mi limit. CPU sagging alongside nps distinguishes a
throttling board from an idle one; memory creeping toward the limit
warns of an OOM kill before it lands.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Ship (controller):** push, PR, CI green, merge with merge commit, pin kustomization to the CI image, apply with worker/coordinator scale captured and restored, verify: `/api/state` boards carry `cpu_millicores`/`memory_bytes` within ~30s, and `POST /api/selfplay/start` without force returns 409 while gen 0's dataset record exists.

## Self-review notes

- Spec coverage: collector/poller/State join (T1), Board fields + omitempty (T1), UI sparklines + helpers (T3), RBAC (T3), failure latch logging (T1), guardrail addendum (T2 — approved by Pascal alongside the spec).
- `TestCoordinatorCanNeverExceedOne` adjustment is called out explicitly in T2 so the walk keeps exercising scaling.
- Types consistent: `PodMetrics`, `NoteMetrics`, `Metrics`, `CoordinatorGeneration`, `metricScale`/`fmtMillicores`/`fmtBytes` named once and reused.
