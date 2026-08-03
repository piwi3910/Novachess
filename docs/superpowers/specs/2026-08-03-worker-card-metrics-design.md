# Worker-card CPU and memory graphs — design

Date: 2026-08-03
Status: approved (Pascal), pre-implementation

## Purpose

Each worker card on the training dashboard gains two mini-graphs: CPU and
memory for that worker's pod. CPU corroborates the nodes/sec signal (a healthy
searching worker sits pinned near its 1-core request; sagging CPU alongside
sagging nps distinguishes throttling from idling), and memory creeping toward
the 512Mi limit warns of an OOM-kill before it happens.

Scope is deliberately the worker cards only. Coordinator, NATS and the
dashboard itself stay off the board grid.

## Data source

metrics-server via the `metrics.k8s.io/v1beta1` PodMetrics API — present and
healthy on the cluster (k3s ships it). Rejected alternatives: workers
self-reporting cgroup usage in heartbeats (touches worker code and the event
schema for no gain) and a Prometheus stack (two sparklines do not justify a
monitoring system; handover pending-work item 4 remains open and separate).

## Architecture

1. **Cluster interface** gains one method:
   `Metrics(ctx context.Context) (map[string]PodMetrics, error)` where
   `PodMetrics{CPUMillicores int64, MemoryBytes int64}`, keyed by pod name,
   listing the `novachess` namespace. Implemented with the metrics client
   (`k8s.io/metrics/pkg/client/clientset/versioned`) alongside the existing
   clientset; the fake for tests is a map on the existing fakeCluster.
2. **Poller**: a goroutine owned by the server (started like the followers,
   bound to the server-lifetime context) calls `Metrics` every 15 seconds —
   metrics-server's own resolution; faster polling yields duplicate samples —
   and hands the result to `State.NoteMetrics(map[string]PodMetrics)`.
3. **State**: `NoteMetrics` joins by pod name onto the boards map
   (`worker_id` is the pod name), stores the two numbers on the Board, and
   notifies. Boards keep metrics until the board itself is evicted; a board
   with no sample yet has none. Metrics for pods that are not boards
   (coordinator, NATS, dashboard) are ignored at the join.
4. **API**: `Board` gains `cpu_millicores` and `memory_bytes`, both
   `omitempty` — absent until the first sample arrives (metrics-server lags
   ~30s behind pod start), so the UI renders a blank rather than a fake zero.
5. **UI**: two mini-sparklines per card under the nps one, same hand-rolled
   SVG and the same client-side rolling-window pattern (~30 samples). CPU is
   scaled against the 1000m request, memory against the 512Mi limit; both
   also print the current value ("970m", "212Mi"). A pure helper
   `metricScale(value, max)` clamps to [0,1] for the bar/sparkline height
   and is vitest-covered.
6. **RBAC**: one new rule in the dashboard Role — apiGroup
   `metrics.k8s.io`, resource `pods`, verbs get/list.

## Failure behavior

metrics-server unavailable → the poller logs once per failure streak (not
per tick), boards render exactly as today with no graphs, and nothing else
degrades. This feature is read-only decoration on the pipeline.

## Testing

- Fake `Metrics` map on the existing fakeCluster; poller test with injected
  interval (same pattern as the follower tests); State join test including
  "metrics for unknown pods are dropped" and interplay with eviction.
- Snapshot JSON: fields absent before the first sample, present after.
- vitest: `metricScale` clamping and formatting of millicores/bytes.

## Out of scope

- Node-level (board-level) metrics, historical retention, alerts.
- Any change to workers, events, or the bus.
