# Novachess training dashboard — design

Date: 2026-08-03
Status: approved (Pascal), pre-implementation

## Purpose

One always-on place to see the training loop and drive it: the 8 boards and
their live search speed, the current generation's progress, live training runs,
improvement across generations, and controls for every step an operator does
today with `kubectl` and YAML edits.

Decisions made during brainstorming:

- **In-cluster web app**, a new service in this repo, deployed like the other
  binaries.
- **Full loop control**: pause/resume/stop/start self-play, launch trainer and
  gatekeeper, advance the generation on promotion — no YAML editing.
- **Live training + history**: loss per epoch while the trainer runs, and a
  per-generation history that survives restarts.
- **LAN LoadBalancer, no auth**: reachable from any browser on the home LAN.
- **Approach**: client-go for Kubernetes, React (Vite + TypeScript) frontend.

## Architecture

- `cmd/dashboard` — new Go binary, env-configured like the others
  (`NOVA_BUS_URLS`, `NOVA_DATA_DIR`, plus `NOVA_NAMESPACE`, default
  `novachess`, and `NOVA_HTTP_ADDR`, default `:8080`).
- `internal/dashboard` — collectors, history store, control actions, HTTP
  server (logic outside `cmd`, testable, matching worker/coordinator layout).
- `web/dashboard/` — React SPA, Vite + TypeScript. `npm run build` output is
  embedded into the Go binary via `embed.FS`; the runtime image stays
  distroless and single-binary.
- Dockerfile gains a Node build stage for the SPA before the Go stage. The
  Go build embeds `web/dashboard/dist`. The six existing binaries are joined
  by a seventh; the binary-presence assertion in the Dockerfile is extended.
- `deploy/dashboard.yaml`: Deployment (1 replica), ServiceAccount, Role +
  RoleBinding (namespace-scoped), LoadBalancer Service (port 80 → 8080).
  Mounts the `novachess-data` PVC (history file, network artifacts metadata).

### RBAC (least privilege, namespace `novachess`)

| Resource | Verbs | Why |
|---|---|---|
| pods | get, list, watch | board/pod status, restarts |
| pods/log | get | trainer/gatekeeper live logs |
| deployments | get, list, patch | scale workers/coordinator, advance-generation env patch |
| deployments/scale | get, patch | pause/resume/stop/start |
| cronjobs | get | job templates for trainer/gatekeeper |
| jobs | get, list, create | launch and track training/gating runs |

## Data collectors

1. **Heartbeats** — core NATS subscription to `nova.worker.heartbeat`
   (fire-and-forget, not JetStream — cannot be replayed, so the collector
   keeps last-seen state per worker in memory). A board is `stale` after 45s
   (three missed 15s beats). Fields already carried: worker id, node name,
   current unit, nodes/sec, games completed, uptime, engine version.
2. **Events** — a durable JetStream consumer on `NOVA_EVENTS` with the
   dashboard's own durable name (`dashboard`). Fan-out on this stream is by
   design; `NOVA_WORK` is never consumed. Subjects used:
   `nova.games.produced` (live per-generation batch/position counting),
   `nova.dataset.ready`, `nova.net.candidate`, `nova.net.promoted`,
   `nova.net.rejected`.
3. **Kubernetes** — client-go informers/polling for: worker and coordinator
   replica counts and pod phases, restart counts, running/finished trainer
   and gatekeeper Jobs and their exit-code verdicts. While a trainer Job
   runs, stream its pod log and parse epoch/loss lines for the live chart.
   The gatekeeper's exit code IS the verdict (0 promoted, 1 rejected,
   2 inconclusive, 3 error) — read from the Job's container status.

## History store

`/data/dashboard/history.jsonl` on the shared PVC. Append-only JSONL records:

- `dataset` — generation, positions, batches, completed-at (from
  `nova.dataset.ready`).
- `training` — generation, job name, started/finished, final loss, epochs.
- `gate` — generation, job name, verdict, SPRT score/match record parsed
  from the gatekeeper log's summary line.
- `promotion` — generation, network URI, when.

On startup, replay whatever the 7-day `NOVA_EVENTS` window still holds and
merge by (type, generation, id) so restarts and reinstalls never double-count.
The file is the long-term memory; the stream is the recent-events feed.

## HTTP API

Read:

- `GET /api/state` — full snapshot (boards, generation progress, jobs,
  incumbent network, replica counts).
- `GET /api/history` — generation history from the store.
- `GET /api/stream` — SSE: heartbeat updates, batch counted, job progress
  (epoch/loss), job finished, verdicts, scale changes.

Control (POST, JSON, no auth by decision — LAN only):

- `/api/selfplay/pause` — workers → 0 (units return to the queue via SIGTERM,
  redelivered on resume; safe by design).
- `/api/selfplay/resume` — workers → `NOVA_WORKER_REPLICAS` (dashboard env,
  default 8).
- `/api/selfplay/stop` — workers → 0 and coordinator → 0.
- `/api/selfplay/start` — coordinator → 1, workers → `NOVA_WORKER_REPLICAS`.
- `/api/trainer/start` — create Job from the suspended trainer CronJob
  template with the target generation's paths.
- `/api/gatekeeper/start` — same, gatekeeper template.
- `/api/generation/advance` — patch coordinator env
  (`NOVA_GENERATION`, `NOVA_NETWORK_URI`), then coordinator → 1.

### Server-side guardrails (repo correctness rules, enforced in Go)

- Coordinator replicas are clamped to {0, 1}. No API path can produce 2.
- Worker search configuration (threads, nodes, seeds) is not writable.
  Scaling replica count is the only worker mutation.
- `advance` refuses unless a `promotion` record exists for the target
  generation and the network file exists on the volume.
- `trainer/start` warns (requires `force: true`) if the coordinator is still
  running for that generation.
- The two NATS streams are read-only to the dashboard; it can never publish
  to `nova.work.*` or consume from `NOVA_WORK`.

## UI

Single page, three areas:

1. **Boards** — 8 cards: node name, nodes/sec with a short sparkline, current
   unit, games completed, engine version, stale/down highlight. A board
   thermally throttling shows as a visibly lower sparkline — the heartbeat's
   whole purpose.
2. **Generation** — progress bar (positions vs `NOVA_POSITIONS` target),
   positions/hour, ETA, batches counted, pause/resume/stop/start.
3. **Training & history** — live loss-per-epoch chart while a trainer Job
   runs (log-tail driven); otherwise the generation timeline: positions,
   final loss, gate verdict with SPRT score, incumbent network. Launch
   trainer / launch gatekeeper / advance generation buttons appear in
   context (e.g. advance only after a promotion).

## Failure behavior

- The dashboard is an observer plus a kubectl-equivalent; its death cannot
  corrupt the pipeline. No pipeline component depends on it.
- SSE clients auto-reconnect; the NATS connection uses the same reconnect
  behavior as `internal/bus`.
- Control actions are idempotent Kubernetes operations; failures surface as
  UI toasts with the API error text.
- If the history file is missing or truncated, the dashboard rebuilds what it
  can from the event stream and says so in the UI rather than guessing.

## Testing

- Collectors against a real NATS broker, following the `internal/pipeline`
  end-to-end pattern (assert invariants, not exact counts or timings).
- History store: round-trip, merge/dedup on replay, truncated-file recovery.
- Control handlers against a fake Kubernetes API (`httptest`), asserting the
  guardrails hard: coordinator can never reach 2, worker env never patched,
  advance refused without a promotion record.
- Frontend: vitest for pure logic (ETA math, loss-line parsing, staleness);
  no browser-automation suite in v1.
- CI: Node build + vitest join the pipeline; gofmt/vet/test -race/arm64
  gates unchanged (the SPA is arch-neutral, embedded before compile).

## Out of scope (deliberate)

- Live game/board viewer (needs worker changes; possible later addition).
- Prometheus metrics endpoints (handover item 4) — the collectors make this
  cheap later, but v1 exposes only the dashboard's own API.
- Auth (LAN-only by decision; basic auth is a small later addition if wanted).
- Automated generation hand-off without a human click (handover item 1's
  full automation) — this dashboard deliberately keeps the operator's
  deliberate click while removing the YAML editing.
