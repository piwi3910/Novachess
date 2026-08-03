# Architecture

Novachess is a chess engine plus a distributed self-play training pipeline. The
two have very different shapes, and the boundary between them is the single most
important structural decision in the project.

## The boundary: what is a service and what is not

**The engine is not a microservice, and its internals must never become
services.** Alpha-beta search evaluates millions of positions per second. Every
node calls move generation and evaluation directly, and the whole design depends
on those calls being function calls — inlined, allocation-free, sharing an L1
cache with the search stack.

A network round trip is on the order of 100µs. An in-process evaluation call is
on the order of 100ns. Putting a network boundary inside the search loop is a
factor of a thousand, which does not degrade the engine so much as delete it: an
engine searching thousands of nodes per second instead of millions plays
materially worse chess than one running on a phone.

This is not a preference about coupling. It is the reason the engine exists as a
single linked binary, and it is why `internal/chess`, `internal/search` and
`internal/eval` are packages rather than deployments. Even in-process foreign
function calls were rejected for evaluation — cgo's ~50ns of call overhead is
already too expensive at this call frequency.

**Everything above the engine is genuinely distributed**, and is built as
services. Self-play generation, training, gating and the Lichess bot have
independent lifecycles, different scaling profiles, and are naturally
asynchronous. They coordinate over a message bus.

The rule: services are processes *around* the engine, not pieces *of* it. Each
service links the engine as a library.

## Services

| Service | Role | Replicas |
|---|---|---|
| `novachess` | UCI engine binary; also the library every service links | n/a |
| `coordinator` | issues work units, tracks generations, assembles datasets | 1 |
| `selfplay-worker` | plays games, emits scored positions | many |
| `trainer` | consumes a dataset, produces a candidate network | 1, on demand |
| `gatekeeper` | runs SPRT matches, promotes or rejects a candidate | bursts wide |
| `novabot` | Lichess Bot API client, plays with the promoted network | 1 |

`selfplay-worker` is where nearly all the compute goes. Data generation is
roughly 80-90% of the pipeline's wall clock; everything else is comparatively
cheap and intermittent.

## Message bus: NATS with JetStream

NATS is a single static Go binary of around 15MB with first-class arm64
support. On a cluster of ARM single-board computers that matters: a JVM-based
broker would spend memory and operational attention on throughput this pipeline
does not need. Self-play coordination is a low-rate, small-message workload —
tens of messages per second, not millions.

JetStream supplies the delivery semantics the pipeline actually depends on:

- **Durable work queues.** A work unit is delivered to exactly one worker in the
  queue group. If that worker's pod is evicted mid-unit, the unassigned message
  is redelivered rather than lost.
- **At-least-once with explicit ack.** Workers ack only after their batch is
  durably stored, so a crash between "generated" and "uploaded" replays the unit
  instead of silently dropping games.
- **Replay from a sequence.** A new consumer can rebuild state from the stream,
  which makes the coordinator restartable without a separate database.

At-least-once means duplicate batches are possible, so every artifact carries
the work unit ID that produced it and the coordinator deduplicates on it.

## Control plane and data plane are separate

**Bus messages carry references, never bulk data.**

A self-play batch is several megabytes of packed positions. Routing that through
the broker would trade a system that coordinates comfortably for one that
spends its time moving bytes it did not need to touch.

So: artifacts — game batches, datasets, network weights — are written to object
storage (MinIO, on the nodes' NVMe drives), and the bus carries a URI plus
metadata. The bus coordinates; the NVMe drives move the data.

This also makes the bus messages small enough to log, replay and inspect
without special tooling, which matters more than it sounds when debugging a
pipeline spread across dozens of pods.

## Message flow

```
                    ┌───────────────┐
                    │  coordinator  │
                    └───────┬───────┘
                            │ work.assign          (queue group:
                            ▼                       one worker per unit)
                 ┌──────────────────────┐
                 │  selfplay-worker ×N  │
                 └──────────┬───────────┘
              writes batch  │
              to MinIO ─────┤
                            │ games.produced       (URI + stats + NPS)
                            ▼
                    ┌───────────────┐
                    │  coordinator  │  assembles dataset from recent
                    └───────┬───────┘  generations, deduplicates
                            │ dataset.ready
                            ▼
                     ┌────────────┐
                     │  trainer   │
                     └─────┬──────┘
                           │ net.candidate
                           ▼
                    ┌──────────────┐
                    │  gatekeeper  │  SPRT vs. the incumbent
                    └──────┬───────┘
                           │ net.promoted / net.rejected
                           ▼
                  ┌──────────────────┐
                  │ novabot, workers │  pick up the new network
                  └──────────────────┘
```

The loop is closed: a promoted network becomes the network the next round of
self-play uses, so each generation generates better data than the last.

`gatekeeper` is what keeps the loop improving rather than drifting. A candidate
network is only promoted if it beats the incumbent under a sequential
probability ratio test — without that gate, training noise accumulates and the
engine can get quietly worse across generations while every individual step
looks fine.

## Variants

Classical chess and Chess960 share one move generator. Castling is encoded
internally as king-takes-rook, and the rook origins and required-empty squares
are stored per position rather than assumed, so nothing in the generator
branches on the variant. Only two things do: how a castling move is written in
UCI, and how the castling field is written in FEN.

That matters for the pipeline because self-play, training and gating all run on
classical positions. Chess960 support costs the pipeline nothing — no separate
code path to keep verified, no second network — while letting the Lichess bot
accept variant challenges rather than decline them.

## Determinism

Every node in the cluster must produce identical results for identical inputs.
This is a hard requirement, not an aspiration:

- Magic bitboard constants are found from a fixed seed at startup.
- Zobrist keys are generated from a fixed seed.
- Work units carry an explicit RNG seed; workers derive all randomness from it.
- Self-play searches use fixed node counts, never wall-clock time limits.
- **Self-play workers run one search thread each.** Lazy SMP works by letting
  threads race on a shared transposition table, so a multithreaded search is
  non-deterministic by construction. Scale a node by running more worker pods,
  never by raising a worker's thread count. The gatekeeper's match play may use
  as many threads as it likes, since nothing downstream depends on reproducing
  those games.

The last point is the one that is easy to get wrong. A time-based search returns
different results on a thermally throttled Orange Pi than on an idle one, which
would make the training data depend on which node happened to run it — a
correctness bug that is nearly impossible to diagnose from the resulting network.

## Hardware notes

The pipeline targets a Kubernetes cluster of eight Orange Pi 5 Plus boards
(RK3588: 4× Cortex-A76, 4× Cortex-A55, 32GB RAM, NVMe), which is about 32 fast
cores in aggregate.

- **Do not set CPU limits on worker pods.** Kubernetes enforces limits via CFS
  quota; periodic throttling stalls a search mid-node and wrecks throughput. Use
  requests only, or the static CPU manager policy for real pinning.
- **Workers buffer to local NVMe before uploading**, so a coordinator restart or
  rolling update costs one batch rather than a shift's work.
- **Workers report measured NPS** on every batch, which makes thermal throttling
  visible in the data instead of a thing to guess about.
- **The NPU is unused.** RKNN is inference-only and batch-oriented, with
  per-dispatch latency in the milliseconds. NNUE evaluation is a sparse,
  incrementally-updated, one-position-at-a-time workload called millions of
  times per second — the exact opposite of what the NPU accelerates. The useful
  hardware on these boards is the 32 ARM cores.
- **arm64 has no SIMD path in Go.** `simd/archsimd` is amd64-only and gated
  behind `GOEXPERIMENT=simd`, so evaluation runs scalar on the cluster. It sits
  behind an interface with a portable implementation so an accelerated backend
  can be added without touching callers.

## Repository layout

```
cmd/novachess/          UCI engine binary
cmd/novabot/            Lichess bot client
cmd/coordinator/        work distribution and dataset assembly
cmd/selfplay-worker/    game generation
cmd/trainer/            network training
cmd/gatekeeper/         SPRT gating

internal/chess/         board, move generation, Zobrist, perft
internal/search/        negamax, transposition table, time management
internal/eval/          evaluation (hand-crafted, then NNUE)
internal/uci/           UCI protocol
internal/lichess/       Lichess Bot API client
internal/train/         self-play, data format, optimizer

internal/events/        message contracts shared by all services
internal/bus/           message bus abstraction and NATS implementation
internal/store/         artifact storage abstraction
```

`internal/events` and `internal/bus` are the cross-service contracts. They are
deliberately small and free of engine types: a service should be able to consume
the pipeline's messages without linking the search.
