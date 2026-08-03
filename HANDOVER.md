# Novachess — deployment handover

This document is for whoever deploys the Novachess self-play training pipeline
onto the Kubernetes cluster. It assumes you have `kubectl` access to the
cluster and a container registry the cluster can pull from, and that you have
never seen this repository before.

Everything here was verified against the code at merge of PR #10. Where a
number is an estimate rather than a measurement, it says so.

---

## 1. What this system is

Novachess is a UCI chess engine that teaches itself to play by playing itself.
The training loop has four stages, and each stage is a separate binary:

```
  ┌──────────────┐   work units      ┌──────────────────┐
  │ coordinator  │ ────────────────▶ │ selfplay-worker  │  × 8
  │              │                   │ (one per board)  │
  │  issues work │ ◀──────────────── │  plays games     │
  │  counts data │   game batches    └──────────────────┘
  └──────┬───────┘                            │
         │                                    │ writes .novadata
         │ dataset ready                      ▼
         │                            ┌───────────────┐
         │                            │ shared volume │
         │                            └───────────────┘
         ▼                                    │
  ┌──────────────┐                            │ reads all batches
  │   trainer    │ ◀──────────────────────────┘
  │  fits a NNUE │
  └──────┬───────┘
         │ candidate network
         ▼
  ┌──────────────┐
  │  gatekeeper  │   plays candidate vs incumbent, SPRT
  │  promote or  │   exit 0 = promoted, 1 = rejected
  │    reject    │
  └──────────────┘
```

A **generation** is one turn of that loop: workers produce a million positions,
the trainer fits a network to them, the gatekeeper decides whether the new
network is actually better, and if it is, the next generation plays with it.

Generation zero has no network. The workers bootstrap from the hand-crafted
evaluation (`internal/eval`), which is the thing the first network has to beat
in order to be worth using at all.

### The two things that make this different from a normal job queue

**Determinism is a correctness requirement, not a nicety.** The same work unit
replayed on a different board must produce byte-identical games. This is why
each worker searches on **one thread** — lazy SMP is non-deterministic by
construction — and why search is bounded by **node count, never by time**. If
you find yourself tuning `Threads` or adding a time limit to make throughput
look better, you have broken the training data. Scale by adding worker pods.

**Nearly every failure mode in this system is silent.** A worker writing to the
wrong volume, a queue misconfiguration throttling the cluster to one unit at a
time, the coordinator waiting forever for units that already came back — none
of these produce an error in any log. Three of them were caught only by writing
tests that measured throughput rather than checking for errors. Section 6 lists
the ones to actively check for, because nothing will tell you.

---

## 2. Repository layout

| Path | What it is |
|---|---|
| `cmd/novachess` | The UCI engine binary. Not part of the training loop. |
| `cmd/novabot` | The Lichess bot. Not part of the training loop. |
| `cmd/selfplay-worker` | Worker service. Env-configured, runs forever. |
| `cmd/coordinator` | Coordinator service. Env-configured, runs one generation. |
| `cmd/trainer` | CLI. Reads batches, writes a network. Runs once per generation. |
| `cmd/gatekeeper` | CLI. Plays a match, exits with the verdict. Once per generation. |
| `internal/chess` | Board, move generation, magic bitboards, perft. |
| `internal/search` | Alpha-beta, transposition table, lazy SMP. |
| `internal/eval` | Hand-crafted evaluation (the generation-zero baseline). |
| `internal/nnue` | The neural network: features, accumulator, evaluator. |
| `internal/train` | Sample format, self-play generator, the trainer itself. |
| `internal/gate` | SPRT, match runner, the promotion gate. |
| `internal/bus` | NATS JetStream transport. |
| `internal/events` | The message schemas and subject names. |
| `internal/store` | Artifact storage on the shared volume. |
| `internal/worker` | Worker logic (separate from `cmd` so it is testable). |
| `internal/coordinator` | Coordinator logic. |
| `internal/pipeline` | End-to-end tests: real broker, real workers, real volume. |
| `deploy/` | The Kubernetes manifests. Start here. |
| `Dockerfile` | One image, all six binaries. |
| `CLAUDE.md` | Working agreements. Read before changing anything. |

Go 1.26. No cgo anywhere, deliberately — that is what makes the binaries static
and the runtime image `distroless/static`.

---

## 3. Prerequisites

### 3.1 A ReadWriteMany storage class

**This is the one prerequisite not satisfied by the manifests, and the one that
fails silently if you get it wrong.**

Workers on eight different boards write batch files, and the trainer on a ninth
pod reads all of them. That requires a genuinely shared volume. With
`ReadWriteOnce` each pod gets its own view: every write succeeds, every read
returns a fraction of the data, and nothing anywhere reports a problem. You
would discover it as a network that trains on one eighth of its data and plays
badly for no visible reason.

The cluster runs Longhorn, which provides RWX through its share-manager.
Verify before you deploy anything:

```bash
kubectl get storageclass
# expect a class named `longhorn`

# Confirm the share-manager components are present — RWX is not automatic.
kubectl -n longhorn-system get pods | grep -i share-manager
```

If your storage class has a different name, change `storageClassName` in
[storage.yaml](deploy/storage.yaml).

### 3.2 A registry the cluster can pull from

The manifests reference the image `novachess:dev`, which does not exist
anywhere. CI builds and pushes the real image on every push — see section 4.1.
The cluster must be able to pull from `ghcr.io`; if the package is private,
the `novachess` namespace needs an image pull secret.

### 3.3 arm64

The boards are Orange Pi, so `linux/arm64`. CI cross-compiles on every push
(the "Cross-compile for the cluster" job) and the "Build and push image" job
builds the image itself for `linux/arm64`, so an amd64-only image cannot reach
you through CI. If you ever build locally instead, use `docker buildx build
--platform linux/arm64` — a plain `docker build` on an x86 or Apple-silicon
laptop will silently produce an image for the wrong platform that fails with
`exec format error` on the boards.

---

## 4. Deploying

### 4.1 The image

One image contains all six binaries; each Deployment picks one with `command`.
They share the whole engine, so building six images would multiply layers,
pushes and version skew for nothing.

CI's "Build and push image" job builds `linux/arm64` and pushes
`ghcr.io/piwi3910/novachess:<version>` on every push, where `<version>` is
`git describe --always --dirty` for the built commit. `VERSION` is stamped
into every batch the workers produce, so a dataset can later be traced back
to the build that generated it — which is also why the tag is the version
rather than `latest` or `dev`.

The Dockerfile asserts at build time that all six binaries exist, so a missing
binary fails the build rather than the deployment.

Point the manifests at the pushed tag in
[kustomization.yaml](deploy/kustomization.yaml):

```yaml
images:
  - name: novachess
    newName: ghcr.io/piwi3910/novachess
    newTag: <version>
```

### 4.2 Apply

```bash
kubectl apply -k deploy/
```

That creates, in `deploy/`:

- [namespace.yaml](deploy/namespace.yaml) — namespace `novachess`
- [storage.yaml](deploy/storage.yaml) — the 100Gi RWX PVC `novachess-data`
- [nats.yaml](deploy/nats.yaml) — single-node NATS with JetStream (ConfigMap, Service, StatefulSet)
- [worker.yaml](deploy/worker.yaml) — 8 worker replicas, 1 CPU each
- [coordinator.yaml](deploy/coordinator.yaml) — 1 coordinator, `Recreate` strategy
- [jobs.yaml](deploy/jobs.yaml) — trainer and gatekeeper as **suspended** CronJobs

The trainer and gatekeeper do not start. They are suspended CronJobs so that
`kubectl create job --from=cronjob/...` can start one on demand with the spec
already in the cluster. See section 5.

### 4.3 Verify, in order

Do not skip to the logs. Check each layer, because a failure at a lower layer
looks like "nothing is happening" at a higher one.

**Storage bound and actually shared:**
```bash
kubectl -n novachess get pvc novachess-data
# STATUS must be Bound, ACCESS MODES must be RWX
```

Then prove it is really shared — this is worth the two minutes. You cannot
`kubectl exec ... ls` into the pipeline pods: the image is distroless/static
and has no shell or coreutils. Mount the PVC from a throwaway pod instead:
```bash
kubectl -n novachess run vol-check --restart=Never --image=busybox --overrides='
{"spec":{"containers":[{"name":"vol-check","image":"busybox",
  "command":["sh","-c","ls /data/gen0"],
  "volumeMounts":[{"name":"data","mountPath":"/data"}]}],
 "volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"novachess-data"}}],
 "restartPolicy":"Never"}}'
kubectl -n novachess logs vol-check   # once Completed
kubectl -n novachess delete pod vol-check
# after workers have run, this must show batches from multiple workers
```
The coordinator also proves sharing continuously: it checksum-verifies every
batch (`store.Verify`) from a different node than the one that wrote it, so
`counted a batch` lines naming several distinct workers are themselves
evidence the volume is shared.

**NATS up:**
```bash
kubectl -n novachess get statefulset nova-nats
kubectl -n novachess port-forward svc/nova-nats 8222:8222 &
curl -s localhost:8222/healthz
curl -s localhost:8222/jsz | jq
# expect two streams once services connect: NOVA_WORK and NOVA_EVENTS
```

**Workers connected and spread:**
```bash
kubectl -n novachess get pods -l app=novachess-worker -o wide
# 8 Running, on 8 different nodes
```

**The loop turning:**
```bash
kubectl -n novachess logs -l app=novachess-coordinator -f
```
The coordinator logs each batch it counts and the running total against the
target. If you see units issued and no batches coming back after a few minutes,
go to section 6.

---

## 5. Running a generation, by hand

**The manifests run one generation and stop.** Automating the hand-off is the
top item in section 8. Until then, the loop turns with an operator in it.

### 5.1 Watch generation zero fill

```bash
kubectl -n novachess logs -l app=novachess-coordinator -f
```

The default target is 1,000,000 positions at 20,000 nodes per move
(`NOVA_POSITIONS`, `NOVA_NODES` in [coordinator.yaml](deploy/coordinator.yaml)).

**How long this takes is unknown.** No measurement exists on this hardware.
Everything so far is CI runners and a dev container, where timings vary by more
than 4× run to run, so any figure I gave you would be invented. The first real
number comes from the heartbeats — see 5.4. If it looks like days rather than
hours, lower `NOVA_POSITIONS` for the first generation; a smaller generation
that completes teaches you more than a large one that is still running.

### 5.2 Train

When the coordinator announces the dataset:

```bash
kubectl -n novachess create job --from=cronjob/novachess-trainer train-gen0
kubectl -n novachess logs -f job/train-gen0
```

The trainer reads `/data/gen0/` (a directory — it globs `*.novadata` and sorts,
so batch order is stable) and writes `/data/networks/gen0.nnue`, creating the
directory if needed.

Flags, if you want to change them, are in [jobs.yaml](deploy/jobs.yaml):
`-epochs 10`, `-batch 1024`, `-lr 0.001`, plus `-result-weight` (blend between
search score and game result), `-validation` (held-out fraction) and `-seed`.

### 5.3 Gate

```bash
kubectl -n novachess create job --from=cronjob/novachess-gatekeeper gate-gen0
kubectl -n novachess get job gate-gen0 -o jsonpath='{.status}'
```

**The exit status is the decision**, so the Job succeeding or failing *is* the
promotion verdict:

| exit | meaning | Job shows |
|---|---|---|
| 0 | promoted | Complete |
| 1 | rejected | Failed |
| 2 | inconclusive — the evidence never arrived, which is not a promotion | Failed |
| 3 | error | Failed |

`backoffLimit: 0` on this Job is deliberate. A rejection is a correct outcome,
not a transient failure; rerunning until it passed would defeat the entire
point of the gate.

A rejection at generation zero is a perfectly normal result — it means the
network trained on that data does not beat the hand-crafted evaluation yet.
The response is more data or different training hyperparameters, not rerunning
the gate.

### 5.4 Advance the generation

If promoted, edit [coordinator.yaml](deploy/coordinator.yaml):

```yaml
- name: NOVA_GENERATION
  value: "1"                              # was 0
- name: NOVA_NETWORK_URI
  value: "file:///data/networks/gen0.nnue"  # was empty
```

and `kubectl apply -k deploy/`. The `Recreate` strategy means the old
coordinator is fully gone before the new one starts — see section 7.

Then the trainer and gatekeeper Job specs need their gen0 paths bumped to gen1
before the next round. That editing is exactly what section 8's first item
replaces.

---

## 6. The silent failures to check for

These are all real; each one was found in this codebase and fixed, and each one
produced no error message anywhere while it was happening. Knowing the symptom
is the only way to catch a recurrence.

### Only one worker is ever busy

**Symptom:** adding boards does not increase throughput at all. Eight workers
running, one unit in flight.

**Cause:** `MaxAckPending` is enforced per *consumer*, and every member of a
NATS queue group shares a single consumer. Setting it to "one unit per worker"
throttles the whole cluster to one unit. The fix in place sets
`DefaultMaxInFlight = 512` (group-wide ceiling) and enforces one-at-a-time
per worker through `PullMaxMessages(1)` instead.

**Check:**
```bash
kubectl -n novachess logs -l app=novachess-worker --prefix --tail=50 | grep '"msg":"starting"'
# each worker logs `starting` with its unit ID when it picks one up.
# Distinct pods should be holding distinct units concurrently, not one at a time.
```

### The generation stalls partway

**Symptom:** the coordinator stops issuing work; positions plateau below the
target; no errors.

**Cause (two of them, both fixed):** the coordinator measured outstanding units
against the set of units that *contributed data*, so a unit that came back
rejected or empty stayed "in flight" forever. And a work unit whose games all
ended inside the random opening produced zero positions, and the worker
returned without announcing anything at all.

**Check:** the coordinator does not log issuance, so watch the work queue
itself. On `curl -s localhost:8222/jsz?streams=true` (port-forward 8222 from
`svc/nova-nats`), `NOVA_WORK`'s `last_seq` is the number of units ever issued
and `messages` is the queue depth, which the coordinator keeps topped up to
`NOVA_UNITS_IN_FLIGHT`. Healthy: `last_seq` climbing and depth steady at the
ceiling. This failure mode: depth draining toward zero and `last_seq` frozen
while workers sit idle and the counted total is below target. The unit IDs in
`counted a batch` lines rising monotonically is the same signal from the log
side. A unit that comes back empty is also logged now
(`a unit produced no positions`), so the original silent variant of this
stall would leave a trace.

### The trainer sees a fraction of the data

**Cause:** the volume is not really RWX. Section 3.1.

**Check:** the trainer prints `read N samples from M files` on startup. `M`
should equal the number of batches the coordinator counted, and `N` should be
close to its announced position count. A fraction — an eighth, with eight
boards — is the signature of per-pod volumes.

### Batches counted that cannot be read

The coordinator verifies size and checksum of every artifact before counting it
(`store.Verify`). A batch that fails verification is logged at ERROR and
discarded — `rejecting a batch that does not verify`. Occasional ones are
survivable; a stream of them means the volume is misbehaving.

---

## 7. Operational notes and gotchas

**Exactly one coordinator, always.** Two coordinators issuing units for the
same generation would each count half the batches coming back, and neither
would ever reach its target — a deadlock with no error anywhere. Hence
`replicas: 1` and `strategy: Recreate` rather than `RollingUpdate`. Do not
scale this Deployment. Do not change the strategy.

**The coordinator exits when a generation completes.** It is a Deployment, so
Kubernetes restarts the container, and it starts issuing generation-zero work
again — its state is in memory and rebuilt from the bus, and its durable
consumer has already acked the batches it counted. Expect a restart count above
zero on that pod and expect it to resume producing data for a generation you
considered finished. **Scale the coordinator to zero once a generation's
dataset is announced and you are training**, then scale it back up after
bumping `NOVA_GENERATION`. Fixing this properly is section 8, item 2.

```bash
kubectl -n novachess scale deploy/novachess-coordinator --replicas=0
```

**Heartbeats are not stored.** Workers publish to `nova.worker.heartbeat` over
core NATS, deliberately not JetStream — a heartbeat is worth nothing a second
later, and storing them would fill the stream with the one message type nobody
replays. So you cannot read them back from a stream. To watch them live:

```bash
kubectl -n novachess run nats-box --rm -it --image=natsio/nats-box -- \
  nats -s nats://nova-nats:4222 sub 'nova.worker.heartbeat'
```

Each carries the worker's measured nodes per second. **This is the practical
way to spot a board that is thermally throttling or has lost its fan**, because
it does not require the board to have finished anything recently. It is also
where your first real throughput number comes from.

**Worker shutdown.** `terminationGracePeriodSeconds: 30` — long enough for a
worker to notice SIGTERM and return its unit to the queue, not long enough to
finish the unit. A unit dropped this way is redelivered to another worker.
`DefaultAckWait` is 30 minutes, generous by design: a self-play unit is minutes
of games, and the NATS default of 30 seconds would redeliver every unit
repeatedly while the first worker was still playing it.

**At-least-once delivery.** The coordinator deduplicates batches by work unit
ID. Duplicate work is wasted, not corrupting.

**Subjects, for debugging with `nats`:**

| Subject | Carries | Durable |
|---|---|---|
| `nova.work.assign` | `WorkUnit` → worker queue group | yes, work-queue retention |
| `nova.games.produced` | `GameBatch` → coordinator | yes, 7-day retention |
| `nova.dataset.ready` | `DatasetReady` | yes — **nothing subscribes yet** |
| `nova.net.candidate` | `NetworkCandidate` | yes — **nothing publishes or subscribes** |
| `nova.net.promoted` | `NetworkVerdict` | yes — **nothing publishes or subscribes** |
| `nova.net.rejected` | `NetworkVerdict` | yes — **nothing publishes or subscribes** |
| `nova.worker.heartbeat` | `WorkerHeartbeat` | **no** — core NATS, fire and forget |

The three "nothing publishes or subscribes" rows are the automation gap. The
schemas and the transport exist; the trainer and gatekeeper are pure CLI tools
that never touch the bus. That is section 8, item 1.

**Streams:** `NOVA_WORK` (WorkQueue retention) and `NOVA_EVENTS` (Limits, 7-day
max age). Two streams rather than one because a work-queue stream permits only
one consumer per subject, which is precisely the fan-out that promotion
announcements need. Do not merge them.

**Resource sizing.** One CPU per worker, because one worker searches on one
thread; raising it does not make anything faster. To use more of a board's
cores, raise `replicas` and let two or three workers share a board — never
raise thread count. If the boards have other work, lower `requests.cpu` so the
scheduler does not overcommit; self-play is entirely CPU-bound and a throttled
board shows up immediately in its heartbeats.

**NATS is single-node.** One replica is right here: the pipeline exchanges tens
of messages per second, and a three-node cluster would spend memory on
replication the boards would rather spend on search. Work survives a restart
because JetStream writes to disk, which is what actually matters — a broker
being briefly unavailable delays the loop, it does not lose it.

---

## 8. Pending work

Ordered by what unblocks the most. Items 1–3 are what stand between this and
running unattended.

### 1. Automate the generation hand-off — **the big one**

Right now a human runs the trainer, runs the gatekeeper, reads an exit code,
edits two YAML files and reapplies. Everything needed to close this loop
already exists except the code that connects it:

- The subjects and schemas are defined (`nova.dataset.ready`,
  `nova.net.candidate`, `nova.net.promoted`, `nova.net.rejected`).
- The coordinator already publishes `nova.dataset.ready` when a generation
  completes.
- **Nothing subscribes to it.** `cmd/trainer` and `cmd/gatekeeper` do not
  import `internal/bus` at all.

The work: give the trainer and gatekeeper bus awareness (or write a small
controller that watches the subjects and creates Jobs), have the gatekeeper
publish its verdict, and have the coordinator subscribe to `nova.net.promoted`
so it advances its own generation and network URI rather than being edited.

Design decision to make first: a controller pod that creates Jobs, versus
long-lived trainer/gatekeeper services that wait on the bus. The Jobs approach
keeps the "a completed run is a fact" property that the current CronJob
structure was chosen for.

### 2. The coordinator should not restart into the generation it just finished

Described in section 7. It exits 0 on completion and Kubernetes restarts it.
Either it should wait for a promotion event and advance (which item 1 gives it
for free), or, as a stopgap, block instead of exiting so the pod stays up
without producing anything.

### 3. Per-generation paths in the Job specs

[jobs.yaml](deploy/jobs.yaml) hardcodes `/data/gen0/` and
`/data/networks/gen0.nnue`. Every generation currently requires editing that
file. It should take the generation from an env var or from the dataset event.

### 4. No monitoring

There are no metrics, no Prometheus endpoint, no alerts. Given how many of this
system's failure modes are silent, this matters more here than in a typical
service. The minimum worth having:

- positions per second, per worker and cluster-wide (heartbeats already carry
  nodes/sec — expose them)
- units issued vs resolved on the coordinator (catches a stall directly)
- artifacts that failed verification (catches a sick volume)
- generation progress against target

### 5. No health probes on the application pods

NATS has readiness and liveness probes; the worker and coordinator have none.
A worker that has lost its bus connection looks healthy to Kubernetes.

### 6. Incremental NNUE accumulator updates

`internal/nnue` currently does a **full accumulator refresh on every
`Evaluate`** rather than updating incrementally from make/unmake in the search.
This is correct but leaves a large amount of search speed on the table — the
whole point of the NNUE accumulator design is that a move changes only a
handful of features.

This is the single biggest engine-strength item on the list, and it is pure
engine work with no cluster dependency. It can be done in parallel with the
deployment. Note the constraint: the search shares one evaluator across lazy
SMP threads, and the evaluator is currently stateless for that reason. Making
the accumulator incremental means giving each search thread its own.

### 7. Secrets and the Lichess bot

`cmd/novabot` is not deployed by these manifests at all. When it is, it needs
`LICHESS_TOKEN` as a Kubernetes Secret — it is supplied by env var and must
never be committed. Note also that `novabot -upgrade` irreversibly converts a
Lichess account to a BOT account and requires an account that has never played
a rated game.

### 8. Backup

The PVC holds every generation of training data and every network. Longhorn
supports recurring snapshots; none are configured. A million positions is
roughly 32MB of packed records, so retaining a lot of history is cheap.

---

## 9. Rules that must not be broken

From [CLAUDE.md](CLAUDE.md), which is the working agreement for this repo. If
you change engine or pipeline code, these are the things that will bite:

**Before pushing anything:**
```bash
gofmt -l .                              # must be silent
go vet ./...                            # must be clean
go test -race ./...                     # must pass
GOOS=linux GOARCH=arm64 go build ./...  # must succeed
```
The arm64 build matters because the workers run on arm64 and a break there is
otherwise only found at deploy time.

**Measurement discipline.** Timings on CI runners and dev containers vary by
more than 4× run to run. Prove a change is behaviour-neutral with **node
counts, not milliseconds**. Compare performance only back to back on the same
host with a cold transposition table on both sides. State plainly when a number
is indicative rather than measured.

**Correctness properties that must not regress:**
- Fixed-depth and fixed-node search is deterministic at one thread. The
  distributed pipeline replays work units on different boards; training data
  must not depend on which machine ran them.
- The principal variation is legal move by move. This is what catches
  corruption in the PV table, the transposition table, or make/unmake — all of
  which otherwise show up only as mysteriously bad play.
- Perft counts are exact. Anything touching move generation re-runs the
  full-depth suite, classical and Chess960.

**Testing style.** Assert invariants, not node counts or exact scores — those
change whenever a heuristic is touched, and a test that breaks on every tuning
change gets deleted rather than fixed. When a test fails, check the fixture
before the code: a large share of failures in this repo have been bad test
positions.

---

## 10. Quick reference

```bash
# Namespace-wide status
kubectl -n novachess get all

# Follow the loop
kubectl -n novachess logs -l app=novachess-coordinator -f
kubectl -n novachess logs -l app=novachess-worker --prefix -f

# Live heartbeats (nodes/sec per board)
kubectl -n novachess run nats-box --rm -it --image=natsio/nats-box -- \
  nats -s nats://nova-nats:4222 sub 'nova.worker.heartbeat'

# JetStream state
kubectl -n novachess port-forward svc/nova-nats 8222:8222
curl -s localhost:8222/jsz | jq

# What is on the volume — the images are distroless, so exec has no `ls`;
# mount the PVC from a throwaway busybox pod instead (see section 4.3)

# Run a generation's trainer and gate
kubectl -n novachess create job --from=cronjob/novachess-trainer   train-gen0
kubectl -n novachess create job --from=cronjob/novachess-gatekeeper gate-gen0

# Park the coordinator while training (see section 7)
kubectl -n novachess scale deploy/novachess-coordinator --replicas=0
```

### Coordinator environment

| Variable | Default | Meaning |
|---|---|---|
| `NOVA_BUS_URLS` | `nats://nova-nats:4222` | Comma-separated NATS URLs |
| `NOVA_DATA_DIR` | `/data` | Shared volume for artifacts |
| `NOVA_GENERATION` | `0` | Generation to run; a restart resumes here |
| `NOVA_POSITIONS` | `1000000` | Positions needed before training |
| `NOVA_GAMES_PER_UNIT` | `50` | Games in one work unit |
| `NOVA_NODES` | `20000` | Search nodes per move |
| `NOVA_RANDOM_PLIES` | `8` | Random moves opening each game |
| `NOVA_UNITS_IN_FLIGHT` | `64` | Units kept queued |
| `NOVA_SEED` | `1` | Derives every unit's seed |
| `NOVA_NETWORK_URI` | *(empty)* | Network to play with; empty bootstraps from the hand-crafted evaluation |

### Worker environment

| Variable | Default | Meaning |
|---|---|---|
| `NOVA_BUS_URLS` | `nats://nova-nats:4222` | Comma-separated NATS URLs |
| `NOVA_DATA_DIR` | `/data` | Shared volume for artifacts |
| `NOVA_WORKER_ID` | hostname | Identity in heartbeats and batches; set to the pod name |
| `NOVA_NODE_NAME` | *(empty)* | The board this runs on; set from `spec.nodeName` |
| `NOVA_HASH_MB` | `64` | Transposition table size per search |
