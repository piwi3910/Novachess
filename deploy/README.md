# Deploying Novachess

Manifests for running the training loop on a Kubernetes cluster of arm64
boards.

```
kubectl apply -k deploy/
```

That creates the namespace, a single-node NATS server with JetStream, the
shared volume, one worker per board, and the coordinator. Generation zero
starts immediately: there is no network yet, so the workers bootstrap from the
hand-crafted evaluation.

## What you need first

**A ReadWriteMany storage class.** Workers write batches and the trainer reads
them, from different nodes, so the volume has to be shared. Longhorn provides
this through its share-manager; the claim in `storage.yaml` asks for
`longhorn` and `ReadWriteMany`. Change the class if yours is named differently.

This is the one prerequisite that is not in these files. Without it the workers
will each write to their own disk and the trainer will see a fraction of the
data — with no error, because from each pod's point of view everything worked.

**An image.** Build for arm64 and push somewhere the cluster can reach:

```
docker buildx build --platform linux/arm64 \
  --build-arg VERSION=$(git describe --always --dirty) \
  -t your-registry/novachess:v1 --push .
```

Then set that image in `kustomization.yaml`.

## Sizing

`worker.yaml` runs one worker per board with `replicas: 8`. Each worker
searches on **one thread** — that is a correctness requirement rather than a
tuning choice, because lazy SMP is non-deterministic and a replayed work unit
has to produce identical games. Use more of a board's cores by raising the
replica count and letting two or three workers share a board, not by giving one
worker more threads.

The CPU requests assume the boards have nothing else to do. If they do, lower
`requests.cpu` so the scheduler does not overcommit them; self-play is entirely
CPU-bound and a throttled board shows up immediately in its heartbeats.

## Watching it

```
kubectl -n novachess logs -l app=novachess-coordinator -f
kubectl -n novachess logs -l app=novachess-worker --prefix -f
```

The coordinator logs each batch it counts and the running total against the
generation's target. Workers publish a heartbeat every fifteen seconds carrying
their measured nodes per second, which is the practical way to spot a board that
is throttling or has lost its fan — it does not require the board to have
finished anything recently.

## The generation loop

These manifests run **one** generation and stop. That is deliberate: the trainer
and gatekeeper are Jobs rather than Deployments, because each runs once per
generation and a completed run is a fact rather than something to keep alive.

Once a generation's dataset is announced:

```
kubectl -n novachess create job --from=cronjob/novachess-trainer train-gen0
kubectl -n novachess create job --from=cronjob/novachess-gatekeeper gate-gen0
```

Then raise `NOVA_GENERATION` and set `NOVA_NETWORK_URI` to the promoted network
in `coordinator.yaml`, and apply again. Automating that hand-off is the next
piece of work; until then the loop turns with an operator in it, which is a
reasonable place to be for the first few generations when the numbers are worth
looking at anyway.

## Exit codes

The gatekeeper's exit status is the decision, so a Job's success or failure is
the promotion decision itself:

| code | meaning |
|---|---|
| 0 | promoted |
| 1 | rejected |
| 2 | inconclusive — the evidence never arrived, which is not a promotion |
| 3 | error |
