# Novachess

A UCI-compatible chess engine in Go, with a native Lichess bot client and a
self-play training pipeline designed to run distributed across a cluster.

## Status

Under construction. The move generator is complete and perft-verified; search,
evaluation, UCI and the Lichess client are not yet implemented.

| Component | State |
|---|---|
| Board representation, bitboards, magics | done |
| Legal move generation, make/unmake, Zobrist | done, perft-verified |
| Pipeline message contracts and bus abstraction | done |
| Search (alpha-beta, TT, quiescence) | not started |
| Evaluation (hand-crafted, then NNUE) | not started |
| UCI protocol | not started |
| Lichess bot client | not started |
| Pipeline services (coordinator, worker, trainer, gatekeeper) | not started |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the service topology and the
reasoning behind where the boundary between engine and services falls.

## Design

**Search.** Classical alpha-beta rather than MCTS: negamax with iterative
deepening, aspiration windows, a Zobrist transposition table, quiescence search
with SEE, null-move pruning, late move reductions, and killer/history move
ordering. Lazy SMP for multithreading.

**Evaluation.** A hand-crafted evaluation first, so the engine plays early and
can generate the initial self-play data. NNUE replaces it behind the same
interface once the training pipeline works, in the Stockfish lineage rather than
the AlphaZero one — it runs well on CPU and needs no GPU.

**Learning.** Self-play generates positions, those positions train a network,
the stronger network generates better positions, and the loop repeats. Each new
network must beat the incumbent under SPRT before it is promoted, which is what
keeps the loop improving rather than drifting.

**Why Go.** Goroutines make lazy SMP and the Lichess client straightforward, the
standard library streams Lichess's NDJSON API with no dependencies, and the
absence of cgo means `GOOS=linux GOARCH=arm64 go build` produces a static binary
for the cluster from any machine. The cost is SIMD: Go's `simd/archsimd` is
amd64-only and experiment-gated, so NNUE inference will sit behind an interface
with a portable scalar implementation and an optional accelerated one.

**Architecture.** The pipeline around the engine is a set of services
coordinating over NATS: a coordinator issuing work, self-play workers generating
games, a trainer, and a gatekeeper that SPRT-tests each candidate network before
promoting it. The engine itself stays a single linked binary — its search
evaluates millions of positions per second, so a network boundary anywhere
inside that loop would cost a factor of a thousand. Services are processes
around the engine, not pieces of it.

## Layout

```
cmd/novachess/          UCI engine binary
cmd/novabot/            Lichess bot client
cmd/coordinator/        work distribution and dataset assembly
cmd/selfplay-worker/    game generation
cmd/trainer/            network training
cmd/gatekeeper/         SPRT gating

internal/chess/         board, move generation, Zobrist, perft
internal/search/        negamax, transposition table, time management
internal/eval/          evaluation
internal/uci/           UCI protocol
internal/lichess/       Lichess Bot API client
internal/train/         self-play, data format, optimizer

internal/events/        message contracts shared by all services
internal/bus/           message bus abstraction and NATS implementation
```

## Move generation

Moves are generated legal-only rather than pseudo-legal-then-filtered. Two masks
computed up front make every generated move legal by construction: a check-evasion
mask restricting non-king pieces to capturing the checker or interposing, and a
pin mask restricting pinned pieces to their pin ray. En passant is the one case
these cannot express — it vacates two squares on a rank at once — so it is
verified by simulating the resulting occupancy directly.

Sliding attacks use magic bitboards. The magic constants are searched for at
startup from a fixed seed rather than hardcoded, which keeps several hundred
opaque constants out of the source. The fixed seed matters beyond tidiness: the
distributed self-play pipeline requires every node to produce identical results
for identical inputs.

## Correctness

Move generation is gated by perft — counting the leaf nodes of the move tree and
comparing against published exact values. Seven standard positions are verified,
chosen to cover the rules that are easiest to get wrong: en passant pins,
castling rights revoked by rook captures, and underpromotions.

```
go test ./...                      # shallow perft, seconds
go test ./... -perft-full          # full depth, ~610M nodes
```

Make/unmake reversibility and incremental-vs-recomputed Zobrist agreement are
tested separately, since perft alone would not catch a hash that drifts or an
unmake that restores an equivalent-but-different position.

## Building

Requires Go 1.24 or later.

```
go build ./...
go test ./...
```
