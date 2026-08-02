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
| Game termination rules (draws, mate, stalemate) | done |
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

Move generation is checked four independent ways, because each catches
something the others can miss.

**Perft** counts the leaf nodes of the move tree and compares against published
exact values, across seven standard positions verified to full depth (~610M
nodes). These positions are chosen to cover the rules that are easiest to get
wrong: en passant pins, castling rights revoked by rook captures, and
underpromotions.

**A differential test against a naive reference generator** compares move sets,
not just counts. The reference generates every pseudo-legal move by brute force,
plays each one, and keeps it if the king is not left attacked — legality
transcribed directly, with no pin masks to be wrong about. It shares nothing
with the real generator but the attack tables, and the two must agree exactly
across tens of thousands of positions reached by random play. Perft alone could
not catch two compensating bugs that cancel in the total.

**A mirror symmetry test** requires perft counts to be unchanged when a position
is reflected vertically and its colors swapped. Any difference is a bug treating
one color differently from the other. Its value is that it needs no published
counts, so it applies to positions nobody has tabulated.

**Make/unmake reversibility and incremental-vs-recomputed Zobrist agreement**
are tested separately, since perft would not catch a hash that drifts or an
unmake that restores an equivalent-but-differently-represented position.

```
go test ./...                      # everything except the deep counts, seconds
go test ./... -perft-full          # full depth, ~610M nodes
```

## Rules coverage

Classical chess is fully implemented.

**Moves.** All piece movement. Castling on both wings for both colors, with
rights revoked when the king moves, when a rook moves, and when a rook is
captured on its home square — the last being the case that is easiest to miss.
Castling is refused out of check, through an attacked square and into check,
while correctly permitting the queenside castle when only b1/b8 is attacked,
since the king does not cross it. En passant, including the discovered-check
case where vacating two squares on one rank exposes the king, and including
captures that resolve a check. Promotion is mandatory on reaching the back rank
and offers all four pieces.

**Game end.** Checkmate, stalemate, and four drawing rules. FIDE distinguishes
*claimable* draws (threefold repetition, fifty-move) from *automatic* ones
(fivefold, seventy-five-move); all four are implemented. Adjudication uses the
claimable thresholds, which is universal engine convention and what self-play
requires — a game continuing past a threefold repetition because neither side
"claimed" would never end. The automatic thresholds are exposed separately for
callers that must agree with an external arbiter, which is the Lichess bot's
situation.

**Positions are normalized on parse.** A FEN asserting a castling right whose
rook is absent, or an en passant target no pawn can legally use, describes
something the board contradicts. Both are corrected rather than believed. The
first was a crash — the generator emitted a castling move and playing it
relocated a piece that was not there — and FENs arrive from GUIs and web APIs.
The second matters for draw claims: FIDE distinguishes positions by the moves
available in them, so an unusable en passant target must not make an otherwise
identical position hash differently, or a threefold repetition goes undetected.

**Known limits.** Insufficient material covers the material-based subset of
FIDE 5.2.2 — bare kings, king and minor piece, and same-colored bishops. Full
"dead position" detection, which includes completely locked pawn structures, is
not tractable and no engine attempts it; the difference only ever costs a draw
claim in positions that are drawn anyway.

**Chess960 is not supported.** Castling is encoded as the king's origin and
destination on the standard squares, correct for classical chess and wrong for
Fischer random. The Lichess bot should decline Chess960 challenges.

## Building

Requires Go 1.24 or later.

```
go build ./...
go test ./...
```
