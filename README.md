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
| Chess960 | done, verified on all 960 arrangements |
| Game termination rules (draws, mate, stalemate, dead positions) | done |
| Pipeline message contracts and bus abstraction | done |
| Search (alpha-beta, TT, quiescence) | done |
| Evaluation, hand-crafted | done |
| UCI protocol and engine binary | done |
| Lichess bot client | done |
| Evaluation, NNUE | not started |
| Pipeline services (coordinator, worker, trainer, gatekeeper) | not started |

The engine plays. `go build ./cmd/novachess` produces a UCI binary that any
GUI or match runner can load.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the service topology and the
reasoning behind where the boundary between engine and services falls.

## Design

**Search.** Classical alpha-beta rather than MCTS: negamax with iterative
deepening, aspiration windows, a Zobrist transposition table, quiescence search,
null-move pruning, reverse futility, late move reductions, and killer/history
move ordering. Lazy SMP for multithreading, via the `Threads` option.

Threads share nothing but the transposition table, which uses lockless hashing:
each entry is one packed word stored alongside that word XOR-ed with the
position key, so a reader that catches a writer mid-update sees a mismatch and
treats the entry as a miss. Locking a structure probed millions of times a
second would defeat the point of threading, and leaving it unsynchronized is a
data race — undefined behaviour in Go, not merely untidy.

Two properties are treated as requirements rather than nice-to-haves. The
principal variation must be legal move by move, which is what catches
corruption in the PV table, the transposition table or make/unmake — all of
which otherwise surface only as mysteriously bad play. And a fixed-depth or
fixed-node search must be deterministic **at one thread**, because the
distributed pipeline replays work units on different nodes and the training
data cannot be allowed to depend on which machine ran them.

That second property is why `Threads` defaults to 1 and why the thread count is
a policy decision rather than a performance knob. Lazy SMP works precisely by
letting threads race on the shared table, so more than one thread is
non-deterministic by construction. Self-play must run one thread per worker —
scale by running more workers, not more threads per worker. Match play and
analysis, where reproducibility does not matter, should use every core.

**Evaluation.** A hand-crafted evaluation first, so the engine plays early and
can generate the initial self-play data. It is tapered between middlegame and
endgame by material phase, which is what lets the king want the corner while
queens are on and the centre once they are gone. NNUE replaces it behind the
same interface once the training pipeline works, in the Stockfish lineage
rather than the AlphaZero one — it runs well on CPU and needs no GPU.

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
cmd/novabot/            Lichess bot binary
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

**The Chess960 suite** applies both of the above to all 960 starting
arrangements, eight random plies deep so that castling is genuinely in play.
Chess960 is where a castling generator is most likely to be wrong, because the
overlapping king and rook paths cannot occur in classical chess at all.

**Make/unmake reversibility and incremental-vs-recomputed Zobrist agreement**
are tested separately, since perft would not catch a hash that drifts or an
unmake that restores an equivalent-but-differently-represented position.

```
go test ./...                      # everything except the deep counts, seconds
go test ./... -perft-full          # full depth, ~610M nodes
```

## Rules coverage

Classical chess and Chess960 are both fully implemented.

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

**Dead positions.** FIDE 5.2.2 declares a position drawn when no sequence of
legal moves can produce checkmate. Both halves are detected. The material half
covers bare kings, a lone minor piece, and any number of bishops confined to one
color complex. The structural half proves a locked pawn position dead: only
kings and pawns remain, every pawn is blocked with no capture available, and
neither king can ever reach a capture — from which the only moves that will ever
exist are king moves, and two lone kings cannot mate.

That second proof needs two observations to fire on real positions. A king may
never stand on a square an enemy pawn attacks, and frozen pawns never stop
attacking it, so pawn-attacked squares are permanent walls exactly like the
pawns; this is what seals the kings apart in the classic interlocked wall. And a
pawn defended by another pawn can never be taken by a king, since the capture
would be a move into check. Where the analysis remains approximate it errs
toward saying no, which can only leave a genuine draw unclaimed — the opposite
error would adjudicate a winnable game as drawn.

**Chess960 is supported.** Castling is encoded internally as king-takes-rook,
the only unambiguous form once the king's destination can already be occupied by
the castling rook, or the king need not move at all. Rook origins and the
squares that must be empty are stored per position rather than assumed. FEN
reads and writes both the classical `KQkq` and Shredder `AHah` castling fields,
and UCI moves are rendered in whichever form the variant expects.

Verification uses the full 960-arrangement perft suite from Ethereal, checked to
depth 5 across every arrangement — 635 million leaf nodes, all exact — with a
sampled depth-6 pass under `-perft-full`. The orthodox arrangement written in
Shredder notation is separately required to reproduce the classical perft counts
exactly, which pins the generalized code to ground truth everyone agrees on.

## Running

```
go build ./cmd/novachess
./novachess
```

It speaks UCI on stdin and stdout, so any UCI GUI or match runner can load it.
Driven by hand it looks like this:

```
position startpos
go depth 12
info depth 12 seldepth 31 score cp 24 nodes 2551987 nps 813404 time 3137 ...
  pv e2e4 e7e5 g1f3 g8f6 b1c3 b8c6 d2d4 e5d4 f3d4 c6d4 d1d4 f8d6
bestmove e2e4 ponder e7e5
```

Beyond the protocol there are three commands that make manual testing bearable:
`d` prints the board, `eval` prints the static evaluation, and `perft <depth>`
runs a divide from the current position.

### Playing on Lichess

`cmd/novabot` plays games on Lichess directly, without the Python bridge. It
needs a token with the `bot:play` scope from an account that has been upgraded
to a bot account — an upgrade that only works while the account has never
played a rated game, and cannot be undone.

```
export LICHESS_TOKEN=lip_xxxxxxxxxxxx
go build ./cmd/novabot
./novabot -upgrade          # once, irreversibly
./novabot
```

It accepts standard and Chess960 challenges, rated or casual, and declines
anything else with a reason rather than leaving the challenger waiting —
including variants whose rules the engine does not implement, where accepting
would mean playing illegal moves rather than merely playing badly. Each game
runs in its own goroutine with its own searcher, so a search never blocks the
bot from noticing events in its other games.

## Building

Requires Go 1.26.5 or later, the current stable release. CI tracks `stable`,
so it follows new releases as they land rather than pinning to a version that
goes quietly out of date.

1.26 is also where `simd` and `simd/archsimd` enter the standard library.
Nothing uses them yet — the engine is scalar Go throughout — but NNUE inference
is the one place hand-vectorized code will eventually pay for itself. See the
note on SIMD under Design for the caveats: `archsimd` is amd64-only and sits
behind `GOEXPERIMENT=simd`.

```
go build ./...
go test ./...
```
