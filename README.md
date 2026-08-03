# Novachess

A UCI-compatible chess engine in Go, with a native Lichess bot client and a
self-play training pipeline designed to run distributed across a cluster.

## Status

Under construction. The learning loop closes and is self-correcting: self-play
generates labelled positions, the trainer fits a network to them, the gatekeeper
makes each candidate win a match before it replaces the incumbent, and the
engine plays with whatever was promoted. What remains is the services that run
the loop unattended across a cluster.

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
| Training data format and self-play generation | done |
| Evaluation, NNUE | network, inference and accumulator done |
| Trainer | done |
| Gatekeeper (SPRT match gating) | done |
| Pipeline services (coordinator, worker) | not started |

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
queens are on and the centre once they are gone.

NNUE replaces it behind the same interface, in the Stockfish lineage rather than
the AlphaZero one — it runs well on CPU and needs no GPU. The network is a
768->256x2->1 perspective net: one input per colour, piece type and square,
accumulated separately from each side's point of view, with the output layer
reading the side to move's half first. See [Trained evaluation](#trained-evaluation)
for what that shape buys.

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
cmd/trainer/            network training (implemented)
cmd/gatekeeper/         SPRT gating (implemented)

internal/chess/         board, move generation, Zobrist, perft
internal/search/        negamax, transposition table, time management
internal/eval/          evaluation
internal/uci/           UCI protocol
internal/lichess/       Lichess Bot API client
internal/gate/          SPRT, match running, promotion decisions
internal/nnue/          trained evaluation: network, accumulator, inference
internal/train/         self-play generation, packed data format, optimizer

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

## Trained evaluation

The network is a perspective net, 768->256x2->1. Its input is one feature per
colour, piece type and square; those features are accumulated into a hidden
layer twice, once from each side's point of view, and the output layer reads the
side to move's half first and the opponent's second. Weights are stored
quantized as integers rather than floats, because the search cannot afford float
maths per node and because integer arithmetic is bit-for-bit identical on every
machine — which the pipeline requires, since a network that evaluated
differently on two boards would make self-play irreproducible.

Two properties follow from that shape, and both are load-bearing.

**The evaluation is symmetric by construction.** Mirroring a position — swapping
colours and flipping ranks — swaps the two accumulator halves and swaps the side
to move, so the output layer sees the identical pair of inputs and returns the
identical number. This holds for any weights whatsoever, trained or random,
which makes it a test of the feature indexing rather than of the training. It is
the check that catches a wrong rank flip or a missed colour swap, which is the
mistake this architecture invites.

**The accumulator is incremental.** A move changes a handful of features, so the
hidden layer is updated by adding and subtracting a few rows instead of being
recomputed — the entire reason the architecture is shaped this way, since the
first layer is the expensive one and a search evaluates millions of positions
that differ from their parent by one piece. The incremental result is required
to equal a full recomputation exactly; a drift there would have no symptom
except bad play.

The evaluator holds no mutable state, because the search shares one across every
lazy SMP thread. An accumulator hanging off it would be written by all of them
at once, which is a data race and would surface only as evaluations that are
wrong when threads happen to overlap. Making the accumulator persist across plies
means keeping one per ply and updating it in step with make and unmake, which is
a change to the search rather than to the network.

A network is loaded with the `EvalFile` UCI option, and unset by setting it
empty. A load failure falls back to the hand-crafted evaluation and says so:
UCI gives an engine no way to refuse an option, and a network silently not in
use is invisible to a strength test.

## Training data

Self-play produces labelled positions in a packed 32-byte record: an occupancy
bitboard, one nibble per occupied square, and the side to move, castling rights
and en passant file in two more bytes, leaving the search score, the eventual
game result and the ply. Storing FENs instead would roughly triple a dataset
meant to reach a billion positions. Files carry a magic number and a version,
and a reader refuses anything it does not recognize rather than reading old
bytes under new rules — the resulting positions would look plausible and their
labels would be meaningless, which is the one corruption nothing downstream can
detect.

**Score and result are both White-relative**, deliberately unlike the engine's
own side-to-move convention. A file mixing the two orientations cannot be
interpreted without knowing which record uses which, and reading it wrong trains
the network to prefer losing positions.

Positions where the side to move is in check, where the best move is a capture,
or whose score is a mate score are dropped. Their evaluation describes a tactic
about to resolve rather than the position itself, and training on them teaches
the network to imitate the search instead of judging the quiet positions it will
actually be asked about at the leaves. Games that hit the ply cap without
finishing are discarded whole, since their positions have no trustworthy label.

Generation is deterministic: a work unit fixes a seed and a node budget, and
replaying it produces byte-identical samples on any machine. This is what makes
a unit safe to redeliver when a cluster node is evicted mid-job. Two things
follow from it. Self-play searches by nodes and never by time, since a slower
node given a time limit would search less and play different moves; and it runs
one thread, because lazy SMP works by letting threads race and is
non-deterministic by construction. Scale with more workers, not more threads per
worker.

## Training

The trainer reads packed self-play files and writes a network:

```
go build ./cmd/trainer
./trainer -out gen1.nnue data/*.novadata
./novachess
> setoption name EvalFile value gen1.nnue
```

It trains in float32 and quantizes at the end. The engine needs integer weights
— its arithmetic has to be identical on every machine — but gradients through
integers are meaningless, so the two representations are kept apart and the
conversion is **measured rather than assumed**. Every run reports how far the
quantized network drifted from the model that was trained, against that
network's own output range, and how many weights did not fit the integer range
at all. A clipped weight means the saved network is not the model whose loss was
printed, which is a distinction nothing downstream could otherwise recover.

Each sample carries two labels and the trainer blends them, controlled by
`-result-weight`. The search score is precise but only as good as the evaluation
that produced it, so training on it alone teaches the network to imitate its
predecessor and the loop stops improving. The game result is ground truth but
very noisy, since one position in a lost game may well have been winning.
Neither extreme is what you want.

The run is deterministic: the same inputs with the same seed produce
byte-identical weights, so a generation can be reproduced or audited rather than
merely trusted. Input files are sorted before reading, because a shell glob's
order depends on the locale and would otherwise silently change the result.

The reported losses are diagnostics, not a verdict. A network can improve its
loss against the training objective and still play worse — which is exactly what
the gatekeeper's SPRT match exists to catch, and why nothing here decides
whether a network is promoted.

## Gating

A trained network does not get used because its loss went down. It has to win a
match:

```
go build ./cmd/gatekeeper
./gatekeeper -candidate gen2.nnue -incumbent gen1.nnue
```

The exit status is the decision — 0 promoted, 1 rejected, 2 inconclusive, 3
error — so a pipeline can branch on it without parsing text.

This is what makes the loop self-correcting rather than merely self-feeding.
Training loss measures how well a network fit the data it was shown, which is a
different question from whether it plays better; a network can improve its loss
and lose games. Promoted on loss alone it would then generate the next
generation's data, and the whole loop would drift somewhere worse with nothing
to notice.

**An inconclusive match is not a promotion.** Running out of games means the
evidence never arrived, and treating that as success would promote every
candidate that failed to prove itself either way — which across generations is a
random walk, not a training loop.

The match is judged by a sequential probability ratio test rather than a fixed
number of games. The two hypotheses are Elo differences rather than "better" and
"worse": testing against zero would eventually promote a network a fraction of
an Elo stronger, at the cost of running forever to prove it. A clearly better or
clearly worse candidate is identified in a handful of games and only genuinely
marginal ones run long.

**Games are played in colour-reversed pairs from a shared opening.** Openings are
not perfectly balanced, so playing each one once would measure the openings as
much as the networks. This also gives an exact test of the harness rather than a
statistical one: two deterministic engines with the same evaluation play the
identical game whichever colour they are given, so a network matched against
itself must score *exactly* 50% — not approximately, and however few games are
played. Any bias breaks it immediately, and a statistical check would need
thousands of games to see what this sees in twelve.

Matches search by nodes and run one thread, for the same reasons self-play does:
a match decided partly by which process got more CPU measures nothing, and a
gating decision that cannot be reproduced cannot be audited when a promotion
later looks like a mistake.

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
