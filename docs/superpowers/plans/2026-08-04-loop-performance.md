# Loop Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the four measured bottlenecks in the training loop: full accumulator refresh per node (~5-8× NNUE wall-clock), sequential gate matches (~5h verdicts), the generation replay tax (~2.7h per extension), and scratch-only training (no LR decay, no fine-tuning).

**Architecture:** (1) An incremental NNUE evaluator with an accumulator stack driven by search make/unmake, per search thread, bit-identical to Refresh by construction and by property test. (2) A concurrent match runner in internal/gate. (3) Coordinator startup that verifies existing artifacts and issues only missing units. (4) Trainer LR halving schedule and `-init` fine-tuning, wired through jobs.yaml and the dashboard's job-launch rewriter.

**Tech Stack:** Pure Go, existing packages only. No new dependencies.

## Global Constraints

- Before ANY push: `gofmt -l .` silent, `go vet ./...` clean, `go test -race ./...` passing, `GOOS=linux GOARCH=arm64 go build ./...` succeeding.
- **Fixed-depth and fixed-node search at one thread stays deterministic, and Task 2 must prove behaviour-neutrality with NODE COUNTS, not milliseconds** (CLAUDE.md measurement discipline). Incremental evaluation must produce bit-identical scores to the refresh path — same evals → same tree → same node counts.
- The PV-legality and perft suites are the regression net; move generation is untouched, but run everything anyway.
- Tests assert invariants, never exact scores or node counts as magic numbers — Task 2's proof asserts EQUALITY between two configurations of the same build, not stored counts.
- One thread per search worker, always. Parallel gate games are N single-threaded games, never one multi-threaded search.
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

| Path | Responsibility |
|---|---|
| `internal/nnue/incremental.go` (new) | `IncrementalEvaluator`: accumulator stack, Push/Pop/Reset |
| `internal/nnue/incremental_test.go` (new) | bit-identical property tests vs Refresh |
| `internal/eval/eval.go` (modify) | optional `MoveAware` + `PerThread` interfaces |
| `internal/search/search.go`, `negamax.go` (modify) | per-thread evaluator instances; Push/Pop at make/unmake |
| `internal/search/incremental_proof_test.go` (new) | node-count equality proof |
| `internal/gate/match.go` (modify) | `Concurrency` in MatchConfig, worker-pool match runner |
| `cmd/gatekeeper/main.go` (modify) | `-concurrency` flag |
| `internal/coordinator/coordinator.go` (modify) | resume: verify existing artifacts, skip their units |
| `internal/train/` + `cmd/trainer/main.go` (modify) | LR halving schedule, `-init` |
| `internal/dashboard/control.go` (modify) | rewriter handles `-init`; `-concurrency` untouched (no path) |
| `deploy/jobs.yaml` (modify) | gatekeeper `-concurrency 6` + cpu 6; trainer `-init` |

---

### Task 1: Incremental NNUE evaluator (bit-identical by test)

**Files:**
- Create: `internal/nnue/incremental.go`, `internal/nnue/incremental_test.go`
- Modify: `internal/eval/eval.go` (add the optional interfaces — no existing implementor changes)

**Interfaces produced (later tasks depend on these exact names):**

```go
// internal/eval/eval.go — optional capabilities an Evaluator may implement.

// MoveAware is implemented by evaluators that maintain internal state
// incrementally as the search walks the tree. Push is called immediately
// BEFORE p.MakeMove(m) (the pre-move position is what the delta is computed
// from); Pop after UnmakeMove; PushNull/PopNull around null moves; Reset
// whenever the search starts from a fresh root position.
type MoveAware interface {
	Reset(p *chess.Position)
	Push(p *chess.Position, m chess.Move)
	Pop()
	PushNull()
	PopNull()
}

// PerThread is implemented by evaluators that carry per-thread state. The
// search calls NewThreadEvaluator once per search thread instead of sharing
// one instance; stateless evaluators (the HCE) don't implement it and stay
// shared exactly as today.
type PerThread interface {
	NewThreadEvaluator() Evaluator
}
```

```go
// internal/nnue/incremental.go
const maxStack = 256 // > MaxPlies anywhere in the engine; Reset guards depth

type IncrementalEvaluator struct { /* net *Network; stack [maxStack]Accumulator; sp int; valid bool */ }

func NewIncremental(net *Network) *IncrementalEvaluator
func (e *IncrementalEvaluator) Evaluate(p *chess.Position) int // net.Evaluate(&stack[sp], p.SideToMove()); if !valid, Refresh first (safety net, must not happen in normal operation)
// Reset/Push/Pop/PushNull/PopNull per eval.MoveAware
func (e *IncrementalEvaluator) NewThreadEvaluator() eval.Evaluator // fresh IncrementalEvaluator over the same net
```

Delta rules in Push (computed from the PRE-move position `p` and move `m` — read `internal/chess`'s Move encoding first; every case below must consult the real API for how promotions, castling and en passant are represented):
- copy `stack[sp]` to `stack[sp+1]`, `sp++`, then on the new top:
- plain move: `Move(net, piece, from, to)`
- capture: `Remove(net, captured, capSq)` first (capSq differs from `to` only for en passant), then the mover's `Move`
- promotion: `Remove(net, pawn, from)`, `Add(net, promoPiece, to)` (plus the capture removal when it is one)
- castling: king `Move` + rook `Move` (rook squares from the chess package's castling definitions)
- stack overflow (sp == maxStack-1): set a flag and fall back to Refresh in Evaluate — correctness over speed at absurd depths.

- [ ] **Step 1: Failing property test — the heart of the task.** In `incremental_test.go`:

```go
// TestIncrementalMatchesRefreshOnRandomPlayouts is the correctness anchor:
// integer adds are exact, so if the deltas are right the incremental
// accumulator is BIT-IDENTICAL to a from-scratch refresh at every node.
// Any divergence is a delta bug, full stop.
func TestIncrementalMatchesRefreshOnRandomPlayouts(t *testing.T) {
	net := testNetwork(t) // reuse/adapt however existing nnue tests build a network with non-trivial weights; random int16 weights from a seeded rng are fine and are BETTER than zeros
	for seed := int64(0); seed < 20; seed++ {
		rng := rand.New(rand.NewSource(seed))
		p := chess.StartingPosition() // exact constructor per the chess package
		ev := NewIncremental(net)
		ev.Reset(p)
		for ply := 0; ply < 120; ply++ {
			moves := p.LegalMoves() // exact API per the chess package
			if len(moves) == 0 { break }
			m := moves[rng.Intn(len(moves))]
			ev.Push(p, m)
			p.MakeMove(m)
			var want Accumulator
			want.Refresh(net, p)
			if !ev.top().equal(&want) { // add small unexported helpers for the test
				t.Fatalf("seed %d ply %d move %v: incremental accumulator diverged from refresh", seed, ply, m)
			}
		}
	}
}

// TestPushPopRoundTrip: push a move, pop it, top must be bit-identical to before.
// Include castling, en passant and promotion positions specifically (fixture
// FENs — VERIFY each FEN is legal and actually contains the special move; bad
// fixtures are this repo's most common test failure).
// TestNullMovePushPop: PushNull/PopNull round-trips and the top equals the
// pre-null accumulator (a null move changes no features).
```

- [ ] **Step 2: Verify failure** (package won't build). **Step 3: Implement** `incremental.go` and the eval.go interfaces. **Step 4: Tests green under `-race`.**
- [ ] **Step 5: Full gates, commit** ("Add an incremental NNUE evaluator, bit-identical to refresh by property test — the machinery for ending the full-refresh-per-node cost the handover flagged as item 6.")

---

### Task 2: Search integration + node-count-neutrality proof

**Files:**
- Modify: `internal/search/search.go` (thread setup — find where per-thread `state` is built and where `st.s.evaluator` is bound), `internal/search/negamax.go` (the six call sites found at negamax.go:88/90/124/162/262/264 plus root make/unmake if separate — grep again, don't trust these line numbers)
- Create: `internal/search/incremental_proof_test.go`
- Modify: `internal/nnue/evaluator.go` — `nnue.Evaluator` (the existing stateless one) gains `NewThreadEvaluator() eval.Evaluator` returning `NewIncremental(e.net)`, so every existing construction site transparently upgrades.

**Design:**
- Each search thread's `state` gets `ev eval.Evaluator`: `s.evaluator.(eval.PerThread).NewThreadEvaluator()` when implemented, else the shared evaluator. All `st.s.evaluator.Evaluate(st.pos)` become `st.ev.Evaluate(st.pos)`.
- At thread start (root position set): if `st.ev` implements `eval.MoveAware`, `Reset(root)`.
- Around every MakeMove/UnmakeMove and null-move pair in negamax/quiescence/root: `Push(pos, m)` BEFORE MakeMove, `Pop()` after UnmakeMove; `PushNull/PopNull` around the null-move pair. Use a tiny helper on `state` (`st.make(m)` / `st.unmake()`) so a future call site cannot forget the pairing — replace ALL raw call sites with the helpers.
- HCE path: implements neither interface, zero behavior change.

- [ ] **Step 1: The proof test (write first):**

```go
// TestIncrementalSearchIsBehaviourNeutral is the CLAUDE.md-mandated proof:
// same network, same positions, fixed depth, one thread — the node count
// and best move with the incremental evaluator must EQUAL those with a
// wrapper that forces the old full-refresh path. Node counts are compared
// between the two configurations in the same process, never against stored
// constants.
//
// forceRefresh wraps the same *nnue.Network in the old stateless evaluator
// (nnue.NewEvaluator or equivalent) WITHOUT PerThread, so the search uses
// the shared refresh-per-call path.
func TestIncrementalSearchIsBehaviourNeutral(t *testing.T) {
	net := testNetworkForSearch(t)
	fens := []string{ /* 6-10 varied, VERIFIED positions: middlegame, endgame, in-check, promotion-rich */ }
	for _, fen := range fens {
		nodesInc, moveInc := searchFixedDepth(t, fen, 6, nnueIncremental(net))
		nodesRef, moveRef := searchFixedDepth(t, fen, 6, nnueRefreshOnly(net))
		if nodesInc != nodesRef || moveInc != moveRef {
			t.Fatalf("%s: incremental (nodes %d, move %v) diverged from refresh (nodes %d, move %v)", fen, nodesInc, moveInc, nodesRef, moveRef)
		}
	}
}
```

  Also extend/verify the existing single-thread determinism test still passes with the NNUE evaluator (find it in search_test.go / selfplay_test.go — it must now cover the incremental path too, since gen-1 self-play will use it).
- [ ] **Step 2-4: red, implement, green** (full package under `-race`, incl. SMP tests — each thread having its own evaluator is exactly what makes this race-free; the old shared-stateless path was only safe because it was stateless).
- [ ] **Step 5: Indicative timing note** (allowed as indicative, not proof): run the existing bench or a quick fixed-node search with each path and record the wall-clock ratio in the commit message as "indicative".
- [ ] **Step 6: Full gates, commit.**

---

### Task 3: Concurrent gate matches

**Files:**
- Modify: `internal/gate/match.go`, `internal/gate/gate.go` (result ingestion loop), `cmd/gatekeeper/main.go`, `deploy/jobs.yaml`
- Test: extend `internal/gate`'s tests

**Design:**
- `MatchConfig.Concurrency int` (default in `DefaultMatchConfig`: 4; `validate()`: >= 1). Read match.go fully first: game i's opening/seed must be a pure function of (Seed, i) — if it currently derives openings from a sequentially-advanced rng, refactor to per-game derived seeds (e.g. splitmix on Seed+i) so concurrency does not change which openings are played. EVERY game remains a single-threaded search pair.
- Worker pool: `Concurrency` goroutines pull game indices from a channel, play, send `gameResult{index, outcome, nodes}` on a results channel. The SPRT loop ingests results in ARRIVAL order and stops when a bound is crossed; in-flight games are cancelled via context. Comment honestly: the stopping point (and thus Report.Games) varies run to run with concurrency > 1 — the VERDICT's error guarantees are unchanged (each game is an iid sample either way).
- Each concurrent game needs its own transposition tables/engine instances — check what the sequential runner reuses between games and give each worker goroutine its own pair.
- `cmd/gatekeeper`: `-concurrency` flag defaulting to `match.Concurrency`.
- jobs.yaml gatekeeper: add `- -concurrency` / `- "6"` to the command and raise cpu request `"4"` → `"6"` (self-play is parked during gates; the board has 8).
- Tests: with a deterministic fake/light evaluator, (a) a match at Concurrency 4 produces the same SET of game outcomes for the same seeds as Concurrency 1 when run to MaxGames without early stop (order-independent equality on the multiset of per-index results); (b) early SPRT stop cancels in-flight games promptly (bounded wait); (c) `-race` clean.

- [ ] Red → implement → green → gates → commit.

---

### Task 4: Coordinator resumes a partially-complete generation

**Files:**
- Modify: `internal/coordinator/coordinator.go` (startup path), possibly `internal/store` (a Verify-existing helper) — read both plus `internal/train`'s file-reading API first.
- Test: extend `internal/coordinator` tests (they run against the memory bus + a temp dir).

**Design:**
- On `starting a generation`, before issuing: scan the generation's artifact directory (`store` knows the layout; unit ID ↔ filename mapping is deterministic `g<G>-u<NNNNNN>.novadata`). For each existing file: validate it (parse header/samples via the train package's reader or a cheaper dedicated validation in store — pick whichever exists; a corrupt/truncated file counts as absent and its unit is reissued). Sum positions, mark those unit IDs complete, log one line: `resumed a generation, counted N positions from M existing artifacts`.
- Issue only units not already on disk; the completion check and dataset announcement use the combined total. If the target is already met by existing artifacts alone, announce immediately and exit as usual.
- The at-least-once bus dedup stays untouched — a live batch announcement for an already-counted unit is ignored by the existing WorkUnitID dedup.
- Tests: (a) pre-populate a temp dir with K valid artifacts from a tiny real generation run (the coordinator tests already know how to produce artifacts), restart the coordinator with a larger target: it must issue only the missing units and its announced total must include the pre-existing positions; (b) a truncated file is reissued, not counted; (c) target-already-met announces without issuing anything.

- [ ] Red → implement → green → gates → commit.

---

### Task 5: Trainer LR schedule + fine-tuning, wired end to end

**Files:**
- Modify: `internal/train/trainer.go`, `cmd/trainer/main.go`, `internal/dashboard/control.go` (+ its tests), `deploy/jobs.yaml`

**Design:**
- `Params.LRDropEpochs []int` (validated ascending, in range): at the START of each listed epoch the learning rate halves. Default in `DefaultParams()`: `[]int{8, 12}`. Flag `-lr-drops "8,12"` (comma list, empty = none). Unit test the pure schedule function (epoch → multiplier) — not training outcomes.
- `-init <path>`: when set and the file exists, load the network (`nnue.Load`) and dequantize into the trainer's float parameters as the starting point (inverse of the quantization the trainer already performs; document the precision loss — int16/QA quantization round-trip — in a comment; it is small relative to what an epoch changes and the property that matters is a WARM start, not an exact one). When the flag is set but the file is missing, log and train from scratch rather than failing: the first attempt of a generation has no predecessor.
- Round-trip test: train a tiny net, quantize+write, `-init` it back, assert the dequantized params produce evaluations close to the loaded net (tolerance = one quantization step per layer — an invariant, not an exact score).
- jobs.yaml trainer command gains `- -init` / `- /data/networks/gen0.nnue` and `- -lr-drops` / `- "8,12"`.
- Dashboard rewriter (`generationPathRewriter`/`gatekeeperRewriter` area in control.go — read the current implementation): the element after `-init` is rewritten to `/data/networks/gen<G-1>.nnue` for generation G > 0 (fine-tune from the promoted predecessor) and left as `gen0.nnue` for G = 0 (continue from the previous attempt); in BOTH cases, if the target file does not exist on the volume, DROP the `-init` pair (mirror of the `-incumbent` drop logic — reuse that mechanism). Tests: gen-0 launch keeps `-init gen0.nnue` when the file exists, drops it when absent; gen-2 launch rewrites to `gen1.nnue`; tuned flags still survive verbatim (extend the existing survive-verbatim tests).

- [ ] Red → implement → green → gates → commit.

---

### Task 6: Ship and deploy (controller)

- [ ] Full gates + `cd web/dashboard && npm test` (frontend untouched; embed sanity) one last time.
- [ ] Push, draft PR, CI green (rerun the known `TestNATSSubscriptionCloseStopsDelivery` flake if it appears), merge with a merge commit, restart branch.
- [ ] Pin kustomization to the CI image, apply. NOTE: the 30M fill is running — Task 4 makes the coordinator roll SAFE (it resumes instead of replaying). Workers roll too (image change): units in flight return via SIGTERM and redeliver. Verify after apply: coordinator logs `resumed a generation` with a sane count, fill continues from where it was, fleet healthy.
- [ ] When the 30M dataset announces: stop self-play, launch trainer (now with `-init` + `-lr-drops`), then the gate (now concurrent + incremental NNUE) — verdict in well under an hour instead of five.

## Self-review notes

- Task 1's property test is the single load-bearing correctness artifact; Task 2's node-count equality is the CLAUDE.md proof. Both compare two live configurations, never stored numbers.
- Task 3 changes Report.Games run-to-run variance under concurrency — documented where it happens; determinism of individual games and of the single-threaded search is untouched.
- Task 4 preserves at-least-once semantics; dedup logic untouched.
- Task 5's rewriter change follows the existing `-incumbent` drop mechanism and extends the existing tests that pin tuned-flag survival.
- Type names used consistently: `eval.MoveAware`, `eval.PerThread`, `nnue.IncrementalEvaluator`, `MatchConfig.Concurrency`, `Params.LRDropEpochs`.
