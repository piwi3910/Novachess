// Package search finds the best move in a position.
//
// The search is alpha-beta negamax with iterative deepening, not MCTS. Every
// technique here exists to visit fewer nodes for the same answer: the ordering
// heuristics make cutoffs happen on the first move tried, the transposition
// table avoids re-searching positions reached by a different move order, and
// the pruning rules skip subtrees that provably, or very probably, cannot
// affect the result.
//
// Correctness has a specific meaning in a search: the move returned must be the
// best one the evaluation and depth can justify. Almost every bug here degrades
// play quietly rather than crashing, which is why the tests check invariants —
// that a mate is found, that the principal variation is legal, that repetition
// is scored as a draw — rather than exact node counts, which change whenever
// any heuristic is touched.
package search

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// MaxPly bounds search depth and the size of the per-ply stacks.
const MaxPly = 128

// Limits describes when the search must stop.
type Limits struct {
	// Depth stops after completing this iteration. Zero means no depth limit.
	Depth int
	// Nodes stops after approximately this many nodes. Zero means no limit.
	Nodes uint64
	// MoveTime allots a fixed time for this move, overriding the clock.
	MoveTime time.Duration

	// Clock state, used when MoveTime is unset.
	WhiteTime, BlackTime time.Duration
	WhiteInc, BlackInc   time.Duration
	MovesToGo            int

	// Infinite searches until explicitly stopped.
	Infinite bool
}

// Info is a progress report, emitted once per completed iteration.
type Info struct {
	Depth    int
	SelDepth int
	Score    int
	Mate     bool
	MateIn   int
	Nodes    uint64
	Elapsed  time.Duration
	PV       []chess.Move
	HashFull int
}

// NodesPerSecond returns the search rate, or zero before any time has passed.
func (i Info) NodesPerSecond() uint64 {
	if i.Elapsed <= 0 {
		return 0
	}
	return uint64(float64(i.Nodes) / i.Elapsed.Seconds())
}

// Result is the outcome of a search.
type Result struct {
	Best   chess.Move
	Ponder chess.Move
	Info   Info
}

// Searcher runs searches. One instance owns the transposition table and is
// reused across moves so that the table survives between them, which is most of
// its value.
type Searcher struct {
	tt        *TT
	evaluator eval.Evaluator
	stopped   atomic.Bool

	// threads is atomic because SetThreads may be called from a different
	// goroutine than the one searching. The UCI layer stops the search first,
	// but the package API does not require callers to, and an unsynchronized
	// read here would be a data race rather than merely a stale value.
	threads atomic.Int32
	// totalNodes aggregates every thread's node count so that progress reports
	// describe the whole search rather than one thread's share.
	totalNodes atomic.Uint64
}

// New returns a searcher with a table of the given size in megabytes, searching
// on a single thread.
func New(evaluator eval.Evaluator, hashMB int) *Searcher {
	s := &Searcher{
		tt:        NewTT(hashMB),
		evaluator: evaluator,
	}
	s.threads.Store(1)
	return s
}

// SetThreads sets how many threads a search uses.
//
// One thread is not merely the default but a distinct guarantee: a single
// thread produces identical results for identical inputs, and more than one
// does not. Lazy SMP works precisely by letting threads race on a shared
// transposition table, so which thread reaches a position first — and therefore
// what the search concludes — depends on timing.
//
// That makes the choice a policy decision rather than a performance knob.
// Self-play must run one thread per worker, because the pipeline replays work
// units on different nodes and the training data cannot depend on which machine
// ran them. Match play and analysis, where reproducibility does not matter,
// should use as many as are available.
func (s *Searcher) SetThreads(n int) {
	if n < 1 {
		n = 1
	}
	if n > MaxThreads {
		n = MaxThreads
	}
	s.threads.Store(int32(n))
}

// Threads reports the configured thread count.
func (s *Searcher) Threads() int { return int(s.threads.Load()) }

// MaxThreads bounds the thread count. Each thread holds its own search state of
// a few hundred kilobytes, so the limit is about memory rather than usefulness.
const MaxThreads = 256

// SetHashSize replaces the transposition table. Existing contents are lost,
// which is why UCI only allows it between games.
func (s *Searcher) SetHashSize(megabytes int) { s.tt = NewTT(megabytes) }

// SetEvaluator swaps the evaluation, which is how a trained network replaces
// the hand-crafted one without the search knowing the difference.
//
// It must not be called while a search is running: the search threads read the
// evaluator without synchronization, on the hot path where a mutex would cost
// more than the evaluation itself. The table is also stale afterwards, holding
// scores the old evaluation produced, so it is cleared rather than left to
// blend two evaluations together.
func (s *Searcher) SetEvaluator(e eval.Evaluator) {
	s.evaluator = e
	s.Clear()
}

// Evaluator returns the evaluation in use, so that a caller reporting a static
// score reports the one the search would actually see.
func (s *Searcher) Evaluator() eval.Evaluator { return s.evaluator }

// Clear empties the transposition table, for "ucinewgame".
func (s *Searcher) Clear() { s.tt.Clear() }

// Stop asks the running search to finish as soon as it can. It is safe to call
// from another goroutine, which is how UCI's "stop" is implemented.
func (s *Searcher) Stop() { s.stopped.Store(true) }

// state is the per-search working set. It is not shared between searches, so
// killers and history start clean for each move while the transposition table
// persists.
type state struct {
	s   *Searcher
	pos *chess.Position

	// ev is this thread's evaluator. It is the shared evaluator when the
	// configured one carries no state (the HCE), or a private instance from
	// eval.PerThread's NewThreadEvaluator when it does (NNUE): the whole
	// reason a state exists per thread rather than being shared is that only
	// one thread at a time may drive an incremental evaluator's accumulator
	// stack.
	ev eval.Evaluator
	// moveAware is ev again, asserted once at construction, or nil when ev
	// carries no state to keep in step. make/unmake/makeNull/unmakeNull are
	// the only places that touch it, so a call site cannot forget to pair a
	// Push with its Pop: there is no other way to advance st.pos.
	moveAware eval.MoveAware

	nodes    uint64
	selDepth int

	// killers are quiet moves that produced a cutoff at the same ply
	// elsewhere in the tree. Two per ply is the usual compromise: a third adds
	// very little and costs ordering time.
	killers [MaxPly][2]chess.Move
	// history scores quiet moves by how often they caused cutoffs, indexed by
	// mover and destination. Unlike killers it is global to the search.
	history [chess.ColorCount][chess.SquareCount][chess.SquareCount]int32

	// moveScores is one ordering-score array per ply, pre-allocated so that
	// scoring moves at a node costs nothing. A local array here would escape to
	// the heap and allocate 2KB at every node.
	moveScores [MaxPly][chess.MaxMoves]int32

	// Triangular principal-variation table.
	pv       [MaxPly][MaxPly]chess.Move
	pvLength [MaxPly]int

	deadline time.Time
	hardStop bool
	maxNodes uint64
	ctx      context.Context
	aborted  bool

	// reported is how much of nodes has already been added to the searcher's
	// shared total, so each flush contributes only the delta.
	reported uint64
	// threadID is zero for the thread whose result is reported; helpers use it
	// to stagger their starting depth.
	threadID int
}

// Search finds the best move. onInfo, when non-nil, is called once per
// completed iteration of the reporting thread.
func (s *Searcher) Search(ctx context.Context, pos *chess.Position, limits Limits, onInfo func(Info)) Result {
	s.stopped.Store(false)
	s.tt.NewSearch()
	s.totalNodes.Store(0)

	// A legal move must be returned even if the search is stopped before it
	// completes anything, so the result is seeded before searching.
	var root chess.MoveList
	pos.GenerateLegalMoves(&root)
	if root.Count == 0 {
		return Result{}
	}

	start := time.Now()
	soft, hard := budget(limits, pos.SideToMove())

	maxDepth := limits.Depth
	if maxDepth <= 0 || maxDepth > MaxPly-1 {
		maxDepth = MaxPly - 1
	}

	newState := func(id int) *state {
		st := &state{
			s:        s,
			pos:      pos.Clone(),
			ctx:      ctx,
			maxNodes: limits.Nodes,
			threadID: id,
		}
		if pt, ok := s.evaluator.(eval.PerThread); ok {
			// A stateful evaluator (NNUE) gets its own accumulator stack per
			// thread; sharing one would mean every lazy SMP thread writing
			// the same accumulator at once.
			st.ev = pt.NewThreadEvaluator()
		} else {
			// Stateless (the HCE): sharing is exactly as safe as it always
			// was, so every thread reads the one evaluator on the searcher.
			st.ev = s.evaluator
		}
		if ma, ok := st.ev.(eval.MoveAware); ok {
			st.moveAware = ma
			// Prime the accumulator for this thread's root before any
			// Push/Pop walks the tree, so nothing from a previous search on
			// this evaluator can leak in.
			ma.Reset(st.pos)
		}
		if hard > 0 {
			st.deadline = start.Add(hard)
			st.hardStop = true
		}
		return st
	}

	// Helper threads search the same position concurrently, sharing only the
	// transposition table. They report nothing; their contribution is the
	// entries they leave behind, which deepen and improve the main thread's
	// search. This is lazy SMP: no work is partitioned and no thread waits for
	// another.
	threads := s.Threads()
	var wg sync.WaitGroup
	for id := 1; id < threads; id++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			newState(id).iterate(limits, maxDepth, soft, start, nil)
		}(id)
	}

	result := newState(0).iterate(limits, maxDepth, soft, start, onInfo)
	if result.Best.IsNone() {
		result.Best = root.Moves[0]
	}

	// The main thread has its answer, so the helpers' work can no longer
	// change it.
	s.stopped.Store(true)
	wg.Wait()

	return result
}

// iterate runs iterative deepening for one thread.
func (st *state) iterate(limits Limits, maxDepth int, soft time.Duration, start time.Time, onInfo func(Info)) Result {
	var result Result

	// Helper threads begin one or two plies deeper than the main thread so
	// that they immediately explore different parts of the tree instead of
	// duplicating its early iterations.
	depth := 1
	if st.threadID > 0 {
		depth += st.threadID % 3
		if depth > maxDepth {
			depth = maxDepth
		}
	}

	var lastScore int
	for ; depth <= maxDepth; depth++ {
		score, ok := st.searchRoot(depth, lastScore)
		if !ok {
			// The iteration was abandoned; its result is not trustworthy, so
			// the previous iteration's move stands.
			break
		}
		lastScore = score

		result.Best = st.pv[0][0]
		result.Ponder = chess.MoveNone
		if st.pvLength[0] > 1 {
			result.Ponder = st.pv[0][1]
		}
		// Only the reporting thread builds Info. It samples the table for
		// occupancy and allocates the principal variation, none of which a
		// helper's discarded result needs.
		if st.threadID == 0 {
			result.Info = st.info(depth, score, time.Since(start))
			if onInfo != nil {
				onInfo(result.Info)
			}
		}

		if eval.IsMateScore(score) && !limits.Infinite {
			// A forced mate is found; searching deeper cannot improve on it.
			break
		}
		// Stop before starting an iteration there is not enough time to
		// finish. Iterations grow by roughly a constant factor, so if most of
		// the soft budget is already gone the next one will certainly overrun.
		if soft > 0 && time.Since(start) > soft*2/3 {
			break
		}
		if st.stopping() {
			break
		}
	}

	st.flushNodes()
	return result
}

func (st *state) info(depth, score int, elapsed time.Duration) Info {
	pv := make([]chess.Move, st.pvLength[0])
	copy(pv, st.pv[0][:st.pvLength[0]])

	// Node counts describe the whole search, not this thread's share, so that
	// the figure a GUI displays matches the work actually done.
	st.flushNodes()

	info := Info{
		Depth:    depth,
		SelDepth: st.selDepth,
		Score:    score,
		Nodes:    st.s.totalNodes.Load(),
		Elapsed:  elapsed,
		PV:       pv,
		HashFull: st.s.tt.Hashfull(),
	}
	if eval.IsMateScore(score) {
		info.Mate = true
		plies := eval.MateDistance(score)
		// UCI reports mate in moves, signed by who is delivering it, and
		// rounds a mate that takes an odd number of plies up.
		if plies > 0 {
			info.MateIn = (plies + 1) / 2
		} else {
			info.MateIn = -((-plies + 1) / 2)
		}
	}
	return info
}

// searchRoot runs one iteration, returning the score and whether the iteration
// completed. Aspiration windows start narrow around the previous score and
// widen on failure, which searches far fewer nodes when the score is stable.
func (st *state) searchRoot(depth, prevScore int) (int, bool) {
	alpha, beta := -eval.Infinite, eval.Infinite

	// Only use a narrow window once there is a previous score to centre it on
	// and the score is not a mate, where the window is meaningless.
	window := 25
	if depth >= 4 && !eval.IsMateScore(prevScore) {
		alpha = prevScore - window
		beta = prevScore + window
	}

	for {
		score := st.negamax(alpha, beta, depth, 0, true)
		if st.aborted {
			return 0, false
		}

		switch {
		case score <= alpha:
			// Failed low: the true score is worse than expected. Widen
			// downward, keeping beta so the re-search stays cheap.
			beta = (alpha + beta) / 2
			alpha -= window
			window *= 2
		case score >= beta:
			beta += window
			window *= 2
		default:
			return score, true
		}

		if window > 1000 {
			alpha, beta = -eval.Infinite, eval.Infinite
		}
		if st.stopping() {
			return 0, false
		}
	}
}

// stopping reports whether the search must end now.
func (st *state) stopping() bool {
	if st.s.stopped.Load() {
		return true
	}
	if st.ctx != nil && st.ctx.Err() != nil {
		return true
	}
	// The node limit counts every thread's work, so that "go nodes N" means the
	// same amount of computation however many threads are configured.
	if st.maxNodes > 0 && st.s.totalNodes.Load() >= st.maxNodes {
		return true
	}
	if st.hardStop && time.Now().After(st.deadline) {
		return true
	}
	return false
}

// checkStop is called periodically from the tree. Consulting the clock at every
// node would cost more than the search saves, so it is sampled — and the same
// sampling point publishes this thread's node count to the shared total.
func (st *state) checkStop() {
	if st.nodes&2047 != 0 {
		return
	}
	st.flushNodes()
	if st.stopping() {
		st.aborted = true
	}
}

// flushNodes publishes the nodes searched since the last flush.
func (st *state) flushNodes() {
	if delta := st.nodes - st.reported; delta > 0 {
		st.s.totalNodes.Add(delta)
		st.reported = st.nodes
	}
}

// budget converts the clock into a soft and hard time allowance.
//
// The soft limit decides whether to begin another iteration; the hard limit
// aborts mid-iteration. They differ because abandoning an iteration wastes
// everything spent on it, so it is better to decline to start one than to be
// cut off halfway through.
func budget(l Limits, side chess.Color) (soft, hard time.Duration) {
	if l.Infinite {
		return 0, 0
	}
	if l.MoveTime > 0 {
		return l.MoveTime, l.MoveTime
	}

	remaining, increment := l.WhiteTime, l.WhiteInc
	if side == chess.Black {
		remaining, increment = l.BlackTime, l.BlackInc
	}
	if remaining <= 0 {
		// No clock information: rely on depth or node limits instead.
		return 0, 0
	}

	// Reserve a margin so that a slow move or a scheduling hiccup cannot lose
	// on time. On a thermally throttled cluster node this matters more than it
	// looks.
	const overhead = 30 * time.Millisecond

	moves := l.MovesToGo
	if moves <= 0 {
		// Without a move count, assume the game has a while to run. Dividing
		// by a constant spends more early and less later, which matches how
		// games actually go.
		moves = 30
	}

	soft = remaining/time.Duration(moves) + increment*3/4
	hard = soft * 3

	if limit := remaining - overhead; hard > limit {
		hard = limit
	}
	if hard < 0 {
		hard = 0
	}
	if soft > hard {
		soft = hard
	}
	return soft, hard
}
