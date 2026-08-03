package search

import (
	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// negamax searches a node and returns its score from the side to move's
// perspective.
//
// allowNull guards against two null moves in a row, which would search a
// position two of the opponent's moves ahead and prove nothing.
func (st *state) negamax(alpha, beta, depth, ply int, allowNull bool) int {
	isRoot := ply == 0
	isPV := beta-alpha > 1

	st.pvLength[ply] = ply
	if ply > st.selDepth {
		st.selDepth = ply
	}

	st.checkStop()
	if st.aborted {
		return 0
	}

	if depth <= 0 {
		return st.quiescence(alpha, beta, ply)
	}

	st.nodes++

	if !isRoot {
		// Draws are scored before anything else. Missing a repetition here
		// makes the engine walk into drawn lines while believing it is winning.
		if st.pos.IsRepetition() || st.pos.IsFiftyMoveRule() || st.pos.IsDeadPosition() {
			return eval.DrawScore
		}
		if ply >= MaxPly-1 {
			return st.s.evaluator.Evaluate(st.pos)
		}

		// Mate distance pruning. If a mate has already been found closer to the
		// root than anything this subtree could deliver, the subtree cannot
		// matter. This also stops the search chasing longer mates once a short
		// one exists.
		alpha = max(alpha, -eval.MateScore+ply)
		beta = min(beta, eval.MateScore-ply-1)
		if alpha >= beta {
			return alpha
		}
	}

	ttMove, ttScore, _, ttUsable := st.s.tt.Probe(st.pos.Key(), depth, ply, alpha, beta)
	// A table cutoff is never taken on a principal-variation node: the score
	// would be correct but the move sequence behind it would not be recoverable,
	// leaving the reported line truncated or wrong.
	if ttUsable && !isPV && !isRoot {
		return ttScore
	}

	inCheck := st.pos.InCheck()

	// The static evaluation drives the pruning decisions below. It is
	// meaningless while in check, where the side to move is not free to stand
	// still.
	staticEval := 0
	if !inCheck {
		staticEval = st.s.evaluator.Evaluate(st.pos)
	}

	if !isPV && !inCheck && !isRoot {
		// Reverse futility: if the position is already so far above beta that
		// even a substantial loss would not bring it below, the opponent will
		// avoid this line and it need not be searched.
		if depth <= 6 && staticEval-120*depth >= beta && !eval.IsMateScore(beta) {
			return staticEval
		}

		// Null-move pruning. Give the opponent a free move; if the position is
		// still above beta, it is good enough to cut.
		//
		// This is unsound in zugzwang, where being obliged to move is itself
		// the problem, so it is disabled when the side to move has nothing but
		// king and pawns — which is where zugzwang actually occurs.
		if allowNull && depth >= 3 && staticEval >= beta && st.hasNonPawnMaterial() {
			reduction := 3 + depth/6
			st.pos.MakeNullMove()
			score := -st.negamax(-beta, -beta+1, depth-reduction, ply+1, false)
			st.pos.UnmakeNullMove()

			if st.aborted {
				return 0
			}
			if score >= beta && !eval.IsMateScore(score) {
				return score
			}
		}
	}

	var ml chess.MoveList
	st.pos.GenerateLegalMoves(&ml)

	if ml.Count == 0 {
		if inCheck {
			// Mate scored relative to the root, so that a faster mate scores
			// higher than a slower one and the engine actually finishes games.
			return -eval.MateScore + ply
		}
		return eval.DrawScore
	}

	scores := st.scoreMoves(&ml, ttMove, ply)

	bestScore := -eval.Infinite
	bestMove := chess.MoveNone
	bound := BoundUpper
	searched := 0

	for i := 0; i < ml.Count; i++ {
		m := pickMove(&ml, scores, i)
		isQuiet := !st.pos.IsCapture(m) && m.Kind() != chess.KindPromotion

		st.pos.MakeMove(m)

		var score int
		switch {
		case searched == 0:
			// The first move is searched with the full window; the ordering
			// heuristics exist to make this the best move most of the time.
			score = -st.negamax(-beta, -alpha, depth-1, ply+1, true)

		default:
			// Late move reductions: moves ordered last are unlikely to be
			// best, so they are searched shallower first and only re-searched
			// at full depth if they surprise us.
			reduction := 0
			if depth >= 3 && isQuiet && !inCheck && searched >= 4 {
				reduction = 1 + searched/12
				if isPV {
					reduction--
				}
				if reduction < 0 {
					reduction = 0
				}
				if reduction > depth-2 {
					reduction = depth - 2
				}
			}

			// Zero-window search first: we only need to know whether this move
			// beats alpha, not by how much.
			score = -st.negamax(-alpha-1, -alpha, depth-1-reduction, ply+1, true)
			if score > alpha && reduction > 0 {
				score = -st.negamax(-alpha-1, -alpha, depth-1, ply+1, true)
			}
			if score > alpha && score < beta {
				score = -st.negamax(-beta, -alpha, depth-1, ply+1, true)
			}
		}

		st.pos.UnmakeMove()
		searched++

		if st.aborted {
			return 0
		}

		if score > bestScore {
			bestScore = score
			bestMove = m

			if score > alpha {
				alpha = score
				bound = BoundExact
				st.updatePV(m, ply)

				if alpha >= beta {
					// Beta cutoff. Quiet moves that cause cutoffs are recorded
					// so they are tried earlier next time; captures are already
					// ordered well by their victim.
					if isQuiet {
						st.recordKiller(m, ply)
						st.recordHistory(m, depth)
					}
					bound = BoundLower
					break
				}
			}
		}
	}

	if !st.aborted {
		st.s.tt.Store(st.pos.Key(), bestMove, bestScore, staticEval, depth, ply, bound)
	}
	return bestScore
}

// quiescence searches only forcing moves, so that the evaluation is never
// applied to a position with a capture hanging.
//
// Without it the search is catastrophically wrong: a node cut off immediately
// after its own queen is attacked would be scored as if the queen were safe.
// This is the difference between an engine that plays chess and one that hangs
// pieces at every horizon.
func (st *state) quiescence(alpha, beta, ply int) int {
	st.nodes++
	st.checkStop()
	if st.aborted {
		return 0
	}

	if ply > st.selDepth {
		st.selDepth = ply
	}
	if ply >= MaxPly-1 {
		return st.s.evaluator.Evaluate(st.pos)
	}
	if st.pos.IsRepetition() || st.pos.IsFiftyMoveRule() || st.pos.IsDeadPosition() {
		return eval.DrawScore
	}

	inCheck := st.pos.InCheck()

	bestScore := -eval.Infinite
	if !inCheck {
		// Standing pat: the side to move is not obliged to capture, so the
		// static evaluation is a lower bound on what it can achieve.
		bestScore = st.s.evaluator.Evaluate(st.pos)
		if bestScore >= beta {
			return bestScore
		}
		if bestScore > alpha {
			alpha = bestScore
		}
	}

	var ml chess.MoveList
	st.pos.GenerateCaptures(&ml)

	// In check GenerateCaptures returns every evasion, so an empty list here
	// means checkmate rather than a quiet position.
	if ml.Count == 0 && inCheck {
		return -eval.MateScore + ply
	}

	scores := st.scoreMoves(&ml, chess.MoveNone, ply)

	for i := 0; i < ml.Count; i++ {
		m := pickMove(&ml, scores, i)

		// Delta pruning: if even winning the captured piece outright leaves the
		// position far below alpha, the capture cannot rescue it. Skipped while
		// in check, where the move may be forced rather than optional.
		if !inCheck && !eval.IsMateScore(alpha) {
			const margin = 200
			if bestScore+captureValue(st.pos, m)+margin < alpha {
				continue
			}
		}

		st.pos.MakeMove(m)
		score := -st.quiescence(-beta, -alpha, ply+1)
		st.pos.UnmakeMove()

		if st.aborted {
			return 0
		}

		if score > bestScore {
			bestScore = score
			if score > alpha {
				alpha = score
				if alpha >= beta {
					break
				}
			}
		}
	}

	return bestScore
}

// hasNonPawnMaterial reports whether the side to move has a piece other than
// pawns and the king, which is the condition under which null-move pruning is
// sound enough to use.
func (st *state) hasNonPawnMaterial() bool {
	us := st.pos.SideToMove()
	return st.pos.ColorPiecesOfType(us, chess.Knight)|
		st.pos.ColorPiecesOfType(us, chess.Bishop)|
		st.pos.ColorPiecesOfType(us, chess.Rook)|
		st.pos.ColorPiecesOfType(us, chess.Queen) != 0
}

// updatePV records a move as the best line at this ply and appends the child's
// line to it.
func (st *state) updatePV(m chess.Move, ply int) {
	st.pv[ply][ply] = m
	for next := ply + 1; next < st.pvLength[ply+1]; next++ {
		st.pv[ply][next] = st.pv[ply+1][next]
	}
	st.pvLength[ply] = st.pvLength[ply+1]
	if st.pvLength[ply] <= ply {
		st.pvLength[ply] = ply + 1
	}
}

func (st *state) recordKiller(m chess.Move, ply int) {
	if st.killers[ply][0] == m {
		return
	}
	st.killers[ply][1] = st.killers[ply][0]
	st.killers[ply][0] = m
}

func (st *state) recordHistory(m chess.Move, depth int) {
	us := st.pos.SideToMove()
	h := &st.history[us][m.From()][m.To()]
	*h += int32(depth * depth)

	// Keep the table bounded so that one dominant move cannot swamp the
	// ordering for the rest of the search.
	if *h > 1<<20 {
		for from := range st.history[us] {
			for to := range st.history[us][from] {
				st.history[us][from][to] /= 2
			}
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
