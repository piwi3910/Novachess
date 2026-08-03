package search

import "github.com/piwi3910/novachess/internal/chess"

// Move ordering.
//
// Alpha-beta's efficiency depends almost entirely on trying the best move
// first: with perfect ordering it examines the square root of the nodes that
// perfect anti-ordering would. Nothing else in the search buys as much for as
// little, which is why the ordering heuristics get this much attention.
//
// Moves are scored once and then selected one at a time by scanning for the
// highest remaining score. A full sort would be wasted work, since most nodes
// cut off after the first move or two.

// Ordering score bands, kept far enough apart that no two categories can
// overlap through their internal scoring.
const (
	scoreTTMove      = 1 << 24
	scorePromotion   = 1 << 22
	scoreGoodCapture = 1 << 21
	scoreKiller1     = 1 << 20
	scoreKiller2     = scoreKiller1 - 1
	scoreQuietBase   = 0
)

// pieceValue is a coarse material scale used only for ordering and for delta
// pruning. It is deliberately separate from the evaluation's material values:
// ordering wants a stable, simple ranking, not a tuned one.
var pieceValue = [chess.PieceTypeCount]int{100, 320, 330, 500, 900, 20000}

// scoreMoves assigns an ordering score to each move in the list, returning the
// scores in a parallel array.
func (st *state) scoreMoves(ml *chess.MoveList, ttMove chess.Move, ply int) *[chess.MaxMoves]int {
	var scores [chess.MaxMoves]int
	us := st.pos.SideToMove()

	for i, m := range ml.Slice() {
		switch {
		case m == ttMove && !ttMove.IsNone():
			// The table's move was best last time this position was reached,
			// which makes it by far the best guess available.
			scores[i] = scoreTTMove

		case m.Kind() == chess.KindPromotion:
			// Queen promotions first; underpromotions are almost never best
			// and are pushed below ordinary captures.
			if m.Promotion() == chess.Queen {
				scores[i] = scorePromotion + pieceValue[chess.Queen]
			} else {
				scores[i] = scoreQuietBase + int(m.Promotion())
			}

		case st.pos.IsCapture(m):
			// Most Valuable Victim, Least Valuable Attacker: taking a queen
			// with a pawn is tried before taking a pawn with a queen, because
			// the first is far more likely to win material.
			scores[i] = scoreGoodCapture + captureValue(st.pos, m)*16 -
				pieceValue[st.pos.PieceAt(m.From()).Type()]

		case m == st.killers[ply][0]:
			scores[i] = scoreKiller1
		case m == st.killers[ply][1]:
			scores[i] = scoreKiller2

		default:
			// Quiet moves are ranked by how often they have caused cutoffs.
			scores[i] = scoreQuietBase + int(st.history[us][m.From()][m.To()])
		}
	}

	return &scores
}

// pickMove swaps the highest-scoring remaining move into position i and returns
// it. Selection is done lazily so that a node cutting off after one move pays
// for one scan rather than a full sort.
func pickMove(ml *chess.MoveList, scores *[chess.MaxMoves]int, i int) chess.Move {
	best := i
	for j := i + 1; j < ml.Count; j++ {
		if scores[j] > scores[best] {
			best = j
		}
	}
	if best != i {
		ml.Moves[i], ml.Moves[best] = ml.Moves[best], ml.Moves[i]
		scores[i], scores[best] = scores[best], scores[i]
	}
	return ml.Moves[i]
}

// captureValue returns the material value of what a move captures, which is
// zero for a quiet move. En passant captures a pawn even though the destination
// square is empty.
func captureValue(p *chess.Position, m chess.Move) int {
	if m.Kind() == chess.KindEnPassant {
		return pieceValue[chess.Pawn]
	}
	if victim := p.PieceAt(m.To()); victim != chess.NoPiece && m.Kind() != chess.KindCastling {
		return pieceValue[victim.Type()]
	}
	return 0
}
