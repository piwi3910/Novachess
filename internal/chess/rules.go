package chess

// Game termination rules.
//
// Move generation answers "what may be played". This file answers "is the game
// over", which is a separate question and not one perft can check: perft counts
// moves, and every rule here is about positions in which the move count is
// irrelevant or zero.
//
// These rules are not optional polish. Self-play cannot function without them —
// two engines with no draw detection shuffle pieces until the ply limit and
// emit a game with no result, which is worse than useless as training data
// because it is indistinguishable from a real draw.

// Outcome is the result of a finished game, or Ongoing.
type Outcome uint8

const (
	Ongoing Outcome = iota
	WhiteWins
	BlackWins
	Draw
)

func (o Outcome) String() string {
	switch o {
	case WhiteWins:
		return "1-0"
	case BlackWins:
		return "0-1"
	case Draw:
		return "1/2-1/2"
	}
	return "*"
}

// Score returns the result from White's perspective: 1 for a White win, 0 for a
// Black win, 0.5 for a draw. This is the target the training pipeline blends
// with search evaluations, hence the fixed orientation.
func (o Outcome) Score() float64 {
	switch o {
	case WhiteWins:
		return 1
	case BlackWins:
		return 0
	default:
		return 0.5
	}
}

// TerminationReason explains why a game ended.
type TerminationReason uint8

const (
	NotTerminated TerminationReason = iota
	Checkmate
	Stalemate
	FiftyMoveRule
	ThreefoldRepetition
	InsufficientMaterial
)

func (r TerminationReason) String() string {
	switch r {
	case Checkmate:
		return "checkmate"
	case Stalemate:
		return "stalemate"
	case FiftyMoveRule:
		return "fifty-move rule"
	case ThreefoldRepetition:
		return "threefold repetition"
	case InsufficientMaterial:
		return "insufficient material"
	}
	return "not terminated"
}

// Outcome reports whether the game has ended and why.
//
// Checkmate and stalemate are checked before the drawing rules deliberately:
// delivering mate on the hundredth reversible move wins the game, it does not
// draw it. The order here is the rule, not a preference.
func (p *Position) Outcome() (Outcome, TerminationReason) {
	var ml MoveList
	p.GenerateLegalMoves(&ml)

	if ml.Count == 0 {
		if p.InCheck() {
			if p.side == White {
				return BlackWins, Checkmate
			}
			return WhiteWins, Checkmate
		}
		return Draw, Stalemate
	}

	if p.IsInsufficientMaterial() {
		return Draw, InsufficientMaterial
	}
	if p.IsThreefoldRepetition() {
		return Draw, ThreefoldRepetition
	}
	if p.IsFiftyMoveRule() {
		return Draw, FiftyMoveRule
	}
	return Ongoing, NotTerminated
}

// FIDE draws come in two kinds, and the distinction matters for anything that
// has to agree with an external arbiter.
//
// The fifty-move rule and threefold repetition are *claimable*: the game
// continues unless a player claims the draw. The seventy-five-move rule and
// fivefold repetition are *automatic*: the arbiter ends the game whether or not
// anyone asks.
//
// This engine adjudicates on the claimable thresholds, which is universal
// engine convention and what self-play needs — a game that continues past a
// threefold repetition because neither side "claimed" would run forever. The
// automatic thresholds are exposed separately for callers that must match a
// server's adjudication, which is the Lichess bot's situation.
const (
	fiftyMovePlies       = 100
	seventyFiveMovePlies = 150
)

// IsFiftyMoveRule reports whether a hundred plies have passed with no capture
// and no pawn move. This is the claimable draw of FIDE 9.3.
//
// The caller must still confirm the side to move has a legal move: a position
// that is checkmate on the hundredth ply is a win, not a draw. Outcome handles
// that ordering.
func (p *Position) IsFiftyMoveRule() bool { return p.halfmove >= fiftyMovePlies }

// IsSeventyFiveMoveRule reports whether a hundred and fifty plies have passed
// with no capture and no pawn move. This is the automatic draw of FIDE 9.6.2,
// which needs no claim from either player.
//
// Outcome never reports it, because it draws at the claimable fifty-move
// threshold first. It exists for callers reconciling with an external arbiter.
func (p *Position) IsSeventyFiveMoveRule() bool { return p.halfmove >= seventyFiveMovePlies }

// RepetitionCount returns how many times the current position has previously
// occurred in this game with the same side to move.
//
// Only positions since the last irreversible move are considered. A capture, a
// pawn move or a change of castling rights makes every earlier position
// permanently unreachable, so the halfmove clock bounds exactly how far back it
// is worth looking — which is both correct and much cheaper than scanning the
// entire game.
//
// Note this compares Zobrist keys rather than full positions. A 64-bit
// collision would report a repetition that did not happen; at the rate
// positions are examined this is vanishingly unlikely and is the same trade
// every engine makes.
func (p *Position) RepetitionCount() int {
	// Positions earlier than the last irreversible move cannot recur.
	end := len(p.history) - p.halfmove
	if end < 0 {
		end = 0
	}

	// Step back two plies at a time: only positions with the same side to move
	// can be repetitions of this one.
	count := 0
	for i := len(p.history) - 2; i >= end; i -= 2 {
		if p.history[i].key == p.key {
			count++
		}
	}
	return count
}

// IsThreefoldRepetition reports whether the current position has occurred three
// times, counting the present occurrence. This is the claimable draw of FIDE
// 9.2.
func (p *Position) IsThreefoldRepetition() bool { return p.RepetitionCount() >= 2 }

// IsFivefoldRepetition reports whether the current position has occurred five
// times. This is the automatic draw of FIDE 9.6.1, which needs no claim.
//
// As with the seventy-five-move rule, Outcome never reports it because it draws
// at the threefold threshold first; it exists for callers reconciling with an
// external arbiter.
func (p *Position) IsFivefoldRepetition() bool { return p.RepetitionCount() >= 4 }

// IsRepetition reports whether the current position has occurred before at all.
//
// This is the test a search wants, rather than the threefold one: inside the
// tree, a single repetition already means the side to move can force the
// position again and claim the draw, so treating the second occurrence as
// drawn is both sound and far more effective at cutting off repetition lines.
// Game adjudication uses IsThreefoldRepetition instead.
func (p *Position) IsRepetition() bool { return p.RepetitionCount() >= 1 }

// IsInsufficientMaterial reports whether neither side can deliver checkmate by
// any sequence of legal moves.
//
// This covers the four positions in which mate is strictly impossible: bare
// kings, king and bishop, king and knight, and king and bishop each with the
// bishops on the same color. Positions where mate is merely unreachable against
// correct defence — king and two knights against a bare king, for instance —
// are not draws by this rule, because mate remains possible with cooperation.
//
// FIDE 5.2.2 is strictly broader than this: a "dead position" is any position
// from which no series of legal moves can produce mate, which includes
// pawn structures locked so completely that neither king can ever break
// through. Detecting that in general is not tractable, and no engine attempts
// it; the material-based subset here is the universal practical reading. The
// difference only ever costs a draw claim in positions that are drawn anyway.
func (p *Position) IsInsufficientMaterial() bool {
	// Any pawn, rook or queen leaves mate possible.
	if p.byType[Pawn]|p.byType[Rook]|p.byType[Queen] != 0 {
		return false
	}

	knights := p.byType[Knight]
	bishops := p.byType[Bishop]
	minors := knights | bishops

	switch minors.Count() {
	case 0:
		// King against king.
		return true
	case 1:
		// A single minor piece cannot force mate.
		return true
	case 2:
		// Two bishops on the same color square can never mate, whichever side
		// owns them: they control only one color complex, and the defending
		// king simply stays on the other.
		if bishops.Count() != 2 {
			// Two knights, or a knight and a bishop: mate is possible with
			// cooperation, so this is not a dead position.
			return false
		}
		first := bishops.First()
		second := (bishops & (bishops - 1)).First()
		return squareIsDark(first) == squareIsDark(second)
	default:
		return false
	}
}

// squareIsDark reports whether a square is a dark square. A1 is dark, so the
// parity of file+rank determines the color.
func squareIsDark(s Square) bool { return (s.File()+s.Rank())%2 == 0 }

// IsDraw reports whether the position is drawn by any of the drawing rules. It
// does not consider stalemate, which requires generating moves; use Outcome for
// a complete answer.
func (p *Position) IsDraw() bool {
	return p.IsFiftyMoveRule() || p.IsThreefoldRepetition() || p.IsInsufficientMaterial()
}
