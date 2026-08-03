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
	// DeadPosition is the locked-pawn-structure half of FIDE 5.2.2, reported
	// separately from InsufficientMaterial because the two are established by
	// completely different arguments.
	DeadPosition
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
	case DeadPosition:
		return "dead position"
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
	if p.IsBlockedDeadPosition() {
		return Draw, DeadPosition
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

	switch (knights | bishops).Count() {
	case 0:
		// King against king.
		return true
	case 1:
		// A single minor piece cannot mate at all.
		return true
	}

	// Any knight beyond this point leaves mate possible with cooperation —
	// king and two knights can mate a bare king, it merely cannot be forced.
	if knights != 0 {
		return false
	}

	// Any number of bishops, on either side, confined to a single color
	// complex. Such a position is dead however many bishops are present: they
	// can only ever attack squares of their own color, so a king standing on
	// the other complex can never be attacked by anything but the enemy king,
	// and a lone king cannot deliver mate.
	//
	// This generalizes the two-bishop case, and covers arrangements a
	// piece-counting rule misses entirely — three same-colored bishops against
	// one, say, which arises from underpromotion.
	return bishopsOnOneColorComplex(bishops)
}

// bishopsOnOneColorComplex reports whether every bishop stands on a square of
// the same color. An empty set trivially satisfies this, so callers must
// handle the no-bishops case before asking.
func bishopsOnOneColorComplex(bishops Bitboard) bool {
	return bishops&darkSquares == 0 || bishops&lightSquares == 0
}

// darkSquares and lightSquares partition the board. A1 is dark.
const (
	darkSquares  Bitboard = 0xAA55AA55AA55AA55
	lightSquares Bitboard = ^darkSquares
)

// squareIsDark reports whether a square is a dark square. A1 is dark, so the
// parity of file+rank determines the color.
func squareIsDark(s Square) bool { return (s.File()+s.Rank())%2 == 0 }

// IsBlockedDeadPosition reports whether the position is dead because the pawn
// structure is permanently locked.
//
// This is the second half of FIDE 5.2.2, the part material counting cannot see:
// a board full of pawns where no sequence of legal moves can ever produce
// checkmate, because nothing can ever move except the kings.
//
// The test is a proof, not a heuristic, and it only answers yes when all three
// conditions hold:
//
//  1. Only kings and pawns remain. Any other piece can move freely and could
//     eventually break the structure or deliver mate.
//  2. Every pawn is immobile — blocked in front, with no enemy pawn on either
//     capturing diagonal. Since the only other mobile pieces are kings, and
//     kings cannot be captured, no pawn can ever acquire a capture. The
//     structure is a fixpoint.
//  3. Neither king can ever capture a pawn. Capturing is the one remaining way
//     to unfreeze the structure, so if no capture is reachable, the structure
//     is permanent.
//
// Given all three, the only moves that will ever exist are king moves, and two
// lone kings cannot deliver mate — a king never attacks the enemy king, and a
// frozen pawn can never give check, since the defending king would have to step
// into the attack voluntarily. The position is therefore dead.
//
// Condition 3 needs two observations to be tight enough to be useful. A king
// may never stand on a square an enemy pawn attacks, and those pawns never
// move, so pawn-attacked squares are permanent walls exactly like the pawns
// themselves — this is what seals the kings apart in the classic interlocked
// wall, where every square of the rank behind the wall is either a pawn or
// attacked by one. And a pawn defended by another pawn can never be taken by a
// king at all, since the capture would be a move into check. Without both, the
// detector is sound but fires on almost nothing.
//
// Where the analysis is still approximate it errs toward saying no: the flood
// fill ignores the opposing king, and defense by a king is not counted. Both
// can only turn a genuine draw into an unclaimed one, never the reverse.
// Wrongly claiming a draw in a live position would be a far worse defect.
func (p *Position) IsBlockedDeadPosition() bool {
	// Condition 1: kings and pawns only.
	if p.byType[Knight]|p.byType[Bishop]|p.byType[Rook]|p.byType[Queen] != 0 {
		return false
	}
	pawns := p.byType[Pawn]
	if pawns == 0 {
		return false // bare kings are handled by the material rule
	}

	// A pending en passant capture means a pawn just double-pushed, so the
	// structure demonstrably was not frozen a ply ago.
	if p.epSquare != NoSquare {
		return false
	}

	// Condition 2: every pawn permanently immobile.
	//
	// The blocker must be another pawn, not merely any piece. A pawn standing
	// behind a king is not blocked at all — the king moves and it advances.
	for c := White; c <= Black; c++ {
		push := PawnPush(c)
		theirPawns := p.ColorPiecesOfType(c.Opposite(), Pawn)
		for bb := p.ColorPiecesOfType(c, Pawn); bb != 0; {
			var s Square
			s, bb = bb.PopFirst()

			if !pawns.Has(Square(int(s) + push)) {
				return false // can still advance
			}
			if PawnAttacks[c][s]&theirPawns != 0 {
				return false // can still capture
			}
		}
	}

	// Condition 3: neither king can ever capture a pawn.
	for c := White; c <= Black; c++ {
		theirPawns := p.ColorPiecesOfType(c.Opposite(), Pawn)
		theirPawnAttacks := pawnAttackSpan(theirPawns, c.Opposite())

		ksq := p.KingSquare(c)
		if theirPawnAttacks.Has(ksq) {
			// In check from a pawn. With the structure frozen this cannot
			// arise from play, so the position is not the steady state this
			// argument describes.
			return false
		}

		// Squares attacked by enemy pawns are walls as solid as the pawns
		// themselves: a king may never stand on one, and since these pawns
		// never move, the exclusion is permanent rather than an estimate.
		reach := kingReachableSquares(ksq, pawns|theirPawnAttacks)

		// A pawn defended by another pawn can never be taken by a king —
		// capturing it would be moving into check. Only undefended pawns are
		// genuinely at risk, and only those adjacent to somewhere the king can
		// actually stand.
		undefended := theirPawns &^ theirPawnAttacks
		for bb := undefended; bb != 0; {
			var s Square
			s, bb = bb.PopFirst()
			if reach&KingAttacks[s] != 0 {
				return false
			}
		}
	}

	return true
}

// pawnAttackSpan returns every square attacked by the given pawns of one color.
func pawnAttackSpan(pawns Bitboard, c Color) Bitboard {
	if c == White {
		return pawns.Shifted(North+East) | pawns.Shifted(North+West)
	}
	return pawns.Shifted(South+East) | pawns.Shifted(South+West)
}

// kingReachableSquares floods outward from the king over every square not
// occupied by a pawn, returning everywhere it could eventually stand.
//
// Pawns are the only walls: with the structure frozen they never move, so the
// board's topology is fixed. The opposing king is not treated as a wall, since
// it moves and cannot permanently block anything.
func kingReachableSquares(from Square, walls Bitboard) Bitboard {
	reach := from.BB()
	for {
		grown := reach
		for bb := reach; bb != 0; {
			var s Square
			s, bb = bb.PopFirst()
			grown |= KingAttacks[s]
		}
		grown &^= walls
		if grown == reach {
			return reach
		}
		reach = grown
	}
}

// IsDeadPosition reports whether checkmate is impossible for either side by any
// sequence of legal moves, covering both the material and the locked-structure
// cases.
func (p *Position) IsDeadPosition() bool {
	return p.IsInsufficientMaterial() || p.IsBlockedDeadPosition()
}

// IsDraw reports whether the position is drawn by any of the drawing rules. It
// does not consider stalemate, which requires generating moves; use Outcome for
// a complete answer.
func (p *Position) IsDraw() bool {
	return p.IsFiftyMoveRule() || p.IsThreefoldRepetition() || p.IsDeadPosition()
}
