package chess

// Legal move generation.
//
// Moves are generated legal-only rather than pseudo-legal-then-filtered. The
// classic alternative — generate everything, make each move, throw away the
// ones leaving the king in check — is simpler but pays a make/unmake for every
// candidate. Instead we compute two masks up front:
//
//   - check evasion: when in check, non-king pieces may only capture the
//     checker or interpose on the line between it and the king
//   - pins: a piece shielding its king from a slider may only move along that
//     line, and never off it
//
// Between them, every generated move is legal by construction. The one case
// these masks cannot express is en passant, which removes two pieces from one
// rank at once and so can expose a king in a way no pin mask predicts. That
// move is verified explicitly.

// GenerateLegalMoves fills ml with every legal move in the position. Any
// existing contents are discarded.
func (p *Position) GenerateLegalMoves(ml *MoveList) {
	ml.Count = 0

	us := p.side
	them := us.Opposite()
	ksq := p.KingSquare(us)
	occ := p.Occupied()
	ourPieces := p.byColor[us]
	checkers := p.AttackersTo(ksq, occ) & p.byColor[them]

	// King moves are legal regardless of check state, so they are always
	// generated.
	p.genKingMoves(ml, ksq, them, occ, ourPieces)

	// Under double check no interposition or capture can resolve both threats
	// at once, so only the king may move.
	if checkers.MoreThanOne() {
		return
	}

	var targets Bitboard
	if checkers != 0 {
		// Single check: capture the checker or block the line to it. A knight
		// or pawn checker has an empty BetweenBB, leaving capture as the only
		// non-king option, which is correct.
		checkerSq := checkers.First()
		targets = BetweenBB[ksq][checkerSq] | checkerSq.BB()
	} else {
		targets = ^ourPieces
		p.genCastling(ml, us, occ)
	}

	pinned := p.blockersForKing(us) & ourPieces

	p.genPawnMoves(ml, us, targets, pinned, ksq, occ)
	p.genPieceMoves(ml, us, targets, pinned, ksq, occ)
}

// LegalMoves returns the legal moves as a freshly allocated slice. This
// allocates and is meant for tests and tooling; the search uses
// GenerateLegalMoves with a stack-allocated MoveList.
func (p *Position) LegalMoves() []Move {
	var ml MoveList
	p.GenerateLegalMoves(&ml)
	out := make([]Move, ml.Count)
	copy(out, ml.Slice())
	return out
}

// blockersForKing returns the pieces of either color that stand between the
// king of color c and an enemy slider, such that removing them would expose the
// king. Intersected with our own pieces this gives the pinned set; the enemy
// pieces in it are candidates for discovered attacks, which the search will
// want later.
func (p *Position) blockersForKing(c Color) Bitboard {
	ksq := p.KingSquare(c)
	them := c.Opposite()
	occ := p.Occupied()

	// Sliders that would attack the king if the board were empty. Anything
	// else can never pin, no matter what moves.
	snipers := ((RookAttacks(ksq, EmptyBB) & (p.byType[Rook] | p.byType[Queen])) |
		(BishopAttacks(ksq, EmptyBB) & (p.byType[Bishop] | p.byType[Queen]))) &
		p.byColor[them]

	// Exclude the snipers themselves so that two aligned enemy sliders do not
	// count each other as the blocker.
	occupancy := occ &^ snipers

	var blockers Bitboard
	for snipers != 0 {
		var sniperSq Square
		sniperSq, snipers = snipers.PopFirst()

		// Exactly one piece in between means that piece is pinned. Zero means
		// the king is already in check by this slider; two or more means no
		// single move can expose it.
		between := BetweenBB[ksq][sniperSq] & occupancy
		if between != 0 && !between.MoreThanOne() {
			blockers |= between
		}
	}
	return blockers
}

// genKingMoves generates king moves, excluding any square attacked by the enemy.
func (p *Position) genKingMoves(ml *MoveList, ksq Square, them Color, occ, ourPieces Bitboard) {
	// The king must be removed from the occupancy before testing destinations:
	// otherwise it blocks the very slider that is checking it, and stepping
	// straight backwards along the check ray would look safe.
	occWithoutKing := occ &^ ksq.BB()

	for attacks := KingAttacks[ksq] &^ ourPieces; attacks != 0; {
		var to Square
		to, attacks = attacks.PopFirst()
		if p.AttackersTo(to, occWithoutKing)&p.byColor[them] == 0 {
			ml.Add(NewMove(ksq, to))
		}
	}
}

// genCastling generates castling moves. It is only called when the side to move
// is not in check, so that condition is not retested here.
func (p *Position) genCastling(ml *MoveList, us Color, occ Bitboard) {
	them := us.Opposite()

	// kingSide and queenSide differ only by squares, so they are tabulated:
	// the right, the squares that must be empty, the squares the king crosses
	// (which must not be attacked), and the king's origin and destination.
	type castlingSpec struct {
		right     CastlingRights
		emptyPath Bitboard
		kingPath  [2]Square
		from, to  Square
		rookFrom  Square
	}
	var specs [2]castlingSpec
	if us == White {
		specs = [2]castlingSpec{
			{WhiteKingSide, F1.BB() | G1.BB(), [2]Square{F1, G1}, E1, G1, H1},
			{WhiteQueenSide, B1.BB() | C1.BB() | D1.BB(), [2]Square{D1, C1}, E1, C1, A1},
		}
	} else {
		specs = [2]castlingSpec{
			{BlackKingSide, F8.BB() | G8.BB(), [2]Square{F8, G8}, E8, G8, H8},
			{BlackQueenSide, B8.BB() | C8.BB() | D8.BB(), [2]Square{D8, C8}, E8, C8, A8},
		}
	}

	for _, s := range specs {
		if !p.castling.Has(s.right) || occ&s.emptyPath != 0 {
			continue
		}
		// Rights are sanitized when a position is parsed and revoked whenever
		// a king or rook square is touched, so this should never fire. It is
		// checked anyway because the alternative failure mode is relocating a
		// piece that does not exist, which panics rather than plays a bad move.
		if p.board[s.rookFrom] != MakePiece(us, Rook) {
			continue
		}
		// b1/b8 may be attacked during queenside castling — only the squares
		// the king actually crosses matter, which is why kingPath is separate
		// from emptyPath.
		if p.IsAttacked(s.kingPath[0], them) || p.IsAttacked(s.kingPath[1], them) {
			continue
		}
		ml.Add(NewCastling(s.from, s.to))
	}
}

// genPieceMoves generates knight, bishop, rook and queen moves.
func (p *Position) genPieceMoves(ml *MoveList, us Color, targets, pinned Bitboard, ksq Square, occ Bitboard) {
	for pt := Knight; pt <= Queen; pt++ {
		for bb := p.ColorPiecesOfType(us, pt); bb != 0; {
			var from Square
			from, bb = bb.PopFirst()

			attacks := AttacksFrom(pt, us, from, occ) & targets
			if pinned.Has(from) {
				// A pinned piece may only move along the pin ray. For a
				// knight this is always empty, so pinned knights correctly
				// generate nothing.
				attacks &= LineBB[ksq][from]
			}

			for attacks != 0 {
				var to Square
				to, attacks = attacks.PopFirst()
				ml.Add(NewMove(from, to))
			}
		}
	}
}

// genPawnMoves generates pawn pushes, captures, promotions and en passant.
func (p *Position) genPawnMoves(ml *MoveList, us Color, targets, pinned Bitboard, ksq Square, occ Bitboard) {
	push := PawnPush(us)
	theirPieces := p.byColor[us.Opposite()]

	promoRank, startRank := 7, 1
	if us == Black {
		promoRank, startRank = 0, 6
	}

	for bb := p.ColorPiecesOfType(us, Pawn); bb != 0; {
		var from Square
		from, bb = bb.PopFirst()

		pinMask := FullBB
		if pinned.Has(from) {
			pinMask = LineBB[ksq][from]
		}

		// Pushes. Pawns can never stand on the back ranks, so stepping one
		// square forward is always on the board.
		if to := Square(int(from) + push); !occ.Has(to) {
			if targets.Has(to) && pinMask.Has(to) {
				addPawnMove(ml, from, to, promoRank)
			}
			if from.Rank() == startRank {
				if to2 := Square(int(to) + push); !occ.Has(to2) &&
					targets.Has(to2) && pinMask.Has(to2) {
					ml.Add(NewMove(from, to2))
				}
			}
		}

		// Captures.
		for caps := PawnAttacks[us][from] & theirPieces & targets & pinMask; caps != 0; {
			var to Square
			to, caps = caps.PopFirst()
			addPawnMove(ml, from, to, promoRank)
		}

		// En passant. Neither the check-evasion mask nor the pin mask applies
		// cleanly here: the captured pawn sits on a different square from the
		// destination, so an en passant capture can resolve a check without
		// landing on a target square, and it vacates two squares on one rank
		// at once. Both cases are settled by simulating the resulting
		// occupancy outright.
		if p.epSquare != NoSquare && PawnAttacks[us][from].Has(p.epSquare) {
			if p.enPassantLegal(from, p.epSquare, us, ksq, occ) {
				ml.Add(NewEnPassant(from, p.epSquare))
			}
		}
	}
}

// addPawnMove appends a pawn move, expanding it into the four promotion choices
// when it lands on the back rank.
func addPawnMove(ml *MoveList, from, to Square, promoRank int) {
	if to.Rank() != promoRank {
		ml.Add(NewMove(from, to))
		return
	}
	// Queen first: move ordering in the search will want the best promotion
	// examined first, and underpromotions are almost never best.
	ml.Add(NewPromotion(from, to, Queen))
	ml.Add(NewPromotion(from, to, Rook))
	ml.Add(NewPromotion(from, to, Bishop))
	ml.Add(NewPromotion(from, to, Knight))
}

// enPassantLegal reports whether an en passant capture leaves the king safe. It
// builds the exact occupancy the move produces and asks whether anything
// attacks the king in it.
//
// This exists because en passant is the only move that removes two pieces from
// a rank simultaneously — the capturing pawn leaves and the captured pawn
// vanishes beside it. With a king and an enemy rook on that same rank, the move
// can expose the king even though neither pawn was pinned by any ordinary test.
func (p *Position) enPassantLegal(from, to Square, us Color, ksq Square, occ Bitboard) bool {
	them := us.Opposite()
	capSq := Square(int(to) - PawnPush(us))

	occAfter := (occ &^ from.BB() &^ capSq.BB()) | to.BB()

	theirs := p.byColor[them]
	if RookAttacks(ksq, occAfter)&(p.byType[Rook]|p.byType[Queen])&theirs != 0 {
		return false
	}
	if BishopAttacks(ksq, occAfter)&(p.byType[Bishop]|p.byType[Queen])&theirs != 0 {
		return false
	}
	// Knights are unaffected by the occupancy change, but an existing knight
	// check would persist through the capture and must still be rejected.
	if KnightAttacks[ksq]&p.byType[Knight]&theirs != 0 {
		return false
	}
	// The captured pawn is gone; any other enemy pawn checking the king still
	// counts.
	if PawnAttacks[us][ksq]&p.byType[Pawn]&theirs&^capSq.BB() != 0 {
		return false
	}
	return true
}
