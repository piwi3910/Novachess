package chess

import "math/bits"

// Castling, generalized over both classical chess and Chess960.
//
// Chess960 breaks three assumptions that classical castling code can get away
// with. The king does not start on e1; the rooks do not start on a1 and h1; and
// the squares the king and rook pass through can overlap, up to the king not
// moving at all. What does *not* change is where they finish: in Chess960 the
// king and rook end on exactly the squares they would in classical chess —
// g1/f1 for kingside, c1/d1 for queenside.
//
// Because the destinations are fixed but the origins are not, a castling move
// is encoded internally as "king takes own rook": origin is the king's square,
// destination is the rook's. That is the only encoding that stays unambiguous
// when the king's destination is already occupied by the castling rook, or when
// the king does not move at all. UCI output converts back to the king's
// destination in classical mode, which is what GUIs expect there.

// Castling right indices, one per right, used to index the per-position rook
// squares and paths.
const (
	idxWhiteKingSide = iota
	idxWhiteQueenSide
	idxBlackKingSide
	idxBlackQueenSide
	castlingIndexCount
)

// rightAt returns the CastlingRights bit for an index.
func rightAt(idx int) CastlingRights { return CastlingRights(1 << idx) }

// indexOfRight returns the index of a single castling right bit.
func indexOfRight(cr CastlingRights) int { return bits.TrailingZeros8(uint8(cr)) }

// castlingIndex returns the index for a color and side.
func castlingIndex(c Color, kingSide bool) int {
	idx := idxWhiteKingSide
	if c == Black {
		idx = idxBlackKingSide
	}
	if !kingSide {
		idx++
	}
	return idx
}

// colorRights returns both castling rights belonging to a color.
func colorRights(c Color) CastlingRights {
	if c == White {
		return WhiteKingSide | WhiteQueenSide
	}
	return BlackKingSide | BlackQueenSide
}

// castlingKingTo returns the square the king finishes on. These are fixed in
// Chess960 exactly as in classical chess.
func castlingKingTo(idx int) Square {
	switch idx {
	case idxWhiteKingSide:
		return G1
	case idxWhiteQueenSide:
		return C1
	case idxBlackKingSide:
		return G8
	default:
		return C8
	}
}

// castlingRookTo returns the square the rook finishes on.
func castlingRookTo(idx int) Square {
	switch idx {
	case idxWhiteKingSide:
		return F1
	case idxWhiteQueenSide:
		return D1
	case idxBlackKingSide:
		return F8
	default:
		return D8
	}
}

// isKingSideCastling reports whether a castling move is to the king's side.
// Since the move encodes the rook's square as its destination, the rook being
// to the right of the king identifies the kingside castle — which holds in
// Chess960 too, where "kingside" means the rook starting right of the king.
func isKingSideCastling(m Move) bool { return m.To() > m.From() }

// CastlingRookSquare returns the starting square of the rook belonging to a
// castling right. It is meaningful only when the position holds that right.
func (p *Position) CastlingRookSquare(cr CastlingRights) Square {
	return p.castlingRook[indexOfRight(cr)]
}

// IsChess960 reports whether the position is being interpreted as Chess960.
// The only observable differences are how castling moves are written in UCI and
// how the castling field is rendered in FEN; move generation is identical, and
// correct for both.
func (p *Position) IsChess960() bool { return p.chess960 }

// setCastlingRight records a castling right together with the rook it belongs
// to, precomputing the squares that must be empty and the rights each square
// revokes when touched.
//
// The king's square must already be set on the board when this is called.
func (p *Position) setCastlingRight(idx int, rookFrom Square) {
	us := castlingColorOf(idx)
	kingFrom := p.KingSquare(us)
	kingTo := castlingKingTo(idx)
	rookTo := castlingRookTo(idx)

	p.castling |= rightAt(idx)
	p.castlingRook[idx] = rookFrom

	// Every square the king or rook must travel through or land on, minus the
	// two they already occupy. In Chess960 these ranges overlap freely, and
	// either piece may already be standing on its own destination, so the two
	// origins are removed after the union rather than special-cased.
	p.castlingPath[idx] = (BetweenBB[kingFrom][kingTo] | kingTo.BB() |
		BetweenBB[rookFrom][rookTo] | rookTo.BB()) &^
		(kingFrom.BB() | rookFrom.BB())

	// Moving the king forfeits both of its rights; moving or losing this rook
	// forfeits just this one.
	p.crMask[kingFrom] &^= colorRights(us)
	p.crMask[rookFrom] &^= rightAt(idx)
}

// castlingColorOf returns the color owning a castling index.
func castlingColorOf(idx int) Color {
	if idx < idxBlackKingSide {
		return White
	}
	return Black
}

// findCastlingRook locates the rook a classical KQkq castling flag refers to.
//
// In X-FEN, a KQkq flag denotes the outermost rook on that side of the king, so
// the search starts at the edge and works inward. For an orthodox position this
// simply finds a1 or h1; for a Chess960 position written with classical flags
// it resolves correctly anyway.
func (p *Position) findCastlingRook(us Color, kingSide bool) (Square, bool) {
	ksq := p.KingSquare(us)
	rank := 0
	if us == Black {
		rank = 7
	}
	if ksq.Rank() != rank {
		// A king off its back rank cannot have castling rights.
		return NoSquare, false
	}

	rook := MakePiece(us, Rook)
	if kingSide {
		for file := 7; file > ksq.File(); file-- {
			if sq := MakeSquare(file, rank); p.board[sq] == rook {
				return sq, true
			}
		}
	} else {
		for file := 0; file < ksq.File(); file++ {
			if sq := MakeSquare(file, rank); p.board[sq] == rook {
				return sq, true
			}
		}
	}
	return NoSquare, false
}

// castlingFEN renders the castling field of a FEN.
//
// Classical positions use the familiar KQkq. Chess960 positions use Shredder
// notation, naming each rook by its file — uppercase for White — because KQkq
// cannot describe a position whose rooks both start on the same side of the
// king.
func (p *Position) castlingFEN() string {
	if p.castling == NoCastling {
		return "-"
	}
	if !p.chess960 {
		return p.castling.String()
	}

	var b []byte
	for idx := 0; idx < castlingIndexCount; idx++ {
		if !p.castling.Has(rightAt(idx)) {
			continue
		}
		file := byte('A' + p.castlingRook[idx].File())
		if castlingColorOf(idx) == Black {
			file += 'a' - 'A'
		}
		b = append(b, file)
	}
	return string(b)
}

// MoveToUCI renders a move in the notation a GUI expects for this position's
// variant.
//
// Castling is the only move whose UCI text depends on the variant. Classical
// UCI writes the king's origin and destination ("e1g1"); Chess960 UCI writes
// king-takes-rook ("e1h1"), which is also how moves are stored internally,
// because the classical form becomes ambiguous once a rook can already occupy
// the king's destination.
//
// Call this before the move is played: it reads the position to determine which
// side is castling.
func (p *Position) MoveToUCI(m Move) string {
	if m.Kind() != KindCastling || p.chess960 {
		return m.String()
	}
	us := p.board[m.From()].Color()
	idx := castlingIndex(us, isKingSideCastling(m))
	return m.From().String() + castlingKingTo(idx).String()
}

// ParseUCIMove converts a move written in UCI notation into a Move, resolving
// it against the legal moves of this position.
//
// Resolution against actual legal moves is what makes castling unambiguous:
// "e1g1" may be a castle or an ordinary king move depending on the position,
// and in Chess960 "e1h1" may be a castle or a king capturing on h1. Rather than
// guess from the text, the matching legal move is found.
func (p *Position) ParseUCIMove(s string) (Move, bool) {
	var ml MoveList
	p.GenerateLegalMoves(&ml)

	// Exact match on the notation this position would produce.
	for _, m := range ml.Slice() {
		if p.MoveToUCI(m) == s {
			return m, true
		}
	}
	// Also accept the internal king-takes-rook form for castling, so that a
	// GUI sending Chess960 notation while the engine is in classical mode is
	// understood rather than rejected.
	for _, m := range ml.Slice() {
		if m.Kind() == KindCastling && m.String() == s {
			return m, true
		}
	}
	return MoveNone, false
}
