package chess

// Move is a packed move, encoded in 16 bits:
//
//	bits  0-5   origin square
//	bits  6-11  destination square
//	bits 12-13  promotion piece, as PieceType-Knight (0=N, 1=B, 2=R, 3=Q)
//	bits 14-15  move kind (normal, promotion, en passant, castling)
//
// Keeping moves at 16 bits matters: the transposition table stores one per
// entry, and move lists are copied constantly during search. Note that the
// promotion field is only meaningful when the kind is KindPromotion.
//
// For castling, the destination is the king's final square (g1/c1/g8/c8),
// not the rook's — the standard convention for non-Chess960 play.
type Move uint16

// MoveNone is the zero Move, used as "no move". It is never a legal move
// because it would encode a1-a1.
const MoveNone Move = 0

// MoveKind distinguishes moves that need special handling in make/unmake.
type MoveKind uint16

const (
	KindNormal MoveKind = iota << 14
	KindPromotion
	KindEnPassant
	KindCastling
)

// NewMove builds an ordinary move, including captures — a capture needs no flag
// because the destination square tells the position what was taken.
func NewMove(from, to Square) Move {
	return Move(uint16(from) | uint16(to)<<6)
}

// NewPromotion builds a pawn promotion to the given piece type.
func NewPromotion(from, to Square, promo PieceType) Move {
	return Move(uint16(from) | uint16(to)<<6 | uint16(promo-Knight)<<12 | uint16(KindPromotion))
}

// NewEnPassant builds an en passant capture. The destination is the square the
// capturing pawn lands on, not the square of the captured pawn.
func NewEnPassant(from, to Square) Move {
	return Move(uint16(from) | uint16(to)<<6 | uint16(KindEnPassant))
}

// NewCastling builds a castling move, with to set to the king's final square.
func NewCastling(from, to Square) Move {
	return Move(uint16(from) | uint16(to)<<6 | uint16(KindCastling))
}

// From returns the origin square.
func (m Move) From() Square { return Square(m & 0x3F) }

// To returns the destination square.
func (m Move) To() Square { return Square((m >> 6) & 0x3F) }

// Kind returns the move kind.
func (m Move) Kind() MoveKind { return MoveKind(m & 0xC000) }

// Promotion returns the piece a pawn promotes to. It is only meaningful when
// Kind is KindPromotion.
func (m Move) Promotion() PieceType { return PieceType((m>>12)&3) + Knight }

// IsNone reports whether this is the null "no move" value.
func (m Move) IsNone() bool { return m == MoveNone }

// String renders the move in the long algebraic notation UCI uses, for example
// "e2e4" or "e7e8q". Castling is rendered as the king's actual origin and
// destination ("e1g1"), which is what UCI expects outside Chess960.
func (m Move) String() string {
	if m.IsNone() {
		return "0000"
	}
	s := m.From().String() + m.To().String()
	if m.Kind() == KindPromotion {
		s += string(pieceChars[MakePiece(Black, m.Promotion())])
	}
	return s
}

// MaxMoves bounds the number of legal moves in any reachable position. The true
// maximum is 218; the headroom keeps the bound obviously safe.
const MaxMoves = 256

// MoveList is a fixed-capacity move buffer. It is a value type with an inline
// array so that generating moves during search allocates nothing — move
// generation happens millions of times per second and must not touch the heap.
type MoveList struct {
	Moves [MaxMoves]Move
	Count int
}

// Add appends a move to the list.
func (ml *MoveList) Add(m Move) {
	ml.Moves[ml.Count] = m
	ml.Count++
}

// Slice returns the populated prefix of the list.
func (ml *MoveList) Slice() []Move { return ml.Moves[:ml.Count] }
