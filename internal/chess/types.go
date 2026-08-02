// Package chess implements board representation, move generation and the rules
// of chess. It is the foundation every other package builds on, so correctness
// here is gated by an exhaustive perft suite rather than by unit tests alone.
package chess

// Color is the side to move. The zero value is White, which lets a zeroed
// Position start out as white-to-move.
type Color uint8

const (
	White Color = iota
	Black
	ColorCount = 2
)

// Opposite returns the other color.
func (c Color) Opposite() Color { return c ^ 1 }

func (c Color) String() string {
	if c == White {
		return "white"
	}
	return "black"
}

// PieceType identifies a piece independent of its color.
type PieceType uint8

const (
	Pawn PieceType = iota
	Knight
	Bishop
	Rook
	Queen
	King
	PieceTypeCount = 6
)

// Piece is a colored piece, packed as color*6+type so that the color can be
// recovered by division and the type by remainder. NoPiece sits past the end of
// the valid range and marks an empty square.
type Piece uint8

const (
	WhitePawn Piece = iota
	WhiteKnight
	WhiteBishop
	WhiteRook
	WhiteQueen
	WhiteKing
	BlackPawn
	BlackKnight
	BlackBishop
	BlackRook
	BlackQueen
	BlackKing
	NoPiece
	PieceCount = 12
)

// MakePiece combines a color and a piece type into a Piece.
func MakePiece(c Color, pt PieceType) Piece { return Piece(uint8(c)*PieceTypeCount + uint8(pt)) }

// Color returns the color of the piece. It is only meaningful when p != NoPiece.
func (p Piece) Color() Color { return Color(p / PieceTypeCount) }

// Type returns the type of the piece. It is only meaningful when p != NoPiece.
func (p Piece) Type() PieceType { return PieceType(p % PieceTypeCount) }

// pieceChars maps each piece to its FEN character, indexed by Piece.
const pieceChars = "PNBRQKpnbrqk"

func (p Piece) String() string {
	if p >= NoPiece {
		return "."
	}
	return string(pieceChars[p])
}

// Square is a board square in little-endian rank-file order: A1=0, B1=1, ...,
// H8=63. This mapping makes north a shift of +8 and east a shift of +1, which
// keeps the direction arithmetic in the move generator readable.
type Square uint8

const (
	A1 Square = iota
	B1
	C1
	D1
	E1
	F1
	G1
	H1
	A2
	B2
	C2
	D2
	E2
	F2
	G2
	H2
	A3
	B3
	C3
	D3
	E3
	F3
	G3
	H3
	A4
	B4
	C4
	D4
	E4
	F4
	G4
	H4
	A5
	B5
	C5
	D5
	E5
	F5
	G5
	H5
	A6
	B6
	C6
	D6
	E6
	F6
	G6
	H6
	A7
	B7
	C7
	D7
	E7
	F7
	G7
	H7
	A8
	B8
	C8
	D8
	E8
	F8
	G8
	H8

	SquareCount = 64
	// NoSquare marks the absence of a square, used for the en passant target
	// when no double pawn push was just played.
	NoSquare Square = 64
)

// MakeSquare builds a square from a zero-based file and rank.
func MakeSquare(file, rank int) Square { return Square(rank*8 + file) }

// File returns the zero-based file (0=a) of the square.
func (s Square) File() int { return int(s) & 7 }

// Rank returns the zero-based rank (0=rank 1) of the square.
func (s Square) Rank() int { return int(s) >> 3 }

func (s Square) String() string {
	if s >= NoSquare {
		return "-"
	}
	return string([]byte{byte('a' + s.File()), byte('1' + s.Rank())})
}

// parseSquare parses algebraic coordinates such as "e4". It reports false for
// anything that is not exactly two characters in range.
func parseSquare(s string) (Square, bool) {
	if len(s) != 2 || s[0] < 'a' || s[0] > 'h' || s[1] < '1' || s[1] > '8' {
		return NoSquare, false
	}
	return MakeSquare(int(s[0]-'a'), int(s[1]-'1')), true
}

// CastlingRights is a bitmask of the four castling privileges. Rights are
// revoked permanently once a king or rook moves, so this only ever loses bits
// as a game progresses.
type CastlingRights uint8

const (
	WhiteKingSide CastlingRights = 1 << iota
	WhiteQueenSide
	BlackKingSide
	BlackQueenSide

	NoCastling  CastlingRights = 0
	AllCastling                = WhiteKingSide | WhiteQueenSide | BlackKingSide | BlackQueenSide
)

// Has reports whether all of the given rights are present.
func (cr CastlingRights) Has(r CastlingRights) bool { return cr&r != 0 }

func (cr CastlingRights) String() string {
	if cr == NoCastling {
		return "-"
	}
	var b []byte
	for i, ch := range []byte("KQkq") {
		if cr&(1<<uint(i)) != 0 {
			b = append(b, ch)
		}
	}
	return string(b)
}
