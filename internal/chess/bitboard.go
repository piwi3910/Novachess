package chess

import "math/bits"

// Bitboard is a set of squares, one bit per square, indexed so that bit 0 is A1
// and bit 63 is H8.
//
// The operations here compile to single instructions on the architectures we
// care about: OnesCount64 becomes POPCNT on amd64 and CNT+ADDV on arm64, and
// TrailingZeros64 becomes TZCNT/RBIT+CLZ. That is the whole reason this engine
// is bitboard-based rather than mailbox-based.
type Bitboard uint64

const (
	EmptyBB Bitboard = 0
	FullBB  Bitboard = ^Bitboard(0)
)

// File masks, indexed by zero-based file.
var FileBB = [8]Bitboard{
	0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
}

// Rank masks, indexed by zero-based rank.
var RankBB = [8]Bitboard{
	0x00000000000000FF, 0x000000000000FF00, 0x0000000000FF0000, 0x00000000FF000000,
	0x000000FF00000000, 0x0000FF0000000000, 0x00FF000000000000, 0xFF00000000000000,
}

// SquareBB[s] is the bitboard containing only square s.
var SquareBB [SquareCount]Bitboard

func init() {
	for s := Square(0); s < SquareCount; s++ {
		SquareBB[s] = 1 << s
	}
}

// BB returns the single-square bitboard for s.
func (s Square) BB() Bitboard { return 1 << s }

// Has reports whether the square is a member of the set.
func (b Bitboard) Has(s Square) bool { return b&(1<<s) != 0 }

// Set returns the set with s added.
func (b Bitboard) Set(s Square) Bitboard { return b | (1 << s) }

// Clear returns the set with s removed.
func (b Bitboard) Clear(s Square) Bitboard { return b &^ (1 << s) }

// Count returns the number of squares in the set.
func (b Bitboard) Count() int { return bits.OnesCount64(uint64(b)) }

// IsEmpty reports whether the set has no members.
func (b Bitboard) IsEmpty() bool { return b == 0 }

// MoreThanOne reports whether the set has at least two members. This is the
// double-check test in the move generator, and clearing the low bit is cheaper
// than a full popcount for it.
func (b Bitboard) MoreThanOne() bool { return b&(b-1) != 0 }

// First returns the lowest-numbered square in the set. The result is undefined
// for an empty set; callers must check IsEmpty first.
func (b Bitboard) First() Square { return Square(bits.TrailingZeros64(uint64(b))) }

// PopFirst removes and returns the lowest-numbered square in the set. This is
// the standard way to iterate a bitboard:
//
//	for bb != 0 {
//	    var sq Square
//	    sq, bb = bb.PopFirst()
//	    ...
//	}
func (b Bitboard) PopFirst() (Square, Bitboard) {
	return Square(bits.TrailingZeros64(uint64(b))), b & (b - 1)
}

// Shift directions, as signed square deltas. North is toward rank 8.
const (
	North = 8
	South = -8
	East  = 1
	West  = -1
)

// Shifted moves every square in the set by delta, discarding squares that fall
// off the board. Files are masked before shifting so that an east shift from
// the h-file does not wrap onto the a-file of the next rank.
func (b Bitboard) Shifted(delta int) Bitboard {
	switch delta {
	case North:
		return b << 8
	case South:
		return b >> 8
	case East:
		return (b &^ FileBB[7]) << 1
	case West:
		return (b &^ FileBB[0]) >> 1
	case North + East:
		return (b &^ FileBB[7]) << 9
	case North + West:
		return (b &^ FileBB[0]) << 7
	case South + East:
		return (b &^ FileBB[7]) >> 7
	case South + West:
		return (b &^ FileBB[0]) >> 9
	default:
		panic("chess: unsupported shift delta")
	}
}

// PawnPush returns the one-square push direction for the given color.
func PawnPush(c Color) int {
	if c == White {
		return North
	}
	return South
}

// String renders the bitboard as an 8x8 grid with rank 8 on top, for debugging.
func (b Bitboard) String() string {
	buf := make([]byte, 0, 8*17)
	for rank := 7; rank >= 0; rank-- {
		for file := 0; file < 8; file++ {
			if b.Has(MakeSquare(file, rank)) {
				buf = append(buf, 'X')
			} else {
				buf = append(buf, '.')
			}
			buf = append(buf, ' ')
		}
		buf = append(buf, '\n')
	}
	return string(buf)
}
