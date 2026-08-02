package chess

import "math/bits"

// Attack tables.
//
// Leaper attacks (knight, king, pawn) are plain lookup tables. Slider attacks
// (bishop, rook, queen) use magic bitboards: for each square we mask the
// occupancy down to the squares that can actually block, multiply by a magic
// constant that maps every distinct blocker pattern to a distinct index, and
// shift down into a per-square table.
//
// The magics are searched for at startup with a fixed PRNG seed rather than
// hardcoded. It costs a few milliseconds once, and it keeps the source free of
// several hundred opaque 64-bit constants. The fixed seed means the search is
// deterministic — important because the whole distributed self-play pipeline
// depends on every node computing identical results.

var (
	// KnightAttacks[s] is the set of squares a knight on s attacks.
	KnightAttacks [SquareCount]Bitboard
	// KingAttacks[s] is the set of squares a king on s attacks.
	KingAttacks [SquareCount]Bitboard
	// PawnAttacks[c][s] is the set of squares a pawn of color c on s attacks.
	PawnAttacks [ColorCount][SquareCount]Bitboard

	// BetweenBB[a][b] holds the squares strictly between a and b when they
	// share a rank, file or diagonal, and is empty otherwise. Used to find
	// blocking squares when in check.
	BetweenBB [SquareCount][SquareCount]Bitboard
	// LineBB[a][b] holds every square on the infinite line through a and b
	// (including both endpoints) when they are aligned, and is empty
	// otherwise. Used to constrain pinned pieces to their pin ray.
	LineBB [SquareCount][SquareCount]Bitboard
)

var (
	bishopDirs = [4]int{North + East, North + West, South + East, South + West}
	rookDirs   = [4]int{North, South, East, West}
)

// magic holds the per-square parameters for one slider on one square.
type magic struct {
	mask    Bitboard   // relevant occupancy squares
	magic   uint64     // multiplier mapping blockers to a table index
	attacks []Bitboard // indexed by (occ & mask) * magic >> shift
	shift   uint
}

func (m *magic) index(occ Bitboard) uint64 {
	return (uint64(occ&m.mask) * m.magic) >> m.shift
}

var (
	rookMagics   [SquareCount]magic
	bishopMagics [SquareCount]magic
)

func init() {
	initLeaperAttacks()
	initMagics(&rookMagics, rookDirs)
	initMagics(&bishopMagics, bishopDirs)
	initLineTables()
}

// safeDestination returns the bitboard for the square delta steps away from s,
// or the empty set if that would wrap around the board or fall off it. The
// distance check is what stops a knight on h1 from appearing on a3.
func safeDestination(s Square, delta int) Bitboard {
	to := int(s) + delta
	if to < 0 || to >= SquareCount {
		return EmptyBB
	}
	// Reject any move that changes the file by more than two, which can only
	// happen by wrapping around the edge.
	if abs(Square(to).File()-s.File()) > 2 {
		return EmptyBB
	}
	return Square(to).BB()
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func initLeaperAttacks() {
	knightDeltas := [8]int{-17, -15, -10, -6, 6, 10, 15, 17}
	kingDeltas := [8]int{-9, -8, -7, -1, 1, 7, 8, 9}

	for s := Square(0); s < SquareCount; s++ {
		for _, d := range knightDeltas {
			KnightAttacks[s] |= safeDestination(s, d)
		}
		for _, d := range kingDeltas {
			KingAttacks[s] |= safeDestination(s, d)
		}
		PawnAttacks[White][s] = safeDestination(s, North+East) | safeDestination(s, North+West)
		PawnAttacks[Black][s] = safeDestination(s, South+East) | safeDestination(s, South+West)
	}
}

// slidingAttacks walks outward from s in each direction until it runs off the
// board or hits an occupied square, which is included as a capture. This is the
// slow, obviously-correct reference used to build the magic tables — and, in
// tests, to verify them.
func slidingAttacks(s Square, occ Bitboard, dirs [4]int) Bitboard {
	var attacks Bitboard
	for _, d := range dirs {
		for cur := s; ; {
			next := safeDestination(cur, d)
			if next == 0 {
				break
			}
			attacks |= next
			cur = next.First()
			if occ.Has(cur) {
				break
			}
		}
	}
	return attacks
}

// initMagics finds and stores a magic multiplier for every square.
func initMagics(table *[SquareCount]magic, dirs [4]int) {
	// A fixed seed keeps startup deterministic across machines.
	rng := newXorshift(0x246C_CB2D_3B40_2853)

	// Reused across squares to avoid reallocating on every failed candidate.
	var occupancies, references [4096]Bitboard
	var epoch [4096]int
	currentEpoch := 0

	for s := Square(0); s < SquareCount; s++ {
		m := &table[s]

		// The relevant occupancy mask excludes edge squares: a blocker on the
		// far edge cannot hide anything behind it, so it does not affect the
		// resulting attack set and need not be part of the index.
		edges := ((RankBB[0] | RankBB[7]) &^ RankBB[s.Rank()]) |
			((FileBB[0] | FileBB[7]) &^ FileBB[s.File()])
		m.mask = slidingAttacks(s, EmptyBB, dirs) &^ edges
		m.shift = uint(64 - m.mask.Count())

		// Enumerate every subset of the mask using the carry-rippler trick,
		// recording the attack set each one produces. The sequence returns to
		// the empty set exactly once it has visited all 2^n subsets.
		size := 0
		occ := EmptyBB
		for {
			occupancies[size] = occ
			references[size] = slidingAttacks(s, occ, dirs)
			size++
			occ = (occ - m.mask) & m.mask
			if occ == 0 {
				break
			}
		}
		m.attacks = make([]Bitboard, size)

		// Try random sparse multipliers until one indexes every blocker
		// pattern without a destructive collision. Collisions are tolerated
		// when both patterns yield the same attack set — that is what makes
		// the tables far smaller than the theoretical worst case.
		for {
			// ANDing three random values biases toward few set bits, which
			// finds working magics far faster than uniform randomness.
			m.magic = rng.next() & rng.next() & rng.next()

			// Cheap rejection: a usable magic must scatter the mask's high
			// bits into the index range. Candidates failing this almost never
			// work, and rejecting them here avoids the full collision scan.
			if bits.OnesCount64((uint64(m.mask)*m.magic)>>56) < 6 {
				continue
			}

			currentEpoch++
			ok := true
			for i := 0; i < size; i++ {
				idx := m.index(occupancies[i])
				if epoch[idx] != currentEpoch {
					epoch[idx] = currentEpoch
					m.attacks[idx] = references[i]
					continue
				}
				if m.attacks[idx] != references[i] {
					ok = false
					break
				}
			}
			if ok {
				break
			}
		}
	}
}

// initLineTables fills BetweenBB and LineBB from the slider attack tables.
func initLineTables() {
	for a := Square(0); a < SquareCount; a++ {
		for _, dirs := range [2][4]int{bishopDirs, rookDirs} {
			// Squares this slider reaches from a on an empty board. Any b in
			// here is aligned with a, so exactly one of the two direction sets
			// writes each pair and they never conflict.
			fromA := slidingAttacks(a, EmptyBB, dirs)
			for bb := fromA; bb != 0; {
				var b Square
				b, bb = bb.PopFirst()

				// Both endpoints plus everything on the infinite line.
				LineBB[a][b] = (fromA & slidingAttacks(b, EmptyBB, dirs)) | a.BB() | b.BB()

				// Squares strictly between: what a sees when blocked at b,
				// intersected with what b sees when blocked at a.
				BetweenBB[a][b] = slidingAttacks(a, b.BB(), dirs) &
					slidingAttacks(b, a.BB(), dirs)
			}
		}
	}
}

// RookAttacks returns the squares a rook on s attacks given the occupancy occ.
// Occupied squares are included, since they are capturable; the caller filters
// out its own pieces.
func RookAttacks(s Square, occ Bitboard) Bitboard {
	m := &rookMagics[s]
	return m.attacks[m.index(occ)]
}

// BishopAttacks returns the squares a bishop on s attacks given occupancy occ.
func BishopAttacks(s Square, occ Bitboard) Bitboard {
	m := &bishopMagics[s]
	return m.attacks[m.index(occ)]
}

// QueenAttacks returns the squares a queen on s attacks given occupancy occ.
func QueenAttacks(s Square, occ Bitboard) Bitboard {
	return RookAttacks(s, occ) | BishopAttacks(s, occ)
}

// AttacksFrom returns the squares attacked by a piece of type pt and color c
// standing on s under occupancy occ.
func AttacksFrom(pt PieceType, c Color, s Square, occ Bitboard) Bitboard {
	switch pt {
	case Pawn:
		return PawnAttacks[c][s]
	case Knight:
		return KnightAttacks[s]
	case Bishop:
		return BishopAttacks(s, occ)
	case Rook:
		return RookAttacks(s, occ)
	case Queen:
		return QueenAttacks(s, occ)
	case King:
		return KingAttacks[s]
	}
	return EmptyBB
}

// Aligned reports whether the three squares lie on a common rank, file or
// diagonal. Used to test whether a pinned piece stays on its pin ray.
func Aligned(a, b, c Square) bool { return LineBB[a][b].Has(c) }

// xorshift is a small deterministic PRNG used only for the magic search. It is
// deliberately not crypto-grade and deliberately reproducible.
type xorshift struct{ state uint64 }

func newXorshift(seed uint64) *xorshift { return &xorshift{state: seed} }

func (r *xorshift) next() uint64 {
	r.state ^= r.state >> 12
	r.state ^= r.state << 25
	r.state ^= r.state >> 27
	return r.state * 2685821657736338717
}
