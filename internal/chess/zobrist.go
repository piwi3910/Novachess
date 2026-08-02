package chess

// Zobrist hashing.
//
// Every position gets a 64-bit key built by XOR-ing together one random value
// per (piece, square) pair, plus values for the side to move, castling rights
// and the en passant file. Because XOR is its own inverse, the key can be
// updated incrementally as moves are made and unmade rather than recomputed —
// which is what makes the transposition table affordable.
//
// The keys are generated from a fixed seed. That is not cosmetic: the
// distributed self-play pipeline requires every node to produce byte-identical
// results for the same inputs, and a hash that varied per process would break
// reproducibility of anything TT-dependent.

var (
	zobristPieces   [PieceCount][SquareCount]uint64
	zobristCastling [16]uint64
	zobristEPFile   [8]uint64
	zobristSide     uint64
)

func init() {
	rng := newXorshift(0x9E37_79B9_7F4A_7C15)

	for p := Piece(0); p < PieceCount; p++ {
		for s := Square(0); s < SquareCount; s++ {
			zobristPieces[p][s] = rng.next()
		}
	}
	for i := range zobristCastling {
		zobristCastling[i] = rng.next()
	}
	for i := range zobristEPFile {
		zobristEPFile[i] = rng.next()
	}
	zobristSide = rng.next()
}

// computeKey recalculates the position key from scratch. Make/unmake maintains
// the key incrementally; this exists to seed it after parsing a FEN, and to
// verify the incremental updates in tests.
func (p *Position) computeKey() uint64 {
	var key uint64
	for s := Square(0); s < SquareCount; s++ {
		if pc := p.board[s]; pc != NoPiece {
			key ^= zobristPieces[pc][s]
		}
	}
	key ^= zobristCastling[p.castling]
	// Only the file matters for en passant, since the rank is implied by the
	// side to move.
	if p.epSquare != NoSquare {
		key ^= zobristEPFile[p.epSquare.File()]
	}
	if p.side == Black {
		key ^= zobristSide
	}
	return key
}
