package chess

import "fmt"

// Perft counts the leaf nodes of the move tree to the given depth. It is the
// standard correctness test for a move generator: the counts for well-known
// positions are published and exact, so any discrepancy — a missed en passant
// pin, a castling right not revoked when a rook is captured, a promotion
// under-generated — shows up as a wrong number rather than as a mysterious
// weakness months later.
//
// Every other component in this engine assumes move generation is correct, so
// this is the gate they all sit behind.
func Perft(p *Position, depth int) uint64 {
	if depth <= 0 {
		return 1
	}

	var ml MoveList
	p.GenerateLegalMoves(&ml)

	// At depth 1 the move count is the answer; making and unmaking each move
	// to count one leaf apiece is pure waste.
	if depth == 1 {
		return uint64(ml.Count)
	}

	var nodes uint64
	for _, m := range ml.Slice() {
		p.MakeMove(m)
		nodes += Perft(p, depth-1)
		p.UnmakeMove()
	}
	return nodes
}

// PerftDivide returns the per-move node counts at the root, which is how a
// perft mismatch is actually debugged: compare against a known-good engine,
// find the move whose subtree count differs, descend into it, repeat.
func PerftDivide(p *Position, depth int) (map[string]uint64, uint64) {
	result := make(map[string]uint64)
	var total uint64

	var ml MoveList
	p.GenerateLegalMoves(&ml)

	for _, m := range ml.Slice() {
		p.MakeMove(m)
		n := Perft(p, depth-1)
		p.UnmakeMove()

		result[m.String()] = n
		total += n
	}
	return result, total
}

// PrintPerftDivide writes a divide breakdown in the format most engines use, so
// the output can be diffed directly against a reference.
func PrintPerftDivide(p *Position, depth int) {
	divide, total := PerftDivide(p, depth)
	for _, m := range p.LegalMoves() {
		fmt.Printf("%s: %d\n", m, divide[m.String()])
	}
	fmt.Printf("\nNodes searched: %d\n", total)
}
