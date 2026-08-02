package chess

import (
	"sort"
	"strings"
	"testing"
)

// This file cross-checks the fast move generator against a deliberately naive
// one, and checks a symmetry property that needs no external ground truth.
//
// Perft is strong evidence but it is aggregate: it compares node *counts* on
// seven positions. Two compensating bugs — a move wrongly generated in one
// place and a legal move missed in another — would cancel out in the total. And
// the seven positions, however carefully chosen, are seven positions.
//
// The reference generator here shares nothing with the real one but the attack
// tables (themselves verified independently against slow ray-walking in
// TestMagicAttacksMatchReference). It generates every pseudo-legal move by
// brute force, plays each one, and keeps it if the mover's king is not attacked
// afterwards. That is the definition of legality transcribed directly, with no
// pin masks, no check masks and no cleverness to be wrong about.

// pseudoLegalMoves generates every move that obeys the movement rules of the
// pieces, ignoring entirely whether it leaves the king in check.
func (p *Position) pseudoLegalMoves() []Move {
	var out []Move
	us := p.side
	occ := p.Occupied()
	ourPieces := p.byColor[us]

	promoRank, startRank := 7, 1
	if us == Black {
		promoRank, startRank = 0, 6
	}
	push := PawnPush(us)

	for from := Square(0); from < SquareCount; from++ {
		pc := p.board[from]
		if pc == NoPiece || pc.Color() != us {
			continue
		}

		if pc.Type() == Pawn {
			// Single push.
			if to := Square(int(from) + push); !occ.Has(to) {
				appendPawn(&out, from, to, promoRank)
				// Double push.
				if from.Rank() == startRank {
					if to2 := Square(int(to) + push); !occ.Has(to2) {
						out = append(out, NewMove(from, to2))
					}
				}
			}
			// Captures, including en passant.
			for caps := PawnAttacks[us][from]; caps != 0; {
				var to Square
				to, caps = caps.PopFirst()
				if p.board[to] != NoPiece && p.board[to].Color() != us {
					appendPawn(&out, from, to, promoRank)
				} else if to == p.epSquare && p.epSquare != NoSquare {
					out = append(out, NewEnPassant(from, to))
				}
			}
			continue
		}

		for attacks := AttacksFrom(pc.Type(), us, from, occ) &^ ourPieces; attacks != 0; {
			var to Square
			to, attacks = attacks.PopFirst()
			out = append(out, NewMove(from, to))
		}
	}

	out = append(out, p.pseudoLegalCastling(us, occ)...)
	return out
}

func appendPawn(out *[]Move, from, to Square, promoRank int) {
	if to.Rank() == promoRank {
		for _, pt := range []PieceType{Knight, Bishop, Rook, Queen} {
			*out = append(*out, NewPromotion(from, to, pt))
		}
		return
	}
	*out = append(*out, NewMove(from, to))
}

// pseudoLegalCastling handles castling separately, because the rule that the
// king may not pass through an attacked square cannot be expressed by making
// the move and inspecting the result — the intermediate square is not the final
// one. This walks every square from the king's origin to its destination and
// requires all of them to be safe, which is a different formulation from the
// generator's tabulated version.
func (p *Position) pseudoLegalCastling(us Color, occ Bitboard) []Move {
	var out []Move
	them := us.Opposite()

	type spec struct {
		right            CastlingRights
		kingFrom, kingTo Square
		rookFrom         Square
	}
	specs := []spec{
		{WhiteKingSide, E1, G1, H1},
		{WhiteQueenSide, E1, C1, A1},
		{BlackKingSide, E8, G8, H8},
		{BlackQueenSide, E8, C8, A8},
	}

	for _, s := range specs {
		if MakePiece(us, King) != p.board[s.kingFrom] || !p.castling.Has(s.right) {
			continue
		}
		// Every square strictly between king and rook must be empty.
		if BetweenBB[s.kingFrom][s.rookFrom]&occ != 0 {
			continue
		}
		// Every square the king occupies or crosses must be unattacked,
		// including the one it starts on (no castling out of check).
		step := 1
		if s.kingTo < s.kingFrom {
			step = -1
		}
		safe := true
		for sq := s.kingFrom; ; sq = Square(int(sq) + step) {
			if p.IsAttacked(sq, them) {
				safe = false
				break
			}
			if sq == s.kingTo {
				break
			}
		}
		if safe {
			out = append(out, NewCastling(s.kingFrom, s.kingTo))
		}
	}
	return out
}

// referenceLegalMoves is legality by definition: play every pseudo-legal move
// and keep the ones that do not leave the mover's king attacked.
func (p *Position) referenceLegalMoves() []Move {
	us := p.side
	var out []Move
	for _, m := range p.pseudoLegalMoves() {
		p.MakeMove(m)
		safe := !p.IsAttacked(p.KingSquare(us), us.Opposite())
		p.UnmakeMove()
		if safe {
			out = append(out, m)
		}
	}
	return out
}

func moveStrings(moves []Move) []string {
	out := make([]string, len(moves))
	for i, m := range moves {
		out[i] = m.String()
	}
	sort.Strings(out)
	return out
}

// TestLegalMovesMatchReference walks a large number of positions reached by
// random play and requires the fast generator's move set to equal the reference
// set exactly — not merely to be the same size. Any move generated that should
// not be, or missing that should be there, fails here with both lists.
func TestLegalMovesMatchReference(t *testing.T) {
	roots := []string{
		StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
		"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
		// Endgames rich in promotions and pawn races, where en passant and
		// underpromotion arise far more often than in the middlegame.
		"8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
		"4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1",
		"8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 1",
	}

	rng := newXorshift(0x0BADC0DE_5EED_1234)
	const gamesPerRoot = 60
	const maxPlies = 120

	positionsChecked := 0

	for _, fen := range roots {
		for game := 0; game < gamesPerRoot; game++ {
			p, err := ParseFEN(fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", fen, err)
			}

			for ply := 0; ply < maxPlies; ply++ {
				fast := moveStrings(p.LegalMoves())
				ref := moveStrings(p.referenceLegalMoves())
				positionsChecked++

				if strings.Join(fast, ",") != strings.Join(ref, ",") {
					t.Fatalf("move sets differ at %s\n  generator: %v\n  reference: %v\n  only in generator: %v\n  only in reference: %v",
						p.FEN(), fast, ref, difference(fast, ref), difference(ref, fast))
				}

				if len(fast) == 0 {
					break // checkmate or stalemate
				}
				// Halfmove clock caps runaway shuffling games.
				if p.HalfmoveClock() >= 100 {
					break
				}

				moves := p.LegalMoves()
				p.MakeMove(moves[rng.next()%uint64(len(moves))])
			}
		}
	}

	t.Logf("cross-checked %d positions against the reference generator", positionsChecked)
}

func difference(a, b []string) []string {
	inB := make(map[string]bool, len(b))
	for _, s := range b {
		inB[s] = true
	}
	var out []string
	for _, s := range a {
		if !inB[s] {
			out = append(out, s)
		}
	}
	return out
}

// mirrorFEN reflects a position vertically and swaps colors, producing a
// position that is strategically identical for the other side. Rank 1 becomes
// rank 8, white pieces become black, castling rights and the en passant square
// move with them.
func mirrorFEN(t *testing.T, fen string) string {
	t.Helper()

	fields := strings.Fields(fen)
	if len(fields) < 4 {
		t.Fatalf("mirrorFEN: malformed FEN %q", fen)
	}

	ranks := strings.Split(fields[0], "/")
	for i, j := 0, len(ranks)-1; i < j; i, j = i+1, j-1 {
		ranks[i], ranks[j] = ranks[j], ranks[i]
	}
	for i, r := range ranks {
		ranks[i] = swapCase(r)
	}
	board := strings.Join(ranks, "/")

	side := "b"
	if fields[1] == "b" {
		side = "w"
	}

	castling := swapCase(fields[2])
	if castling != "-" {
		// Re-order into the canonical KQkq sequence.
		var b strings.Builder
		for _, want := range "KQkq" {
			if strings.ContainsRune(castling, want) {
				b.WriteRune(want)
			}
		}
		castling = b.String()
	}

	ep := fields[3]
	if ep != "-" {
		sq, ok := parseSquare(ep)
		if !ok {
			t.Fatalf("mirrorFEN: bad en passant square %q", ep)
		}
		ep = MakeSquare(sq.File(), 7-sq.Rank()).String()
	}

	rest := " 0 1"
	if len(fields) >= 6 {
		rest = " " + fields[4] + " " + fields[5]
	}
	return board + " " + side + " " + castling + " " + ep + rest
}

func swapCase(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r - 32
		case r >= 'A' && r <= 'Z':
			return r + 32
		}
		return r
	}, s)
}

// TestPerftMirrorSymmetry checks that mirroring a position vertically and
// swapping colors leaves the perft counts unchanged. Chess is symmetric under
// this transformation, so any difference is a bug that treats one color
// differently from the other — a wrong pawn direction, a castling right
// tabulated for only one back rank, an en passant rank hardcoded.
//
// The value of this test is that it needs no published node counts: it can be
// applied to any position at all, including ones nobody has ever tabulated.
func TestPerftMirrorSymmetry(t *testing.T) {
	for _, tc := range perftTests {
		t.Run(tc.name, func(t *testing.T) {
			original, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}
			mirroredFEN := mirrorFEN(t, tc.fen)
			mirrored, err := ParseFEN(mirroredFEN)
			if err != nil {
				t.Fatalf("ParseFEN(mirror %q) = %q: %v", tc.fen, mirroredFEN, err)
			}

			for depth := 1; depth <= 4; depth++ {
				want := Perft(original, depth)
				got := Perft(mirrored, depth)
				if got != want {
					t.Errorf("depth %d: mirrored perft = %d, original = %d\n  original: %s\n  mirrored: %s",
						depth, got, want, tc.fen, mirroredFEN)
				}
			}
		})
	}
}

// TestEveryGeneratedMoveIsReversible plays every legal move in a set of
// positions and confirms make/unmake restores the position exactly, for each
// move kind independently. Perft's aggregate counts would survive an unmake
// that restored an equivalent-but-differently-represented position; this would
// not.
func TestEveryGeneratedMoveIsReversible(t *testing.T) {
	fens := []string{
		StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
		"8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 1",
		"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
	}

	for _, fen := range fens {
		p, err := ParseFEN(fen)
		if err != nil {
			t.Fatalf("ParseFEN(%q): %v", fen, err)
		}

		for _, m := range p.LegalMoves() {
			before := *p
			beforeFEN := p.FEN()

			p.MakeMove(m)
			p.UnmakeMove()

			if got := p.FEN(); got != beforeFEN {
				t.Errorf("%s after %s: FEN = %q, want %q", fen, m, got, beforeFEN)
			}
			if p.key != before.key {
				t.Errorf("%s after %s: key = %#x, want %#x", fen, m, p.key, before.key)
			}
			if p.board != before.board || p.byColor != before.byColor || p.byType != before.byType {
				t.Errorf("%s after %s: board state not restored", fen, m)
			}
			if p.castling != before.castling || p.epSquare != before.epSquare ||
				p.halfmove != before.halfmove || p.fullmove != before.fullmove {
				t.Errorf("%s after %s: position metadata not restored", fen, m)
			}
		}
	}
}

// TestAllMoveKindsAreExercised guards the tests themselves. The cross-check
// above is only meaningful if the positions it walks actually produce castling,
// en passant and underpromotions; a fixture change that quietly stopped
// generating them would leave the suite green and the coverage gone.
func TestAllMoveKindsAreExercised(t *testing.T) {
	seen := map[MoveKind]int{}
	seenPromo := map[PieceType]int{}

	roots := []string{
		StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
		"8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 1",
	}

	var walk func(p *Position, depth int)
	walk = func(p *Position, depth int) {
		if depth == 0 {
			return
		}
		for _, m := range p.LegalMoves() {
			seen[m.Kind()]++
			if m.Kind() == KindPromotion {
				seenPromo[m.Promotion()]++
			}
			p.MakeMove(m)
			walk(p, depth-1)
			p.UnmakeMove()
		}
	}

	for _, fen := range roots {
		p, err := ParseFEN(fen)
		if err != nil {
			t.Fatalf("ParseFEN(%q): %v", fen, err)
		}
		walk(p, 3)
	}

	for kind, name := range map[MoveKind]string{
		KindNormal:    "normal",
		KindPromotion: "promotion",
		KindEnPassant: "en passant",
		KindCastling:  "castling",
	} {
		if seen[kind] == 0 {
			t.Errorf("no %s moves generated anywhere in the test positions", name)
		}
	}
	for _, pt := range []PieceType{Knight, Bishop, Rook, Queen} {
		if seenPromo[pt] == 0 {
			t.Errorf("no promotions to %s generated; underpromotion coverage is missing",
				MakePiece(White, pt))
		}
	}
	t.Logf("move kinds seen: normal=%d promotion=%d en-passant=%d castling=%d",
		seen[KindNormal], seen[KindPromotion], seen[KindEnPassant], seen[KindCastling])
}
