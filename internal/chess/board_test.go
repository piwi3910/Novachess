package chess

import (
	"testing"
)

// TestMagicAttacksMatchReference checks the magic bitboard tables against the
// slow ray-walking implementation they were built from, across randomly chosen
// occupancies. Perft would catch most magic errors indirectly, but only for
// occupancies that arise in those particular trees; this exercises the tables
// directly and reports the failing square and occupancy when something is off.
func TestMagicAttacksMatchReference(t *testing.T) {
	rng := newXorshift(0xDEAD_BEEF_CAFE_F00D)

	for s := Square(0); s < SquareCount; s++ {
		for i := 0; i < 256; i++ {
			// AND three draws together to get sparse, board-like occupancies
			// rather than half the squares being full.
			occ := Bitboard(rng.next() & rng.next() & rng.next())

			if got, want := RookAttacks(s, occ), slidingAttacks(s, occ, rookDirs); got != want {
				t.Fatalf("RookAttacks(%v, occ):\ngot:\n%vwant:\n%voccupancy:\n%v", s, got, want, occ)
			}
			if got, want := BishopAttacks(s, occ), slidingAttacks(s, occ, bishopDirs); got != want {
				t.Fatalf("BishopAttacks(%v, occ):\ngot:\n%vwant:\n%voccupancy:\n%v", s, got, want, occ)
			}
		}
	}
}

func TestFENRoundTrip(t *testing.T) {
	fens := []string{
		StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
		"4k3/8/8/8/8/8/8/4K3 w - - 0 1",
		// The en passant target survives because the pawn on e5 can actually
		// capture on d6.
		"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
		"8/8/8/8/8/8/6k1/4K2R w K - 13 47",
	}

	for _, fen := range fens {
		p, err := ParseFEN(fen)
		if err != nil {
			t.Errorf("ParseFEN(%q): %v", fen, err)
			continue
		}
		if got := p.FEN(); got != fen {
			t.Errorf("FEN round trip: got %q, want %q", got, fen)
		}
	}
}

// TestFENNormalization documents where a FEN deliberately does not round trip.
// A castling right whose rook is absent, or an en passant target no pawn can
// use, describes something the board contradicts. Both are dropped, so the
// rendered FEN is the corrected one — see sanitize for why believing them is
// worse than rewriting them.
func TestFENNormalization(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "unusable en passant target is dropped",
			in:   "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
			want: "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
		},
		{
			name: "castling rights without rooks are dropped",
			in:   "4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1",
			want: "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
		},
		{
			name: "only the unsupported right is dropped",
			in:   "4k3/8/8/8/8/8/8/R3K3 w KQ - 0 1",
			want: "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.in)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.in, err)
			}
			if got := p.FEN(); got != tc.want {
				t.Errorf("normalized FEN = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestParseFENRejectsInvalid(t *testing.T) {
	cases := []struct {
		name string
		fen  string
	}{
		{"too few fields", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"},
		{"seven ranks", "rnbqkbnr/pppppppp/8/8/8/8/RNBQKBNR w KQkq - 0 1"},
		{"no white king", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR w KQkq - 0 1"},
		{"two white kings", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RKBQKBNR w KQkq - 0 1"},
		{"bad side to move", "4k3/8/8/8/8/8/8/4K3 x - - 0 1"},
		{"rank too short", "4k3/8/8/8/8/8/8/4K2 w - - 0 1"},
		{"invalid piece", "4k3/8/8/8/8/8/8/4K2X w - - 0 1"},
		{"pawn on back rank", "4k3/8/8/8/8/8/8/P3K3 w - - 0 1"},
		// White is in check from the f1 rook but it is Black to move, so White
		// must have failed to address the check on the previous ply.
		{"side not to move is in check", "4k3/8/8/8/8/8/8/4Kr2 b - - 0 1"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseFEN(tc.fen); err == nil {
				t.Errorf("ParseFEN(%q) succeeded, want error", tc.fen)
			}
		})
	}
}

// TestLegalMovesInTrickyPositions covers the rules that are easiest to get
// wrong and that shallow perft can miss, each isolated so a failure names the
// specific rule rather than a node count.
func TestLegalMovesInTrickyPositions(t *testing.T) {
	cases := []struct {
		name string
		fen  string
		// wantCount, when non-nil, is the exact number of legal moves.
		wantCount *int
		present   []string // moves that must be generated
		absent    []string // moves that must not be generated
		// kingOnly asserts every legal move is a king move, which is the
		// defining property of a double check.
		kingOnly bool
	}{
		{
			// The classic en passant discovered check. White king a5, white
			// pawn e5, black pawn d5 and black rook h5 all share rank 5.
			// Capturing exd6 e.p. vacates both d5 and e5 at once, opening the
			// rook's line to the king. No pin mask predicts this, because
			// neither pawn is pinned on its own.
			name:   "en passant would expose king along the rank",
			fen:    "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",
			absent: []string{"e5d6"},
			// The ordinary push is unaffected and must still be there.
			present: []string{"e5e6"},
		},
		{
			// Identical geometry with the rook removed, so the same capture is
			// legal. This is the control for the case above: without it, a
			// generator that simply never emits en passant would pass.
			name:    "en passant legal when no slider is behind the pawns",
			fen:     "8/8/8/K2pP3/8/8/8/7k w - d6 0 1",
			present: []string{"e5d6"},
		},
		{
			// Knight on e2 pinned against the king on e1 by the rook on e8. A
			// knight's attack set never intersects its pin ray, so a pinned
			// knight has no legal move whatsoever.
			name:      "pinned knight cannot move",
			fen:       "4rk2/8/8/8/8/8/4N3/4K3 w - - 0 1",
			absent:    []string{"e2c1", "e2c3", "e2d4", "e2f4", "e2g3", "e2g1"},
			wantCount: intPtr(4), // king to d1, d2, f1, f2
		},
		{
			// The rook on f4 attacks f1, which the king crosses when castling
			// kingside. Queenside is untouched and stays legal.
			name:    "cannot castle through an attacked square",
			fen:     "4k3/8/8/8/5r2/8/8/R3K2R w KQ - 0 1",
			absent:  []string{"e1g1"},
			present: []string{"e1c1"},
		},
		{
			// Only the squares the king crosses matter. During queenside
			// castling the rook passes over b1, but the king does not, so an
			// attack on b1 does not prevent it. This is the mirror of the case
			// above and catches generators that mask too broadly.
			name:    "may castle queenside when only b1 is attacked",
			fen:     "4k3/8/8/8/1r6/8/8/R3K2R w KQ - 0 1",
			present: []string{"e1c1", "e1g1"},
		},
		{
			// In check from the e1 rook, so neither castle is available. The
			// white king sits on g1 rather than h1 so that the black h8 rook
			// is not itself giving check on the wrong turn.
			name:   "cannot castle out of check",
			fen:    "r3k2r/8/8/8/8/8/8/4R1K1 b kq - 0 1",
			absent: []string{"e8g8", "e8c8"},
		},
		{
			// Rook on e8 and knight on d3 both check the king on e1. No single
			// capture or interposition answers both, so only king moves are
			// legal. The black king stands on f7, off the e-file, so that it
			// does not block its own rook.
			name:      "double check allows only king moves",
			fen:       "4r3/5k2/8/8/8/3n4/8/4K3 w - - 0 1",
			kingOnly:  true,
			wantCount: intPtr(3), // d1, d2, f1
		},
		{
			name:      "stalemate has no legal moves",
			fen:       "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
			wantCount: intPtr(0),
		},
		{
			name:      "checkmate has no legal moves",
			fen:       "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
			wantCount: intPtr(0),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}

			moves := p.LegalMoves()
			got := make(map[string]bool, len(moves))
			for _, m := range moves {
				got[m.String()] = true
			}

			if tc.wantCount != nil && len(moves) != *tc.wantCount {
				t.Errorf("got %d legal moves, want %d: %v", len(moves), *tc.wantCount, moves)
			}
			for _, want := range tc.present {
				if !got[want] {
					t.Errorf("legal move %s was not generated; got %v", want, moves)
				}
			}
			for _, bad := range tc.absent {
				if got[bad] {
					t.Errorf("illegal move %s was generated", bad)
				}
			}
			if tc.kingOnly {
				ksq := p.KingSquare(p.SideToMove())
				for _, m := range moves {
					if m.From() != ksq {
						t.Errorf("non-king move %s generated under double check", m)
					}
				}
			}
		})
	}
}

func intPtr(n int) *int { return &n }

// TestStalemateAndCheckmateDistinguished confirms the position reports check
// correctly, which is what separates the two zero-move outcomes.
func TestStalemateAndCheckmateDistinguished(t *testing.T) {
	stalemate, err := ParseFEN("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	if len(stalemate.LegalMoves()) != 0 {
		t.Error("expected no legal moves in stalemate position")
	}
	if stalemate.InCheck() {
		t.Error("stalemate position should not be in check")
	}

	mate, err := ParseFEN("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
	if err != nil {
		t.Fatal(err)
	}
	if len(mate.LegalMoves()) != 0 {
		t.Error("expected no legal moves in checkmate position")
	}
	if !mate.InCheck() {
		t.Error("checkmate position should be in check")
	}
}

func TestMoveEncoding(t *testing.T) {
	cases := []struct {
		move Move
		want string
		kind MoveKind
	}{
		{NewMove(E2, E4), "e2e4", KindNormal},
		{NewMove(A1, H8), "a1h8", KindNormal},
		{NewPromotion(E7, E8, Queen), "e7e8q", KindPromotion},
		{NewPromotion(B2, A1, Knight), "b2a1n", KindPromotion},
		{NewPromotion(B2, A1, Rook), "b2a1r", KindPromotion},
		{NewPromotion(B2, A1, Bishop), "b2a1b", KindPromotion},
		{NewEnPassant(E5, D6), "e5d6", KindEnPassant},
		{NewCastling(E1, G1), "e1g1", KindCastling},
		{NewCastling(E8, C8), "e8c8", KindCastling},
	}

	for _, tc := range cases {
		if got := tc.move.String(); got != tc.want {
			t.Errorf("Move.String() = %q, want %q", got, tc.want)
		}
		if got := tc.move.Kind(); got != tc.kind {
			t.Errorf("%s: Kind() = %d, want %d", tc.want, got, tc.kind)
		}
	}

	// From and To must survive encoding for every square pair, since the whole
	// search depends on this packing being lossless.
	for from := Square(0); from < SquareCount; from++ {
		for to := Square(0); to < SquareCount; to++ {
			m := NewMove(from, to)
			if m.From() != from || m.To() != to {
				t.Fatalf("NewMove(%v,%v) decoded as (%v,%v)", from, to, m.From(), m.To())
			}
		}
	}
}

func TestSquareAndBitboardBasics(t *testing.T) {
	if A1.File() != 0 || A1.Rank() != 0 {
		t.Errorf("A1 = file %d rank %d, want 0,0", A1.File(), A1.Rank())
	}
	if H8.File() != 7 || H8.Rank() != 7 {
		t.Errorf("H8 = file %d rank %d, want 7,7", H8.File(), H8.Rank())
	}
	if got := E4.String(); got != "e4" {
		t.Errorf("E4.String() = %q, want %q", got, "e4")
	}

	// Shifts must not wrap around the board edges.
	if got := H1.BB().Shifted(East); got != EmptyBB {
		t.Errorf("east shift off the h-file wrapped: %v", got)
	}
	if got := A1.BB().Shifted(West); got != EmptyBB {
		t.Errorf("west shift off the a-file wrapped: %v", got)
	}
	if got := A8.BB().Shifted(North); got != EmptyBB {
		t.Errorf("north shift off rank 8 wrapped: %v", got)
	}

	// A knight in the corner reaches exactly two squares.
	if got := KnightAttacks[A1].Count(); got != 2 {
		t.Errorf("knight on a1 attacks %d squares, want 2", got)
	}
	if got := KnightAttacks[D4].Count(); got != 8 {
		t.Errorf("knight on d4 attacks %d squares, want 8", got)
	}
	if got := KingAttacks[A1].Count(); got != 3 {
		t.Errorf("king on a1 attacks %d squares, want 3", got)
	}
}

func TestBetweenAndLineTables(t *testing.T) {
	if got, want := BetweenBB[A1][A4], A2.BB()|A3.BB(); got != want {
		t.Errorf("BetweenBB[a1][a4] = %v, want %v", got, want)
	}
	if got, want := BetweenBB[A1][C3], B2.BB(); got != want {
		t.Errorf("BetweenBB[a1][c3] = %v, want %v", got, want)
	}
	if got := BetweenBB[A1][B3]; got != EmptyBB {
		t.Errorf("BetweenBB for unaligned squares = %v, want empty", got)
	}
	if !LineBB[A1][A4].Has(A8) {
		t.Error("LineBB[a1][a4] should extend to a8")
	}
	if LineBB[A1][B3] != EmptyBB {
		t.Error("LineBB for unaligned squares should be empty")
	}
	if !Aligned(A1, D4, H8) {
		t.Error("a1, d4, h8 should be aligned on a diagonal")
	}
	if Aligned(A1, D4, H7) {
		t.Error("a1, d4, h7 should not be aligned")
	}
}
