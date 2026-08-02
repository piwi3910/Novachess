package chess

import "testing"

// Compliance tests for the rules that perft cannot reach.
//
// Perft validates move generation from positions it is given. It cannot check
// what happens when a position is described wrongly, and it cannot check the
// rules about when a game ends, because both are about positions rather than
// about counting moves.

// TestCastlingRightsWithoutRookAreDropped covers a crash, not merely a wrong
// move. A FEN asserting castling rights with no rook on the corner square would
// previously generate the castling move, and playing it relocated a piece that
// was not there. FENs arrive from GUIs and web APIs, so a malformed one must
// not be able to take the engine down mid-game.
func TestCastlingRightsWithoutRookAreDropped(t *testing.T) {
	cases := []struct {
		name string
		fen  string
		want CastlingRights
	}{
		{"no rooks at all", "4k3/8/8/8/8/8/8/4K3 w KQ - 0 1", NoCastling},
		{"kingside rook missing", "4k3/8/8/8/8/8/8/R3K3 w KQ - 0 1", WhiteQueenSide},
		{"queenside rook missing", "4k3/8/8/8/8/8/8/4K2R w KQ - 0 1", WhiteKingSide},
		{"king off its home square", "4k3/8/8/8/8/8/8/R2K3R w KQ - 0 1", NoCastling},
		{"black rights with no black rooks", "4k3/8/8/8/8/8/8/R3K2R w KQkq - 0 1", WhiteKingSide | WhiteQueenSide},
		{"a piece that is not a rook on the corner", "4k3/8/8/8/8/8/8/N3K2R w KQ - 0 1", WhiteKingSide},
		{"all rights genuinely available", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", AllCastling},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}
			if got := p.CastlingRights(); got != tc.want {
				t.Errorf("castling rights = %v, want %v", got, tc.want)
			}

			// Whatever survived sanitization must be playable without panic.
			for _, m := range p.LegalMoves() {
				if m.Kind() != KindCastling {
					continue
				}
				p.MakeMove(m)
				p.UnmakeMove()
			}
		})
	}
}

// TestPhantomEnPassantSquareIsCleared checks that an en passant target no pawn
// can use is discarded.
//
// This is a rules requirement rather than tidiness. FIDE distinguishes
// positions by the moves available in them, so two positions differing only by
// an unusable en passant target are the same position. Recording it would hash
// them differently and hide a threefold repetition, losing a draw the rules
// grant.
func TestPhantomEnPassantSquareIsCleared(t *testing.T) {
	cases := []struct {
		name string
		fen  string
		want Square
	}{
		{
			name: "no pawn of either color near the target",
			fen:  "4k3/8/8/8/8/8/4P3/4K3 w - d6 0 1",
			want: NoSquare,
		},
		{
			// Black played d7-d5, but White has no pawn on c5 or e5 to take it.
			name: "double push with no capturing pawn adjacent",
			fen:  "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
			want: NoSquare,
		},
		{
			// White pawn on e5 genuinely can capture d6.
			name: "capture is available, target is kept",
			fen:  "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
			want: D6,
		},
		{
			// The capture is available on paper but leaves the king exposed
			// along the fifth rank, so no en passant possibility exists.
			name: "capture exists but is illegal, target is cleared",
			fen:  "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",
			want: NoSquare,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}
			if got := p.EnPassantSquare(); got != tc.want {
				t.Errorf("en passant square = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestUnusableEnPassantDoesNotAffectHash is the consequence that matters: two
// descriptions of the same position must produce the same Zobrist key, or
// repetition detection silently fails.
func TestUnusableEnPassantDoesNotAffectHash(t *testing.T) {
	withTarget, err := ParseFEN("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
	if err != nil {
		t.Fatal(err)
	}
	without, err := ParseFEN("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
	if err != nil {
		t.Fatal(err)
	}
	if withTarget.Key() != without.Key() {
		t.Errorf("keys differ (%#x vs %#x); no pawn can capture on d6, so these are the same position",
			withTarget.Key(), without.Key())
	}
}

// TestEnPassantRecordedOnlyWhenCapturable checks the same rule on the
// make-move path rather than the parsing path.
func TestEnPassantRecordedOnlyWhenCapturable(t *testing.T) {
	// White pawn on e5: after black plays d7-d5, exd6 is available.
	p, err := ParseFEN("rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	p.MakeMove(NewMove(D7, D5))
	if got := p.EnPassantSquare(); got != D6 {
		t.Errorf("en passant square = %v, want d6 (exd6 is available)", got)
	}

	// Same double push with the white pawn on e4 instead: nothing can capture.
	q, err := ParseFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	q.MakeMove(NewMove(D7, D5))
	if got := q.EnPassantSquare(); got != NoSquare {
		t.Errorf("en passant square = %v, want none (no pawn can capture on d6)", got)
	}
}

// TestRepetitionAcrossDifferentMoveOrders is the practical payoff. The same
// position reached two ways must be recognized as a repetition even though one
// route passed through a double pawn push.
func TestRepetitionAcrossDifferentMoveOrders(t *testing.T) {
	p := NewPosition()

	// A double push whose en passant target no pawn can use, then a knight
	// shuffle back and forth. The position after the shuffle must count as a
	// repetition of the position right after the push.
	p.MakeMove(NewMove(E2, E4))
	if p.EnPassantSquare() != NoSquare {
		t.Fatalf("setup: expected no usable en passant target after 1.e4, got %v", p.EnPassantSquare())
	}
	keyAfterPush := p.Key()

	for _, m := range []Move{
		NewMove(G8, F6), NewMove(G1, F3),
		NewMove(F6, G8), NewMove(F3, G1),
	} {
		p.MakeMove(m)
	}

	if p.Key() != keyAfterPush {
		t.Errorf("key after shuffling back = %#x, want %#x", p.Key(), keyAfterPush)
	}
	if !p.IsRepetition() {
		t.Error("IsRepetition() = false, want true after returning to the same position")
	}
}

// TestCastlingRulesInDetail enumerates the castling conditions individually, so
// a failure names the specific rule rather than a node count.
func TestCastlingRulesInDetail(t *testing.T) {
	cases := []struct {
		name    string
		fen     string
		present []string
		absent  []string
	}{
		{
			name:    "both sides available for both colors",
			fen:     "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
			present: []string{"e1g1", "e1c1"},
		},
		{
			name:    "blocked by an own piece on f1",
			fen:     "r3k2r/8/8/8/8/8/8/R3KB1R w KQkq - 0 1",
			absent:  []string{"e1g1"},
			present: []string{"e1c1"},
		},
		{
			name:    "blocked by an own piece on b1 (queenside only)",
			fen:     "r3k2r/8/8/8/8/8/8/RN2K2R w KQkq - 0 1",
			absent:  []string{"e1c1"},
			present: []string{"e1g1"},
		},
		{
			name:    "blocked by an enemy piece in the path",
			fen:     "r3k2r/8/8/8/8/8/8/R2nK2R w KQkq - 0 1",
			absent:  []string{"e1c1"},
			present: []string{"e1g1"},
		},
		{
			name:   "rights absent from the FEN",
			fen:    "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1",
			absent: []string{"e1g1", "e1c1"},
		},
		{
			name:    "black to move, both sides available",
			fen:     "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
			present: []string{"e8g8", "e8c8"},
		},
		{
			// The g8 square is attacked by the bishop on a2, so kingside is
			// out; queenside is unaffected.
			name:    "black cannot castle through an attacked square",
			fen:     "r3k2r/8/8/8/8/8/B7/4K3 b kq - 0 1",
			absent:  []string{"e8g8"},
			present: []string{"e8c8"},
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
			for _, want := range tc.present {
				if !got[want] {
					t.Errorf("castling move %s missing; generated %v", want, moves)
				}
			}
			for _, bad := range tc.absent {
				if got[bad] {
					t.Errorf("castling move %s generated but should be illegal", bad)
				}
			}
		})
	}
}

// TestCastlingRightsRevocation covers every way a right can be lost. The rook
// capture case is the one that is easy to miss: taking a rook on its home
// square removes the opponent's right even though the opponent did not move.
func TestCastlingRightsRevocation(t *testing.T) {
	cases := []struct {
		name string
		fen  string
		move Move
		want CastlingRights
	}{
		{
			name: "king move revokes both white rights",
			fen:  "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
			move: NewMove(E1, E2),
			want: BlackKingSide | BlackQueenSide,
		},
		{
			name: "h1 rook move revokes white kingside only",
			fen:  "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
			move: NewMove(H1, H2),
			want: WhiteQueenSide | BlackKingSide | BlackQueenSide,
		},
		{
			name: "a1 rook move revokes white queenside only",
			fen:  "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
			move: NewMove(A1, A2),
			want: WhiteKingSide | BlackKingSide | BlackQueenSide,
		},
		{
			name: "castling itself revokes both rights",
			fen:  "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
			move: NewCastling(E1, G1),
			want: BlackKingSide | BlackQueenSide,
		},
		{
			// The rook is captured on h8 without Black having moved anything.
			name: "capturing a rook on its home square revokes the owner's right",
			fen:  "r3k2r/7R/8/8/8/8/8/R3K3 w Qkq - 0 1",
			move: NewMove(H7, H8),
			want: WhiteQueenSide | BlackQueenSide,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}
			before := p.CastlingRights()

			p.MakeMove(tc.move)
			if got := p.CastlingRights(); got != tc.want {
				t.Errorf("after %s: rights = %v, want %v", tc.move, got, tc.want)
			}

			// Rights must come back on unmake, and only then.
			p.UnmakeMove()
			if got := p.CastlingRights(); got != before {
				t.Errorf("after unmake: rights = %v, want %v", got, before)
			}
		})
	}
}

// TestEnPassantRules enumerates the en passant conditions individually.
func TestEnPassantRules(t *testing.T) {
	t.Run("capture is available only on the immediately following ply", func(t *testing.T) {
		p, err := ParseFEN("rnbqkbnr/ppppppp1/8/4P2p/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
		if err != nil {
			t.Fatal(err)
		}
		p.MakeMove(NewMove(F7, F5))
		if p.EnPassantSquare() != F6 {
			t.Fatalf("setup: en passant square = %v, want f6", p.EnPassantSquare())
		}

		hasEP := func(p *Position) bool {
			for _, m := range p.LegalMoves() {
				if m.Kind() == KindEnPassant {
					return true
				}
			}
			return false
		}
		if !hasEP(p) {
			t.Error("exf6 should be available immediately after the double push")
		}

		// Interpose a quiet move for each side; the chance must be gone.
		p.MakeMove(NewMove(B1, C3))
		p.MakeMove(NewMove(B8, C6))
		if p.EnPassantSquare() != NoSquare {
			t.Errorf("en passant square = %v, want none two plies later", p.EnPassantSquare())
		}
		if hasEP(p) {
			t.Error("en passant capture still available two plies after the double push")
		}
	})

	t.Run("capturing pawn pinned on the file cannot capture", func(t *testing.T) {
		// The white pawn on e5 is pinned against the king on e1 by the rook on
		// e8, so it may not step aside to d6. Since no legal en passant capture
		// exists, the target square is also dropped as unusable — either route
		// produces the same required outcome, which is that no en passant move
		// is generated.
		p, err := ParseFEN("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range p.LegalMoves() {
			if m.Kind() == KindEnPassant {
				t.Errorf("generated en passant %s with the capturing pawn pinned", m)
			}
		}
	})

	t.Run("en passant may resolve a check by capturing the checking pawn", func(t *testing.T) {
		// Black has just played f7-f5 giving check from f5; exf6 en passant
		// removes the checking pawn. The destination f6 is not on the line
		// between king and checker, so a naive evasion mask would miss it.
		p, err := ParseFEN("8/8/8/4Pp2/6K1/8/8/4k3 w - f6 0 1")
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, m := range p.LegalMoves() {
			if m.Kind() == KindEnPassant {
				found = true
			}
		}
		if !found {
			t.Errorf("en passant capture not generated; legal moves were %v", p.LegalMoves())
		}
	})
}

// TestPromotionRules checks that promotion is mandatory and offers exactly the
// four choices.
func TestPromotionRules(t *testing.T) {
	p, err := ParseFEN("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}

	promos := map[PieceType]int{}
	quietToBackRank := 0
	for _, m := range p.LegalMoves() {
		if m.From() != B7 || m.To() != B8 {
			continue
		}
		if m.Kind() == KindPromotion {
			promos[m.Promotion()]++
		} else {
			quietToBackRank++
		}
	}

	if quietToBackRank != 0 {
		t.Error("generated a non-promotion pawn move to the back rank; promotion is mandatory")
	}
	for _, pt := range []PieceType{Knight, Bishop, Rook, Queen} {
		if promos[pt] != 1 {
			t.Errorf("promotions to %v = %d, want exactly 1", MakePiece(White, pt), promos[pt])
		}
	}
	if len(promos) != 4 {
		t.Errorf("got %d distinct promotion pieces, want 4", len(promos))
	}
}

// TestPromotionByCaptureRevokesCastlingRights combines two rules that interact:
// a pawn promoting by capturing a rook on its home square must also remove the
// castling right that rook carried.
func TestPromotionByCaptureRevokesCastlingRights(t *testing.T) {
	p, err := ParseFEN("r3k2r/1P6/8/8/8/8/8/4K3 w kq - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	p.MakeMove(NewPromotion(B7, A8, Queen))

	if got := p.CastlingRights(); got != BlackKingSide {
		t.Errorf("rights after bxa8=Q = %v, want k only (queenside rook was captured)", got)
	}
	if p.PieceAt(A8) != WhiteQueen {
		t.Errorf("piece on a8 = %v, want a white queen", p.PieceAt(A8))
	}

	p.UnmakeMove()
	if got := p.CastlingRights(); got != BlackKingSide|BlackQueenSide {
		t.Errorf("rights after unmake = %v, want kq", got)
	}
	if p.PieceAt(A8) != BlackRook || p.PieceAt(B7) != WhitePawn {
		t.Error("promotion capture was not correctly reversed")
	}
}

// TestAutomaticDrawThresholds covers the FIDE rules that need no claim. The
// engine adjudicates on the claimable thresholds, so these are never what ends
// a game here, but the Lichess bot has to agree with a server that uses them.
func TestAutomaticDrawThresholds(t *testing.T) {
	t.Run("seventy-five move rule", func(t *testing.T) {
		p, err := ParseFEN("4k3/8/8/8/8/8/3R4/4K3 w - - 149 90")
		if err != nil {
			t.Fatal(err)
		}
		if p.IsSeventyFiveMoveRule() {
			t.Error("IsSeventyFiveMoveRule() = true at 149 plies, want false")
		}
		p.MakeMove(NewMove(D2, D3))
		if !p.IsSeventyFiveMoveRule() {
			t.Error("IsSeventyFiveMoveRule() = false at 150 plies, want true")
		}
		// The claimable rule fired long ago and still holds.
		if !p.IsFiftyMoveRule() {
			t.Error("IsFiftyMoveRule() should also hold at 150 plies")
		}
	})

	t.Run("fivefold repetition", func(t *testing.T) {
		p := NewPosition()
		cycle := []Move{
			NewMove(G1, F3), NewMove(G8, F6),
			NewMove(F3, G1), NewMove(F6, G8),
		}

		for i := 1; i <= 4; i++ {
			for _, m := range cycle {
				p.MakeMove(m)
			}
			if got := p.RepetitionCount(); got != i {
				t.Fatalf("after %d cycles: RepetitionCount() = %d, want %d", i, got, i)
			}
			wantFivefold := i >= 4
			if got := p.IsFivefoldRepetition(); got != wantFivefold {
				t.Errorf("after %d cycles: IsFivefoldRepetition() = %v, want %v", i, got, wantFivefold)
			}
		}
	})
}
