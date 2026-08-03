package chess

import "testing"

func TestOutcomeCheckmateAndStalemate(t *testing.T) {
	cases := []struct {
		name       string
		fen        string
		wantResult Outcome
		wantReason TerminationReason
	}{
		{
			name:       "fool's mate, white is mated",
			fen:        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
			wantResult: BlackWins,
			wantReason: Checkmate,
		},
		{
			name:       "back rank mate is threatened but not yet delivered",
			fen:        "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1",
			wantResult: Ongoing,
			wantReason: NotTerminated,
		},
		{
			name:       "stalemate",
			fen:        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
			wantResult: Draw,
			wantReason: Stalemate,
		},
		{
			name:       "ordinary middlegame is ongoing",
			fen:        StartingFEN,
			wantResult: Ongoing,
			wantReason: NotTerminated,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}
			gotResult, gotReason := p.Outcome()
			if gotResult != tc.wantResult || gotReason != tc.wantReason {
				t.Errorf("Outcome() = %v (%v), want %v (%v)",
					gotResult, gotReason, tc.wantResult, tc.wantReason)
			}
		})
	}
}

func TestInsufficientMaterial(t *testing.T) {
	cases := []struct {
		name string
		fen  string
		want bool
	}{
		{"bare kings", "4k3/8/8/8/8/8/8/4K3 w - - 0 1", true},
		{"king and bishop", "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1", true},
		{"king and knight", "4k3/8/8/8/8/8/8/2N1K3 w - - 0 1", true},
		{"king and knight versus king and knight", "2n1k3/8/8/8/8/8/8/2N1K3 w - - 0 1", false},
		// Both bishops on dark squares (c1 and f8): neither side can ever
		// attack a light square, so the defending king is safe forever.
		{"bishops both on dark squares", "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1", true},
		// c1 is dark and c8 is light, so between them the bishops cover both
		// color complexes and mate is possible with cooperation.
		{"bishops on opposite colors", "2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1", false},
		{"king and two knights", "4k3/8/8/8/8/8/8/1NN1K3 w - - 0 1", false},
		{"king and rook", "4k3/8/8/8/8/8/8/2R1K3 w - - 0 1", false},
		{"king and queen", "4k3/8/8/8/8/8/8/2Q1K3 w - - 0 1", false},
		{"a single pawn is sufficient", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", false},
		{"king and two bishops same side", "4k3/8/8/8/8/8/8/2BBK3 w - - 0 1", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}
			if got := p.IsInsufficientMaterial(); got != tc.want {
				t.Errorf("IsInsufficientMaterial() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestThreefoldRepetition plays a knight shuffle that returns both sides to the
// starting arrangement twice over.
func TestThreefoldRepetition(t *testing.T) {
	p := NewPosition()

	// Ng1-f3 Ng8-f6 Nf3-g1 Nf6-g8 returns to the start: that is the second
	// occurrence. Repeating the cycle produces the third.
	cycle := []Move{
		NewMove(G1, F3), NewMove(G8, F6),
		NewMove(F3, G1), NewMove(F6, G8),
	}

	if got := p.RepetitionCount(); got != 0 {
		t.Fatalf("initial RepetitionCount() = %d, want 0", got)
	}

	for _, m := range cycle {
		p.MakeMove(m)
	}
	if got := p.RepetitionCount(); got != 1 {
		t.Errorf("after one cycle RepetitionCount() = %d, want 1", got)
	}
	if !p.IsRepetition() {
		t.Error("IsRepetition() = false after one cycle, want true")
	}
	if p.IsThreefoldRepetition() {
		t.Error("IsThreefoldRepetition() = true after one cycle, want false")
	}

	for _, m := range cycle {
		p.MakeMove(m)
	}
	if got := p.RepetitionCount(); got != 2 {
		t.Errorf("after two cycles RepetitionCount() = %d, want 2", got)
	}
	if !p.IsThreefoldRepetition() {
		t.Error("IsThreefoldRepetition() = false after two cycles, want true")
	}

	result, reason := p.Outcome()
	if result != Draw || reason != ThreefoldRepetition {
		t.Errorf("Outcome() = %v (%v), want Draw (threefold repetition)", result, reason)
	}
}

// TestRepetitionResetsOnIrreversibleMove checks that a pawn move erases the
// repetition history. Positions before an irreversible move can never recur, so
// counting them would report draws that are not draws.
func TestRepetitionResetsOnIrreversibleMove(t *testing.T) {
	p := NewPosition()

	for _, m := range []Move{
		NewMove(G1, F3), NewMove(G8, F6),
		NewMove(F3, G1), NewMove(F6, G8),
	} {
		p.MakeMove(m)
	}
	if got := p.RepetitionCount(); got != 1 {
		t.Fatalf("setup: RepetitionCount() = %d, want 1", got)
	}

	// A pawn move is irreversible and resets the halfmove clock.
	p.MakeMove(NewMove(E2, E4))
	if got := p.HalfmoveClock(); got != 0 {
		t.Errorf("halfmove clock after pawn move = %d, want 0", got)
	}
	if got := p.RepetitionCount(); got != 0 {
		t.Errorf("RepetitionCount() after pawn move = %d, want 0", got)
	}
}

func TestFiftyMoveRule(t *testing.T) {
	// A position with the halfmove clock already at 99: one more reversible
	// move reaches 100 and the rule applies. The rook sits off the e-file so
	// that it is not already checking the black king.
	p, err := ParseFEN("4k3/8/8/8/8/8/3R4/4K3 w - - 99 60")
	if err != nil {
		t.Fatal(err)
	}
	if p.IsFiftyMoveRule() {
		t.Error("IsFiftyMoveRule() = true at 99 plies, want false")
	}

	p.MakeMove(NewMove(D2, D3))
	if !p.IsFiftyMoveRule() {
		t.Error("IsFiftyMoveRule() = false at 100 plies, want true")
	}
	result, reason := p.Outcome()
	if result != Draw || reason != FiftyMoveRule {
		t.Errorf("Outcome() = %v (%v), want Draw (fifty-move rule)", result, reason)
	}
}

// TestCheckmateBeatsFiftyMoveRule pins down the rule ordering. A position that
// is both checkmate and past the hundredth reversible ply is a win: the game
// ended when mate was delivered, and a draw claim cannot follow it.
func TestCheckmateBeatsFiftyMoveRule(t *testing.T) {
	p, err := ParseFEN("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 60")
	if err != nil {
		t.Fatal(err)
	}

	// Ra1-a8 is mate: the black king is boxed in by its own pawns.
	p.MakeMove(NewMove(A1, A8))

	if !p.IsFiftyMoveRule() {
		t.Fatal("setup: expected the halfmove clock to have reached 100")
	}
	if len(p.LegalMoves()) != 0 || !p.InCheck() {
		t.Fatal("setup: expected the position to be checkmate")
	}

	result, reason := p.Outcome()
	if result != WhiteWins || reason != Checkmate {
		t.Errorf("Outcome() = %v (%v), want WhiteWins (checkmate) — mate must take precedence over the fifty-move rule",
			result, reason)
	}
}

// TestStalemateBeatsInsufficientMaterial checks the same ordering on the other
// side: with no legal move, the game ended by stalemate rather than by a
// material draw claim. Both are draws, so this is about reporting the reason
// correctly in the training data rather than about the score.
func TestStalemateBeatsInsufficientMaterial(t *testing.T) {
	// Black king on h8 with g7 and g8 covered by the white king on f7, and h7
	// covered by the light-squared bishop on b1. Black is not in check and has
	// no move, and the material is K+B versus K.
	p, err := ParseFEN("7k/5K2/8/8/8/8/8/1B6 b - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	if len(p.LegalMoves()) != 0 {
		t.Fatalf("fixture is not stalemate; legal moves: %v", p.LegalMoves())
	}
	if !p.IsInsufficientMaterial() {
		t.Fatal("fixture should also be insufficient material, or it tests nothing")
	}
	_, reason := p.Outcome()
	if reason != Stalemate {
		t.Errorf("reason = %v, want stalemate", reason)
	}
}

func TestOutcomeScore(t *testing.T) {
	cases := []struct {
		outcome Outcome
		want    float64
	}{
		{WhiteWins, 1},
		{BlackWins, 0},
		{Draw, 0.5},
	}
	for _, tc := range cases {
		if got := tc.outcome.Score(); got != tc.want {
			t.Errorf("%v.Score() = %v, want %v", tc.outcome, got, tc.want)
		}
	}
}

// TestSelfPlayGamesTerminate is the property the training pipeline depends on:
// a game of random moves must reach a definite result rather than run forever.
// Without the drawing rules a shuffling game never ends, and a self-play worker
// would emit positions labelled with a result it never actually reached.
func TestSelfPlayGamesTerminate(t *testing.T) {
	rng := newXorshift(0xC0FFEE_1234_5678)

	const games = 200
	const plyLimit = 1000
	unfinished := 0

	for g := 0; g < games; g++ {
		p := NewPosition()
		finished := false

		for ply := 0; ply < plyLimit; ply++ {
			result, _ := p.Outcome()
			if result != Ongoing {
				finished = true
				break
			}
			moves := p.LegalMoves()
			p.MakeMove(moves[rng.next()%uint64(len(moves))])
		}
		if !finished {
			unfinished++
		}
	}

	if unfinished > 0 {
		t.Errorf("%d of %d random games did not terminate within %d plies", unfinished, games, plyLimit)
	}
}
