package eval

import (
	"strings"
	"testing"

	"github.com/piwi3910/novachess/internal/chess"
)

// mirrorFEN reflects a position vertically and swaps colors, producing a
// position that is strategically identical for the other side.
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

	side := "b"
	if fields[1] == "b" {
		side = "w"
	}

	castling := swapCase(fields[2])
	if castling != "-" {
		var b strings.Builder
		for _, want := range "KQkq" {
			if strings.ContainsRune(castling, want) {
				b.WriteRune(want)
			}
		}
		if b.Len() > 0 {
			castling = b.String()
		}
	}

	ep := fields[3]
	if ep != "-" && len(ep) == 2 {
		ep = string(ep[0]) + string(byte('1'+('8'-ep[1])))
	}

	return strings.Join(ranks, "/") + " " + side + " " + castling + " " + ep + " 0 1"
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

// TestEvaluationIsColorSymmetric is the single most valuable evaluation test.
//
// Chess is symmetric under reflecting the board and swapping colors, so an
// evaluation must return the same score for a position and its mirror. Almost
// every evaluation bug — a piece-square table applied without flipping, a pawn
// term that only understands White's direction of travel, a king-safety mask
// built for one back rank — shows up here and is otherwise very hard to see,
// because the engine simply plays slightly worse as one color.
func TestEvaluationIsColorSymmetric(t *testing.T) {
	e := NewHCE()

	fens := []string{
		chess.StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		"8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
		"4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
		"r2q1rk1/pp2ppbp/2n2np1/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 1",
		"8/5k2/8/8/8/8/5K2/6Q1 w - - 0 1",
		"2r3k1/5ppp/8/8/8/8/5PPP/2R3K1 w - - 0 1",
	}

	for _, fen := range fens {
		t.Run(fen, func(t *testing.T) {
			p, err := chess.ParseFEN(fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", fen, err)
			}
			mirrored := mirrorFEN(t, fen)
			q, err := chess.ParseFEN(mirrored)
			if err != nil {
				t.Fatalf("ParseFEN(mirror) = %q: %v", mirrored, err)
			}

			if got, want := e.Evaluate(q), e.Evaluate(p); got != want {
				t.Errorf("mirrored score = %d, original = %d\n  original: %s\n  mirrored: %s",
					got, want, fen, mirrored)
			}
		})
	}
}

// TestEvaluationSymmetryOverRandomPlay extends the symmetry property past a
// handful of fixtures, since the terms that break symmetry often only fire in
// positions nobody thought to write down.
func TestEvaluationSymmetryOverRandomPlay(t *testing.T) {
	e := NewHCE()
	rng := newTestRNG(0x5EED_1234_ABCD_0001)

	checked := 0
	for game := 0; game < 40; game++ {
		p, err := chess.ParseFEN(chess.StartingFEN)
		if err != nil {
			t.Fatal(err)
		}

		for ply := 0; ply < 60; ply++ {
			moves := p.LegalMoves()
			if len(moves) == 0 {
				break
			}

			fen := p.FEN()
			mirrored := mirrorFEN(t, fen)
			q, err := chess.ParseFEN(mirrored)
			if err != nil {
				// A mirrored position can be rejected only if the mirroring is
				// wrong, which is worth knowing about.
				t.Fatalf("mirror of %q = %q: %v", fen, mirrored, err)
			}
			if got, want := e.Evaluate(q), e.Evaluate(p); got != want {
				t.Fatalf("asymmetric evaluation: %d vs %d\n  original: %s\n  mirrored: %s",
					got, want, fen, mirrored)
			}
			checked++

			p.MakeMove(moves[rng.next()%uint64(len(moves))])
		}
	}
	t.Logf("checked symmetry on %d positions", checked)
}

// TestStartingPositionIsBalanced checks the obvious sanity property: the
// starting position is equal apart from the tempo bonus.
func TestStartingPositionIsBalanced(t *testing.T) {
	e := NewHCE()
	p := chess.NewPosition()

	score := e.Evaluate(p)
	if score != tempoBonus {
		t.Errorf("starting position = %d, want exactly the tempo bonus %d", score, tempoBonus)
	}
}

// TestMaterialDominatesEvaluation checks that the positional terms cannot
// outweigh a whole piece. An evaluation whose mobility or king-safety weights
// drown out material plays bizarre chess, and the failure is easy to introduce
// while tuning.
func TestMaterialDominatesEvaluation(t *testing.T) {
	e := NewHCE()

	cases := []struct {
		name    string
		fen     string
		wantMin int
	}{
		{"a queen up", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 800},
		{"a rook up", "4k3/8/8/8/8/8/8/3RK3 w - - 0 1", 400},
		{"a knight up", "4k3/8/8/8/8/8/8/3NK3 w - - 0 1", 200},
		{"three pawns up", "4k3/8/8/8/8/8/PPP5/4K3 w - - 0 1", 200},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := chess.ParseFEN(tc.fen)
			if err != nil {
				t.Fatal(err)
			}
			if got := e.Evaluate(p); got < tc.wantMin {
				t.Errorf("score = %d, want at least %d", got, tc.wantMin)
			}
		})
	}
}

// TestSideToMovePerspective pins down the negamax convention. Getting this
// backwards produces an engine that reliably plays the worst move available,
// so it is worth an explicit test rather than trusting the comment.
func TestSideToMovePerspective(t *testing.T) {
	e := NewHCE()

	// White is a queen up. With White to move the score must be strongly
	// positive; with Black to move, strongly negative.
	white, err := chess.ParseFEN("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	black, err := chess.ParseFEN("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
	if err != nil {
		t.Fatal(err)
	}

	if got := e.Evaluate(white); got <= 0 {
		t.Errorf("White to move, a queen up: score = %d, want positive", got)
	}
	if got := e.Evaluate(black); got >= 0 {
		t.Errorf("Black to move, a queen down: score = %d, want negative", got)
	}
}

// TestTaperingMovesKingTowardTheCentre checks that the phase interpolation is
// actually wired up, by confirming the king's evaluation flips sign between a
// full board and a bare endgame. If tapering were broken, one of the two king
// tables would simply never be used.
func TestTaperingMovesKingTowardTheCentre(t *testing.T) {
	e := NewHCE()

	// Bare kings and pawns: a centralized king should beat a cornered one.
	central, err := chess.ParseFEN("8/8/8/3K4/8/8/5k2/8 w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	cornered, err := chess.ParseFEN("K7/8/8/8/8/8/5k2/8 w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	if e.Evaluate(central) <= e.Evaluate(cornered) {
		t.Errorf("in the endgame a central king (%d) should beat a cornered one (%d)",
			e.Evaluate(central), e.Evaluate(cornered))
	}
}

func TestPassedPawnIsWorthMoreWhenAdvanced(t *testing.T) {
	e := NewHCE()

	near, err := chess.ParseFEN("4k3/8/8/8/8/P7/8/4K3 w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	far, err := chess.ParseFEN("4k3/8/P7/8/8/8/8/4K3 w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	if e.Evaluate(far) <= e.Evaluate(near) {
		t.Errorf("an advanced passed pawn (%d) should outscore a home one (%d)",
			e.Evaluate(far), e.Evaluate(near))
	}
}

func TestEvaluationIsDeterministic(t *testing.T) {
	e := NewHCE()
	p, err := chess.ParseFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	first := e.Evaluate(p)
	for i := 0; i < 100; i++ {
		if got := e.Evaluate(p); got != first {
			t.Fatalf("evaluation varied between calls: %d then %d", first, got)
		}
	}
}

func TestMateScoreHelpers(t *testing.T) {
	if !IsMateScore(MateScore - 5) {
		t.Error("MateScore-5 should be a mate score")
	}
	if !IsMateScore(-(MateScore - 5)) {
		t.Error("-(MateScore-5) should be a mate score")
	}
	if IsMateScore(500) {
		t.Error("an ordinary evaluation should not be a mate score")
	}
	if IsMateScore(0) {
		t.Error("zero should not be a mate score")
	}
	if got := MateDistance(MateScore - 7); got != 7 {
		t.Errorf("MateDistance = %d, want 7", got)
	}
	if got := MateDistance(-(MateScore - 7)); got != -7 {
		t.Errorf("MateDistance for a losing mate = %d, want -7", got)
	}
}

func BenchmarkEvaluate(b *testing.B) {
	e := NewHCE()
	p, err := chess.ParseFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Evaluate(p)
	}
}

// testRNG is a small deterministic generator, kept local so the tests do not
// depend on anything unexported from the chess package.
type testRNG struct{ state uint64 }

func newTestRNG(seed uint64) *testRNG { return &testRNG{state: seed} }

func (r *testRNG) next() uint64 {
	r.state ^= r.state >> 12
	r.state ^= r.state << 25
	r.state ^= r.state >> 27
	return r.state * 2685821657736338717
}
