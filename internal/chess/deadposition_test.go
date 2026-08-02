package chess

import "testing"

func TestDarkSquareMaskMatchesParity(t *testing.T) {
	for s := Square(0); s < SquareCount; s++ {
		if got, want := darkSquares.Has(s), squareIsDark(s); got != want {
			t.Fatalf("%v: mask says dark=%v, parity says dark=%v", s, got, want)
		}
	}
	if darkSquares.Count() != 32 || lightSquares.Count() != 32 {
		t.Fatalf("masks split %d/%d, want 32/32", darkSquares.Count(), lightSquares.Count())
	}
	if darkSquares&lightSquares != 0 {
		t.Error("color masks overlap")
	}
	if darkSquares|lightSquares != FullBB {
		t.Error("color masks do not cover the board")
	}
}

// TestBishopsOnOneColorComplex covers the generalization past two bishops.
// Positions with three or four same-colored bishops arise from underpromotion
// and are just as dead as the two-bishop case, since bishops can only ever
// attack their own color complex however many of them there are.
func TestBishopsOnOneColorComplex(t *testing.T) {
	cases := []struct {
		name string
		fen  string
		want bool
	}{
		{
			// c1 and f8 are both dark.
			name: "two bishops, both dark",
			fen:  "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",
			want: true,
		},
		{
			// c1, f8 and a3 are all dark: three bishops, one complex.
			name: "three bishops, all dark",
			fen:  "4kb2/8/8/8/8/B7/8/2B1K3 w - - 0 1",
			want: true,
		},
		{
			// Four bishops all on light squares.
			name: "four bishops, all light",
			fen:  "3bkb2/8/8/8/8/8/8/2B1KB2 w - - 0 1",
			want: false,
		},
		{
			// One light bishop among dark ones breaks the argument: between
			// them the bishops cover both complexes.
			name: "three dark bishops and one light",
			fen:  "4kb2/8/8/8/8/B7/8/2B1KB2 w - - 0 1",
			want: false,
		},
		{
			name: "a knight alongside bishops leaves mate possible",
			fen:  "4kb2/8/8/8/8/8/8/1NB1K3 w - - 0 1",
			want: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}
			if got := p.IsInsufficientMaterial(); got != tc.want {
				t.Errorf("IsInsufficientMaterial() = %v, want %v\n%s", got, tc.want, p)
			}
		})
	}
}

// TestBlockedDeadPosition covers the locked-structure half of FIDE 5.2.2.
//
// Each "want true" case must be genuinely dead: no sequence of legal moves by
// either side can produce checkmate. Each "want false" case must be rejected,
// and those matter more — wrongly claiming a draw in a live position is a far
// worse defect than failing to claim one in a drawn position.
func TestBlockedDeadPosition(t *testing.T) {
	cases := []struct {
		name string
		fen  string
		want bool
		why  string
	}{
		{
			// The classic interlocked wall. Every pawn is blocked head-on and
			// no pawn has a capture, because the files alternate between two
			// ranks. Rank 5 is completely occupied, and every square of rank 4
			// is either a white pawn or attacked by a black one, so the white
			// king can never leave ranks 1-3; rank 6 seals the black king into
			// ranks 7-8 the same way. Neither king can ever touch a pawn.
			name: "classic interlocked wall, kings sealed apart",
			fen:  "4k3/8/p1p1p1p1/PpPpPpPp/1P1P1P1P/8/8/4K3 w - - 0 1",
			want: true,
			why:  "no pawn can ever move and neither king can reach one",
		},
		{
			// The same wall with the white king already on the far side of it,
			// where it can walk to b6 and take the undefended a6 pawn.
			name: "same wall but a king is loose among the enemy pawns",
			fen:  "8/1K2k3/p1p1p1p1/PpPpPpPp/1P1P1P1P/8/8/8 w - - 0 1",
			want: false,
			why:  "the white king reaches b6 and captures the undefended a6 pawn",
		},
		{
			// Two adjacent full ranks look like a wall but are nothing of the
			// sort: every pawn has a diagonal capture available.
			name: "adjacent full ranks offer captures everywhere",
			fen:  "4k3/8/8/pppppppp/PPPPPPPP/8/8/4K3 w - - 0 1",
			want: false,
			why:  "every pawn can capture diagonally, so nothing is frozen",
		},
		{
			// The interlocked wall with one pawn removed, opening a file.
			name: "interlocked wall with a hole",
			fen:  "4k3/8/p1p1p1p1/Pp1pPpPp/1P1P1P1P/8/8/4K3 w - - 0 1",
			want: false,
			why:  "the c-file is open, so pawns can advance and kings can cross",
		},
		{
			name: "ordinary middlegame",
			fen:  StartingFEN,
			want: false,
			why:  "nothing is blocked at all",
		},
		{
			name: "a piece on the board disqualifies the argument entirely",
			fen:  "4k3/8/p1p1p1p1/PpPpPpPp/1P1P1P1P/8/8/R3K3 w - - 0 1",
			want: false,
			why:  "a rook moves freely regardless of the pawn structure",
		},
		{
			name: "bare kings are material-dead, not blocked-dead",
			fen:  "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
			want: false,
			why:  "no pawns, so the material rule covers it instead",
		},
		{
			// A pawn "blocked" by a king is not blocked: the king steps aside.
			name: "pawn blocked only by a king",
			fen:  "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1",
			want: false,
			why:  "the black king can move and let the e2 pawn advance",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}
			if got := p.IsBlockedDeadPosition(); got != tc.want {
				t.Errorf("IsBlockedDeadPosition() = %v, want %v (%s)\n%s", got, tc.want, tc.why, p)
			}
		})
	}
}

// TestBlockedDeadPositionIsReportedByOutcome checks the detector is actually
// wired into adjudication, not merely available.
func TestBlockedDeadPositionIsReportedByOutcome(t *testing.T) {
	p, err := ParseFEN("4k3/8/p1p1p1p1/PpPpPpPp/1P1P1P1P/8/8/4K3 w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	result, reason := p.Outcome()
	if result != Draw || reason != DeadPosition {
		t.Errorf("Outcome() = %v (%v), want Draw (dead position)", result, reason)
	}
	if !p.IsDeadPosition() {
		t.Error("IsDeadPosition() = false, want true")
	}
}

// TestBlockedDeadPositionNeverClaimsALivePosition is the safety property. A
// wrong "yes" here would adjudicate a winnable game as drawn, and in self-play
// would label every position in it with the wrong result. The detector is
// therefore run over a large sample of ordinary play, where it must never fire:
// no position reachable from the opening within a few plies is dead.
func TestBlockedDeadPositionNeverClaimsALivePosition(t *testing.T) {
	rng := newXorshift(0xDEAD_9051_7104_5EED)

	const games = 300
	const plies = 60
	checked := 0

	for g := 0; g < games; g++ {
		p := NewPosition()
		for ply := 0; ply < plies; ply++ {
			moves := p.LegalMoves()
			if len(moves) == 0 {
				break
			}
			checked++

			if p.IsBlockedDeadPosition() {
				// Prove the claim false the only way that is decisive: the
				// detector says no move can ever change anything, so any pawn
				// move or capture available right now is a contradiction.
				for _, m := range moves {
					if p.PieceAt(m.From()).Type() == Pawn || p.PieceAt(m.To()) != NoPiece {
						t.Fatalf("claimed dead but %s is available:\n%s", m, p)
					}
				}
				t.Fatalf("claimed dead in a position with pieces still on the board:\n%s", p)
			}
			p.MakeMove(moves[rng.next()%uint64(len(moves))])
		}
	}
	t.Logf("detector correctly declined %d live positions", checked)
}

// TestKingReachableSquares checks the flood fill directly, since the
// correctness of the whole detector rests on it.
func TestKingReachableSquares(t *testing.T) {
	// An empty board: the king reaches everywhere.
	if got := kingReachableSquares(A1, EmptyBB); got != FullBB {
		t.Errorf("on an empty board the king reached %d squares, want 64", got.Count())
	}

	// A full-width wall on rank 4 seals ranks 1-3 off from ranks 5-8.
	wall := RankBB[3]
	got := kingReachableSquares(A1, wall)
	want := RankBB[0] | RankBB[1] | RankBB[2]
	if got != want {
		t.Errorf("king behind a rank-4 wall reached:\n%vwant:\n%v", got, want)
	}

	// A wall with one hole lets the king through to the whole board.
	holed := wall &^ E4.BB()
	got = kingReachableSquares(A1, holed)
	if got != FullBB&^holed {
		t.Errorf("king should reach every non-wall square through the gap; reached %d", got.Count())
	}
}
