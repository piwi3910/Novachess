package chess

import (
	"sort"
	"strings"
	"testing"
)

// Tests for the two facilities the search needs beyond plain move generation:
// capture-only generation for quiescence, and null moves for pruning.

// TestGenerateCapturesMatchesFilteredLegalMoves pins the capture generator to
// the full legal move list rather than to a hand-written expectation.
//
// Quiescence is where most of a search's nodes are spent, so a capture
// generator that quietly misses a move type — en passant, or an
// underpromotion — would degrade play everywhere while leaving perft
// untouched, since perft never calls it.
func TestGenerateCapturesMatchesFilteredLegalMoves(t *testing.T) {
	roots := []string{
		StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
		"8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
		"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
		"bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
	}

	rng := newXorshift(0xCA97_1234_5678_9ABC)
	positions, withCaptures := 0, 0

	for _, fen := range roots {
		p, err := ParseFEN(fen)
		if err != nil {
			t.Fatalf("ParseFEN(%q): %v", fen, err)
		}

		for ply := 0; ply < 400; ply++ {
			var all, caps MoveList
			p.GenerateLegalMoves(&all)
			p.GenerateCaptures(&caps)
			positions++

			// In check, quiescence needs every evasion, so the two lists must
			// be identical. Otherwise captures and promotions exactly.
			var want []string
			if p.InCheck() {
				want = moveStrings(all.Slice())
			} else {
				for _, m := range all.Slice() {
					if p.IsCapture(m) || m.Kind() == KindPromotion {
						want = append(want, m.String())
					}
				}
				sort.Strings(want)
			}

			got := moveStrings(caps.Slice())
			if strings.Join(got, ",") != strings.Join(want, ",") {
				t.Fatalf("captures differ at %s (in check: %v)\n  generated: %v\n  expected:  %v\n  extra:     %v\n  missing:   %v",
					p.FEN(), p.InCheck(), got, want,
					difference(got, want), difference(want, got))
			}
			if len(got) > 0 {
				withCaptures++
			}

			if all.Count == 0 || p.HalfmoveClock() >= 100 {
				break
			}
			p.MakeMove(all.Moves[rng.next()%uint64(all.Count)])
		}
	}

	// Guard the test itself: if the walk stopped producing captures, the
	// comparison above would be trivially satisfied by an empty list.
	if withCaptures < positions/10 {
		t.Errorf("only %d of %d positions had captures; coverage is too thin to be meaningful",
			withCaptures, positions)
	}
	t.Logf("checked %d positions, %d with captures available", positions, withCaptures)
}

// TestGenerateCapturesIncludesPromotionPushes checks the case that is easy to
// drop: a pawn promoting without capturing is a quiet move by every ordinary
// definition, but it swings material more than most captures and quiescence
// must see it.
func TestGenerateCapturesIncludesPromotionPushes(t *testing.T) {
	// b7 can promote by pushing to b8 or by capturing on a8 or c8.
	p, err := ParseFEN("r1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}

	var caps MoveList
	p.GenerateCaptures(&caps)

	got := map[string]bool{}
	for _, m := range caps.Slice() {
		got[p.MoveToUCI(m)] = true
	}

	for _, want := range []string{"b7b8q", "b7a8q", "b7c8q", "b7b8n", "b7a8r"} {
		if !got[want] {
			t.Errorf("promotion %s missing from captures; got %v", want, got)
		}
	}
	// A plain king move is quiet and must not appear.
	if got["e1e2"] {
		t.Error("quiet king move e1e2 generated in captures-only mode")
	}
}

// TestGenerateCapturesReturnsAllEvasionsInCheck covers the rule that keeps
// quiescence from hallucinating checkmate.
func TestGenerateCapturesReturnsAllEvasionsInCheck(t *testing.T) {
	// The white king on e1 is checked by the rook on e8. The only escapes are
	// quiet king steps; none of them is a capture.
	p, err := ParseFEN("4r3/8/8/8/8/8/8/4K2k w - - 0 1")
	if err != nil {
		t.Fatal(err)
	}
	if !p.InCheck() {
		t.Fatal("fixture should be in check")
	}

	var all, caps MoveList
	p.GenerateLegalMoves(&all)
	p.GenerateCaptures(&caps)

	if caps.Count == 0 {
		t.Fatal("no moves generated in check; quiescence would score this as mate")
	}
	if caps.Count != all.Count {
		t.Errorf("got %d evasions, want all %d legal moves", caps.Count, all.Count)
	}
}

// TestNullMoveIsReversible checks that passing the turn and taking it back
// restores the position exactly. Null-move pruning runs at nearly every node,
// so a leak here would corrupt the search everywhere.
func TestNullMoveIsReversible(t *testing.T) {
	fens := []string{
		StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		// An en passant target must be cleared by the null move and restored
		// by the unmake.
		"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
		"bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
	}

	for _, fen := range fens {
		p, err := ParseFEN(fen)
		if err != nil {
			t.Fatalf("ParseFEN(%q): %v", fen, err)
		}
		before := *p
		beforeFEN := p.FEN()

		p.MakeNullMove()

		if p.SideToMove() == before.side {
			t.Errorf("%s: side to move did not change", fen)
		}
		if p.EnPassantSquare() != NoSquare {
			t.Errorf("%s: en passant square survived the null move", fen)
		}
		if p.Key() == before.key {
			t.Errorf("%s: key unchanged after passing the turn", fen)
		}

		p.UnmakeNullMove()

		if got := p.FEN(); got != beforeFEN {
			t.Errorf("%s: FEN after unmake = %q", fen, got)
		}
		if p.Key() != before.key {
			t.Errorf("%s: key after unmake = %#x, want %#x", fen, p.Key(), before.key)
		}
		if p.board != before.board || p.byColor != before.byColor || p.byType != before.byType {
			t.Errorf("%s: board not restored after null move", fen)
		}
		if len(p.history) != len(before.history) {
			t.Errorf("%s: history depth %d, want %d", fen, len(p.history), len(before.history))
		}
	}
}

// TestNullMoveInterleavesWithRealMoves exercises the pattern the search
// actually produces: real moves and null moves nested arbitrarily.
func TestNullMoveInterleavesWithRealMoves(t *testing.T) {
	p := NewPosition()
	beforeFEN := p.FEN()
	beforeKey := p.Key()

	p.MakeMove(NewMove(E2, E4))
	p.MakeNullMove()
	p.MakeMove(NewMove(D2, D4))
	p.MakeNullMove()

	p.UnmakeNullMove()
	p.UnmakeMove()
	p.UnmakeNullMove()
	p.UnmakeMove()

	if got := p.FEN(); got != beforeFEN {
		t.Errorf("FEN = %q, want %q", got, beforeFEN)
	}
	if p.Key() != beforeKey {
		t.Errorf("key = %#x, want %#x", p.Key(), beforeKey)
	}
	if len(p.history) != 0 {
		t.Errorf("history depth = %d, want 0", len(p.history))
	}
}

// TestNullMoveDoesNotCreateFalseRepetitions checks that repetition detection
// cannot see across a null move. The positions on either side of one are not
// reachable from each other by legal play, so counting them as repetitions
// would invent draws.
func TestNullMoveDoesNotCreateFalseRepetitions(t *testing.T) {
	p := NewPosition()
	for _, m := range []Move{
		NewMove(G1, F3), NewMove(G8, F6),
		NewMove(F3, G1), NewMove(F6, G8),
	} {
		p.MakeMove(m)
	}
	if p.RepetitionCount() != 1 {
		t.Fatalf("setup: RepetitionCount() = %d, want 1", p.RepetitionCount())
	}

	p.MakeNullMove()
	if got := p.RepetitionCount(); got != 0 {
		t.Errorf("RepetitionCount() across a null move = %d, want 0", got)
	}

	p.UnmakeNullMove()
	if got := p.RepetitionCount(); got != 1 {
		t.Errorf("RepetitionCount() after unmake = %d, want 1", got)
	}
}

func TestIsCapture(t *testing.T) {
	p, err := ParseFEN("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
	if err != nil {
		t.Fatal(err)
	}

	var ml MoveList
	p.GenerateLegalMoves(&ml)
	for _, m := range ml.Slice() {
		// The definition IsCapture must agree with: something is removed from
		// the board when this move is played.
		beforeCount := p.Occupied().Count()
		p.MakeMove(m)
		removed := beforeCount - p.Occupied().Count()
		p.UnmakeMove()

		if got := p.IsCapture(m); got != (removed == 1) {
			t.Errorf("%s: IsCapture = %v but occupancy changed by %d", m, got, removed)
		}
	}
}
