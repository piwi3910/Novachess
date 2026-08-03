package search

import (
	"context"
	"testing"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// TestSelfPlayGameCompletes plays the engine against itself to a finished game.
//
// This is the closest thing to an end-to-end test the engine has, and it is
// exactly what a self-play worker will do millions of times. It exercises
// search, evaluation, make/unmake, repetition detection and the termination
// rules together, over hundreds of consecutive positions, in the one
// configuration that actually matters for the training pipeline.
//
// Everything asserted here is a property the pipeline depends on: every move
// played is legal, the game reaches a definite result rather than running
// forever, and the result is one the trainer can use as a label.
func TestSelfPlayGameCompletes(t *testing.T) {
	const (
		games        = 4
		nodesPerMove = 3000
		plyLimit     = 400
	)

	for game := 0; game < games; game++ {
		p := chess.NewPosition()
		// One searcher per side, each with its own transposition table, which
		// is how the gatekeeper will run its matches.
		white := New(eval.NewHCE(), 4)
		black := New(eval.NewHCE(), 4)

		var moves int
		var outcome chess.Outcome
		var reason chess.TerminationReason

		for ply := 0; ply < plyLimit; ply++ {
			outcome, reason = p.Outcome()
			if outcome != chess.Ongoing {
				break
			}

			engine := white
			if p.SideToMove() == chess.Black {
				engine = black
			}

			res := engine.Search(context.Background(), p, Limits{Nodes: nodesPerMove}, nil)
			if res.Best.IsNone() {
				t.Fatalf("game %d ply %d: no move returned in an ongoing position %s",
					game, ply, p.FEN())
			}

			// The returned move must be legal in the position it was searched
			// from. A search that hands back a move from a stale or corrupted
			// board would poison every game it produced.
			var legal chess.MoveList
			p.GenerateLegalMoves(&legal)
			found := false
			for _, m := range legal.Slice() {
				if m == res.Best {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("game %d ply %d: %s is not legal in %s", game, ply, res.Best, p.FEN())
			}

			p.MakeMove(res.Best)
			moves++
		}

		if outcome == chess.Ongoing {
			t.Errorf("game %d did not finish within %d plies (last position %s)",
				game, plyLimit, p.FEN())
			continue
		}
		if score := outcome.Score(); score != 0 && score != 0.5 && score != 1 {
			t.Errorf("game %d produced an unusable label %v", game, score)
		}
		t.Logf("game %d: %d plies, %v by %v", game, moves, outcome, reason)
	}
}

// TestSelfPlayIsReproducible is the property that makes distributed self-play
// possible at all. The same game, replayed with the same fixed node budget,
// must produce the same moves — otherwise a work unit redelivered to a
// different worker yields different training data than the one it replaced.
func TestSelfPlayIsReproducible(t *testing.T) {
	play := func() []string {
		p := chess.NewPosition()
		white := New(eval.NewHCE(), 4)
		black := New(eval.NewHCE(), 4)

		var line []string
		for ply := 0; ply < 30; ply++ {
			if outcome, _ := p.Outcome(); outcome != chess.Ongoing {
				break
			}
			engine := white
			if p.SideToMove() == chess.Black {
				engine = black
			}
			res := engine.Search(context.Background(), p, Limits{Nodes: 2000}, nil)
			if res.Best.IsNone() {
				break
			}
			line = append(line, p.MoveToUCI(res.Best))
			p.MakeMove(res.Best)
		}
		return line
	}

	first := play()
	if len(first) < 10 {
		t.Fatalf("game was too short to be a meaningful check: %v", first)
	}

	for run := 1; run < 3; run++ {
		again := play()
		if len(again) != len(first) {
			t.Fatalf("run %d played %d moves, first run played %d", run, len(again), len(first))
		}
		for i := range first {
			if again[i] != first[i] {
				t.Fatalf("run %d diverged at ply %d: %s vs %s\n  first: %v\n  again: %v",
					run, i, again[i], first[i], first, again)
			}
		}
	}
}
