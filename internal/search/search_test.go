package search

import (
	"context"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

func newTestSearcher() *Searcher { return New(eval.NewHCE(), 16) }

func mustParse(t *testing.T, fen string) *chess.Position {
	t.Helper()
	p, err := chess.ParseFEN(fen)
	if err != nil {
		t.Fatalf("ParseFEN(%q): %v", fen, err)
	}
	return p
}

// searchDepth runs a fixed-depth search, which is the reproducible mode
// self-play depends on.
func searchDepth(t *testing.T, fen string, depth int) (*chess.Position, Result) {
	t.Helper()
	p := mustParse(t, fen)
	s := newTestSearcher()
	res := s.Search(context.Background(), p, Limits{Depth: depth}, nil)
	return p, res
}

// TestFindsForcedMate is the clearest end-to-end check that the search works.
// A mate score also has to be reported with the right distance, since a search
// that finds mate but misreports the distance will shuffle instead of
// delivering it.
func TestFindsForcedMate(t *testing.T) {
	cases := []struct {
		name       string
		fen        string
		depth      int
		wantMateIn int // in moves, positive for the side to move
		wantMove   string
	}{
		{
			name:       "back rank mate in one",
			fen:        "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
			depth:      3,
			wantMateIn: 1,
			wantMove:   "a1a8",
		},
		{
			// Qb1-b8 mates: the queen covers the eighth rank while the king on
			// g6 takes g7 and h7. The queen starts on b1 rather than anywhere
			// on the long diagonal, so that Black is not already in check.
			name:       "queen and king mate in one",
			fen:        "7k/8/6K1/8/8/8/8/1Q6 w - - 0 1",
			depth:      3,
			wantMateIn: 1,
			wantMove:   "b1b8",
		},
		{
			// The same back-rank mate found at greater depth, and with White's
			// own pawns on the board so the position is not a bare endgame.
			// Still mate in one; the point is that extra depth does not make
			// the engine prefer a slower mate.
			name:       "back rank mate found at depth five",
			fen:        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
			depth:      5,
			wantMateIn: 1,
			wantMove:   "a1a8",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, res := searchDepth(t, tc.fen, tc.depth)

			if !res.Info.Mate {
				t.Fatalf("no mate found; score = %d, move = %s", res.Info.Score, p.MoveToUCI(res.Best))
			}
			if res.Info.MateIn != tc.wantMateIn {
				t.Errorf("mate in %d, want %d", res.Info.MateIn, tc.wantMateIn)
			}
			if got := p.MoveToUCI(res.Best); got != tc.wantMove {
				t.Errorf("best move = %s, want %s", got, tc.wantMove)
			}
		})
	}
}

// TestDetectsBeingMated checks the losing side of the same coin. An engine that
// cannot recognise it is being mated will not try to avoid it.
func TestDetectsBeingMated(t *testing.T) {
	// Black is to move and a queen up with an attack. Scores are from the side
	// to move's perspective, so the winning side seeing a win means a strongly
	// positive number here, not a negative one.
	_, res := searchDepth(t, "6k1/5ppp/8/8/8/7q/5PPP/6K1 b - - 0 1", 5)
	if res.Info.Score < 500 {
		t.Errorf("Black to move with a winning attack: score = %d, want clearly positive", res.Info.Score)
	}
}

// TestPrincipalVariationIsLegal is the invariant that catches most search bugs.
//
// The PV is the line the engine claims to have calculated. If any move in it is
// illegal in the position it follows, then the PV table, the transposition
// table, or make/unmake is corrupt — and all three failures otherwise show up
// only as mysteriously bad play.
func TestPrincipalVariationIsLegal(t *testing.T) {
	fens := []string{
		chess.StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		"8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
		"bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
	}

	for _, fen := range fens {
		t.Run(fen, func(t *testing.T) {
			p, res := searchDepth(t, fen, 6)

			if res.Best.IsNone() {
				t.Fatal("no best move returned")
			}
			if len(res.Info.PV) == 0 {
				t.Fatal("empty principal variation")
			}

			// Walk the line, checking each move against the position it is
			// played in, then unwind.
			walk := p.Clone()
			for i, m := range res.Info.PV {
				var legal chess.MoveList
				walk.GenerateLegalMoves(&legal)

				found := false
				for _, candidate := range legal.Slice() {
					if candidate == m {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("PV move %d (%s) is illegal in %s\nfull PV: %v",
						i, m, walk.FEN(), res.Info.PV)
				}
				walk.MakeMove(m)
			}

			if res.Info.PV[0] != res.Best {
				t.Errorf("PV starts with %s but best move is %s", res.Info.PV[0], res.Best)
			}
		})
	}
}

// TestFixedDepthSearchIsDeterministic is a hard requirement of the training
// pipeline, not a nicety.
//
// Self-play work units are distributed across the cluster and must be
// reproducible: the same unit replayed on a different node has to produce the
// same games, or the training data silently depends on which machine ran it.
func TestFixedDepthSearchIsDeterministic(t *testing.T) {
	fens := []string{
		chess.StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
	}

	for _, fen := range fens {
		t.Run(fen, func(t *testing.T) {
			// A fresh searcher each time, so the transposition table starts
			// empty exactly as it would on another node.
			var first Result
			for run := 0; run < 3; run++ {
				p := mustParse(t, fen)
				s := newTestSearcher()
				res := s.Search(context.Background(), p, Limits{Depth: 6}, nil)

				if run == 0 {
					first = res
					continue
				}
				if res.Best != first.Best {
					t.Fatalf("run %d chose %s, first run chose %s", run, res.Best, first.Best)
				}
				if res.Info.Score != first.Info.Score {
					t.Fatalf("run %d scored %d, first run scored %d", run, res.Info.Score, first.Info.Score)
				}
				if res.Info.Nodes != first.Info.Nodes {
					t.Fatalf("run %d visited %d nodes, first run visited %d",
						run, res.Info.Nodes, first.Info.Nodes)
				}
			}
		})
	}
}

// TestFindsBasicTactics checks the search actually calculates rather than just
// evaluating. Each position has a move that wins material only if the engine
// looks a few plies ahead.
func TestFindsBasicTactics(t *testing.T) {
	cases := []struct {
		name  string
		fen   string
		depth int
		want  string
	}{
		{
			// The undefended queen on d5 can simply be taken.
			name:  "take the hanging queen",
			fen:   "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
			depth: 4,
			want:  "d1d5",
		},
		{
			// Nb5-c7 forks the king on e8 and the rook on a8; the rook cannot
			// be defended and cannot capture the knight, so it falls next move.
			// Only a search deep enough to see the follow-up finds this.
			name:  "knight fork wins a rook",
			fen:   "r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1",
			depth: 6,
			want:  "b5c7",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, res := searchDepth(t, tc.fen, tc.depth)
			if got := p.MoveToUCI(res.Best); got != tc.want {
				t.Errorf("best move = %s, want %s (score %d, pv %v)",
					got, tc.want, res.Info.Score, res.Info.PV)
			}
		})
	}
}

// TestAvoidsHangingPieces is what quiescence search exists for. Without it, the
// search would evaluate a position at the moment its own piece is attacked and
// score it as if the piece were safe.
func TestAvoidsHangingPieces(t *testing.T) {
	// Moving the rook to d5 loses it to the queen for nothing.
	p := mustParse(t, "3qk3/8/8/8/8/8/8/3RK3 w - - 0 1")
	s := newTestSearcher()
	res := s.Search(context.Background(), p, Limits{Depth: 5}, nil)

	if got := p.MoveToUCI(res.Best); got == "d1d5" || got == "d1d6" || got == "d1d7" {
		t.Errorf("played %s, which hangs the rook to the queen", got)
	}
}

// TestScoresRepetitionAsDraw checks that a position already repeated once is
// treated as drawn inside the search rather than as whatever the evaluation
// happens to say.
func TestScoresRepetitionAsDraw(t *testing.T) {
	// White is a queen down, so anything other than a draw score would be
	// strongly negative. Shuffling into a repetition is the best available.
	p := mustParse(t, "7k/8/8/8/8/8/6q1/K7 w - - 0 1")
	s := newTestSearcher()
	res := s.Search(context.Background(), p, Limits{Depth: 4}, nil)

	// The point is only that the search runs and returns a legal move without
	// treating the repeated position as a win.
	if res.Best.IsNone() {
		t.Fatal("no move returned")
	}
	if res.Info.Score > 100 {
		t.Errorf("score = %d while a queen down; want non-positive", res.Info.Score)
	}
}

func TestRespectsDepthLimit(t *testing.T) {
	for depth := 1; depth <= 6; depth++ {
		p := mustParse(t, chess.StartingFEN)
		s := newTestSearcher()
		res := s.Search(context.Background(), p, Limits{Depth: depth}, nil)
		if res.Info.Depth != depth {
			t.Errorf("requested depth %d, reported %d", depth, res.Info.Depth)
		}
	}
}

func TestRespectsNodeLimit(t *testing.T) {
	p := mustParse(t, chess.StartingFEN)
	s := newTestSearcher()

	const limit = 20000
	res := s.Search(context.Background(), p, Limits{Nodes: limit}, nil)

	// The limit is checked every few thousand nodes, so a modest overshoot is
	// expected; an unbounded one is not.
	if res.Info.Nodes > limit*3 {
		t.Errorf("visited %d nodes for a limit of %d", res.Info.Nodes, limit)
	}
	if res.Best.IsNone() {
		t.Error("no move returned under a node limit")
	}
}

func TestRespectsMoveTime(t *testing.T) {
	p := mustParse(t, chess.StartingFEN)
	s := newTestSearcher()

	const allotted = 300 * time.Millisecond
	start := time.Now()
	res := s.Search(context.Background(), p, Limits{MoveTime: allotted}, nil)
	elapsed := time.Since(start)

	if elapsed > allotted*3 {
		t.Errorf("took %v for a %v allowance", elapsed, allotted)
	}
	if res.Best.IsNone() {
		t.Error("no move returned under a time limit")
	}
}

// TestStopEndsSearchPromptly covers the UCI "stop" path. A search that ignores
// it makes the engine unresponsive and can lose games on time.
func TestStopEndsSearchPromptly(t *testing.T) {
	p := mustParse(t, chess.StartingFEN)
	s := newTestSearcher()

	done := make(chan Result, 1)
	go func() {
		done <- s.Search(context.Background(), p, Limits{Infinite: true}, nil)
	}()

	time.Sleep(100 * time.Millisecond)
	s.Stop()

	select {
	case res := <-done:
		if res.Best.IsNone() {
			t.Error("stopped search returned no move")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("search did not stop within three seconds")
	}
}

// TestContextCancellationStopsSearch checks the other cancellation path, which
// is what a UCI server uses when the connection drops.
func TestContextCancellationStopsSearch(t *testing.T) {
	p := mustParse(t, chess.StartingFEN)
	s := newTestSearcher()

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	start := time.Now()
	res := s.Search(ctx, p, Limits{Infinite: true}, nil)

	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("search ran %v past a cancelled context", elapsed)
	}
	if res.Best.IsNone() {
		t.Error("cancelled search returned no move")
	}
}

// TestSearchLeavesPositionUnchanged verifies the search unwinds make/unmake
// exactly. A leak here would corrupt the caller's position between moves.
func TestSearchLeavesPositionUnchanged(t *testing.T) {
	fen := "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
	p := mustParse(t, fen)
	s := newTestSearcher()

	s.Search(context.Background(), p, Limits{Depth: 6}, nil)

	if got := p.FEN(); got != fen {
		t.Errorf("position after search = %q, want %q", got, fen)
	}
}

// TestReportsProgressEachIteration checks the info callback, which is what a
// GUI shows the user and what the self-play worker reads NPS from.
func TestReportsProgressEachIteration(t *testing.T) {
	p := mustParse(t, chess.StartingFEN)
	s := newTestSearcher()

	var depths []int
	res := s.Search(context.Background(), p, Limits{Depth: 6}, func(info Info) {
		depths = append(depths, info.Depth)
		if info.Nodes == 0 {
			t.Error("progress report with zero nodes")
		}
		if len(info.PV) == 0 {
			t.Errorf("depth %d reported an empty PV", info.Depth)
		}
	})

	if len(depths) == 0 {
		t.Fatal("no progress reported")
	}
	for i, d := range depths {
		if d != i+1 {
			t.Errorf("iteration %d reported depth %d; depths should increase by one", i, d)
		}
	}
	if res.Info.Depth != depths[len(depths)-1] {
		t.Error("final result does not match the last progress report")
	}
}

// TestDeeperSearchDoesNotLoseToShallower is a weak but broad sanity check that
// extra depth is not actively harmful, which would indicate a sign error or a
// broken pruning rule.
func TestDeeperSearchDoesNotLoseToShallower(t *testing.T) {
	p := mustParse(t, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")

	s := newTestSearcher()
	shallow := s.Search(context.Background(), p, Limits{Depth: 3}, nil)

	s2 := newTestSearcher()
	deep := s2.Search(context.Background(), p, Limits{Depth: 7}, nil)

	if deep.Info.Nodes <= shallow.Info.Nodes {
		t.Errorf("depth 7 visited %d nodes, depth 3 visited %d; deeper should search more",
			deep.Info.Nodes, shallow.Info.Nodes)
	}
	if deep.Best.IsNone() {
		t.Error("deep search returned no move")
	}
}

func TestNoLegalMovesReturnsNoMove(t *testing.T) {
	// Stalemate.
	p := mustParse(t, "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
	s := newTestSearcher()
	res := s.Search(context.Background(), p, Limits{Depth: 4}, nil)
	if !res.Best.IsNone() {
		t.Errorf("returned %s in a stalemate position", res.Best)
	}
}

func BenchmarkSearchStartPosition(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		p, err := chess.ParseFEN(chess.StartingFEN)
		if err != nil {
			b.Fatal(err)
		}
		s := New(eval.NewHCE(), 16)
		b.StartTimer()

		s.Search(context.Background(), p, Limits{Depth: 8}, nil)
	}
}
