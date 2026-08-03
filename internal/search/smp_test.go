package search

import (
	"context"
	"runtime"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// Lazy SMP tests.
//
// Multithreaded search cannot be tested by comparing results, because it has
// none to compare: threads race on a shared transposition table by design, so
// two runs legitimately differ. What can be tested is that it stays correct —
// legal moves, mates still found, limits still honoured — and that turning it
// on does not disturb the single-threaded determinism the pipeline relies on.

func multiThreaded(threads int) *Searcher {
	s := New(eval.NewHCE(), 16)
	s.SetThreads(threads)
	return s
}

// TestSingleThreadRemainsDeterministic is the guarantee that must survive the
// introduction of threading.
//
// Self-play work units are replayed on different nodes of the cluster, so a
// search that varied run to run would make the training data depend on which
// machine happened to run it. Threads=1 is the contract that prevents this, and
// it is checked here explicitly rather than assumed from the earlier tests,
// because it is now a property of a code path that also supports threading.
func TestSingleThreadRemainsDeterministic(t *testing.T) {
	fens := []string{
		chess.StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
	}

	for _, fen := range fens {
		t.Run(fen, func(t *testing.T) {
			var first Result
			for run := 0; run < 3; run++ {
				p := mustParse(t, fen)
				s := multiThreaded(1)
				res := s.Search(context.Background(), p, Limits{Depth: 6}, nil)

				if run == 0 {
					first = res
					continue
				}
				if res.Best != first.Best || res.Info.Score != first.Info.Score ||
					res.Info.Nodes != first.Info.Nodes {
					t.Fatalf("run %d: move %s score %d nodes %d; first run: move %s score %d nodes %d",
						run, res.Best, res.Info.Score, res.Info.Nodes,
						first.Best, first.Info.Score, first.Info.Nodes)
				}
			}
		})
	}
}

// TestDefaultIsSingleThreaded pins the default down. A searcher that silently
// used every core would make self-play non-reproducible without anyone
// choosing that.
func TestDefaultIsSingleThreaded(t *testing.T) {
	s := New(eval.NewHCE(), 16)
	if got := s.Threads(); got != 1 {
		t.Errorf("default threads = %d, want 1", got)
	}
}

func TestSetThreadsClampsToValidRange(t *testing.T) {
	s := New(eval.NewHCE(), 16)

	s.SetThreads(0)
	if got := s.Threads(); got != 1 {
		t.Errorf("SetThreads(0) gave %d, want 1", got)
	}
	s.SetThreads(-5)
	if got := s.Threads(); got != 1 {
		t.Errorf("SetThreads(-5) gave %d, want 1", got)
	}
	s.SetThreads(MaxThreads + 100)
	if got := s.Threads(); got != MaxThreads {
		t.Errorf("SetThreads above the maximum gave %d, want %d", got, MaxThreads)
	}
	s.SetThreads(4)
	if got := s.Threads(); got != 4 {
		t.Errorf("SetThreads(4) gave %d", got)
	}
}

// TestMultiThreadedSearchIsCorrect checks that the answers stay right with
// helpers running. Run under -race this is also the check that the shared
// transposition table's unsynchronized access does not corrupt anything the
// search depends on.
func TestMultiThreadedSearchIsCorrect(t *testing.T) {
	threads := runtime.NumCPU()
	if threads < 2 {
		t.Skip("needs more than one CPU to exercise threading")
	}
	if threads > 4 {
		threads = 4
	}

	cases := []struct {
		name  string
		fen   string
		depth int
		mate  bool
	}{
		{"start position", chess.StartingFEN, 7, false},
		{"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 6, false},
		{"back rank mate", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 5, true},
		{"promotion endgame", "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1", 7, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := mustParse(t, tc.fen)
			s := multiThreaded(threads)
			res := s.Search(context.Background(), p, Limits{Depth: tc.depth}, nil)

			if res.Best.IsNone() {
				t.Fatal("no move returned")
			}

			// The move must be legal in the original position.
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
				t.Fatalf("%s is not legal in %s", res.Best, tc.fen)
			}

			// And the whole principal variation must be playable.
			walk := p.Clone()
			for i, m := range res.Info.PV {
				var ml chess.MoveList
				walk.GenerateLegalMoves(&ml)
				ok := false
				for _, c := range ml.Slice() {
					if c == m {
						ok = true
						break
					}
				}
				if !ok {
					t.Fatalf("PV move %d (%s) illegal in %s", i, m, walk.FEN())
				}
				walk.MakeMove(m)
			}

			if tc.mate && !res.Info.Mate {
				t.Errorf("mate not found with %d threads; score %d", threads, res.Info.Score)
			}
		})
	}
}

// TestMultiThreadedRespectsLimits checks that helper threads do not run past
// the point the search was supposed to end. A helper that ignored the stop flag
// would keep burning CPU after bestmove was sent, which on the cluster means
// every worker slowly starving its neighbours.
func TestMultiThreadedRespectsLimits(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("needs more than one CPU")
	}

	t.Run("move time", func(t *testing.T) {
		p := mustParse(t, chess.StartingFEN)
		s := multiThreaded(4)

		const allotted = 300 * time.Millisecond
		start := time.Now()
		s.Search(context.Background(), p, Limits{MoveTime: allotted}, nil)

		if elapsed := time.Since(start); elapsed > allotted*4 {
			t.Errorf("took %v for a %v allowance", elapsed, allotted)
		}
	})

	t.Run("stop", func(t *testing.T) {
		p := mustParse(t, chess.StartingFEN)
		s := multiThreaded(4)

		done := make(chan struct{})
		go func() {
			s.Search(context.Background(), p, Limits{Infinite: true}, nil)
			close(done)
		}()

		time.Sleep(150 * time.Millisecond)
		s.Stop()

		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("multithreaded search did not stop; a helper is ignoring the stop flag")
		}
	})
}

// TestNodeCountAggregatesAcrossThreads checks that reported nodes describe the
// whole search. Reporting one thread's share would understate the work by a
// factor of the thread count and make nps meaningless.
func TestNodeCountAggregatesAcrossThreads(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("needs more than one CPU")
	}

	p := mustParse(t, chess.StartingFEN)
	single := New(eval.NewHCE(), 16)
	singleRes := single.Search(context.Background(), p, Limits{Depth: 7}, nil)

	p = mustParse(t, chess.StartingFEN)
	multi := multiThreaded(4)
	multiRes := multi.Search(context.Background(), p, Limits{Depth: 7}, nil)

	if multiRes.Info.Nodes <= singleRes.Info.Nodes {
		t.Errorf("four threads reported %d nodes, one thread reported %d; the total should include every thread",
			multiRes.Info.Nodes, singleRes.Info.Nodes)
	}
}

// TestMultiThreadedNodeLimitCountsAllThreads checks that "go nodes N" means the
// same amount of computation regardless of thread count, rather than N per
// thread.
func TestMultiThreadedNodeLimitCountsAllThreads(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("needs more than one CPU")
	}

	p := mustParse(t, chess.StartingFEN)
	s := multiThreaded(4)

	const limit = 50000
	res := s.Search(context.Background(), p, Limits{Nodes: limit}, nil)

	// Each thread only checks the limit every couple of thousand nodes, so
	// four threads can overshoot by that much apiece; an unbounded overshoot
	// would mean the limit is being applied per thread.
	if res.Info.Nodes > limit*4 {
		t.Errorf("searched %d nodes for a limit of %d", res.Info.Nodes, limit)
	}
}

// TestMultiThreadedSelfPlayWouldNotBeReproducible documents, by observation
// rather than assertion, why self-play must not use threads.
//
// It does not fail when results happen to agree — with a shared table and a
// short search they often will. It exists so that the reasoning is written down
// next to the code, and so that anyone tempted to raise the worker thread count
// finds the explanation attached to a running test rather than only in a
// comment.
func TestMultiThreadedSelfPlayWouldNotBeReproducible(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("needs more than one CPU")
	}

	fen := "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

	var nodeCounts []uint64
	for run := 0; run < 4; run++ {
		p := mustParse(t, fen)
		s := multiThreaded(4)
		res := s.Search(context.Background(), p, Limits{MoveTime: 120 * time.Millisecond}, nil)
		nodeCounts = append(nodeCounts, res.Info.Nodes)
	}

	identical := true
	for _, n := range nodeCounts[1:] {
		if n != nodeCounts[0] {
			identical = false
			break
		}
	}
	t.Logf("four threaded runs visited %v nodes (identical: %v) — self-play uses one thread for exactly this reason",
		nodeCounts, identical)
}
