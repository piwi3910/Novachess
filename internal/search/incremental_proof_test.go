package search

import (
	"context"
	"testing"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
	"github.com/piwi3910/novachess/internal/nnue"
)

// searchRNG is a local deterministic generator, matching the one in
// internal/nnue's own tests: a test failure has to reproduce exactly, which
// rules out anything the standard library is free to change between
// releases.
type searchRNG struct{ state uint64 }

func newSearchRNG(seed uint64) *searchRNG {
	if seed == 0 {
		seed = 0x9E3779B97F4A7C15
	}
	return &searchRNG{state: seed}
}

func (r *searchRNG) next() uint64 {
	r.state ^= r.state >> 12
	r.state ^= r.state << 25
	r.state ^= r.state >> 27
	return r.state * 2685821657736338717
}

// testNetworkForSearch builds a network with weights in a plausible range.
// Random weights play badly, which does not matter here: node-count equality
// between the incremental and refresh-only paths must hold for any weights at
// all, which is exactly what makes this test check the wiring rather than the
// training.
func testNetworkForSearch(t *testing.T) *nnue.Network {
	t.Helper()
	r := newSearchRNG(1)
	n := nnue.NewZero()

	fill := func(dst []int16, spread int64) {
		for i := range dst {
			dst[i] = int16(int64(r.next()%uint64(2*spread+1)) - spread)
		}
	}
	fill(n.FeatureWeights, 64)
	fill(n.FeatureBias, 64)
	fill(n.OutputWeights, 64)
	n.OutputBias = int32(int64(r.next()%20001) - 10000)

	if err := n.Validate(); err != nil {
		t.Fatalf("invalid test network: %v", err)
	}
	return n
}

// refreshOnlyEvaluator wraps a network in exactly the full-refresh-per-call
// path evaluator.go's Evaluate has always used, and deliberately implements
// ONLY eval.Evaluator — not eval.PerThread, not eval.MoveAware. That is what
// forces the search onto the old shared, stateless path for comparison: no
// accumulator is ever carried between nodes.
type refreshOnlyEvaluator struct {
	net *nnue.Network
}

func nnueRefreshOnly(net *nnue.Network) eval.Evaluator {
	return &refreshOnlyEvaluator{net: net}
}

func (e *refreshOnlyEvaluator) Evaluate(p *chess.Position) int {
	var acc nnue.Accumulator
	acc.Refresh(e.net, p)
	return e.net.Evaluate(&acc, p.SideToMove())
}

// nnueIncremental wraps the same network in the production nnue.Evaluator,
// which now implements eval.PerThread and so drives the search onto the
// incremental accumulator path exercised by this proof.
func nnueIncremental(net *nnue.Network) eval.Evaluator {
	e, err := nnue.NewEvaluator(net)
	if err != nil {
		panic(err)
	}
	return e
}

// searchFixedDepth runs a fixed-depth, single-threaded search with the given
// evaluator and returns the node count and best move.
func searchFixedDepth(t *testing.T, fen string, depth int, evaluator eval.Evaluator) (uint64, chess.Move) {
	t.Helper()
	p := mustParse(t, fen)
	s := New(evaluator, 16)
	res := s.Search(context.Background(), p, Limits{Depth: depth}, nil)
	return res.Info.Nodes, res.Best
}

// TestIncrementalSearchIsBehaviourNeutral is the CLAUDE.md-mandated proof:
// same network, same positions, fixed depth, one thread — the node count
// and best move with the incremental evaluator must EQUAL those with a
// wrapper that forces the old full-refresh path. Node counts are compared
// between the two configurations in the same process, never against stored
// constants.
//
// Driving the accumulator by delta instead of recomputing it from scratch
// changes nothing about which nodes are visited or what the search concludes;
// if it did, that would mean the incremental path was silently scoring some
// position differently from a full refresh, which is exactly the class of
// bug this test exists to catch before it reaches self-play.
func TestIncrementalSearchIsBehaviourNeutral(t *testing.T) {
	net := testNetworkForSearch(t)

	fens := []string{
		chess.StartingFEN,
		// Middlegame, from a well-known perft/search stress position.
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		// Endgame with a rook and pawns each side.
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		// Side to move in check.
		"6k1/5ppp/8/8/8/7q/5PPP/6K1 b - - 0 1",
		// Promotion-rich endgame.
		"8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
		// Chess960 castling rights, rook and king start squares overlapping
		// with the castling destinations.
		"bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
	}

	for _, fen := range fens {
		t.Run(fen, func(t *testing.T) {
			nodesInc, moveInc := searchFixedDepth(t, fen, 6, nnueIncremental(net))
			nodesRef, moveRef := searchFixedDepth(t, fen, 6, nnueRefreshOnly(net))

			if nodesInc != nodesRef || moveInc != moveRef {
				t.Fatalf("%s: incremental (nodes %d, move %v) diverged from refresh (nodes %d, move %v)",
					fen, nodesInc, moveInc, nodesRef, moveRef)
			}
		})
	}
}

// TestFixedDepthSearchIsDeterministicWithNNUE extends the HCE determinism
// guarantee (search_test.go's TestFixedDepthSearchIsDeterministic) to the
// incremental NNUE path, since gen-1 self-play onward searches through this
// evaluator, not the HCE.
func TestFixedDepthSearchIsDeterministicWithNNUE(t *testing.T) {
	net := testNetworkForSearch(t)

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
				s := New(nnueIncremental(net), 16)
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
