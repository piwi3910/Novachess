package chess

import (
	"testing"
)

// The reference node counts below are the published values for the standard
// perft positions. They are not approximations and there is no tolerance: a
// move generator either reproduces them exactly or it is wrong.
//
// Each position targets a different cluster of rules. Position 3 is dense with
// en passant and promotion interactions; positions 4 and 5 are notorious for
// exposing castling-rights and promotion bugs; position 6 is a quiet middlegame
// that catches ordinary generation errors.
var perftTests = []struct {
	name   string
	fen    string
	counts []uint64 // counts[i] is perft at depth i+1
}{
	{
		name:   "starting position",
		fen:    StartingFEN,
		counts: []uint64{20, 400, 8902, 197281, 4865609, 119060324},
	},
	{
		name:   "kiwipete",
		fen:    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		counts: []uint64{48, 2039, 97862, 4085603, 193690690},
	},
	{
		name:   "position 3 (en passant and promotion heavy)",
		fen:    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		counts: []uint64{14, 191, 2812, 43238, 674624, 11030083},
	},
	{
		name:   "position 4 (promotions, white to move)",
		fen:    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
		counts: []uint64{6, 264, 9467, 422333, 15833292},
	},
	{
		name:   "position 4 mirrored (black to move)",
		fen:    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
		counts: []uint64{6, 264, 9467, 422333, 15833292},
	},
	{
		name:   "position 5",
		fen:    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
		counts: []uint64{44, 1486, 62379, 2103487, 89941194},
	},
	{
		name:   "position 6",
		fen:    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
		counts: []uint64{46, 2079, 89890, 3894594, 164075551},
	},
}

// maxPerftNodes bounds how much work the default test run does. Deeper counts
// are still checked, but only under -perft-full, so that `go test ./...` stays
// fast enough to run on every commit while CI can still go deep.
const maxPerftNodes = 5_000_000

func TestPerft(t *testing.T) {
	for _, tc := range perftTests {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}

			for i, want := range tc.counts {
				depth := i + 1
				if want > maxPerftNodes && testing.Short() {
					t.Logf("skipping depth %d (%d nodes) in short mode", depth, want)
					break
				}
				if want > maxPerftNodes && !*perftFull {
					t.Logf("skipping depth %d (%d nodes); use -perft-full to run", depth, want)
					break
				}

				got := Perft(p, depth)
				if got != want {
					// A divide breakdown is the only practical way to localize
					// the discrepancy, so surface it on failure.
					divide, _ := PerftDivide(p, depth)
					t.Errorf("perft(%d) = %d, want %d\ndivide: %v", depth, got, want, divide)
					return
				}
			}
		})
	}
}

// TestPerftLeavesPositionUnchanged verifies that make/unmake is exactly
// reversible. Perft would still produce correct counts if unmake restored an
// equivalent-but-different position, so this is checked independently: a subtle
// corruption here would surface much later as inexplicable search behavior.
func TestPerftLeavesPositionUnchanged(t *testing.T) {
	for _, tc := range perftTests {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}

			before := *p
			beforeFEN := p.FEN()

			Perft(p, 3)

			if got := p.FEN(); got != beforeFEN {
				t.Errorf("FEN after perft = %q, want %q", got, beforeFEN)
			}
			if p.key != before.key {
				t.Errorf("key after perft = %#x, want %#x", p.key, before.key)
			}
			if p.byColor != before.byColor || p.byType != before.byType {
				t.Error("bitboards not restored after perft")
			}
			if p.board != before.board {
				t.Error("mailbox not restored after perft")
			}
			if len(p.history) != len(before.history) {
				t.Errorf("history depth = %d, want %d", len(p.history), len(before.history))
			}
		})
	}
}

// TestZobristIncrementalMatchesRecomputed checks that the incrementally
// maintained hash agrees with a from-scratch recomputation at every node. The
// transposition table trusts this equality completely; if it ever drifts, the
// search silently reads results belonging to other positions.
func TestZobristIncrementalMatchesRecomputed(t *testing.T) {
	for _, tc := range perftTests {
		t.Run(tc.name, func(t *testing.T) {
			p, err := ParseFEN(tc.fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", tc.fen, err)
			}
			var walk func(depth int) error
			walk = func(depth int) error {
				if want := p.computeKey(); p.key != want {
					t.Errorf("incremental key %#x != recomputed %#x at %s", p.key, want, p.FEN())
					return errStop
				}
				if depth == 0 {
					return nil
				}
				var ml MoveList
				p.GenerateLegalMoves(&ml)
				for _, m := range ml.Slice() {
					p.MakeMove(m)
					err := walk(depth - 1)
					p.UnmakeMove()
					if err != nil {
						return err
					}
				}
				return nil
			}
			_ = walk(3)
		})
	}
}

// errStop unwinds the recursive walk after the first mismatch, so a broken hash
// reports one clear failure rather than thousands.
var errStop = &stopError{}

type stopError struct{}

func (*stopError) Error() string { return "stop" }

func BenchmarkPerft(b *testing.B) {
	p, err := ParseFEN(StartingFEN)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Perft(p, 4)
	}
	// Report generated nodes per second, the number that actually matters when
	// tuning the generator.
	b.SetBytes(197281)
}

func BenchmarkGenerateLegalMoves(b *testing.B) {
	p, err := ParseFEN(perftTests[1].fen) // kiwipete: 48 legal moves, wide branching
	if err != nil {
		b.Fatal(err)
	}
	var ml MoveList
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.GenerateLegalMoves(&ml)
	}
}
