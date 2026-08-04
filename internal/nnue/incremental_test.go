package nnue

import (
	"bufio"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// testNetwork builds a network with non-trivial, seeded random weights. Random
// weights play terribly, which does not matter here: every property below
// holds for any weights at all, which is what makes these tests check the
// code rather than the training. randomNetwork is shared with nnue_test.go.
func testNetwork(t *testing.T) *Network {
	t.Helper()
	return randomNetwork(0xC0FFEE)
}

// equal reports whether two accumulators hold identical values. Unexported
// and test-only: production code never needs to compare accumulators, only to
// keep them right.
func (a *Accumulator) equal(b *Accumulator) bool {
	return a.values == b.values
}

// top returns the accumulator at the current stack pointer, for tests that
// need to inspect it directly rather than through Evaluate.
func (e *IncrementalEvaluator) top() *Accumulator {
	return &e.stack[e.sp]
}

// TestIncrementalMatchesRefreshOnRandomPlayouts is the correctness anchor:
// integer adds are exact, so if the deltas are right the incremental
// accumulator is BIT-IDENTICAL to a from-scratch refresh at every node. Any
// divergence is a delta bug, full stop.
func TestIncrementalMatchesRefreshOnRandomPlayouts(t *testing.T) {
	net := testNetwork(t)
	for seed := int64(0); seed < 20; seed++ {
		rng := rand.New(rand.NewSource(seed))
		p := chess.NewPosition()
		ev := NewIncremental(net)
		ev.Reset(p)
		for ply := 0; ply < 120; ply++ {
			moves := p.LegalMoves()
			if len(moves) == 0 {
				break
			}
			m := moves[rng.Intn(len(moves))]
			ev.Push(p, m)
			p.MakeMove(m)
			var want Accumulator
			want.Refresh(net, p)
			if !ev.top().equal(&want) {
				t.Fatalf("seed %d ply %d move %v: incremental accumulator diverged from refresh", seed, ply, m)
			}
		}
	}
}

// playUCI drives a position through a sequence of moves given in UCI
// notation, resolving each against the position's own legal moves rather than
// trusting the text — which is what catches a typo in the sequence as a test
// failure instead of a silently wrong fixture.
func playUCI(t *testing.T, p *chess.Position, moves ...string) {
	t.Helper()
	for _, s := range moves {
		m, ok := p.ParseUCIMove(s)
		if !ok {
			t.Fatalf("%q is not a legal move at %s", s, p.FEN())
		}
		p.MakeMove(m)
	}
}

// TestPushPopRoundTrip checks that pushing a move and immediately popping it
// leaves the top of the stack bit-identical to what it was before, across the
// move kinds whose deltas are easiest to get wrong: castling, en passant and
// promotion.
//
// Every position below is reached by playing the moves in the comment from
// the starting position and resolving them against the position's own legal
// moves (see playUCI), rather than typing a FEN by hand — this repo's most
// common test failure is a bad fixture, and a played-out sequence cannot
// silently be illegal or miss the claimed move.
func TestPushPopRoundTrip(t *testing.T) {
	net := testNetwork(t)

	type fixture struct {
		name  string
		build func(t *testing.T) *chess.Position
		match func(chess.Move) bool
	}

	cases := []fixture{
		{
			name: "kingside castling available",
			// 1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 leaves White able to castle
			// kingside.
			build: func(t *testing.T) *chess.Position {
				p := chess.NewPosition()
				playUCI(t, p, "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5")
				return p
			},
			match: func(m chess.Move) bool {
				return m.Kind() == chess.KindCastling && m.To() > m.From()
			},
		},
		{
			name: "queenside castling available",
			// 1. d4 d5 2. Nc3 Nc6 3. Bf4 Bf5 4. Qd2 Qd7 leaves White able to
			// castle queenside.
			build: func(t *testing.T) *chess.Position {
				p := chess.NewPosition()
				playUCI(t, p, "d2d4", "d7d5", "b1c3", "b8c6", "c1f4", "c8f5", "d1d2", "d8d7")
				return p
			},
			match: func(m chess.Move) bool {
				return m.Kind() == chess.KindCastling && m.To() < m.From()
			},
		},
		{
			name: "en passant available",
			// 1. e4 a6 2. e5 f5 leaves White able to capture f6 en passant.
			build: func(t *testing.T) *chess.Position {
				p := chess.NewPosition()
				playUCI(t, p, "e2e4", "a7a6", "e4e5", "f7f5")
				return p
			},
			match: func(m chess.Move) bool { return m.Kind() == chess.KindEnPassant },
		},
		{
			name: "promotion available",
			// Reaching a promotion by playing out moves from the start
			// takes dozens of plies of forced captures, so this position is
			// built directly instead: a White pawn on c7, one push from
			// queening, with a Black knight on b8 it could also capture-
			// promote. The fixture is still verified below — the test fails
			// if the position's legal moves do not actually contain a
			// promotion — so a mistyped FEN cannot pass silently.
			build: func(t *testing.T) *chess.Position {
				p, err := chess.ParseFEN("1n2k3/2P5/8/8/8/8/8/4K3 w - - 0 1")
				if err != nil {
					t.Fatalf("ParseFEN: %v", err)
				}
				return p
			},
			match: func(m chess.Move) bool { return m.Kind() == chess.KindPromotion },
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.build(t)

			var m chess.Move
			found := false
			for _, cand := range p.LegalMoves() {
				if tc.match(cand) {
					m, found = cand, true
					break
				}
			}
			if !found {
				t.Fatalf("fixture at %s does not contain the claimed move among its legal moves", p.FEN())
			}

			ev := NewIncremental(net)
			ev.Reset(p)
			before := *ev.top()

			ev.Push(p, m)
			p.MakeMove(m)

			// This is the assertion with content: Push must have landed on
			// the same values a from-scratch Refresh of the post-move
			// position would produce. Checking only that Pop restores the
			// pre-move accumulator (below) would pass even if Push wrote
			// nothing at all, since Pop only moves the stack pointer and
			// never touches the entry Push should have written.
			var want Accumulator
			want.Refresh(net, p)
			if !ev.top().equal(&want) {
				t.Fatalf("push of %v did not match a fresh refresh of the resulting position %s", m, p.FEN())
			}

			ev.Pop()
			p.UnmakeMove()

			if !ev.top().equal(&before) {
				t.Fatalf("push/pop of %v did not round-trip to the pre-move accumulator", m)
			}
		})
	}
}

// TestNullMovePushPop checks that PushNull/PopNull round-trip too: a null move
// changes side to move but no feature on the board, so the top of the stack
// after PushNull must equal what it was before (Evaluate reads it from the
// new side's perspective, but the stored values themselves do not change).
func TestNullMovePushPop(t *testing.T) {
	net := testNetwork(t)
	p := chess.NewPosition()

	ev := NewIncremental(net)
	ev.Reset(p)
	before := *ev.top()

	ev.PushNull()
	if !ev.top().equal(&before) {
		t.Fatal("PushNull changed the accumulator, but a null move changes no feature")
	}
	ev.PopNull()
	if !ev.top().equal(&before) {
		t.Fatal("PushNull/PopNull did not round-trip to the pre-null accumulator")
	}
}

// TestIncrementalSatisfiesMoveAwareAndPerThread is a compile-time check that
// the incremental evaluator implements the optional interfaces the search
// depends on.
func TestIncrementalSatisfiesMoveAwareAndPerThread(t *testing.T) {
	var _ eval.Evaluator = (*IncrementalEvaluator)(nil)
	var _ eval.MoveAware = (*IncrementalEvaluator)(nil)
	var _ eval.PerThread = (*IncrementalEvaluator)(nil)
}

// TestNewThreadEvaluatorIsIndependent checks that per-thread evaluators do not
// share stack state: pushing on one must not disturb the other, which is what
// lets the search run one evaluator per lazy SMP thread over a shared network.
func TestNewThreadEvaluatorIsIndependent(t *testing.T) {
	net := testNetwork(t)
	p := chess.NewPosition()

	ev := NewIncremental(net)
	ev.Reset(p)

	other := ev.NewThreadEvaluator()
	otherIncremental, ok := other.(*IncrementalEvaluator)
	if !ok {
		t.Fatalf("NewThreadEvaluator returned %T, want *IncrementalEvaluator", other)
	}
	otherIncremental.Reset(p)

	moves := p.LegalMoves()
	if len(moves) == 0 {
		t.Fatal("starting position has no legal moves")
	}
	m := moves[0]

	ev.Push(p, m)
	p.MakeMove(m)

	var refreshed Accumulator
	refreshed.Refresh(net, p)
	if !ev.top().equal(&refreshed) {
		t.Fatal("pushing on one thread's evaluator did not update as expected")
	}

	p.UnmakeMove()
	var preMove Accumulator
	preMove.Refresh(net, p)
	if !otherIncremental.top().equal(&preMove) {
		t.Fatal("NewThreadEvaluator shared stack state with its parent")
	}
}

// moveKindLabel names the shape of a move's delta, for coverage tracking
// rather than production logic.
type moveKindLabel string

const (
	kindQuiet            moveKindLabel = "quiet"
	kindCapture          moveKindLabel = "capture"
	kindEnPassant        moveKindLabel = "en passant"
	kindCastleKingside   moveKindLabel = "castle kingside"
	kindCastleQueenside  moveKindLabel = "castle queenside"
	kindPromotion        moveKindLabel = "promotion"
	kindPromotionCapture moveKindLabel = "promotion capture"
)

// allMoveKinds is every shape Push branches on. TestIncrementalCoversAllMoveKinds
// fails unless a playout has exercised every one of them at least once.
var allMoveKinds = []moveKindLabel{
	kindQuiet, kindCapture, kindEnPassant,
	kindCastleKingside, kindCastleQueenside,
	kindPromotion, kindPromotionCapture,
}

// classifyMove labels a legal move by the delta shape Push will take for it.
// It must be called against the PRE-move position, the same requirement Push
// itself has, since a promotion capture is only visible by checking what sits
// on the destination square before the move is played.
func classifyMove(p *chess.Position, m chess.Move) moveKindLabel {
	switch m.Kind() {
	case chess.KindCastling:
		if m.To() > m.From() {
			return kindCastleKingside
		}
		return kindCastleQueenside
	case chess.KindEnPassant:
		return kindEnPassant
	case chess.KindPromotion:
		if p.PieceAt(m.To()) != chess.NoPiece {
			return kindPromotionCapture
		}
		return kindPromotion
	default:
		if p.PieceAt(m.To()) != chess.NoPiece {
			return kindCapture
		}
		return kindQuiet
	}
}

// TestIncrementalCoversAllMoveKinds makes the property test's coverage
// structural instead of incidental. TestIncrementalMatchesRefreshOnRandomPlayouts
// picks uniformly among legal moves, and instrumentation showed that across
// its 20 seeded playouts an en passant capture never once came up — a
// deliberately wrong en passant delta would have sailed through the entire
// suite. Here, whenever a legal move exists whose kind has not yet been seen,
// it is played in preference to a random one; every move actually played is
// still drawn from the position's own LegalMoves, so this cannot manufacture
// an illegal move, only steer which legal one is chosen. The test fails if
// any move kind is never exercised by the time the seed budget runs out.
func TestIncrementalCoversAllMoveKinds(t *testing.T) {
	net := testNetwork(t)
	seen := make(map[moveKindLabel]bool, len(allMoveKinds))

	for seed := int64(0); seed < 200 && len(seen) < len(allMoveKinds); seed++ {
		rng := rand.New(rand.NewSource(seed + 1_000_003))
		p := chess.NewPosition()
		ev := NewIncremental(net)
		ev.Reset(p)

		for ply := 0; ply < 150; ply++ {
			moves := p.LegalMoves()
			if len(moves) == 0 {
				break
			}

			m := moves[rng.Intn(len(moves))]
			kind := classifyMove(p, m)
			for _, cand := range moves {
				if k := classifyMove(p, cand); !seen[k] {
					m, kind = cand, k
					break
				}
			}
			seen[kind] = true

			ev.Push(p, m)
			p.MakeMove(m)

			var want Accumulator
			want.Refresh(net, p)
			if !ev.top().equal(&want) {
				t.Fatalf("seed %d ply %d move %v (%s): incremental accumulator diverged from refresh",
					seed, ply, m, kind)
			}
		}
	}

	for _, k := range allMoveKinds {
		if !seen[k] {
			t.Errorf("move kind %q was never exercised by any playout", k)
		}
	}
	if !t.Failed() {
		t.Logf("exercised all %d move kinds: %v", len(allMoveKinds), allMoveKinds)
	}
}

// TestOverflowFallsBackToRefreshWithoutPanicking is the regression test for
// C1: the overflow guard used to advance sp past the end of the stack array,
// so the very next Evaluate indexed out of bounds and panicked instead of
// falling back to a refresh. newIncrementalCapped builds an evaluator with a
// tiny stack so the overflow path is reached after a handful of plies rather
// than 256, and this test pushes past it, pops back one level while still
// overflowed, and checks Evaluate against a fresh Refresh both times.
func TestOverflowFallsBackToRefreshWithoutPanicking(t *testing.T) {
	net := testNetwork(t)
	const capacity = 4

	p := chess.NewPosition()
	ev := newIncrementalCapped(net, capacity)
	ev.Reset(p)

	// capacity+2 plies guarantees at least two Push calls land past cap-1.
	for played := 0; played < capacity+2; played++ {
		moves := p.LegalMoves()
		if len(moves) == 0 {
			t.Fatalf("ran out of legal moves after %d plies", played)
		}
		m := moves[played%len(moves)]
		ev.Push(p, m)
		p.MakeMove(m)
	}

	assertMatchesFreshRefresh := func(label string) {
		t.Helper()
		var want Accumulator
		want.Refresh(net, p)
		wantScore := net.Evaluate(&want, p.SideToMove())
		if got := ev.Evaluate(p); got != wantScore {
			t.Fatalf("%s: overflowed Evaluate = %d, want %d (fresh refresh)", label, got, wantScore)
		}
	}
	assertMatchesFreshRefresh("while overflowed")

	p.UnmakeMove()
	ev.Pop()
	assertMatchesFreshRefresh("after popping one overflowed level")
}

// chess960FENs samples starting positions from the Chess960 perft fixture
// suite (internal/chess/testdata/chess960_perft.epd), the same file
// internal/chess's own Chess960 tests load — reusing a corpus that is already
// known to be legal Chess960 positions rather than typing new ones. Each line
// there is "<FEN> ;D1 <n> ;D2 <n> ...", already eight plies into a random
// arrangement so castling is genuinely available in most of them; only the
// FEN field is needed here.
func chess960FENs(t *testing.T, count int) []string {
	t.Helper()

	f, err := os.Open("../chess/testdata/chess960_perft.epd")
	if err != nil {
		t.Fatalf("open chess960 testdata: %v", err)
	}
	defer f.Close()

	const totalArrangements = 960
	stride := totalArrangements / count
	if stride == 0 {
		stride = 1
	}

	var fens []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for line := 0; scanner.Scan(); line++ {
		text := strings.TrimSpace(scanner.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		if line%stride == 0 {
			fen := strings.TrimSpace(strings.SplitN(text, ";", 2)[0])
			fens = append(fens, fen)
			if len(fens) >= count {
				break
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("read chess960 testdata: %v", err)
	}
	if len(fens) == 0 {
		t.Fatal("no Chess960 fixtures loaded from testdata")
	}
	return fens
}

// TestIncrementalMatchesRefreshOverChess960Playouts extends the correctness
// anchor to Chess960: the castling delta is the one piece of Push that is
// genuinely different there (king and rook squares are not fixed, and their
// origins and destinations can overlap), so classical-only playouts leave it
// under-exercised. This also structurally raises how often a castling move is
// actually played, since these fixtures start with castling rights already
// live in most arrangements.
func TestIncrementalMatchesRefreshOverChess960Playouts(t *testing.T) {
	net := testNetwork(t)
	fens := chess960FENs(t, 12)

	for i, fen := range fens {
		t.Run(fen, func(t *testing.T) {
			p, err := chess.ParseFEN(fen)
			if err != nil {
				t.Fatalf("ParseFEN(%q): %v", fen, err)
			}
			if !p.IsChess960() {
				t.Fatalf("fixture %q was not recognized as Chess960", fen)
			}

			rng := rand.New(rand.NewSource(int64(i) + 42))
			ev := NewIncremental(net)
			ev.Reset(p)

			for ply := 0; ply < 80; ply++ {
				moves := p.LegalMoves()
				if len(moves) == 0 {
					break
				}
				m := moves[rng.Intn(len(moves))]
				ev.Push(p, m)
				p.MakeMove(m)

				var want Accumulator
				want.Refresh(net, p)
				if !ev.top().equal(&want) {
					t.Fatalf("ply %d move %v: incremental accumulator diverged from refresh at %s", ply, m, p.FEN())
				}
			}
		})
	}
}
