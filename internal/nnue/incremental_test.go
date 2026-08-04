package nnue

import (
	"math/rand"
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
