package nnue

import (
	"bytes"
	"strings"
	"sync"
	"testing"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// rng is a local deterministic generator, so that a test failure can be
// reproduced exactly and does not depend on a standard library implementation
// that is free to change between releases.
type rng struct{ state uint64 }

func newRNG(seed uint64) *rng {
	if seed == 0 {
		seed = 0x9E3779B97F4A7C15
	}
	return &rng{state: seed}
}

func (r *rng) next() uint64 {
	r.state ^= r.state >> 12
	r.state ^= r.state << 25
	r.state ^= r.state >> 27
	return r.state * 2685821657736338717
}

// randomNetwork builds a network with weights in a range a trained one would
// plausibly occupy. Random weights play terribly, which does not matter: every
// property tested here holds for any weights at all, and that is exactly what
// makes these tests check the code rather than the training.
func randomNetwork(seed uint64) *Network {
	r := newRNG(seed)
	n := NewZero()

	fill := func(dst []int16, spread int64) {
		for i := range dst {
			dst[i] = int16(int64(r.next()%uint64(2*spread+1)) - spread)
		}
	}
	fill(n.FeatureWeights, 64)
	fill(n.FeatureBias, 64)
	fill(n.OutputWeights, 64)
	n.OutputBias = int32(int64(r.next()%20001) - 10000)

	return n
}

func mustParse(t *testing.T, fen string) *chess.Position {
	t.Helper()
	p, err := chess.ParseFEN(fen)
	if err != nil {
		t.Fatalf("ParseFEN(%q): %v", fen, err)
	}
	return p
}

// testPositions spans the shapes the accumulator has to handle: a full board, a
// nearly empty one, lopsided material, and pawns on the ranks where the
// perspective flip is easiest to get wrong.
var testPositions = []string{
	chess.StartingFEN,
	"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
	"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
	"4k3/8/8/8/8/8/8/4K3 w - - 0 1",
	"8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
	"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
	"8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
	"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
}

// mirrorFEN reflects a position vertically and swaps colors, giving a position
// that is strategically identical for the other side.
func mirrorFEN(t *testing.T, fen string) string {
	t.Helper()

	fields := strings.Fields(fen)
	if len(fields) < 4 {
		t.Fatalf("mirrorFEN: malformed FEN %q", fen)
	}

	swapCase := func(s string) string {
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

	castling := fields[2]
	if castling != "-" {
		castling = swapCase(castling)
		var b strings.Builder
		for _, want := range "KQkq" {
			if strings.ContainsRune(castling, want) {
				b.WriteRune(want)
			}
		}
		castling = b.String()
	}

	ep := fields[3]
	if ep != "-" && len(ep) == 2 {
		// Rank 3 and rank 6 are each other's reflection, and they are the only
		// ranks an en passant target can occupy.
		flipped := map[byte]byte{'3': '6', '6': '3'}[ep[1]]
		if flipped == 0 {
			t.Fatalf("mirrorFEN: en passant square %q is not on rank 3 or 6", ep)
		}
		ep = string([]byte{ep[0], flipped})
	}

	return strings.Join(ranks, "/") + " " + side + " " + castling + " " + ep + " 0 1"
}

// TestMirrorSymmetry is the load-bearing test for the feature indexing.
//
// A perspective network is symmetric by construction: mirroring a position
// swaps the two accumulator halves and swaps the side to move, so the output
// layer sees the identical pair of inputs and must return the identical number.
// This holds for every possible set of weights, which is what makes it a test
// of the code and not of the training — and it fails loudly if the rank flip or
// the color swap in FeatureIndex is wrong, which is the mistake this
// architecture invites.
func TestMirrorSymmetry(t *testing.T) {
	e, err := NewEvaluator(randomNetwork(0xA5A5))
	if err != nil {
		t.Fatal(err)
	}

	for _, fen := range testPositions {
		t.Run(fen, func(t *testing.T) {
			pos := mustParse(t, fen)
			mirrored := mustParse(t, mirrorFEN(t, fen))

			got, want := e.Evaluate(mirrored), e.Evaluate(pos)
			if got != want {
				t.Errorf("mirrored position evaluates to %d, original to %d\n  original: %s\n  mirrored: %s",
					got, want, fen, mirrorFEN(t, fen))
			}
		})
	}
}

// TestMirrorSymmetryOverRandomPlay extends the same check past hand-picked
// positions, where the interesting cases — promotions, lopsided material,
// nearly empty boards — arise on their own.
func TestMirrorSymmetryOverRandomPlay(t *testing.T) {
	e, err := NewEvaluator(randomNetwork(0x1234_5678))
	if err != nil {
		t.Fatal(err)
	}

	r := newRNG(0xBEEF)
	checked := 0

	for game := 0; game < 15; game++ {
		pos := chess.NewPosition()
		for ply := 0; ply < 90; ply++ {
			moves := pos.LegalMoves()
			if len(moves) == 0 {
				break
			}

			mirrored := mustParse(t, mirrorFEN(t, pos.FEN()))
			if got, want := e.Evaluate(mirrored), e.Evaluate(pos); got != want {
				t.Fatalf("asymmetry at %s: mirrored %d, original %d", pos.FEN(), got, want)
			}
			checked++

			pos.MakeMove(moves[r.next()%uint64(len(moves))])
		}
	}
	t.Logf("checked %d positions for mirror symmetry", checked)
}

// TestIncrementalMatchesRefresh is the invariant the incremental accumulator
// exists for. Adding and removing individual features must land on exactly the
// value a full recomputation would give — if it drifts, the search evaluates
// positions that do not exist, and the only symptom is bad play.
func TestIncrementalMatchesRefresh(t *testing.T) {
	n := randomNetwork(0xFEED)
	r := newRNG(0x2468)
	checked := 0

	for game := 0; game < 20; game++ {
		pos := chess.NewPosition()

		var incremental Accumulator
		incremental.Refresh(n, pos)

		for ply := 0; ply < 60; ply++ {
			moves := pos.LegalMoves()
			if len(moves) == 0 {
				break
			}
			move := moves[r.next()%uint64(len(moves))]

			// Apply the move to the accumulator by hand, the way a search
			// driving it from make/unmake would have to.
			applyMove(n, &incremental, pos, move)
			pos.MakeMove(move)

			var fresh Accumulator
			fresh.Refresh(n, pos)

			if incremental.values != fresh.values {
				t.Fatalf("accumulator drifted after %s at %s", pos.MoveToUCI(move), pos.FEN())
			}
			// The halves are what the output layer reads, so agreement there is
			// what actually matters.
			for _, stm := range []chess.Color{chess.White, chess.Black} {
				if got, want := n.Evaluate(&incremental, stm), n.Evaluate(&fresh, stm); got != want {
					t.Fatalf("evaluation differs for %v at %s: incremental %d, refreshed %d",
						stm, pos.FEN(), got, want)
				}
			}
			checked++
		}
	}
	t.Logf("%d incremental updates matched a full refresh exactly", checked)
}

// applyMove updates an accumulator for a move about to be played, mirroring
// what the search would do around MakeMove. It is written here rather than in
// the package because wiring it into make/unmake is a separate change; keeping
// it in the test proves the primitives are right first.
func applyMove(n *Network, a *Accumulator, pos *chess.Position, m chess.Move) {
	from, to := m.From(), m.To()
	piece := pos.PieceAt(from)
	us := pos.SideToMove()

	switch m.Kind() {
	case chess.KindCastling:
		// The destination is the rook's square, not the king's. Both pieces are
		// lifted before either is placed, because in Chess960 their origins and
		// destinations overlap freely — the king may not move at all, and king
		// and rook may swap squares.
		rook := pos.PieceAt(to)
		kingTo, rookTo := chess.CastlingDestinations(us, to > from)

		a.Remove(n, piece, from)
		a.Remove(n, rook, to)
		a.Add(n, piece, kingTo)
		a.Add(n, rook, rookTo)

	case chess.KindEnPassant:
		// The captured pawn sits beside the destination, not on it.
		captured := chess.Square(int(to) - chess.PawnPush(us))
		a.Remove(n, chess.MakePiece(us.Opposite(), chess.Pawn), captured)
		a.Move(n, piece, from, to)

	case chess.KindPromotion:
		if captured := pos.PieceAt(to); captured != chess.NoPiece {
			a.Remove(n, captured, to)
		}
		a.Remove(n, piece, from)
		a.Add(n, chess.MakePiece(us, m.Promotion()), to)

	default:
		if captured := pos.PieceAt(to); captured != chess.NoPiece {
			a.Remove(n, captured, to)
		}
		a.Move(n, piece, from, to)
	}
}

// TestRefreshIsOrderIndependent checks that the accumulator does not depend on
// the order pieces are added in, which is what lets an incremental update apply
// a move's changes in any convenient sequence.
func TestRefreshIsOrderIndependent(t *testing.T) {
	n := randomNetwork(0x77)
	pos := mustParse(t, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")

	var forward Accumulator
	forward.Refresh(n, pos)

	// Add the same pieces in reverse square order.
	var reverse Accumulator
	for c := range reverse.values {
		for i := 0; i < HiddenSize; i++ {
			reverse.values[c][i] = int32(n.FeatureBias[i])
		}
	}
	squares := []chess.Square{}
	for bb := pos.Occupied(); bb != 0; {
		var sq chess.Square
		sq, bb = bb.PopFirst()
		squares = append(squares, sq)
	}
	for i := len(squares) - 1; i >= 0; i-- {
		reverse.Add(n, pos.PieceAt(squares[i]), squares[i])
	}

	if forward.values != reverse.values {
		t.Error("accumulator depends on the order pieces are added in")
	}
}

// TestAddRemoveAreInverses checks the pair the incremental path is built from.
func TestAddRemoveAreInverses(t *testing.T) {
	n := randomNetwork(0x99)
	pos := chess.NewPosition()

	var acc Accumulator
	acc.Refresh(n, pos)
	before := acc.values

	acc.Add(n, chess.WhiteQueen, chess.E4)
	if acc.values == before {
		t.Fatal("adding a piece changed nothing")
	}
	acc.Remove(n, chess.WhiteQueen, chess.E4)

	if acc.values != before {
		t.Error("removing a piece did not undo adding it")
	}
}

// TestMoveMatchesRemoveThenAdd checks the shortcut against the long way round.
func TestMoveMatchesRemoveThenAdd(t *testing.T) {
	n := randomNetwork(0xABC)
	pos := chess.NewPosition()

	var viaMove, viaPair Accumulator
	viaMove.Refresh(n, pos)
	viaPair.Refresh(n, pos)

	viaMove.Move(n, chess.WhiteKnight, chess.G1, chess.F3)
	viaPair.Remove(n, chess.WhiteKnight, chess.G1)
	viaPair.Add(n, chess.WhiteKnight, chess.F3)

	if viaMove.values != viaPair.values {
		t.Error("Move disagrees with Remove followed by Add")
	}
}

func TestNetworkRoundTrip(t *testing.T) {
	original := randomNetwork(0xDEAD)

	var buf bytes.Buffer
	written, err := original.WriteTo(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if want := int64(headerSize + 2*(InputSize*HiddenSize+HiddenSize+2*HiddenSize) + 4); written != want {
		t.Errorf("wrote %d bytes, want %d", written, want)
	}
	if int64(buf.Len()) != written {
		t.Errorf("reported %d bytes written but the buffer holds %d", written, buf.Len())
	}

	loaded, err := ReadFrom(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(int16Bytes(loaded.FeatureWeights), int16Bytes(original.FeatureWeights)) {
		t.Error("feature weights changed through the file")
	}
	if !bytes.Equal(int16Bytes(loaded.FeatureBias), int16Bytes(original.FeatureBias)) {
		t.Error("feature bias changed through the file")
	}
	if !bytes.Equal(int16Bytes(loaded.OutputWeights), int16Bytes(original.OutputWeights)) {
		t.Error("output weights changed through the file")
	}
	if loaded.OutputBias != original.OutputBias {
		t.Errorf("output bias = %d, want %d", loaded.OutputBias, original.OutputBias)
	}

	// The point of a round trip is that it evaluates the same, not merely that
	// the bytes match.
	before, _ := NewEvaluator(original)
	after, _ := NewEvaluator(loaded)
	for _, fen := range testPositions {
		pos := mustParse(t, fen)
		if got, want := after.Evaluate(pos), before.Evaluate(pos); got != want {
			t.Errorf("%s evaluates to %d after a round trip, %d before", fen, got, want)
		}
	}
}

func int16Bytes(v []int16) []byte {
	out := make([]byte, 0, 2*len(v))
	for _, x := range v {
		out = append(out, byte(x), byte(x>>8))
	}
	return out
}

// TestReadRejectsForeignNetworks covers why the file carries a header. A
// network from a build with different dimensions would load without complaint
// and evaluate nonsense, and the gatekeeper would reject it for reasons nobody
// could explain.
func TestReadRejectsForeignNetworks(t *testing.T) {
	good := func() []byte {
		var buf bytes.Buffer
		if _, err := randomNetwork(1).WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}

	cases := []struct {
		name    string
		corrupt func([]byte) []byte
	}{
		{"empty", func([]byte) []byte { return nil }},
		{"truncated header", func(b []byte) []byte { return b[:10] }},
		{"wrong magic", func(b []byte) []byte { c := append([]byte(nil), b...); c[0] = 'X'; return c }},
		{"future version", func(b []byte) []byte { c := append([]byte(nil), b...); c[8] = 99; return c }},
		{"wrong input size", func(b []byte) []byte { c := append([]byte(nil), b...); c[12] = 1; return c }},
		{"wrong hidden size", func(b []byte) []byte { c := append([]byte(nil), b...); c[16] = 7; return c }},
		{"truncated weights", func(b []byte) []byte { return b[:len(b)-4] }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ReadFrom(bytes.NewReader(tc.corrupt(good()))); err == nil {
				t.Error("loaded a network it should have refused")
			}
		})
	}
}

// TestEvaluationStaysInsideMateRange guards the boundary between a pile of
// trained numbers and the search's score conventions. The search reads large
// scores as forced mates and reasons about distance from them; a network that
// could produce one would announce mates that do not exist.
func TestEvaluationStaysInsideMateRange(t *testing.T) {
	// A network built to overflow: every weight at its maximum, so the output
	// is as large as the arithmetic can make it.
	extreme := NewZero()
	for i := range extreme.FeatureWeights {
		extreme.FeatureWeights[i] = 32767
	}
	for i := range extreme.FeatureBias {
		extreme.FeatureBias[i] = 32767
	}

	for _, sign := range []int16{32767, -32768} {
		for i := range extreme.OutputWeights {
			extreme.OutputWeights[i] = sign
		}
		extreme.OutputBias = int32(sign) * 1 << 15

		e, err := NewEvaluator(extreme)
		if err != nil {
			t.Fatal(err)
		}
		for _, fen := range testPositions {
			score := e.Evaluate(mustParse(t, fen))
			if score > EvalLimit || score < -EvalLimit {
				t.Fatalf("%s evaluates to %d, outside the %d limit", fen, score, EvalLimit)
			}
			if eval.IsMateScore(score) {
				t.Fatalf("%s evaluates to %d, which the search would read as a forced mate", fen, score)
			}
		}
	}
}

// TestZeroNetworkIsDrawish checks the degenerate case a trainer starts from.
func TestZeroNetworkIsDrawish(t *testing.T) {
	e, err := NewEvaluator(NewZero())
	if err != nil {
		t.Fatal(err)
	}
	for _, fen := range testPositions {
		if score := e.Evaluate(mustParse(t, fen)); score != 0 {
			t.Errorf("a zero network evaluates %s as %d, want 0", fen, score)
		}
	}
}

// TestEvaluatorSatisfiesTheSearchInterface is a compile-time check that this
// package can stand in for the hand-crafted evaluation without the search
// noticing.
func TestEvaluatorSatisfiesTheSearchInterface(t *testing.T) {
	var e eval.Evaluator
	var err error
	e, err = NewEvaluator(randomNetwork(3))
	if err != nil {
		t.Fatal(err)
	}
	if e.Evaluate(chess.NewPosition()) == 0 {
		t.Log("a random network happened to evaluate the opening as level")
	}
}

// TestConcurrentEvaluationIsRaceFree is why the evaluator holds no accumulator.
// The search shares one evaluator across every lazy SMP thread, so a mutable
// field here would be written by all of them at once — a data race that would
// show up only as evaluations that are wrong when threads overlap. Run under
// -race, this is the test that would catch it.
func TestConcurrentEvaluationIsRaceFree(t *testing.T) {
	e, err := NewEvaluator(randomNetwork(0x5150))
	if err != nil {
		t.Fatal(err)
	}

	positions := make([]*chess.Position, len(testPositions))
	want := make([]int, len(testPositions))
	for i, fen := range testPositions {
		positions[i] = mustParse(t, fen)
		want[i] = e.Evaluate(positions[i])
	}

	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for round := 0; round < 200; round++ {
				for i, pos := range positions {
					// Each goroutine evaluates its own copy, since Position is
					// not itself safe to share.
					if got := e.Evaluate(pos.Clone()); got != want[i] {
						t.Errorf("concurrent evaluation of %s gave %d, want %d", testPositions[i], got, want[i])
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}
