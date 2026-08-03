package train

import (
	"bytes"
	"context"
	"math"
	"testing"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
	"github.com/piwi3910/novachess/internal/events"
	"github.com/piwi3910/novachess/internal/nnue"
)

func testParams() Params {
	p := DefaultParams()
	p.Epochs = 3
	p.BatchSize = 32
	p.ValidationFraction = 0.2
	p.Seed = 0xC0FFEE
	return p
}

// materialCorpus builds a dataset with a pattern a network can actually learn:
// the side with more material is winning. The positions are constructed rather
// than played, so the label is exact and a failure to learn is the trainer's
// fault rather than the data's.
func materialCorpus(t *testing.T, n int) []Sample {
	t.Helper()

	// Each entry is a FEN and the result it deserves, from White's point of
	// view. Extra queens make the imbalance unmistakable.
	fixtures := []struct {
		fen    string
		result Result
		score  int16
	}{
		{"4k3/8/8/8/8/8/8/3QK3 w - - 0 1", WhiteWon, 900},
		{"4k3/8/8/8/8/8/8/2Q1K3 b - - 0 1", WhiteWon, 900},
		{"3qk3/8/8/8/8/8/8/4K3 w - - 0 1", BlackWon, -900},
		{"2q1k3/8/8/8/8/8/8/4K3 b - - 0 1", BlackWon, -900},
		{"4k3/8/8/8/8/8/8/4K3 w - - 0 1", Drawn, 0},
		{"4k3/8/8/8/8/8/8/4K3 b - - 0 1", Drawn, 0},
		{"4k3/8/8/8/8/8/8/R3K3 w - - 0 1", WhiteWon, 500},
		{"r3k3/8/8/8/8/8/8/4K3 b - - 0 1", BlackWon, -500},
	}

	samples := make([]Sample, 0, n)
	for i := 0; i < n; i++ {
		f := fixtures[i%len(fixtures)]
		samples = append(samples, Sample{
			Position: mustParse(t, f.fen),
			Score:    f.score,
			Result:   f.result,
			Ply:      uint16(i % 60),
		})
	}
	return samples
}

// TestGradientsMatchNumericalDifferences is the load-bearing test for the
// backward pass.
//
// Every other test here would still pass with a subtly wrong gradient: the loss
// would go down, just to a worse place, and nothing would say so. Comparing the
// analytic gradient against a finite difference of the loss checks the
// derivative itself, parameter by parameter, and catches a wrong sign or a
// transposed index that "it converges" cannot.
func TestGradientsMatchNumericalDifferences(t *testing.T) {
	tr, err := NewTrainer(testParams())
	if err != nil {
		t.Fatal(err)
	}

	// A handful of positions with both sides to move, so the side-relative
	// orientation of the output layer is exercised in both directions.
	samples := materialCorpus(t, 8)
	data, err := tr.prepare(samples)
	if err != nil {
		t.Fatal(err)
	}

	batch := make([]int32, len(data.examples))
	for i := range batch {
		batch[i] = int32(i)
	}

	m := newModel()
	m.initialize(newRNG(42))
	grad := newModel()
	work := newWorkspace()

	grad.zero()
	tr.accumulateGradients(m, grad, data, batch, work)

	// The step has to be large enough to survive float32 rounding in the
	// forward pass and small enough that curvature does not dominate. Measured
	// across the sampled parameters, the worst relative disagreement is 9.4e-3
	// at h=1e-2, 6.4e-6 at h=1e-3 and 3.6e-5 at h=1e-4 — truncation dominates
	// above, cancellation below, and the middle is the floor.
	//
	// The tolerance sits well above that floor but far below the size of a real
	// mistake: the mutations this test is meant to catch disagree by 100% or
	// more, not by fractions of a percent.
	const h = 1e-3
	const tolerance = 1e-3

	// The reference loss is deliberately accumulated in float64 even though the
	// trainer's own arithmetic rounds through float32. A finite difference
	// subtracts two nearly equal numbers, so it loses precision exactly where
	// the subtraction happens, and computing it in float32 makes it noisier
	// rather than more faithful — measured, 3.4x worse at h=1e-3 and 6.3x worse
	// at h=1e-4. The objective the analytic gradient belongs to does differ by
	// float32 rounding, but that difference is about 1e-6 relative here, three
	// orders of magnitude below the tolerance.
	loss := func() float64 {
		var total float64
		for _, idx := range batch {
			e := data.examples[idx]
			p := sigmoid(float64(forward(m, data, e, work)))
			err := p - float64(e.target)
			total += err * err
		}
		return total
	}

	// Check a spread of each parameter section rather than all 200k of them.
	type target struct {
		name string
		get  func() *float32
		want float32
	}

	var targets []target
	for _, i := range []int{0, 1, 7, 63, 128, 255} {
		i := i
		targets = append(targets, target{"outputWeights", func() *float32 { return &m.outputWeights[i] }, grad.outputWeights[i]})
		targets = append(targets, target{"outputWeights(them)", func() *float32 { return &m.outputWeights[nnue.HiddenSize+i] }, grad.outputWeights[nnue.HiddenSize+i]})
		targets = append(targets, target{"featureBias", func() *float32 { return &m.featureBias[i] }, grad.featureBias[i]})
	}
	// Feature weights on rows that the corpus actually activates; a row no
	// example touches has a zero gradient and proves nothing.
	for _, f := range data.whiteFeatures(data.examples[0]) {
		for _, i := range []int{0, 5, 200} {
			idx := int(f)*nnue.HiddenSize + i
			targets = append(targets, target{"featureWeights", func() *float32 { return &m.featureWeights[idx] }, grad.featureWeights[idx]})
		}
	}
	targets = append(targets, target{"outputBias", func() *float32 { return &m.outputBias }, grad.outputBias})

	checked, skipped := 0, 0
	for _, tc := range targets {
		p := tc.get()
		original := *p

		*p = original + h
		up := loss()
		*p = original - h
		down := loss()
		*p = original

		numeric := (up - down) / (2 * h)
		analytic := float64(tc.want)

		// A clamped unit has a genuinely zero derivative, and the finite
		// difference across the clamp boundary is not comparable. Skip those
		// rather than pretend they agree.
		if numeric == 0 && analytic == 0 {
			skipped++
			continue
		}

		scale := math.Max(1, math.Max(math.Abs(numeric), math.Abs(analytic)))
		if math.Abs(numeric-analytic)/scale > tolerance {
			t.Errorf("%s gradient = %v, finite difference = %v", tc.name, analytic, numeric)
		}
		checked++
	}

	if checked < 10 {
		t.Fatalf("only %d gradients were non-zero (%d skipped); the check is not exercising the backward pass", checked, skipped)
	}
	t.Logf("%d gradients matched finite differences, %d were zero on both sides", checked, skipped)
}

// TestTrainingLearnsMaterial checks the thing the pipeline actually needs: that
// a trained network ranks a winning position above a losing one, with the sign
// the search expects.
//
// The sign is the point. The samples are White-relative and the network is
// side-to-move relative, so a reflection missed anywhere between the two would
// train a network that prefers to be down a queen — and the loss would look
// perfectly healthy while it did so.
func TestTrainingLearnsMaterial(t *testing.T) {
	p := testParams()
	p.Epochs = 60
	p.BatchSize = 64
	p.LearningRate = 0.01

	tr, err := NewTrainer(p)
	if err != nil {
		t.Fatal(err)
	}

	net, stats, err := tr.Train(context.Background(), materialCorpus(t, 512))
	if err != nil {
		t.Fatal(err)
	}

	if len(stats.EpochLoss) < 2 {
		t.Fatal("no per-epoch loss recorded")
	}
	first, last := stats.EpochLoss[0], stats.EpochLoss[len(stats.EpochLoss)-1]
	if last >= first {
		t.Errorf("loss did not fall: started at %v, ended at %v", first, last)
	}

	e, err := nnue.NewEvaluator(net)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		fen  string
		want string // "positive" or "negative", from the side to move
		why  string
	}{
		{"4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "positive", "White has a queen and is to move"},
		{"3qk3/8/8/8/8/8/8/4K3 w - - 0 1", "negative", "White is a queen down and is to move"},
		{"4k3/8/8/8/8/8/8/3QK3 b - - 0 1", "negative", "Black is a queen down and is to move"},
		{"3qk3/8/8/8/8/8/8/4K3 b - - 0 1", "positive", "Black has a queen and is to move"},
	}

	for _, tc := range cases {
		score := e.Evaluate(mustParse(t, tc.fen))
		ok := score > 0
		if tc.want == "negative" {
			ok = score < 0
		}
		if !ok {
			t.Errorf("%s: evaluated %d, want %s (%s)", tc.fen, score, tc.want, tc.why)
		}
	}

	t.Logf("loss %.5f -> %.5f over %d epochs; validation %.5f; quantization error %.1f cp; %d clipped weights",
		first, last, stats.Epochs, stats.ValidationLoss, stats.QuantizationError, stats.ClippedWeights)
}

// TestTrainingIsReproducible is the same requirement self-play has, for the
// same reason: the pipeline reruns work when a node dies, and two runs that
// disagree would make a generation impossible to reproduce or audit.
func TestTrainingIsReproducible(t *testing.T) {
	samples := materialCorpus(t, 256)

	train := func() *nnue.Network {
		tr, err := NewTrainer(testParams())
		if err != nil {
			t.Fatal(err)
		}
		net, _, err := tr.Train(context.Background(), samples)
		if err != nil {
			t.Fatal(err)
		}
		return net
	}

	a, b := train(), train()

	for i := range a.FeatureWeights {
		if a.FeatureWeights[i] != b.FeatureWeights[i] {
			t.Fatalf("feature weight %d differs between runs: %d vs %d", i, a.FeatureWeights[i], b.FeatureWeights[i])
		}
	}
	for i := range a.OutputWeights {
		if a.OutputWeights[i] != b.OutputWeights[i] {
			t.Fatalf("output weight %d differs between runs: %d vs %d", i, a.OutputWeights[i], b.OutputWeights[i])
		}
	}
	if a.OutputBias != b.OutputBias {
		t.Errorf("output bias differs between runs: %d vs %d", a.OutputBias, b.OutputBias)
	}
}

// TestDifferentSeedsTrainDifferentNetworks is the complement: reproducibility
// is only meaningful if the seed is actually reaching the initialization.
func TestDifferentSeedsTrainDifferentNetworks(t *testing.T) {
	samples := materialCorpus(t, 256)

	train := func(seed uint64) *nnue.Network {
		p := testParams()
		p.Seed = seed
		tr, err := NewTrainer(p)
		if err != nil {
			t.Fatal(err)
		}
		net, _, err := tr.Train(context.Background(), samples)
		if err != nil {
			t.Fatal(err)
		}
		return net
	}

	a, b := train(1), train(2)

	same := true
	for i := range a.FeatureWeights {
		if a.FeatureWeights[i] != b.FeatureWeights[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("two seeds produced identical networks; the seed is not reaching initialization")
	}
}

// TestTrainedNetworkKeepsMirrorSymmetry checks that training cannot break the
// architecture's guarantee. The symmetry comes from the feature indexing rather
// than the weights, so a trained network must satisfy it exactly as a random
// one does — if it does not, the trainer has written weights into a layout the
// engine reads differently.
func TestTrainedNetworkKeepsMirrorSymmetry(t *testing.T) {
	tr, err := NewTrainer(testParams())
	if err != nil {
		t.Fatal(err)
	}
	net, _, err := tr.Train(context.Background(), materialCorpus(t, 256))
	if err != nil {
		t.Fatal(err)
	}

	e, err := nnue.NewEvaluator(net)
	if err != nil {
		t.Fatal(err)
	}

	pairs := [][2]string{
		{"4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "3qk3/8/8/8/8/8/8/4K3 b - - 0 1"},
		{"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
			"rnbqk2r/pppp1ppp/5n2/2b1p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1"},
	}
	for _, pair := range pairs {
		a := e.Evaluate(mustParse(t, pair[0]))
		b := e.Evaluate(mustParse(t, pair[1]))
		if a != b {
			t.Errorf("mirrored positions evaluate differently: %d vs %d\n  %s\n  %s", a, b, pair[0], pair[1])
		}
	}
}

// TestQuantizationPreservesTheModel checks the step between "the model that was
// trained" and "the network that will play". A model whose weights do not fit
// the integer range is clipped, and a clipped network is a different network
// from the one whose loss was measured — so both the drift and the clip count
// are reported, and both are checked here.
func TestQuantizationPreservesTheModel(t *testing.T) {
	p := testParams()
	p.Epochs = 20
	p.LearningRate = 0.01

	tr, err := NewTrainer(p)
	if err != nil {
		t.Fatal(err)
	}
	_, stats, err := tr.Train(context.Background(), materialCorpus(t, 512))
	if err != nil {
		t.Fatal(err)
	}

	if stats.ClippedWeights != 0 {
		t.Errorf("%d weights were clipped; the quantized network is not the model that was trained", stats.ClippedWeights)
	}

	// Judged against the network's own output range rather than as a fixed
	// number of centipawns. Rounding the output weights at QB is inherently
	// worth a few centipawns — an absolute threshold would sit just above
	// whatever the last run happened to produce, and would fail on the next
	// change to the architecture or the corpus without anything being wrong.
	// What matters is that the drift is small next to the distinctions the
	// network is drawing.
	if stats.EvaluationScale <= 0 {
		t.Fatal("the network evaluates everything as level; there is no scale to judge quantization against")
	}
	if relative := stats.QuantizationError / stats.EvaluationScale; relative > 0.05 {
		t.Errorf("quantization moved the evaluation by %.1f cp, %.1f%% of its %.0f cp range",
			stats.QuantizationError, 100*relative, stats.EvaluationScale)
	}

	t.Logf("quantization error %.2f cp against a %.0f cp range (%.2f%%), %d clipped",
		stats.QuantizationError, stats.EvaluationScale,
		100*stats.QuantizationError/stats.EvaluationScale, stats.ClippedWeights)
}

// TestValidationSetIsHeldOut checks that held-out positions really are excluded
// from training. A validation split that leaked would report a loss that says
// nothing about generalization, which is the only thing it exists to say.
func TestValidationSetIsHeldOut(t *testing.T) {
	p := testParams()
	p.ValidationFraction = 0.25

	tr, err := NewTrainer(p)
	if err != nil {
		t.Fatal(err)
	}
	_, stats, err := tr.Train(context.Background(), materialCorpus(t, 400))
	if err != nil {
		t.Fatal(err)
	}

	if stats.TrainingSamples+stats.ValidationSamples != stats.Samples {
		t.Errorf("split accounts for %d+%d samples, want %d",
			stats.TrainingSamples, stats.ValidationSamples, stats.Samples)
	}
	if want := 100; stats.ValidationSamples != want {
		t.Errorf("held out %d samples, want %d", stats.ValidationSamples, want)
	}
	if stats.ValidationLoss == 0 {
		t.Error("no validation loss was measured")
	}
}

// TestTargetOrientation pins down the conversion that is invisible once it is
// wrong: samples are White-relative and the network is side-to-move relative.
func TestTargetOrientation(t *testing.T) {
	tr, err := NewTrainer(testParams())
	if err != nil {
		t.Fatal(err)
	}

	// The same winning-for-White position, once with each side to move.
	white := mustParse(t, "4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
	black := mustParse(t, "4k3/8/8/8/8/8/8/3QK3 b - - 0 1")

	data, err := tr.prepare([]Sample{
		{Position: white, Score: 900, Result: WhiteWon},
		{Position: black, Score: 900, Result: WhiteWon},
	})
	if err != nil {
		t.Fatal(err)
	}

	whiteTarget := data.examples[0].target
	blackTarget := data.examples[1].target

	if whiteTarget <= 0.5 {
		t.Errorf("White to move in a won position has target %v, want above 0.5", whiteTarget)
	}
	if blackTarget >= 0.5 {
		t.Errorf("Black to move in a lost position has target %v, want below 0.5", blackTarget)
	}
	if diff := math.Abs(float64(whiteTarget + blackTarget - 1)); diff > 1e-6 {
		t.Errorf("the two targets should reflect to 1, but sum to %v", whiteTarget+blackTarget)
	}
}

// TestResultWeightBlends checks that both halves of the label reach the target,
// since a blend that silently ignored one would train on half the information
// available and nothing would report it.
func TestResultWeightBlends(t *testing.T) {
	pos := "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
	// A sample whose search score and game result disagree, so the two extremes
	// are distinguishable.
	sample := Sample{Position: mustParse(t, pos), Score: 900, Result: BlackWon}

	targetAt := func(weight float64) float32 {
		p := testParams()
		p.ResultWeight = weight
		tr, err := NewTrainer(p)
		if err != nil {
			t.Fatal(err)
		}
		d, err := tr.prepare([]Sample{sample})
		if err != nil {
			t.Fatal(err)
		}
		return d.examples[0].target
	}

	scoreOnly, resultOnly, blended := targetAt(0), targetAt(1), targetAt(0.5)

	if scoreOnly <= 0.5 {
		t.Errorf("with the score alone, a +900 position has target %v, want above 0.5", scoreOnly)
	}
	if resultOnly != 0 {
		t.Errorf("with the result alone, a loss has target %v, want 0", resultOnly)
	}
	if blended <= float32(math.Min(float64(scoreOnly), float64(resultOnly))) ||
		blended >= float32(math.Max(float64(scoreOnly), float64(resultOnly))) {
		t.Errorf("blended target %v is not between %v and %v", blended, resultOnly, scoreOnly)
	}
}

func TestTrainRejectsBadParams(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Params)
	}{
		{"no epochs", func(p *Params) { p.Epochs = 0 }},
		{"no batch", func(p *Params) { p.BatchSize = 0 }},
		{"no learning rate", func(p *Params) { p.LearningRate = 0 }},
		{"result weight above one", func(p *Params) { p.ResultWeight = 1.5 }},
		{"negative result weight", func(p *Params) { p.ResultWeight = -0.1 }},
		{"no score scale", func(p *Params) { p.ScoreScale = 0 }},
		{"all data held out", func(p *Params) { p.ValidationFraction = 1 }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := testParams()
			tc.mutate(&p)
			if _, err := NewTrainer(p); err == nil {
				t.Error("accepted parameters it should have refused")
			}
		})
	}
}

func TestTrainRejectsEmptyData(t *testing.T) {
	tr, err := NewTrainer(testParams())
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := tr.Train(context.Background(), nil); err == nil {
		t.Error("trained on no samples, want an error")
	}
}

func TestTrainHonoursContextCancellation(t *testing.T) {
	p := testParams()
	p.Epochs = 10000

	tr, err := NewTrainer(p)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, _, err := tr.Train(ctx, materialCorpus(t, 64)); err == nil {
		t.Error("training ignored a cancelled context")
	}
}

// TestTrainedNetworkStaysInsideMateRange checks that a trained network cannot
// produce a score the search would read as a forced mate.
func TestTrainedNetworkStaysInsideMateRange(t *testing.T) {
	p := testParams()
	p.Epochs = 30
	p.LearningRate = 0.05

	tr, err := NewTrainer(p)
	if err != nil {
		t.Fatal(err)
	}
	net, _, err := tr.Train(context.Background(), materialCorpus(t, 256))
	if err != nil {
		t.Fatal(err)
	}

	e, err := nnue.NewEvaluator(net)
	if err != nil {
		t.Fatal(err)
	}

	r := newRNG(0x99)
	for game := 0; game < 5; game++ {
		pos := chess.NewPosition()
		for ply := 0; ply < 60; ply++ {
			moves := pos.LegalMoves()
			if len(moves) == 0 {
				break
			}
			if score := e.Evaluate(pos); eval.IsMateScore(score) {
				t.Fatalf("%s evaluates to %d, which the search would read as a forced mate", pos.FEN(), score)
			}
			pos.MakeMove(moves[r.next()%uint64(len(moves))])
		}
	}
}

// TestEndToEndFromSelfPlay is the whole loop in one test: self-play generates
// games, they survive the file format, the trainer fits a network to them, and
// the engine's evaluator loads it.
//
// Each half is covered on its own elsewhere. What this adds is the seams —
// that what the generator labels is what the trainer reads, and that what the
// trainer writes is what the engine can run. Those are the joints where two
// correct components disagree about a convention.
func TestEndToEndFromSelfPlay(t *testing.T) {
	g := NewGenerator(eval.NewHCE(), 4)
	unit := events.WorkUnit{
		ID:           "end-to-end",
		Games:        4,
		RandomPlies:  6,
		NodesPerMove: 1500,
		Seed:         0x5EED,
	}

	samples, genStats, err := g.Generate(context.Background(), unit)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) < 20 {
		t.Fatalf("self-play produced only %d samples; too few to train on", len(samples))
	}

	// Through the file, since that is how a worker hands data to the trainer.
	var buf bytes.Buffer
	w, err := NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.WriteAll(samples); err != nil {
		t.Fatal(err)
	}
	r, err := NewReader(&buf)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := r.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded) != len(samples) {
		t.Fatalf("wrote %d samples, read back %d", len(samples), len(loaded))
	}

	p := testParams()
	p.Epochs = 8
	p.BatchSize = 64
	p.LearningRate = 0.01

	tr, err := NewTrainer(p)
	if err != nil {
		t.Fatal(err)
	}
	net, stats, err := tr.Train(context.Background(), loaded)
	if err != nil {
		t.Fatal(err)
	}

	if stats.ClippedWeights != 0 {
		t.Errorf("%d weights clipped", stats.ClippedWeights)
	}

	// The engine has to be able to run it, and it has to still be a network
	// rather than a constant.
	e, err := nnue.NewEvaluator(net)
	if err != nil {
		t.Fatal(err)
	}

	distinct := map[int]bool{}
	for _, s := range loaded {
		distinct[e.Evaluate(s.Position)] = true
	}
	if len(distinct) < 2 {
		t.Error("the trained network evaluates every position identically")
	}

	t.Logf("%d games -> %d positions -> loss %.5f, %d distinct evaluations, %.0f cp range",
		genStats.Games, len(loaded), stats.TrainingLoss, len(distinct), stats.EvaluationScale)
}
