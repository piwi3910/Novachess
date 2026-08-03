package train

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/nnue"
)

// Params configures a training run.
type Params struct {
	// Epochs is how many passes to make over the data.
	Epochs int

	// BatchSize is how many samples contribute to one weight update.
	BatchSize int

	// LearningRate is Adam's step size.
	LearningRate float64

	// ResultWeight blends the two things a sample knows. At 0 the network is
	// trained purely on the search's evaluation of the position; at 1, purely
	// on who eventually won.
	//
	// Neither extreme is what you want. The search score is precise but only as
	// good as the evaluation that produced it, so training on it alone teaches
	// the network to imitate its predecessor and the loop stops improving. The
	// game result is ground truth but extremely noisy — one position in a lost
	// game may have been winning — so training on it alone learns very slowly.
	ResultWeight float64

	// ScoreScale converts a centipawn score into a win probability through a
	// logistic curve. It is deliberately equal to the network's own output
	// scale, which is what makes the model's raw output feed the sigmoid
	// directly with no further conversion.
	ScoreScale float64

	// ValidationFraction is how much of the data to hold out. Held-out
	// positions are never trained on, so their loss says whether the network
	// has learned the game or merely memorized the file.
	ValidationFraction float64

	// Seed fixes the shuffle and the weight initialization.
	Seed uint64
}

// DefaultParams returns settings that train a usable network from a first
// generation of self-play.
func DefaultParams() Params {
	return Params{
		Epochs:             10,
		BatchSize:          1024,
		LearningRate:       0.001,
		ResultWeight:       0.5,
		ScoreScale:         nnue.Scale,
		ValidationFraction: 0.05,
		Seed:               1,
	}
}

func (p Params) validate() error {
	switch {
	case p.Epochs <= 0:
		return fmt.Errorf("train: %d epochs", p.Epochs)
	case p.BatchSize <= 0:
		return fmt.Errorf("train: batch size %d", p.BatchSize)
	case p.LearningRate <= 0:
		return fmt.Errorf("train: learning rate %v", p.LearningRate)
	case p.ResultWeight < 0 || p.ResultWeight > 1:
		return fmt.Errorf("train: result weight %v is outside 0..1", p.ResultWeight)
	case p.ScoreScale <= 0:
		return fmt.Errorf("train: score scale %v", p.ScoreScale)
	case p.ValidationFraction < 0 || p.ValidationFraction >= 1:
		return fmt.Errorf("train: validation fraction %v is outside 0..1", p.ValidationFraction)
	}
	return nil
}

// Stats reports how a training run went.
type Stats struct {
	Samples           int
	TrainingSamples   int
	ValidationSamples int
	Epochs            int
	Duration          time.Duration

	// TrainingLoss and ValidationLoss are the final epoch's mean squared error.
	TrainingLoss   float64
	ValidationLoss float64

	// EpochLoss is the training loss after each epoch, which is what shows
	// whether the run converged or diverged.
	EpochLoss []float64

	// QuantizationError is the mean absolute difference in centipawns between
	// the float model and the quantized network it was converted into,
	// measured over the validation set.
	//
	// This is reported rather than assumed because quantization is the one step
	// that can silently undo the training: a model whose weights do not fit the
	// integer range is clipped, and a clipped network plays differently from
	// the one whose loss was measured.
	QuantizationError float64

	// EvaluationScale is the mean absolute evaluation, in centipawns, over the
	// validation set.
	//
	// It is what makes QuantizationError interpretable: a few centipawns of
	// rounding is inconsequential for a network that separates positions by
	// hundreds and ruinous for one whose whole output range is ten. It is also
	// a bluntly useful health check on its own, since a network that evaluates
	// everything as nearly level has learned nothing whatever its loss says.
	EvaluationScale float64

	// ClippedWeights counts weights that did not fit the integer range. Any
	// non-zero value means the quantized network is not the model that was
	// trained.
	ClippedWeights int
}

// Trainer fits a network to self-play samples.
//
// Training runs in float32 and quantizes at the end. Integer weights are what
// the engine needs — its arithmetic has to be identical on every machine — but
// gradients through integers are meaningless, so the two representations are
// kept separate and the conversion is measured rather than trusted.
type Trainer struct {
	params Params
}

// NewTrainer returns a trainer with the given settings.
func NewTrainer(p Params) (*Trainer, error) {
	if err := p.validate(); err != nil {
		return nil, err
	}
	return &Trainer{params: p}, nil
}

// model is the float parameter set, laid out to mirror the quantized network
// exactly so the conversion is a scale factor per section and nothing else.
type model struct {
	featureWeights []float32 // InputSize rows of HiddenSize
	featureBias    []float32 // HiddenSize, shared by both halves
	outputWeights  []float32 // 2*HiddenSize: side to move's half, then the opponent's
	outputBias     float32
}

func newModel() *model {
	return &model{
		featureWeights: make([]float32, nnue.InputSize*nnue.HiddenSize),
		featureBias:    make([]float32, nnue.HiddenSize),
		outputWeights:  make([]float32, 2*nnue.HiddenSize),
	}
}

// initialize seeds the weights.
//
// The scale matters more than the distribution: too large and every activation
// starts clamped, where the gradient is exactly zero and the unit never
// recovers. Both layers are scaled by the fan-in that feeds them, which keeps
// the initial activations inside the range the clamp leaves differentiable.
func (m *model) initialize(r *rng) {
	// About 32 features are active at once, so that is the real fan-in of the
	// first layer rather than the 768 inputs it nominally has.
	featureScale := float32(1.0 / math.Sqrt(32))
	outputScale := float32(1.0 / math.Sqrt(2*nnue.HiddenSize))

	for i := range m.featureWeights {
		m.featureWeights[i] = r.normal() * featureScale
	}
	for i := range m.outputWeights {
		m.outputWeights[i] = r.normal() * outputScale
	}
	// Biases start at zero: there is no asymmetry for them to break.
}

// example is a sample reduced to what training needs. Positions are converted
// once rather than re-walked every epoch, which otherwise dominates the run.
type example struct {
	// offset and count index into the trainer's shared feature arena. The
	// White-perspective indices come first, then the Black-perspective ones,
	// each `count` long.
	offset int32
	count  int32

	stm chess.Color

	// target is the blended label, already oriented to the side to move.
	target float32
}

// dataset is the prepared corpus.
type dataset struct {
	features []int32 // arena: white indices then black indices, per example
	examples []example
}

// prepare converts samples into the training representation.
//
// The target is computed here, once, because it is the step where an
// orientation mistake is invisible afterwards. Sample scores and results are
// White-relative by the file format's definition; the network's output is
// relative to the side to move. A position with Black to move therefore needs
// its target reflected, and sigmoid(-x) = 1-sigmoid(x) makes that a
// subtraction from one.
func (t *Trainer) prepare(samples []Sample) (*dataset, error) {
	d := &dataset{
		features: make([]int32, 0, len(samples)*64),
		examples: make([]example, 0, len(samples)),
	}

	for i, s := range samples {
		if s.Position == nil {
			return nil, fmt.Errorf("train: sample %d has no position", i)
		}

		occ := s.Position.Occupied()
		count := int32(occ.Count())
		offset := int32(len(d.features))

		// White perspective first, then Black, each in the same square order so
		// the two runs line up.
		for _, perspective := range []chess.Color{chess.White, chess.Black} {
			for bb := occ; bb != 0; {
				var sq chess.Square
				sq, bb = bb.PopFirst()
				d.features = append(d.features, int32(nnue.FeatureIndex(perspective, s.Position.PieceAt(sq), sq)))
			}
		}

		// White-relative target, from the search score and the game result.
		scoreTarget := sigmoid(float64(s.Score) / t.params.ScoreScale)
		target := (1-t.params.ResultWeight)*scoreTarget + t.params.ResultWeight*s.Result.Score()

		stm := s.Position.SideToMove()
		if stm == chess.Black {
			target = 1 - target
		}

		d.examples = append(d.examples, example{
			offset: offset,
			count:  count,
			stm:    stm,
			target: float32(target),
		})
	}

	return d, nil
}

// whiteFeatures and blackFeatures return an example's active feature rows.
func (d *dataset) whiteFeatures(e example) []int32 {
	return d.features[e.offset : e.offset+e.count]
}

func (d *dataset) blackFeatures(e example) []int32 {
	return d.features[e.offset+e.count : e.offset+2*e.count]
}

// Train fits a network to the samples and returns it quantized.
func (t *Trainer) Train(ctx context.Context, samples []Sample) (*nnue.Network, Stats, error) {
	var stats Stats
	if len(samples) == 0 {
		return nil, stats, fmt.Errorf("train: no samples to train on")
	}

	start := time.Now()

	data, err := t.prepare(samples)
	if err != nil {
		return nil, stats, err
	}

	r := newRNG(t.params.Seed)

	// Shuffle before splitting, so the validation set is not simply the last
	// games generated — those share openings and opponents with each other far
	// more than with the rest, and a held-out set like that flatters the model.
	order := make([]int32, len(data.examples))
	for i := range order {
		order[i] = int32(i)
	}
	shuffle(order, r)

	validation := int(float64(len(order)) * t.params.ValidationFraction)
	if validation >= len(order) {
		validation = len(order) - 1
	}
	trainIdx, valIdx := order[validation:], order[:validation]

	if len(trainIdx) == 0 {
		return nil, stats, fmt.Errorf("train: validation fraction %v left no training data", t.params.ValidationFraction)
	}

	stats.Samples = len(samples)
	stats.TrainingSamples = len(trainIdx)
	stats.ValidationSamples = len(valIdx)

	m := newModel()
	m.initialize(r)
	opt := newAdam(t.params.LearningRate)
	grad := newModel()
	work := newWorkspace()

	epochOrder := make([]int32, len(trainIdx))
	copy(epochOrder, trainIdx)

	for epoch := 0; epoch < t.params.Epochs; epoch++ {
		if ctx.Err() != nil {
			return nil, stats, ctx.Err()
		}

		shuffle(epochOrder, r)

		// Losses are summed weighted by batch size rather than averaged over
		// batches. The last batch of an epoch is usually short, and averaging
		// per batch would give its few samples the same weight as a full
		// batch's — enough to make the reported loss jump when the batch size
		// changes even though nothing about the training did.
		var epochLoss float64
		var counted int

		for from := 0; from < len(epochOrder); from += t.params.BatchSize {
			if ctx.Err() != nil {
				return nil, stats, ctx.Err()
			}

			to := from + t.params.BatchSize
			if to > len(epochOrder) {
				to = len(epochOrder)
			}
			batch := epochOrder[from:to]

			grad.zero()
			loss := t.accumulateGradients(m, grad, data, batch, work)
			opt.step(m, grad, len(batch))

			epochLoss += loss * float64(len(batch))
			counted += len(batch)
		}

		if counted > 0 {
			epochLoss /= float64(counted)
		}
		stats.EpochLoss = append(stats.EpochLoss, epochLoss)
		stats.TrainingLoss = epochLoss
		stats.Epochs = epoch + 1
	}

	if len(valIdx) > 0 {
		stats.ValidationLoss = t.evaluateLoss(m, data, valIdx, work)
	}

	net, clipped := m.quantize()
	stats.ClippedWeights = clipped
	if len(valIdx) > 0 {
		stats.QuantizationError, stats.EvaluationScale = t.quantizationError(m, net, data, valIdx, work)
	}

	stats.Duration = time.Since(start)
	return net, stats, nil
}

// workspace holds the per-example scratch the forward and backward passes need,
// reused across examples so that training does not allocate per sample.
type workspace struct {
	accWhite []float32
	accBlack []float32
	actWhite []float32
	actBlack []float32
	dWhite   []float32
	dBlack   []float32
}

func newWorkspace() *workspace {
	return &workspace{
		accWhite: make([]float32, nnue.HiddenSize),
		accBlack: make([]float32, nnue.HiddenSize),
		actWhite: make([]float32, nnue.HiddenSize),
		actBlack: make([]float32, nnue.HiddenSize),
		dWhite:   make([]float32, nnue.HiddenSize),
		dBlack:   make([]float32, nnue.HiddenSize),
	}
}

// forward computes the model's raw output for one example.
//
// The output is in the same units the quantized network produces before its
// centipawn scaling, which is what lets it feed the sigmoid directly: the
// network's Scale and the trainer's ScoreScale are the same number precisely so
// that no conversion sits between the two.
func forward(m *model, d *dataset, e example, w *workspace) float32 {
	copy(w.accWhite, m.featureBias)
	copy(w.accBlack, m.featureBias)

	for _, f := range d.whiteFeatures(e) {
		row := m.featureWeights[int(f)*nnue.HiddenSize : (int(f)+1)*nnue.HiddenSize]
		for i, v := range row {
			w.accWhite[i] += v
		}
	}
	for _, f := range d.blackFeatures(e) {
		row := m.featureWeights[int(f)*nnue.HiddenSize : (int(f)+1)*nnue.HiddenSize]
		for i, v := range row {
			w.accBlack[i] += v
		}
	}

	for i := 0; i < nnue.HiddenSize; i++ {
		w.actWhite[i] = clamp01(w.accWhite[i])
		w.actBlack[i] = clamp01(w.accBlack[i])
	}

	us, them := w.actWhite, w.actBlack
	if e.stm == chess.Black {
		us, them = w.actBlack, w.actWhite
	}

	out := m.outputBias
	for i := 0; i < nnue.HiddenSize; i++ {
		out += us[i] * m.outputWeights[i]
		out += them[i] * m.outputWeights[nnue.HiddenSize+i]
	}
	return out
}

// accumulateGradients runs forward and backward over a batch, summing gradients
// into grad and returning the mean squared error.
func (t *Trainer) accumulateGradients(m, grad *model, d *dataset, batch []int32, w *workspace) float64 {
	var total float64

	for _, idx := range batch {
		e := d.examples[idx]

		out := forward(m, d, e, w)
		p := float32(sigmoid(float64(out)))

		err := p - e.target
		total += float64(err) * float64(err)

		// d(loss)/d(out) for MSE through a sigmoid.
		dOut := 2 * err * p * (1 - p)

		grad.outputBias += dOut

		// The output layer reads the side to move's half first, so the
		// gradients come back to the halves in that same order.
		usAct, themAct := w.actWhite, w.actBlack
		usAcc, themAcc := w.accWhite, w.accBlack
		usGrad, themGrad := w.dWhite, w.dBlack
		if e.stm == chess.Black {
			usAct, themAct = w.actBlack, w.actWhite
			usAcc, themAcc = w.accBlack, w.accWhite
			usGrad, themGrad = w.dBlack, w.dWhite
		}

		for i := 0; i < nnue.HiddenSize; i++ {
			grad.outputWeights[i] += dOut * usAct[i]
			grad.outputWeights[nnue.HiddenSize+i] += dOut * themAct[i]

			// Through the clamp: a saturated unit passes nothing back, which is
			// why the initialization has to keep activations off the rails.
			du, dt := dOut*m.outputWeights[i], dOut*m.outputWeights[nnue.HiddenSize+i]
			if usAcc[i] <= 0 || usAcc[i] >= 1 {
				du = 0
			}
			if themAcc[i] <= 0 || themAcc[i] >= 1 {
				dt = 0
			}
			usGrad[i], themGrad[i] = du, dt
		}

		// Both halves share one bias vector, so both contribute to it.
		for i := 0; i < nnue.HiddenSize; i++ {
			grad.featureBias[i] += w.dWhite[i] + w.dBlack[i]
		}

		for _, f := range d.whiteFeatures(e) {
			row := grad.featureWeights[int(f)*nnue.HiddenSize : (int(f)+1)*nnue.HiddenSize]
			for i := range row {
				row[i] += w.dWhite[i]
			}
		}
		for _, f := range d.blackFeatures(e) {
			row := grad.featureWeights[int(f)*nnue.HiddenSize : (int(f)+1)*nnue.HiddenSize]
			for i := range row {
				row[i] += w.dBlack[i]
			}
		}
	}

	if len(batch) == 0 {
		return 0
	}
	return total / float64(len(batch))
}

// evaluateLoss measures mean squared error without touching the weights.
func (t *Trainer) evaluateLoss(m *model, d *dataset, idx []int32, w *workspace) float64 {
	if len(idx) == 0 {
		return 0
	}
	var total float64
	for _, i := range idx {
		e := d.examples[i]
		p := float32(sigmoid(float64(forward(m, d, e, w))))
		err := float64(p - e.target)
		total += err * err
	}
	return total / float64(len(idx))
}

// quantizationError reports how far the integer network drifted from the float
// model it came from, and the scale that drift should be judged against, both
// in centipawns.
func (t *Trainer) quantizationError(m *model, net *nnue.Network, d *dataset, idx []int32, w *workspace) (drift, scale float64) {
	if len(idx) == 0 {
		return 0, 0
	}

	var acc nnue.Accumulator

	for _, i := range idx {
		e := d.examples[i]

		floatCP := float64(forward(m, d, e, w)) * nnue.Scale

		// Rebuild the quantized accumulator from the same feature lists, so the
		// comparison isolates the weights rather than the indexing.
		acc = nnue.Accumulator{}
		nnue.RefreshFromFeatures(&acc, net, d.whiteFeatures(e), d.blackFeatures(e))
		intCP := float64(net.Evaluate(&acc, e.stm))

		drift += math.Abs(floatCP - intCP)
		scale += math.Abs(intCP)
	}
	n := float64(len(idx))
	return drift / n, scale / n
}

// quantize converts the float model into the integer network the engine runs,
// reporting how many weights did not fit.
//
// A clipped weight means the quantized network is not the model whose loss was
// measured, so the count is returned rather than logged and forgotten: it is
// the difference between "this network scored well in training" and "this
// network will score well in a match".
func (m *model) quantize() (*nnue.Network, int) {
	net := nnue.NewZero()
	clipped := 0

	quantize := func(v float32, scale float64) int16 {
		x := math.Round(float64(v) * scale)
		if x > math.MaxInt16 {
			clipped++
			return math.MaxInt16
		}
		if x < math.MinInt16 {
			clipped++
			return math.MinInt16
		}
		return int16(x)
	}

	for i, v := range m.featureWeights {
		net.FeatureWeights[i] = quantize(v, nnue.QA)
	}
	for i, v := range m.featureBias {
		net.FeatureBias[i] = quantize(v, nnue.QA)
	}
	for i, v := range m.outputWeights {
		net.OutputWeights[i] = quantize(v, nnue.QB)
	}

	bias := math.Round(float64(m.outputBias) * nnue.QA * nnue.QB)
	if bias > math.MaxInt32 {
		clipped++
		bias = math.MaxInt32
	}
	if bias < math.MinInt32 {
		clipped++
		bias = math.MinInt32
	}
	net.OutputBias = int32(bias)

	return net, clipped
}

// zero clears a gradient accumulator between batches.
func (m *model) zero() {
	for i := range m.featureWeights {
		m.featureWeights[i] = 0
	}
	for i := range m.featureBias {
		m.featureBias[i] = 0
	}
	for i := range m.outputWeights {
		m.outputWeights[i] = 0
	}
	m.outputBias = 0
}

// adam is the optimizer. Plain SGD is workable here but needs its learning rate
// retuned whenever the data changes, and the pipeline is meant to run
// unattended across generations.
type adam struct {
	lr           float64
	beta1, beta2 float64
	epsilon      float64
	step1        int

	mFeature, vFeature []float32
	mFBias, vFBias     []float32
	mOutput, vOutput   []float32
	mOBias, vOBias     float32
}

func newAdam(lr float64) *adam {
	return &adam{
		lr:       lr,
		beta1:    0.9,
		beta2:    0.999,
		epsilon:  1e-8,
		mFeature: make([]float32, nnue.InputSize*nnue.HiddenSize),
		vFeature: make([]float32, nnue.InputSize*nnue.HiddenSize),
		mFBias:   make([]float32, nnue.HiddenSize),
		vFBias:   make([]float32, nnue.HiddenSize),
		mOutput:  make([]float32, 2*nnue.HiddenSize),
		vOutput:  make([]float32, 2*nnue.HiddenSize),
	}
}

// step applies one update. Gradients are summed over the batch, so they are
// divided by its size here rather than at every accumulation.
func (a *adam) step(m, grad *model, batchSize int) {
	if batchSize <= 0 {
		return
	}
	a.step1++

	scale := float32(1) / float32(batchSize)
	correction1 := 1 - math.Pow(a.beta1, float64(a.step1))
	correction2 := 1 - math.Pow(a.beta2, float64(a.step1))
	lr := float32(a.lr * math.Sqrt(correction2) / correction1)
	b1, b2 := float32(a.beta1), float32(a.beta2)
	eps := float32(a.epsilon)

	update := func(w, g, mm, vv []float32) {
		for i := range w {
			gi := g[i] * scale
			mm[i] = b1*mm[i] + (1-b1)*gi
			vv[i] = b2*vv[i] + (1-b2)*gi*gi
			w[i] -= lr * mm[i] / (sqrt32(vv[i]) + eps)
		}
	}

	update(m.featureWeights, grad.featureWeights, a.mFeature, a.vFeature)
	update(m.featureBias, grad.featureBias, a.mFBias, a.vFBias)
	update(m.outputWeights, grad.outputWeights, a.mOutput, a.vOutput)

	gb := grad.outputBias * scale
	a.mOBias = b1*a.mOBias + (1-b1)*gb
	a.vOBias = b2*a.vOBias + (1-b2)*gb*gb
	m.outputBias -= lr * a.mOBias / (sqrt32(a.vOBias) + eps)
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func sqrt32(v float32) float32 { return float32(math.Sqrt(float64(v))) }

// normal returns a standard normal sample, by the Box-Muller transform over the
// unit generator. It reuses rng rather than math/rand so that a training run is
// reproducible for the same reason a work unit is.
func (r *rng) normal() float32 {
	u1 := r.unit()
	if u1 < 1e-12 {
		u1 = 1e-12
	}
	u2 := r.unit()
	return float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
}

// unit returns a float in [0,1).
func (r *rng) unit() float64 {
	return float64(r.next()>>11) / float64(1<<53)
}

// shuffle permutes in place, Fisher-Yates from the seeded generator.
func shuffle(v []int32, r *rng) {
	for i := len(v) - 1; i > 0; i-- {
		j := int(r.next() % uint64(i+1))
		v[i], v[j] = v[j], v[i]
	}
}
