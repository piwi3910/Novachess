package nnue

import "github.com/piwi3910/novachess/internal/chess"

// Accumulator holds the hidden layer, one half per perspective.
//
// The values are int32 rather than the int16 the weights are stored in. A half
// is the sum of up to 32 feature rows plus the bias, and while trained weights
// are small enough that int16 would hold the total, "small enough" is a
// property of the weights rather than of the code. An overflow here would not
// crash; it would wrap, and the search would quietly evaluate a position as its
// own opposite. The wider type removes the question at a cost of one kilobyte
// per accumulator.
//
// A vectorized backend would want int16 halves to double its lane count. That
// is a decision for the code that adds it, and it does not affect the file
// format — the weights on disk are int16 either way.
type Accumulator struct {
	values [chess.ColorCount][HiddenSize]int32
}

// Refresh recomputes both halves from scratch.
//
// This is the slow path, used when there is no parent accumulator to update
// from. Everything else adds and subtracts individual features.
func (a *Accumulator) Refresh(n *Network, p *chess.Position) {
	for c := range a.values {
		for i := 0; i < HiddenSize; i++ {
			a.values[c][i] = int32(n.FeatureBias[i])
		}
	}

	for bb := p.Occupied(); bb != 0; {
		var sq chess.Square
		sq, bb = bb.PopFirst()
		a.Add(n, p.PieceAt(sq), sq)
	}
}

// RefreshFromFeatures rebuilds an accumulator from feature indices that have
// already been computed, one list per perspective.
//
// The trainer works from precomputed index lists rather than positions, because
// re-walking a board every epoch dominates a run. Giving it this entry point
// means the comparison between a trained model and the network it quantized
// into exercises the same weights through the same layout, isolating the
// quantization from the feature indexing.
func RefreshFromFeatures(a *Accumulator, n *Network, white, black []int32) {
	for c := range a.values {
		for i := 0; i < HiddenSize; i++ {
			a.values[c][i] = int32(n.FeatureBias[i])
		}
	}

	for _, list := range [2]struct {
		features []int32
		color    chess.Color
	}{{white, chess.White}, {black, chess.Black}} {
		half := &a.values[list.color]
		for _, f := range list.features {
			row := n.featureRow(int(f))
			for i := 0; i < HiddenSize; i++ {
				half[i] += int32(row[i])
			}
		}
	}
}

// Add turns on the feature for a piece appearing on a square.
func (a *Accumulator) Add(n *Network, piece chess.Piece, sq chess.Square) {
	for c := chess.Color(0); c < chess.ColorCount; c++ {
		row := n.featureRow(FeatureIndex(c, piece, sq))
		half := &a.values[c]
		for i := 0; i < HiddenSize; i++ {
			half[i] += int32(row[i])
		}
	}
}

// Remove turns off the feature for a piece leaving a square.
func (a *Accumulator) Remove(n *Network, piece chess.Piece, sq chess.Square) {
	for c := chess.Color(0); c < chess.ColorCount; c++ {
		row := n.featureRow(FeatureIndex(c, piece, sq))
		half := &a.values[c]
		for i := 0; i < HiddenSize; i++ {
			half[i] -= int32(row[i])
		}
	}
}

// Move relocates a piece, which is the common case and worth not doing as a
// separate Remove and Add: it walks each half once instead of twice.
func (a *Accumulator) Move(n *Network, piece chess.Piece, from, to chess.Square) {
	for c := chess.Color(0); c < chess.ColorCount; c++ {
		out := n.featureRow(FeatureIndex(c, piece, from))
		in := n.featureRow(FeatureIndex(c, piece, to))
		half := &a.values[c]
		for i := 0; i < HiddenSize; i++ {
			half[i] += int32(in[i]) - int32(out[i])
		}
	}
}

// crelu is the activation: clipped ReLU, clamping to the 0..QA range that QA
// was chosen to represent.
func crelu(v int32) int64 {
	if v <= 0 {
		return 0
	}
	if v >= QA {
		return QA
	}
	return int64(v)
}

// Evaluate reads the accumulator through the output layer, from the point of
// view of the side to move.
//
// The side to move's half is read first and the opponent's second, which is
// what makes the network's output side-relative to match the search's
// convention — and what makes the whole evaluation symmetric under mirroring.
func (n *Network) Evaluate(a *Accumulator, stm chess.Color) int {
	us := &a.values[stm]
	them := &a.values[stm.Opposite()]

	// The sum is int64 because nothing bounds the weights of an arbitrary
	// network file. 512 terms of up to QA times a full int16 overflows int32,
	// and an evaluation that wraps would be far harder to diagnose than one
	// that is merely wrong.
	var sum int64
	for i := 0; i < HiddenSize; i++ {
		sum += crelu(us[i]) * int64(n.OutputWeights[i])
	}
	for i := 0; i < HiddenSize; i++ {
		sum += crelu(them[i]) * int64(n.OutputWeights[HiddenSize+i])
	}
	sum += int64(n.OutputBias)

	cp := sum * Scale / (QA * QB)
	if cp > EvalLimit {
		return EvalLimit
	}
	if cp < -EvalLimit {
		return -EvalLimit
	}
	return int(cp)
}
