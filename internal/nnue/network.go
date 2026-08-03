// Package nnue implements the trained evaluation: a small neural network that
// replaces the hand-crafted terms once self-play has produced enough data.
//
// The architecture is a perspective network, 768->256x2->1. Its input is one
// bit per (color, piece type, square), accumulated separately from each side's
// point of view; the output layer reads the side to move's half first and the
// opponent's second. Two properties follow from that shape, and both are relied
// on elsewhere in this package.
//
// The evaluation is symmetric by construction. Mirroring a position — swapping
// colors and flipping ranks — swaps the two accumulator halves and swaps the
// side to move, so the output layer sees exactly the same pair of inputs and
// returns the same number. This holds for any weights whatsoever, trained or
// random, which makes it a test that checks the feature indexing rather than
// the network.
//
// And the accumulator is incremental. A move changes at most a handful of
// features, so the hidden layer can be updated by adding and subtracting a few
// rows instead of being recomputed. That is the entire reason this architecture
// is shaped the way it is: the first layer is the expensive one, and a search
// evaluates millions of positions that differ from their parent by one piece.
package nnue

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/piwi3910/novachess/internal/chess"
)

// Architecture. These are compiled in rather than read from the file: the
// evaluation is on the hot path of the search, and a hidden size known at
// compile time is what lets the accumulator be a fixed-size array instead of a
// slice. A network with different dimensions is rejected at load rather than
// adapted to.
const (
	// InputSize is one feature per color, piece type and square.
	InputSize = chess.ColorCount * chess.PieceTypeCount * 64 // 768

	// HiddenSize is the width of one accumulator half.
	HiddenSize = 256
)

// Quantization. The network is trained in floating point and stored as
// integers, because the search cannot afford float math per node and because
// integer arithmetic is bit-for-bit identical on every machine — which the
// distributed pipeline requires, since a network that evaluated differently on
// two boards would make self-play irreproducible.
const (
	// QA scales the feature weights and, with them, the accumulator. It is also
	// the ceiling of the clipped ReLU: an activation is clamped to 0..QA, which
	// in network terms is 0..1.
	QA = 255

	// QB scales the output weights.
	QB = 64

	// Scale converts the network's output into centipawns.
	Scale = 400
)

// EvalLimit bounds what a network may return.
//
// The search treats scores above a threshold as forced mates and reasons about
// distance-to-mate from them. A network is a pile of trained numbers with no
// notion of that convention, and an untrained or corrupt one can produce
// arbitrarily large output; left alone it would announce mates that do not
// exist and poison the transposition table with them. Clamping here keeps every
// evaluation comfortably inside the ordinary range.
const EvalLimit = 20000

// Magic identifies a Novachess network file.
var Magic = [8]byte{'N', 'O', 'V', 'A', 'N', 'N', 'U', 'E'}

// FormatVersion is the current weight layout.
const FormatVersion uint32 = 1

// Network holds the quantized weights.
//
// The feature weights are stored row-major by feature, so the HiddenSize values
// a single feature contributes are contiguous. That is the layout the
// accumulator wants: turning a feature on means adding one contiguous run.
type Network struct {
	// FeatureWeights is InputSize rows of HiddenSize weights.
	FeatureWeights []int16
	// FeatureBias is the accumulator's starting value, before any piece.
	FeatureBias []int16
	// OutputWeights is 2*HiddenSize: the side to move's half, then the
	// opponent's.
	OutputWeights []int16
	// OutputBias is already scaled by QA*QB, matching the products it is added
	// to.
	OutputBias int32
}

// NewZero returns a network of the right shape with all weights zero. It
// evaluates every position as a draw, which makes it useless for play and
// convenient as a starting point for training and tests.
func NewZero() *Network {
	return &Network{
		FeatureWeights: make([]int16, InputSize*HiddenSize),
		FeatureBias:    make([]int16, HiddenSize),
		OutputWeights:  make([]int16, 2*HiddenSize),
	}
}

// FeatureIndex returns the input index for a piece standing on a square, seen
// from one side's point of view.
//
// From Black's perspective the board is turned around: ranks are flipped and
// the colors are swapped, so that "my pawn on my second rank" is the same
// feature whichever side is asking. Without that, the network would have to
// learn every pattern twice, once per color, from half as much data each time.
func FeatureIndex(perspective chess.Color, piece chess.Piece, sq chess.Square) int {
	color := piece.Color()
	if perspective == chess.Black {
		color = color.Opposite()
		sq ^= 56 // flip the rank, keep the file
	}
	return (int(color)*chess.PieceTypeCount+int(piece.Type()))*64 + int(sq)
}

// featureRow returns the weights a single feature contributes to a half.
func (n *Network) featureRow(index int) []int16 {
	return n.FeatureWeights[index*HiddenSize : (index+1)*HiddenSize]
}

// Validate checks that a network has the shape this build expects. A network
// with the wrong dimensions would index out of range on the first evaluation.
func (n *Network) Validate() error {
	if n == nil {
		return fmt.Errorf("nnue: network is nil")
	}
	if len(n.FeatureWeights) != InputSize*HiddenSize {
		return fmt.Errorf("nnue: %d feature weights, want %d", len(n.FeatureWeights), InputSize*HiddenSize)
	}
	if len(n.FeatureBias) != HiddenSize {
		return fmt.Errorf("nnue: %d feature biases, want %d", len(n.FeatureBias), HiddenSize)
	}
	if len(n.OutputWeights) != 2*HiddenSize {
		return fmt.Errorf("nnue: %d output weights, want %d", len(n.OutputWeights), 2*HiddenSize)
	}
	return nil
}

// headerSize is the fixed prelude: magic, version, and the two dimensions.
const headerSize = 8 + 4 + 4 + 4

// WriteTo serializes the network.
//
// The dimensions are written even though they are compile-time constants here,
// so that a mismatched file is diagnosed as a mismatch rather than read as
// garbage. A file whose hidden size disagrees with this build would otherwise
// load without complaint and evaluate nonsense.
func (n *Network) WriteTo(w io.Writer) (int64, error) {
	if err := n.Validate(); err != nil {
		return 0, err
	}

	var header [headerSize]byte
	copy(header[0:8], Magic[:])
	binary.LittleEndian.PutUint32(header[8:12], FormatVersion)
	binary.LittleEndian.PutUint32(header[12:16], InputSize)
	binary.LittleEndian.PutUint32(header[16:20], HiddenSize)

	if _, err := w.Write(header[:]); err != nil {
		return 0, fmt.Errorf("nnue: write header: %w", err)
	}
	written := int64(headerSize)

	for _, section := range [][]int16{n.FeatureWeights, n.FeatureBias, n.OutputWeights} {
		buf := make([]byte, 2*len(section))
		for i, v := range section {
			binary.LittleEndian.PutUint16(buf[2*i:], uint16(v))
		}
		if _, err := w.Write(buf); err != nil {
			return written, fmt.Errorf("nnue: write weights: %w", err)
		}
		written += int64(len(buf))
	}

	var tail [4]byte
	binary.LittleEndian.PutUint32(tail[:], uint32(n.OutputBias))
	if _, err := w.Write(tail[:]); err != nil {
		return written, fmt.Errorf("nnue: write output bias: %w", err)
	}
	return written + 4, nil
}

// ReadFrom loads a network.
//
// A wrong magic, version or shape is a hard error. Loading a network the build
// does not understand would not fail visibly — it would play, badly, and the
// gatekeeper would reject it for reasons nobody could explain.
func ReadFrom(r io.Reader) (*Network, error) {
	var header [headerSize]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, fmt.Errorf("nnue: read header: %w", err)
	}

	if [8]byte(header[0:8]) != Magic {
		return nil, fmt.Errorf("nnue: not a Novachess network file")
	}
	if v := binary.LittleEndian.Uint32(header[8:12]); v != FormatVersion {
		return nil, fmt.Errorf("nnue: network is format version %d, this build reads version %d", v, FormatVersion)
	}
	if in := binary.LittleEndian.Uint32(header[12:16]); in != InputSize {
		return nil, fmt.Errorf("nnue: network has %d inputs, this build expects %d", in, InputSize)
	}
	if hidden := binary.LittleEndian.Uint32(header[16:20]); hidden != HiddenSize {
		return nil, fmt.Errorf("nnue: network has a hidden layer of %d, this build expects %d", hidden, HiddenSize)
	}

	n := NewZero()
	for _, section := range [][]int16{n.FeatureWeights, n.FeatureBias, n.OutputWeights} {
		buf := make([]byte, 2*len(section))
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("nnue: read weights: %w", err)
		}
		for i := range section {
			section[i] = int16(binary.LittleEndian.Uint16(buf[2*i:]))
		}
	}

	var tail [4]byte
	if _, err := io.ReadFull(r, tail[:]); err != nil {
		return nil, fmt.Errorf("nnue: read output bias: %w", err)
	}
	n.OutputBias = int32(binary.LittleEndian.Uint32(tail[:]))

	return n, nil
}
