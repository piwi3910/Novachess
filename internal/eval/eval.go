// Package eval scores chess positions.
//
// Two implementations are planned. This one is hand-crafted: readable terms
// with hand-chosen weights, which is what lets the engine play at all before
// any training has happened, and what generates the first round of self-play
// data. The second will be an NNUE network trained on that data.
//
// Both sit behind Evaluator so the search never learns which it is talking to.
// That boundary is also where a SIMD-accelerated backend would go; see the
// notes in the README about archsimd being amd64-only.
package eval

import "github.com/piwi3910/novachess/internal/chess"

// Evaluator scores a position from the perspective of the side to move, in
// centipawns. Positive means the side to move is better.
//
// The side-to-move convention rather than a fixed White orientation is what
// negamax requires: the search negates the score at every ply, so an evaluator
// that always spoke from White's point of view would need a sign flip at every
// call site and would eventually get one wrong.
type Evaluator interface {
	Evaluate(p *chess.Position) int
}

// Score bounds. Mate scores live just below Infinite so that the search can
// encode "mate in n" as MateScore-n and still compare scores with ordinary
// integer comparison.
const (
	Infinite  = 32000
	MateScore = 31000
	// MaxMateDepth bounds how deep a mate can be reported. Any score whose
	// magnitude exceeds MateScore-MaxMateDepth is a mate score rather than an
	// evaluation.
	MaxMateDepth = 256
	// DrawScore is returned for stalemate, repetition, the fifty-move rule and
	// dead positions. It is exactly zero rather than a small contempt value:
	// self-play needs the two sides to agree on what a draw is worth, and a
	// contempt term would make the engine's own games mislabelled.
	DrawScore = 0
)

// IsMateScore reports whether a score encodes a forced mate rather than an
// evaluation.
func IsMateScore(score int) bool {
	if score < 0 {
		score = -score
	}
	return score > MateScore-MaxMateDepth
}

// MateDistance returns the number of plies to mate for a mate score. The result
// is meaningless for ordinary scores.
func MateDistance(score int) int {
	if score > 0 {
		return MateScore - score
	}
	return -MateScore - score
}
