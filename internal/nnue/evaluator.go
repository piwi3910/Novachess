package nnue

import (
	"fmt"
	"os"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// Evaluator adapts a network to the interface the search evaluates through, so
// that swapping the hand-crafted evaluation for a trained one changes nothing
// above this line.
//
// It holds no mutable state. The accumulator is a local in Evaluate rather than
// a field, which matters because the search shares one evaluator across every
// lazy SMP thread: an accumulator hanging off the evaluator would be written by
// all of them at once. That is a data race — undefined behaviour in Go, not an
// untidiness — and it would surface as evaluations that are wrong only when
// threads happen to overlap, which is the hardest kind of bug to find in an
// engine that is allowed to play badly.
//
// The cost is a full refresh per call: every piece on the board, rather than
// the piece or two a move actually changed. Making that incremental means
// keeping an accumulator per ply and updating it in step with make and unmake,
// which is a change to the search rather than to this package.
type Evaluator struct {
	net *Network
}

// NewEvaluator wraps a network. The network is validated up front so that a
// mismatched one is refused at load rather than indexing out of range on the
// first position the search reaches.
func NewEvaluator(n *Network) (*Evaluator, error) {
	if err := n.Validate(); err != nil {
		return nil, err
	}
	return &Evaluator{net: n}, nil
}

// Load reads a network from a file and returns an evaluator for it.
func Load(path string) (*Evaluator, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("nnue: open network: %w", err)
	}
	defer f.Close()

	n, err := ReadFrom(f)
	if err != nil {
		return nil, fmt.Errorf("nnue: %s: %w", path, err)
	}
	return NewEvaluator(n)
}

// Network returns the underlying weights.
func (e *Evaluator) Network() *Network { return e.net }

// Evaluate scores a position from the side to move's point of view, in
// centipawns.
func (e *Evaluator) Evaluate(p *chess.Position) int {
	var acc Accumulator
	acc.Refresh(e.net, p)
	return e.net.Evaluate(&acc, p.SideToMove())
}

// NewThreadEvaluator returns an incremental evaluator over the same network,
// private to one search thread. This is what lets the search upgrade from a
// full refresh per node to an accumulator maintained by delta, without the
// search needing to know anything about NNUE beyond eval.MoveAware: it
// satisfies eval.PerThread so that every existing construction site
// (New, Load) transparently gains the faster per-thread path.
func (e *Evaluator) NewThreadEvaluator() eval.Evaluator {
	return NewIncremental(e.net)
}
