package nnue

import (
	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// maxStack bounds the accumulator stack. It is larger than any ply count the
// search reaches — quiescence search included — so Reset never has to guess
// how deep the tree can go; it only has to guard against the absurd case
// where something has gone wrong above this package.
const maxStack = 256

// IncrementalEvaluator maintains an accumulator per ply, updated by delta as
// the search walks the tree, instead of recomputing it from scratch at every
// node.
//
// Every delta in Push is integer addition and subtraction, which is exact:
// there is no rounding for a search to accumulate error from, only the
// possibility that the wrong delta was applied. That is what
// TestIncrementalMatchesRefreshOnRandomPlayouts exists to catch — it compares
// this evaluator's accumulator against Refresh at every ply of random play,
// and any divergence is a bug here, never a acceptable approximation.
//
// This type holds mutable state, so it must not be shared between search
// threads; NewThreadEvaluator hands each thread its own instance over the
// same read-only network, which is what makes PerThread satisfied.
type IncrementalEvaluator struct {
	net   *Network
	stack [maxStack]Accumulator
	sp    int
	// overflowed marks that the stack ran out of room and Push stopped
	// applying deltas. Evaluate falls back to Refresh in that case:
	// correctness over speed at a depth the search should never reach in
	// practice.
	overflowed bool
}

// NewIncremental builds an incremental evaluator over a network. The network
// is shared and read-only; the accumulator stack is private to this
// evaluator.
func NewIncremental(net *Network) *IncrementalEvaluator {
	return &IncrementalEvaluator{net: net}
}

// Evaluate reads the accumulator currently at the top of the stack. It does
// not push or pop; the search advances the stack itself through Push, Pop,
// PushNull and PopNull around make/unmake.
func (e *IncrementalEvaluator) Evaluate(p *chess.Position) int {
	if e.overflowed {
		e.stack[e.sp].Refresh(e.net, p)
		e.overflowed = false
	}
	return e.net.Evaluate(&e.stack[e.sp], p.SideToMove())
}

// Reset rebuilds the accumulator from scratch for a fresh root position and
// resets the stack pointer. The search calls this whenever it starts walking
// a new tree, so that no state from a previous search can leak in.
func (e *IncrementalEvaluator) Reset(p *chess.Position) {
	e.sp = 0
	e.overflowed = false
	e.stack[0].Refresh(e.net, p)
}

// Push computes the accumulator for the position after m is played, from the
// PRE-move position p. It must be called before p.MakeMove(m); the delta is
// read off p's current state (which piece stands where, what occupies the
// destination) because that information is gone once the move is made.
func (e *IncrementalEvaluator) Push(p *chess.Position, m chess.Move) {
	if e.sp >= maxStack-1 {
		// Absurd depth: stop applying deltas and fall back to Refresh in
		// Evaluate. The stack pointer still advances so Pop stays balanced,
		// but the entry it points to is stale until Evaluate repairs it.
		e.sp++
		e.overflowed = true
		return
	}

	e.stack[e.sp+1] = e.stack[e.sp]
	e.sp++
	acc := &e.stack[e.sp]
	if e.overflowed {
		// A previous Push overflowed; this accumulator is only valid once
		// refreshed, which Evaluate will do on demand. Nothing here can be
		// trusted as a base for a delta.
		return
	}

	from, to := m.From(), m.To()
	piece := p.PieceAt(from)
	us := p.SideToMove()

	switch m.Kind() {
	case chess.KindCastling:
		// The destination is the rook's square, not the king's — both pieces
		// are lifted before either is placed, because in Chess960 their
		// origins and destinations can overlap (the king may not move at
		// all, or the two may swap squares).
		rook := p.PieceAt(to)
		kingTo, rookTo := chess.CastlingDestinations(us, to > from)

		acc.Remove(e.net, piece, from)
		acc.Remove(e.net, rook, to)
		acc.Add(e.net, piece, kingTo)
		acc.Add(e.net, rook, rookTo)

	case chess.KindEnPassant:
		// The captured pawn sits beside the destination, not on it.
		captured := chess.Square(int(to) - chess.PawnPush(us))
		acc.Remove(e.net, chess.MakePiece(us.Opposite(), chess.Pawn), captured)
		acc.Move(e.net, piece, from, to)

	case chess.KindPromotion:
		if captured := p.PieceAt(to); captured != chess.NoPiece {
			acc.Remove(e.net, captured, to)
		}
		acc.Remove(e.net, piece, from)
		acc.Add(e.net, chess.MakePiece(us, m.Promotion()), to)

	default:
		if captured := p.PieceAt(to); captured != chess.NoPiece {
			acc.Remove(e.net, captured, to)
		}
		acc.Move(e.net, piece, from, to)
	}
}

// Pop discards the top of the stack after UnmakeMove, returning to the
// accumulator for the position one ply up.
func (e *IncrementalEvaluator) Pop() {
	e.sp--
	// Popping back under the depth that overflowed makes the entry now on
	// top trustworthy again (it was built by a normal Push before the stack
	// ran out), so the refresh fallback is no longer needed.
	if e.sp < maxStack-1 {
		e.overflowed = false
	}
}

// PushNull advances the stack for a null move. A null move changes the side
// to move but touches no feature on the board, so the new top is simply a
// copy of the current one; Evaluate reads it from the flipped side to move
// through p.SideToMove() itself.
func (e *IncrementalEvaluator) PushNull() {
	if e.sp >= maxStack-1 {
		e.sp++
		e.overflowed = true
		return
	}
	e.stack[e.sp+1] = e.stack[e.sp]
	e.sp++
}

// PopNull discards the top of the stack after a null move is undone.
func (e *IncrementalEvaluator) PopNull() {
	e.sp--
	if e.sp < maxStack-1 {
		e.overflowed = false
	}
}

// NewThreadEvaluator returns a fresh IncrementalEvaluator over the same
// network, with its own accumulator stack. The search calls this once per
// thread rather than sharing one instance, because the stack is mutable
// state that concurrent Push/Pop calls would race on.
func (e *IncrementalEvaluator) NewThreadEvaluator() eval.Evaluator {
	return NewIncremental(e.net)
}
