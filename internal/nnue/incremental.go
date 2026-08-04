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
// and any divergence is a bug here, never an acceptable approximation.
//
// This type holds mutable state, so it must not be shared between search
// threads; NewThreadEvaluator hands each thread its own instance over the
// same read-only network, which is what makes PerThread satisfied.
type IncrementalEvaluator struct {
	net   *Network
	stack []Accumulator // len(stack) == cap; a slice so tests can shrink cap without touching production sizing
	cap   int
	sp    int

	// overflowDepth counts Push/PushNull calls that arrived after sp already
	// reached cap-1 and so could not be given their own stack slot. It is
	// compared against zero rather than folded into sp itself, because sp
	// must never leave the array's bounds: incrementing it past cap-1 is
	// exactly the out-of-bounds panic this field exists to prevent. While
	// overflowDepth is nonzero, stack[sp] does not correspond to the current
	// position, so Evaluate must refresh from scratch instead of reading it,
	// and a subsequent Pop must unwind the virtual depth before it is safe to
	// trust stack[sp] again.
	overflowDepth int

	// valid is false until Reset has run at least once. It exists so that a
	// search path which forgot to call Reset self-heals on the first
	// Evaluate — refreshing rather than reading whatever zero-valued
	// accumulator a fresh IncrementalEvaluator starts with — instead of
	// silently evaluating every position as if the board were empty.
	valid bool
}

// NewIncremental builds an incremental evaluator over a network. The network
// is shared and read-only; the accumulator stack is private to this
// evaluator. valid starts false: nothing has primed the accumulator yet, and
// Evaluate will refresh from scratch before trusting it if Reset is never
// called.
func NewIncremental(net *Network) *IncrementalEvaluator {
	return newIncrementalCapped(net, maxStack)
}

// newIncrementalCapped builds an incremental evaluator with a caller-chosen
// stack capacity. Production code always goes through NewIncremental with
// maxStack; tests use this directly to exercise the overflow path without
// pushing 256 real plies.
func newIncrementalCapped(net *Network, capacity int) *IncrementalEvaluator {
	return &IncrementalEvaluator{net: net, stack: make([]Accumulator, capacity), cap: capacity}
}

// Evaluate reads the accumulator for the current node. It does not push or
// pop; the search advances the stack itself through Push, Pop, PushNull and
// PopNull around make/unmake.
func (e *IncrementalEvaluator) Evaluate(p *chess.Position) int {
	if !e.valid {
		// Nothing has ever primed this evaluator's accumulator. Refresh
		// stands in for the missing Reset so a forgotten call degrades to a
		// full recompute rather than an evaluation of an empty board.
		e.stack[e.sp].Refresh(e.net, p)
		e.valid = true
	}

	if e.overflowDepth > 0 {
		// stack[sp] represents the position at the depth the stack ran out
		// of room, not this one. A scratch refresh keeps that entry intact
		// so that once enough Pops unwind the overflow it is trustworthy
		// again.
		var scratch Accumulator
		scratch.Refresh(e.net, p)
		return e.net.Evaluate(&scratch, p.SideToMove())
	}
	return e.net.Evaluate(&e.stack[e.sp], p.SideToMove())
}

// Reset rebuilds the accumulator from scratch for a fresh root position and
// resets the stack pointer. The search calls this whenever it starts walking
// a new tree, so that no state from a previous search can leak in.
func (e *IncrementalEvaluator) Reset(p *chess.Position) {
	e.sp = 0
	e.overflowDepth = 0
	e.stack[0].Refresh(e.net, p)
	e.valid = true
}

// Push computes the accumulator for the position after m is played, from the
// PRE-move position p. It must be called before p.MakeMove(m); the delta is
// read off p's current state (which piece stands where, what occupies the
// destination) because that information is gone once the move is made.
func (e *IncrementalEvaluator) Push(p *chess.Position, m chess.Move) {
	if e.sp >= e.cap-1 {
		// The stack has no free slot for this ply. Record the overflow
		// without moving sp, which must stay inside the array; Evaluate
		// falls back to a full refresh for as long as overflowDepth is
		// nonzero, and Pop below unwinds it one virtual level at a time
		// before sp moves again.
		e.overflowDepth++
		return
	}

	// overflowDepth is necessarily zero here: once it becomes nonzero, sp
	// stays pinned at cap-1 and every subsequent Push takes the branch
	// above, so this line is only reached while the stack still has room.
	e.stack[e.sp+1] = e.stack[e.sp]
	e.sp++
	acc := &e.stack[e.sp]

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
	if e.overflowDepth > 0 {
		// Unwind a virtual level: sp never moved for this Push, so it must
		// not move for the matching Pop either.
		e.overflowDepth--
		return
	}
	e.sp--
}

// PushNull advances the stack for a null move. A null move changes the side
// to move but touches no feature on the board, so the new top is simply a
// copy of the current one; Evaluate reads it from the flipped side to move
// through p.SideToMove() itself.
func (e *IncrementalEvaluator) PushNull() {
	if e.sp >= e.cap-1 {
		e.overflowDepth++
		return
	}
	e.stack[e.sp+1] = e.stack[e.sp]
	e.sp++
}

// PopNull discards the top of the stack after a null move is undone.
func (e *IncrementalEvaluator) PopNull() {
	if e.overflowDepth > 0 {
		e.overflowDepth--
		return
	}
	e.sp--
}

// NewThreadEvaluator returns a fresh IncrementalEvaluator over the same
// network, with its own accumulator stack and valid starting false — a
// thread's evaluator must prime itself through Reset (or self-heal on first
// Evaluate) exactly like one built by NewIncremental, never inherit the
// parent's stack state.
func (e *IncrementalEvaluator) NewThreadEvaluator() eval.Evaluator {
	return newIncrementalCapped(e.net, e.cap)
}
