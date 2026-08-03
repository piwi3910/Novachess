package search

import (
	"sync/atomic"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// Bound describes how a stored score relates to the true value of a position.
//
// Alpha-beta does not compute exact values for most nodes: it establishes that
// a value is at least, or at most, some bound and then stops looking. Storing
// which of those happened is what makes a transposition table usable — an
// entry whose bound does not permit a cutoff is still useful for move ordering
// but must not be returned as a score.
type Bound uint8

const (
	BoundNone Bound = iota
	// BoundUpper means the true score is at most Score: every move failed low,
	// so the search never found anything as good as alpha.
	BoundUpper
	// BoundLower means the true score is at least Score: a move beat beta and
	// the rest were never examined.
	BoundLower
	// BoundExact means Score is the true value; the node was searched fully
	// within its window.
	BoundExact
)

// entry is one transposition table slot, packed into 16 bytes so that four fit
// in a cache line.
//
// The key is truncated to 32 bits. The remaining bits of the position's Zobrist
// key select the bucket, so a collision needs both the bucket and the stored
// half to match. Collisions are still possible and are why the search never
// trusts a table move without checking it is pseudo-legal in the current
// position.
type entry struct {
	key32 uint32     // upper half of the Zobrist key
	move  chess.Move // best move found, for ordering
	score int16
	eval  int16
	depth int8
	gen   uint8 // generation stamp, for replacement
	bound Bound
	_     uint8 // padding to a round 16 bytes
}

// bucketSize is how many entries share a bucket. Four gives a useful choice of
// replacement victims while keeping a bucket inside one 64-byte cache line, so
// a probe touches exactly one line.
const bucketSize = 4

type bucket [bucketSize]entry

// TT is a transposition table.
//
// It is shared unsynchronized across search threads. That is deliberate and is
// how lazy SMP works: races can produce a torn entry, which the key check
// almost always rejects and which the pseudo-legality check catches otherwise.
// The cost of that rare corruption is far smaller than the cost of locking a
// structure probed millions of times a second.
type TT struct {
	buckets []bucket
	mask    uint64
	gen     atomic.Uint32
}

// NewTT returns a table of about the requested size in megabytes. The bucket
// count is rounded down to a power of two so indexing is a mask rather than a
// division.
func NewTT(megabytes int) *TT {
	if megabytes < 1 {
		megabytes = 1
	}
	bytesPerBucket := 64 // one cache line
	want := megabytes * 1024 * 1024 / bytesPerBucket

	count := 1
	for count*2 <= want {
		count *= 2
	}

	return &TT{
		buckets: make([]bucket, count),
		mask:    uint64(count - 1),
	}
}

// Clear empties the table.
func (t *TT) Clear() {
	for i := range t.buckets {
		t.buckets[i] = bucket{}
	}
	t.gen.Store(0)
}

// NewSearch advances the generation counter, marking existing entries as stale
// so they are preferred for replacement. Entries are not erased: a result from
// the previous search is still correct, just less likely to be relevant.
func (t *TT) NewSearch() { t.gen.Add(1) }

// Probe looks up a position. It returns the entry's move and score along with
// whether the entry was found, and whether the score may be used for a cutoff
// at this depth and window.
//
// Mate scores are stored relative to the node they were found at and converted
// back to be relative to the root here; see adjustMateScoreFromTT.
func (t *TT) Probe(key uint64, depth, ply int, alpha, beta int) (move chess.Move, score int, hit, usable bool) {
	b := &t.buckets[key&t.mask]
	want := uint32(key >> 32)

	for i := range b {
		e := &b[i]
		if e.key32 != want || e.bound == BoundNone {
			continue
		}

		move = e.move
		score = adjustMateScoreFromTT(int(e.score), ply)
		hit = true

		// A shallower entry cannot justify a cutoff, but its move is still the
		// best ordering hint available.
		if int(e.depth) < depth {
			return move, score, true, false
		}

		switch e.bound {
		case BoundExact:
			usable = true
		case BoundLower:
			usable = score >= beta
		case BoundUpper:
			usable = score <= alpha
		}
		return move, score, true, usable
	}
	return chess.MoveNone, 0, false, false
}

// Store records a result.
//
// Replacement prefers, in order: an empty slot, a slot holding this same
// position, a slot from an older search, and finally the shallowest entry.
// Depth-preferred replacement matters because a deep entry cost far more to
// compute than a shallow one and is worth far more when it hits again.
func (t *TT) Store(key uint64, move chess.Move, score, staticEval, depth, ply int, bound Bound) {
	b := &t.buckets[key&t.mask]
	want := uint32(key >> 32)
	gen := uint8(t.gen.Load())

	victim := 0
	bestValue := int(1 << 30)

	for i := range b {
		e := &b[i]

		if e.bound == BoundNone || e.key32 == want {
			victim = i
			// An existing entry for this position is replaced unless what is
			// already there was searched deeper and is exact. Keeping a deep
			// exact score over a shallow bound is strictly better.
			if e.key32 == want && e.bound == BoundExact && int(e.depth) > depth && bound != BoundExact {
				return
			}
			break
		}

		// Value a slot by its depth, discounted heavily if it is from an older
		// search. Lower is more replaceable.
		value := int(e.depth)
		if e.gen != gen {
			value -= 64
		}
		if value < bestValue {
			bestValue = value
			victim = i
		}
	}

	// Preserve the existing move when this store has none: a fail-low node
	// records no best move, but the move stored earlier is still the best
	// ordering hint known for the position.
	e := &b[victim]
	if move.IsNone() && e.key32 == want {
		move = e.move
	}

	*e = entry{
		key32: want,
		move:  move,
		score: int16(adjustMateScoreToTT(score, ply)),
		eval:  int16(clampToInt16(staticEval)),
		depth: int8(depth),
		gen:   gen,
		bound: bound,
	}
}

// Mate scores must be stored relative to the node that found them rather than
// to the root.
//
// "Mate in 3 from here" is a property of the position; "mate in 7 from the
// root" is a property of the path taken to reach it. A transposition reaches
// the same position by a different path, so storing the root-relative form
// would report a mate distance that is simply wrong — and, worse, can make the
// engine shuffle instead of delivering mate because every path looks equally
// good.
func adjustMateScoreToTT(score, ply int) int {
	switch {
	case score >= eval.MateScore-eval.MaxMateDepth:
		return score + ply
	case score <= -(eval.MateScore - eval.MaxMateDepth):
		return score - ply
	default:
		return score
	}
}

func adjustMateScoreFromTT(score, ply int) int {
	switch {
	case score >= eval.MateScore-eval.MaxMateDepth:
		return score - ply
	case score <= -(eval.MateScore - eval.MaxMateDepth):
		return score + ply
	default:
		return score
	}
}

func clampToInt16(v int) int {
	if v > 32767 {
		return 32767
	}
	if v < -32768 {
		return -32768
	}
	return v
}

// Hashfull reports approximate table occupancy in permille, which UCI expects
// to report to the GUI. It samples rather than scanning, since the table can be
// gigabytes.
func (t *TT) Hashfull() int {
	const sample = 1000
	if len(t.buckets) == 0 {
		return 0
	}
	gen := uint8(t.gen.Load())
	used, seen := 0, 0
	for i := 0; i < sample && i < len(t.buckets); i++ {
		for j := range t.buckets[i] {
			e := &t.buckets[i][j]
			if e.bound != BoundNone && e.gen == gen {
				used++
			}
			seen++
		}
	}
	if seen == 0 {
		return 0
	}
	return used * 1000 / seen
}
