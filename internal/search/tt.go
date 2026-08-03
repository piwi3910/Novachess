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

// An entry is packed into one 64-bit word, stored alongside that word XOR-ed
// with the position key.
//
// This is lockless hashing, and it solves two problems at once. Threads share
// the table without synchronization — which is the whole basis of lazy SMP —
// and a reader that catches a writer mid-update sees a mismatched pair, so the
// XOR fails to reproduce the key and the entry is simply treated as a miss.
// Without the trick, a torn read could return one position's score under
// another position's key.
//
// The alternative, locking the table, would serialize something probed millions
// of times a second and defeat the point of threading entirely. Plain
// unsynchronized fields are not an option either: in Go that is a data race,
// which is undefined behaviour rather than merely untidy.
type entry struct {
	keyx atomic.Uint64 // key ^ data
	data atomic.Uint64
}

// Field layout within the packed word.
const (
	shiftMove  = 0
	shiftScore = 16
	shiftEval  = 32
	shiftDepth = 48
	shiftBound = 56
	shiftGen   = 58

	maskBound = 0x3
	maskGen   = 0x3F
)

func packEntry(move chess.Move, score, staticEval int, depth int, bound Bound, gen uint8) uint64 {
	return uint64(uint16(move)) |
		uint64(uint16(int16(score)))<<shiftScore |
		uint64(uint16(int16(staticEval)))<<shiftEval |
		uint64(uint8(depth))<<shiftDepth |
		uint64(bound&maskBound)<<shiftBound |
		uint64(gen&maskGen)<<shiftGen
}

func (d unpacked) move() chess.Move { return chess.Move(uint16(d)) }
func (d unpacked) score() int       { return int(int16(uint16(d >> shiftScore))) }
func (d unpacked) depth() int       { return int(uint8(d >> shiftDepth)) }
func (d unpacked) bound() Bound     { return Bound(d>>shiftBound) & maskBound }
func (d unpacked) gen() uint8       { return uint8(d>>shiftGen) & maskGen }

type unpacked uint64

// read returns the entry's contents and whether they belong to the given key.
func (e *entry) read(key uint64) (unpacked, bool) {
	// data is loaded first so that a writer racing between the two loads
	// produces a mismatch rather than a plausible-looking pair.
	data := e.data.Load()
	keyx := e.keyx.Load()
	return unpacked(data), keyx^data == key && data != 0
}

func (e *entry) write(key, data uint64) {
	e.keyx.Store(key ^ data)
	e.data.Store(data)
}

// bucketSize is how many entries share a bucket. Four 16-byte entries fill one
// 64-byte cache line, so a probe touches exactly one line.
const bucketSize = 4

type bucket [bucketSize]entry

// TT is a transposition table shared by every search thread.
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
	const bytesPerBucket = 64
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
//
// It replaces the backing array rather than walking it. The result is the same
// — every entry reads as a miss — but the work is a single allocation of
// already-zero pages instead of two atomic stores per entry, of which a
// default-sized table has several million. That difference is invisible in
// ordinary play and very visible under the race detector, which instruments
// every one of those stores.
//
// Like SetHashSize, which has always replaced the table this way, this must not
// run while a search is in flight: the search threads read the table without
// synchronization, and swapping it underneath them is a data race. UCI only
// reaches here between games.
func (t *TT) Clear() {
	t.buckets = make([]bucket, len(t.buckets))
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

	for i := range b {
		d, ok := b[i].read(key)
		if !ok || d.bound() == BoundNone {
			continue
		}

		move = d.move()
		score = adjustMateScoreFromTT(d.score(), ply)
		hit = true

		// A shallower entry cannot justify a cutoff, but its move is still the
		// best ordering hint available.
		if d.depth() < depth {
			return move, score, true, false
		}

		switch d.bound() {
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
	gen := uint8(t.gen.Load())

	victim := 0
	bestValue := int(1 << 30)

	for i := range b {
		d, matches := b[i].read(key)

		if d.bound() == BoundNone || matches {
			victim = i
			// An existing entry for this position is replaced unless what is
			// already there was searched deeper and is exact. Keeping a deep
			// exact score over a shallow bound is strictly better.
			if matches && d.bound() == BoundExact && d.depth() > depth && bound != BoundExact {
				return
			}
			// Preserve the existing move when this store has none: a fail-low
			// node records no best move, but the move stored earlier is still
			// the best ordering hint known for the position.
			if matches && move.IsNone() {
				move = d.move()
			}
			break
		}

		// Value a slot by its depth, discounted heavily if it is from an older
		// search. Lower is more replaceable.
		value := d.depth()
		if d.gen() != gen {
			value -= 64
		}
		if value < bestValue {
			bestValue = value
			victim = i
		}
	}

	b[victim].write(key, packEntry(move, adjustMateScoreToTT(score, ply), clampToInt16(staticEval),
		clampDepth(depth), bound, gen))
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

func clampDepth(d int) int {
	if d < 0 {
		return 0
	}
	if d > 255 {
		return 255
	}
	return d
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
			d := unpacked(t.buckets[i][j].data.Load())
			if d.bound() != BoundNone && d.gen() == gen {
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
