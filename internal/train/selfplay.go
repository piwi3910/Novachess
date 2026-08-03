package train

import (
	"context"
	"fmt"
	"time"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
	"github.com/piwi3910/novachess/internal/events"
	"github.com/piwi3910/novachess/internal/search"
)

// GameStats summarizes what a work unit produced.
type GameStats struct {
	Games          int
	Positions      int
	Nodes          uint64
	Duration       time.Duration
	WhiteWins      int
	BlackWins      int
	Draws          int
	SkippedNoisy   int
	SkippedTooLong int
}

// NodesPerSecond reports the measured search throughput, which the worker
// reports back so that a thermally throttled cluster node is visible in the
// data rather than something to infer from wall-clock times.
func (s GameStats) NodesPerSecond() float64 {
	if s.Duration <= 0 {
		return 0
	}
	return float64(s.Nodes) / s.Duration.Seconds()
}

// Generator plays self-play games for a work unit.
//
// One generator serves one worker and is used serially. Everything about it is
// deterministic: given the same WorkUnit it produces byte-identical samples,
// on any machine, which is what makes a redelivered unit safe to replay and
// makes the whole distributed pipeline reproducible.
type Generator struct {
	evaluator eval.Evaluator
	hashMB    int
}

// NewGenerator returns a generator using the given evaluation.
func NewGenerator(evaluator eval.Evaluator, hashMB int) *Generator {
	if hashMB < 1 {
		hashMB = 16
	}
	return &Generator{evaluator: evaluator, hashMB: hashMB}
}

// maxPlies stops a game that neither side can finish. The drawing rules make
// this nearly unreachable, but a worker that hung on one game would stall its
// whole unit.
const maxPlies = 600

// Generate plays the games a work unit describes and returns the training
// samples.
func (g *Generator) Generate(ctx context.Context, unit events.WorkUnit) ([]Sample, GameStats, error) {
	if unit.Games <= 0 {
		return nil, GameStats{}, fmt.Errorf("train: work unit %s asks for %d games", unit.ID, unit.Games)
	}
	if unit.NodesPerMove <= 0 {
		// A time limit here would make the unit irreproducible: the same work
		// on a slower node would search less and play differently.
		return nil, GameStats{}, fmt.Errorf("train: work unit %s has no node budget; self-play must not be time-limited", unit.ID)
	}

	rng := newRNG(unit.Seed)
	start := time.Now()

	var samples []Sample
	stats := GameStats{}

	for i := 0; i < unit.Games; i++ {
		if ctx.Err() != nil {
			return samples, stats, ctx.Err()
		}

		opening := chess.StartingFEN
		if n := len(unit.OpeningFENs); n > 0 {
			opening = unit.OpeningFENs[int(rng.next()%uint64(n))]
		}

		gameSamples, gameStats, err := g.playGame(ctx, opening, unit.RandomPlies, unit.NodesPerMove, rng)
		if err != nil {
			return samples, stats, fmt.Errorf("train: unit %s game %d: %w", unit.ID, i, err)
		}

		samples = append(samples, gameSamples...)
		stats.Games++
		stats.Positions += len(gameSamples)
		stats.Nodes += gameStats.Nodes
		stats.WhiteWins += gameStats.WhiteWins
		stats.BlackWins += gameStats.BlackWins
		stats.Draws += gameStats.Draws
		stats.SkippedNoisy += gameStats.SkippedNoisy
		stats.SkippedTooLong += gameStats.SkippedTooLong
	}

	stats.Duration = time.Since(start)
	return samples, stats, nil
}

// playGame plays one game and returns its labelled positions.
func (g *Generator) playGame(ctx context.Context, openingFEN string, randomPlies, nodesPerMove int, rng *rng) ([]Sample, GameStats, error) {
	var stats GameStats

	pos, err := chess.ParseFEN(openingFEN)
	if err != nil {
		return nil, stats, fmt.Errorf("unusable opening %q: %w", openingFEN, err)
	}

	// Random opening plies are what stop every game being the same game.
	// Self-play from one position with a deterministic engine produces
	// identical games and therefore worthless training data — the network
	// would see one line a million times.
	for ply := 0; ply < randomPlies; ply++ {
		moves := pos.LegalMoves()
		if len(moves) == 0 {
			break
		}
		pos.MakeMove(moves[rng.next()%uint64(len(moves))])
	}

	// If the random opening already ended the game, there is nothing to learn
	// from it.
	if outcome, _ := pos.Outcome(); outcome != chess.Ongoing {
		return nil, stats, nil
	}

	// One searcher per side, each with its own table, matching how the
	// gatekeeper runs its matches. Single-threaded: lazy SMP is
	// non-deterministic and would break reproducibility.
	white := search.New(g.evaluator, g.hashMB)
	black := search.New(g.evaluator, g.hashMB)

	type candidate struct {
		pos   *chess.Position
		score int16
		ply   uint16
	}
	var pending []candidate

	outcome := chess.Ongoing
	for ply := 0; ply < maxPlies; ply++ {
		if ctx.Err() != nil {
			return nil, stats, ctx.Err()
		}

		outcome, _ = pos.Outcome()
		if outcome != chess.Ongoing {
			break
		}

		engine := white
		if pos.SideToMove() == chess.Black {
			engine = black
		}

		result := engine.Search(ctx, pos, search.Limits{Nodes: uint64(nodesPerMove)}, nil)
		if result.Best.IsNone() {
			break
		}
		stats.Nodes += result.Info.Nodes

		// Positions where the side to move is in check, or whose best move is
		// a capture, are dropped. Their evaluation reflects a tactic that is
		// about to resolve rather than the position's actual character, and
		// training on them teaches the network to mimic the search rather than
		// to judge quiet positions — which is the only thing it is asked to do
		// at the leaves.
		if pos.InCheck() || pos.IsCapture(result.Best) || eval.IsMateScore(result.Info.Score) {
			stats.SkippedNoisy++
		} else {
			// The search reports from the side to move's perspective; samples
			// are stored White-relative.
			score := result.Info.Score
			if pos.SideToMove() == chess.Black {
				score = -score
			}
			pending = append(pending, candidate{
				pos:   pos.Clone(),
				score: clampScore(score),
				ply:   uint16(ply),
			})
		}

		pos.MakeMove(result.Best)
	}

	if outcome == chess.Ongoing {
		// The game hit the ply cap without finishing. Its positions have no
		// trustworthy label, so they are discarded rather than guessed at.
		stats.SkippedTooLong += len(pending)
		return nil, stats, nil
	}

	label := Drawn
	switch outcome {
	case chess.WhiteWins:
		label = WhiteWon
		stats.WhiteWins++
	case chess.BlackWins:
		label = BlackWon
		stats.BlackWins++
	default:
		stats.Draws++
	}

	samples := make([]Sample, 0, len(pending))
	for _, c := range pending {
		samples = append(samples, Sample{
			Position: c.pos,
			Score:    c.score,
			Result:   label,
			Ply:      c.ply,
		})
	}
	return samples, stats, nil
}

// clampScore keeps an evaluation inside the range the record can hold.
func clampScore(v int) int16 {
	const limit = 30000
	if v > limit {
		return limit
	}
	if v < -limit {
		return -limit
	}
	return int16(v)
}

// rng is the deterministic generator work units are seeded from.
//
// It is defined here rather than taken from math/rand so that the sequence is
// fixed by this code and cannot change under us: a standard library
// implementation is free to alter its algorithm between releases, which would
// silently make every previously generated work unit irreproducible.
type rng struct{ state uint64 }

func newRNG(seed uint64) *rng {
	if seed == 0 {
		// A zero state would produce nothing but zeroes.
		seed = 0x9E3779B97F4A7C15
	}
	return &rng{state: seed}
}

func (r *rng) next() uint64 {
	r.state ^= r.state >> 12
	r.state ^= r.state << 25
	r.state ^= r.state >> 27
	return r.state * 2685821657736338717
}
