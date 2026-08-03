package gate

import (
	"context"
	"fmt"
	"time"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
	"github.com/piwi3910/novachess/internal/search"
)

// MatchConfig describes how two evaluations are compared.
type MatchConfig struct {
	// MaxGames bounds the match when the test never reaches a conclusion. It
	// is rounded down to a whole number of colour-reversed pairs.
	MaxGames int

	// NodesPerMove fixes the search effort. As in self-play this must be a node
	// count and never a time limit — a match decided partly by which process
	// got more CPU is not a measurement of anything.
	NodesPerMove int

	// HashMB is the transposition table size given to each side. Both get the
	// same, since a table size advantage is a strength advantage.
	HashMB int

	// Openings are the positions games start from. Self-play between
	// deterministic engines from a single position produces one game, so a
	// match without opening variety measures that one game rather than the
	// networks.
	Openings []string

	// RandomPlies adds further variety by playing random moves from each
	// opening before the engines take over.
	RandomPlies int

	// Seed fixes the opening randomisation.
	Seed uint64

	// MaxPlies abandons a game neither side can finish. Such a game is scored
	// as a draw, which is what it is.
	MaxPlies int
}

// DefaultMatchConfig returns settings suitable for gating one generation
// against the next.
func DefaultMatchConfig() MatchConfig {
	return MatchConfig{
		MaxGames:     2000,
		NodesPerMove: 20000,
		HashMB:       16,
		RandomPlies:  8,
		Seed:         1,
		MaxPlies:     400,
	}
}

func (c MatchConfig) validate() error {
	switch {
	case c.MaxGames < 2:
		return fmt.Errorf("gate: %d games is not enough for a colour-reversed pair", c.MaxGames)
	case c.NodesPerMove <= 0:
		return fmt.Errorf("gate: no node budget; a match must not be decided by wall-clock speed")
	case c.MaxPlies <= 0:
		return fmt.Errorf("gate: ply cap %d", c.MaxPlies)
	}
	return nil
}

// MatchResult is the tally, always from the candidate's point of view.
type MatchResult struct {
	Wins   int
	Losses int
	Draws  int

	Games    int
	Nodes    uint64
	Duration time.Duration
}

// Score is the candidate's points per game.
func (r MatchResult) Score() float64 {
	if r.Games == 0 {
		return 0
	}
	return (float64(r.Wins) + 0.5*float64(r.Draws)) / float64(r.Games)
}

// outcome of a single game, from White's point of view.
type gameOutcome int

const (
	whiteWon gameOutcome = iota
	blackWon
	drawn
)

// playGame plays one game between two evaluations and reports who won.
//
// Each side gets its own searcher with its own table, and both are cleared
// between games. A table carried over would make a game depend on the ones
// before it, which would both break reproducibility and quietly favour whichever
// side happened to reach familiar positions.
func playGame(ctx context.Context, white, black eval.Evaluator, opening string, cfg MatchConfig) (gameOutcome, uint64, error) {
	pos, err := chess.ParseFEN(opening)
	if err != nil {
		return drawn, 0, fmt.Errorf("gate: unusable opening %q: %w", opening, err)
	}

	// One thread each: lazy SMP is non-deterministic, and a gating decision
	// that cannot be reproduced cannot be audited when a promotion later looks
	// like a mistake.
	engines := map[chess.Color]*search.Searcher{
		chess.White: search.New(white, cfg.HashMB),
		chess.Black: search.New(black, cfg.HashMB),
	}

	var nodes uint64

	for ply := 0; ply < cfg.MaxPlies; ply++ {
		if ctx.Err() != nil {
			return drawn, nodes, ctx.Err()
		}

		outcome, _ := pos.Outcome()
		switch outcome {
		case chess.WhiteWins:
			return whiteWon, nodes, nil
		case chess.BlackWins:
			return blackWon, nodes, nil
		case chess.Draw:
			return drawn, nodes, nil
		}

		result := engines[pos.SideToMove()].Search(ctx, pos, search.Limits{Nodes: uint64(cfg.NodesPerMove)}, nil)
		if result.Best.IsNone() {
			// No legal move with the game not already over should be
			// impossible, but scoring it as a draw is the safe reading.
			return drawn, nodes, nil
		}
		nodes += result.Info.Nodes
		pos.MakeMove(result.Best)
	}

	// The ply cap was reached. Neither side demonstrated a win, so it is a
	// draw — the same conclusion an arbiter would reach.
	return drawn, nodes, nil
}

// RunMatch plays candidate against incumbent and returns the tally.
//
// Games are played in colour-reversed pairs from a shared opening. That is not
// merely tidy: openings are not perfectly balanced, and playing each one once
// would measure the openings as much as the networks. Playing both sides of
// every opening cancels that out exactly.
func RunMatch(ctx context.Context, candidate, incumbent eval.Evaluator, cfg MatchConfig) (MatchResult, error) {
	var result MatchResult

	if err := cfg.validate(); err != nil {
		return result, err
	}

	start := time.Now()
	pairs := cfg.MaxGames / 2

	for pair := 0; pair < pairs; pair++ {
		if ctx.Err() != nil {
			return result, ctx.Err()
		}

		opening, err := cfg.opening(pair)
		if err != nil {
			return result, err
		}

		// The candidate takes White in the first game of the pair and Black in
		// the second.
		for game := 0; game < 2; game++ {
			candidateIsWhite := game == 0

			white, black := candidate, incumbent
			if !candidateIsWhite {
				white, black = incumbent, candidate
			}

			outcome, nodes, err := playGame(ctx, white, black, opening, cfg)
			if err != nil {
				return result, err
			}

			result.Games++
			result.Nodes += nodes

			switch {
			case outcome == drawn:
				result.Draws++
			case (outcome == whiteWon) == candidateIsWhite:
				result.Wins++
			default:
				result.Losses++
			}
		}
	}

	result.Duration = time.Since(start)
	return result, nil
}

// opening returns the starting position for a pair.
func (c MatchConfig) opening(pair int) (string, error) {
	base := chess.StartingFEN
	if n := len(c.Openings); n > 0 {
		base = c.Openings[pair%n]
	}

	if c.RandomPlies <= 0 {
		return base, nil
	}

	pos, err := chess.ParseFEN(base)
	if err != nil {
		return "", fmt.Errorf("gate: unusable opening %q: %w", base, err)
	}

	// Each pair gets its own stream, derived from the seed and the pair number,
	// so that a pair's opening does not depend on how many pairs came before —
	// which is what lets a match be resumed or a single pair reproduced in
	// isolation when a result needs checking.
	r := newRNG(c.Seed + uint64(pair)*0x9E3779B97F4A7C15)

	for i := 0; i < c.RandomPlies; i++ {
		moves := pos.LegalMoves()
		if len(moves) == 0 {
			break
		}
		pos.MakeMove(moves[r.next()%uint64(len(moves))])
	}

	// A randomised opening that already ended is no use as a starting point.
	if outcome, _ := pos.Outcome(); outcome != chess.Ongoing {
		return base, nil
	}
	return pos.FEN(), nil
}

// rng is the same deterministic generator the rest of the pipeline uses,
// defined here rather than taken from math/rand so the sequence is fixed by
// this code and cannot change under a standard library release.
type rng struct{ state uint64 }

func newRNG(seed uint64) *rng {
	if seed == 0 {
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
