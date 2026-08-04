package gate

import (
	"context"
	"errors"
	"fmt"
	"sync"
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

	// Concurrency is how many colour-reversed pairs are played at once. Each
	// pair still runs as a single-threaded search on each side - lazy SMP is
	// non-deterministic and self-play/gating scale by running more of these
	// worker goroutines (or more pods), never more threads per game. Raising
	// this changes only how fast a match finishes, never which openings are
	// played: a pair's opening is a pure function of (Seed, pair index),
	// computed independently of every other pair, so it does not matter which
	// goroutine plays it or when.
	Concurrency int
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
		Concurrency:  4,
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
	case c.Concurrency < 1:
		return fmt.Errorf("gate: concurrency %d must be at least 1", c.Concurrency)
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

// pairResult is what one colour-reversed pair contributed, together with the
// pair index it came from. The index is what lets a concurrent match be
// checked against a sequential one: two runs that produce the same result at
// every index produced the same match, whatever order the goroutines finished
// in.
type pairResult struct {
	index               int
	wins, losses, draws int
	nodes               uint64
}

// playPair plays one colour-reversed pair - the candidate takes White in the
// first game and Black in the second - and returns the candidate's tally for
// it.
//
// The pair's opening is derived solely from cfg and the pair index (see
// MatchConfig.opening), and both games get their own searchers (see
// playGame), so playPair has no state shared with any other pair. That is
// what makes it safe to call from multiple goroutines at once: which pair a
// worker happens to pick up next cannot change what that pair plays.
func playPair(ctx context.Context, candidate, incumbent eval.Evaluator, pair int, cfg MatchConfig) (pairResult, error) {
	result := pairResult{index: pair}

	opening, err := cfg.opening(pair)
	if err != nil {
		return result, err
	}

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

		result.nodes += nodes
		switch {
		case outcome == drawn:
			result.draws++
		case (outcome == whiteWon) == candidateIsWhite:
			result.wins++
		default:
			result.losses++
		}
	}

	return result, nil
}

// runPairs plays every pair from 0 to pairs-1, running up to cfg.Concurrency
// of them at once, and delivers each one's result to onResult as it arrives.
//
// Results arrive in whatever order their pairs finished in, not in index
// order - a slower pair started early does not block a faster one started
// later. onResult returns true to stop the match early (used by the gate's
// SPRT loop once a verdict is reached); runPairs then cancels every pair still
// in flight and waits for them to unwind before returning, so no goroutine is
// left running past the end of the call.
//
// Because every pair is an independent, pure function of (cfg, index), the
// arrival order changes only how soon an early stop can trigger - never which
// games any given pair played, and never the total tallied over a run that
// goes to completion.
func runPairs(ctx context.Context, candidate, incumbent eval.Evaluator, cfg MatchConfig, pairs int, onResult func(pairResult) (stop bool)) error {
	workCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	workers := cfg.Concurrency
	if workers > pairs {
		workers = pairs
	}
	if workers < 1 {
		workers = 1
	}

	indices := make(chan int)
	results := make(chan pairResult)
	errs := make(chan error, 1)

	go func() {
		defer close(indices)
		for i := 0; i < pairs; i++ {
			select {
			case indices <- i:
			case <-workCtx.Done():
				return
			}
		}
	}()

	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for idx := range indices {
				r, err := playPair(workCtx, candidate, incumbent, idx, cfg)
				if err != nil {
					select {
					case errs <- err:
					default:
					}
					cancel()
					return
				}
				// Always delivered, never dropped for a race with cancellation:
				// the consumer below keeps draining results until every worker
				// has exited, so a completed pair's result is never lost - only
				// a pair still in flight when the context is cancelled is.
				results <- r
			}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	for r := range results {
		if onResult(r) {
			// The caller reached a conclusion; cancelling here is what stops
			// the pairs already in flight promptly instead of letting them
			// run to their own completion.
			cancel()
		}
	}

	// A cancellation the caller asked for (via onResult) is not a failure to
	// report - the match concluded exactly as intended. A cancellation of the
	// context passed in by the caller is: it means the run was interrupted
	// from outside, and that has to reach whoever is waiting on the result.
	if ctx.Err() != nil {
		return ctx.Err()
	}

	select {
	case err := <-errs:
		// A per-pair error caused by our own cancellation (context.Canceled)
		// is expected once onResult has asked to stop, and is not itself a
		// failure worth reporting; any other error - an unusable opening, for
		// instance - is.
		if !errors.Is(err, context.Canceled) {
			return err
		}
	default:
	}
	return nil
}

// RunMatch plays candidate against incumbent and returns the tally.
//
// Games are played in colour-reversed pairs from a shared opening. That is not
// merely tidy: openings are not perfectly balanced, and playing each one once
// would measure the openings as much as the networks. Playing both sides of
// every opening cancels that out exactly.
//
// Pairs run cfg.Concurrency at a time. Concurrency changes only how fast the
// match finishes: each pair is played by its own pair of single-threaded
// searchers and derives its opening purely from cfg and its own index, so the
// total tallied here does not depend on how many pairs ran at once or the
// order they finished in.
func RunMatch(ctx context.Context, candidate, incumbent eval.Evaluator, cfg MatchConfig) (MatchResult, error) {
	var result MatchResult

	if err := cfg.validate(); err != nil {
		return result, err
	}

	start := time.Now()
	pairs := cfg.MaxGames / 2

	err := runPairs(ctx, candidate, incumbent, cfg, pairs, func(r pairResult) bool {
		result.Wins += r.wins
		result.Losses += r.losses
		result.Draws += r.draws
		result.Games += r.wins + r.losses + r.draws
		result.Nodes += r.nodes
		return false
	})

	result.Duration = time.Since(start)
	return result, err
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
