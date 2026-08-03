package gate

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/piwi3910/novachess/internal/eval"
)

// Report is the outcome of gating a candidate.
type Report struct {
	Verdict Verdict

	Wins   int
	Losses int
	Draws  int
	Games  int

	// LLR is the log-likelihood ratio the test stopped at, and Lower/Upper the
	// bounds it was compared against.
	LLR   float64
	Lower float64
	Upper float64

	// EloDelta is the point estimate of the strength change and EloMargin its
	// 95% confidence half-width. The margin is reported whether or not the
	// candidate was promoted, because a point estimate from a few hundred games
	// is not a measurement without it.
	EloDelta  float64
	EloMargin float64

	// LOS is the likelihood the candidate is stronger at all, which is a
	// different and weaker question than the one the verdict answers.
	LOS float64

	Nodes    uint64
	Duration time.Duration

	// Reason explains the decision in a form fit for a dashboard or a log.
	Reason string
}

// Promoted reports whether the candidate should replace the incumbent.
//
// An inconclusive match is not a promotion. Running out of games means the
// evidence never arrived, and treating that as success would promote every
// candidate that failed to prove itself either way — which over generations is
// a random walk rather than a training loop.
func (r Report) Promoted() bool { return r.Verdict == Accept }

// Gate runs a match under a sequential test.
type Gate struct {
	sprt  SPRT
	match MatchConfig
}

// New returns a gate with the given test and match settings.
func New(s SPRT, m MatchConfig) (*Gate, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	if err := m.validate(); err != nil {
		return nil, err
	}
	return &Gate{sprt: s, match: m}, nil
}

// Test plays candidate against incumbent, stopping as soon as the sequential
// test reaches a conclusion.
//
// The test is applied after each colour-reversed pair rather than after each
// game. Checking mid-pair would let a match stop on the half of a pair where
// the candidate had White, turning an opening's imbalance into a verdict.
func (g *Gate) Test(ctx context.Context, candidate, incumbent eval.Evaluator) (Report, error) {
	var report Report
	report.Lower, report.Upper = g.sprt.Bounds()

	if candidate == nil || incumbent == nil {
		return report, fmt.Errorf("gate: both a candidate and an incumbent are required")
	}

	start := time.Now()
	pairs := g.match.MaxGames / 2

	for pair := 0; pair < pairs; pair++ {
		if ctx.Err() != nil {
			return report, ctx.Err()
		}

		// One pair at a time, reusing the match runner so that a gated match
		// and a plain one play identically.
		cfg := g.match
		cfg.MaxGames = 2
		cfg.Seed = g.match.Seed + uint64(pair)*0x9E3779B97F4A7C15
		cfg.Openings = g.match.Openings
		if len(cfg.Openings) > 0 {
			cfg.Openings = []string{g.match.Openings[pair%len(g.match.Openings)]}
		}

		result, err := RunMatch(ctx, candidate, incumbent, cfg)
		if err != nil {
			return report, err
		}

		report.Wins += result.Wins
		report.Losses += result.Losses
		report.Draws += result.Draws
		report.Games += result.Games
		report.Nodes += result.Nodes

		verdict, llr := g.sprt.Test(report.Wins, report.Losses, report.Draws)
		report.Verdict, report.LLR = verdict, llr

		if verdict != Inconclusive {
			break
		}
	}

	report.EloDelta, report.EloMargin = Elo(report.Wins, report.Losses, report.Draws)
	report.LOS = LOS(report.Wins, report.Losses)
	report.Duration = time.Since(start)
	report.Reason = g.explain(report)

	return report, nil
}

// explain puts the decision into a sentence, including the numbers that would
// otherwise have to be recovered from the fields to understand it.
func (g *Gate) explain(r Report) string {
	elo := "unbounded"
	if !math.IsInf(r.EloDelta, 0) {
		elo = fmt.Sprintf("%+.1f +/- %.1f Elo", r.EloDelta, r.EloMargin)
	}

	switch r.Verdict {
	case Accept:
		return fmt.Sprintf("promoted after %d games: %s, LLR %.2f crossed %.2f (+%v/-%v/=%v)",
			r.Games, elo, r.LLR, r.Upper, r.Wins, r.Losses, r.Draws)
	case Reject:
		return fmt.Sprintf("rejected after %d games: %s, LLR %.2f crossed %.2f (+%v/-%v/=%v)",
			r.Games, elo, r.LLR, r.Lower, r.Wins, r.Losses, r.Draws)
	default:
		return fmt.Sprintf("inconclusive after %d games, the match limit: %s, LLR %.2f is still between %.2f and %.2f (+%v/-%v/=%v)",
			r.Games, elo, r.LLR, r.Lower, r.Upper, r.Wins, r.Losses, r.Draws)
	}
}
