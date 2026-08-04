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
//
// Pairs are played g.match.Concurrency at a time, and results are folded into
// the running tally - and tested against the SPRT bounds - in the order they
// arrive, which need not be index order. That means the number of games a run
// stops after is no longer fixed by the seed alone: at Concurrency > 1 it can
// vary from one run to the next, because which pair happens to be the one
// that pushes the LLR across a bound depends on how the goroutines were
// scheduled. This does not weaken the test itself - the SPRT's error
// guarantees rest on each game being an independent sample of the same
// distribution, which is exactly what a pair played by its own pair of
// single-threaded searchers, from an opening that is a pure function of its
// index, still is. Only the stopping point moves; the promotion decision it
// protects does not become less reliable for arriving sooner.
//
// The verdict itself latches once reached (see fold): up to Concurrency-1
// other pairs can still be in flight at that instant, and their results are
// delivered afterwards while runPairs is cancelling and draining them. A
// verdict recomputed on every arrival would let one of those stragglers pull
// the LLR back inside the band and turn a legitimately crossed verdict into
// Inconclusive - which is exactly backwards: the SPRT's stopping rule is a
// decision made at the moment of crossing, not a running summary that keeps
// being revised.
func (g *Gate) Test(ctx context.Context, candidate, incumbent eval.Evaluator) (Report, error) {
	var report Report
	report.Lower, report.Upper = g.sprt.Bounds()

	if candidate == nil || incumbent == nil {
		return report, fmt.Errorf("gate: both a candidate and an incumbent are required")
	}

	start := time.Now()
	pairs := g.match.MaxGames / 2

	// The report is finished on every exit path, including an interrupted one.
	// A run stopped after four hundred games knows something worth reporting,
	// and returning a zeroed report would throw that away — and would print an
	// empty summary from a caller that had every reason to expect a partial
	// one. The verdict on such a run is whatever the evidence supported when it
	// stopped, which for an interruption is normally inconclusive, and so not a
	// promotion.
	finish := func(err error) (Report, error) {
		report.EloDelta, report.EloMargin = Elo(report.Wins, report.Losses, report.Draws)
		report.LOS = LOS(report.Wins, report.Losses)
		report.Duration = time.Since(start)
		report.Reason = g.explain(report, err)
		return report, err
	}

	// runPairs plays pairs from the same match config the plain runner would,
	// so a gated match and an ungated one play identically pair for pair; the
	// only difference here is stopping once the SPRT concludes, which cancels
	// whatever pairs are still in flight and drains their results before this
	// function returns.
	err := runPairs(ctx, candidate, incumbent, g.match, pairs, func(r pairResult) bool {
		return fold(&report, g.sprt, r)
	})

	return finish(err)
}

// fold merges one pair's result into report and, if the verdict is not yet
// decided, tests it against the SPRT bounds.
//
// Wins, Losses, Draws, Games and Nodes always fold in every pair delivered,
// including stragglers that arrive after a verdict - the report's tally
// stays an honest count of every game actually played, whether or not it
// still moves the decision. Verdict and LLR are different: once Verdict is
// no longer Inconclusive it latches, and neither field is touched again. That
// is what makes the stopping decision stick at the pair that actually crossed
// a bound, rather than at whichever pair happened to be folded in last.
//
// fold reports whether the match is decided, so the caller can keep using its
// return value as runPairs' stop signal after the latch as before.
func fold(report *Report, sprt SPRT, r pairResult) bool {
	report.Wins += r.wins
	report.Losses += r.losses
	report.Draws += r.draws
	report.Games += r.wins + r.losses + r.draws
	report.Nodes += r.nodes

	if report.Verdict != Inconclusive {
		return true
	}

	verdict, llr := sprt.Test(report.Wins, report.Losses, report.Draws)
	report.Verdict, report.LLR = verdict, llr

	return verdict != Inconclusive
}

// explain puts the decision into a sentence, including the numbers that would
// otherwise have to be recovered from the fields to understand it.
func (g *Gate) explain(r Report, err error) string {
	// A clean sweep has unbounded Elo, and so does a confidence interval whose
	// upper end reaches a perfect score — which happens readily on the handful
	// of games a decisive match takes. Both are described in words rather than
	// printed as "+Inf", which reads like a number and is not one.
	elo := "unbounded"
	switch {
	case math.IsInf(r.EloDelta, 0):
	case math.IsInf(r.EloMargin, 0):
		elo = fmt.Sprintf("%+.1f Elo, margin unbounded", r.EloDelta)
	default:
		elo = fmt.Sprintf("%+.1f +/- %.1f Elo", r.EloDelta, r.EloMargin)
	}

	if err != nil {
		return fmt.Sprintf("stopped after %d games (%v): %s, LLR %.2f (+%v/-%v/=%v)",
			r.Games, err, elo, r.LLR, r.Wins, r.Losses, r.Draws)
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
