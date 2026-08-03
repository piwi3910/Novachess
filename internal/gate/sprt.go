// Package gate decides whether a newly trained network is actually stronger
// than the one it would replace.
//
// This is the step that makes the training loop self-correcting rather than
// merely self-feeding. A network's training loss says how well it fit the data
// it was shown, which is not the same question as whether it plays better — a
// network can improve its loss and lose games, and if it were promoted on loss
// alone it would generate the next generation's data and the whole loop would
// drift somewhere worse with nothing to notice.
//
// So a candidate has to win a match. The match is judged by a sequential
// probability ratio test rather than a fixed number of games, because the
// number of games needed to tell a 10 Elo improvement from noise is enormous
// and mostly wasted: a clearly better or clearly worse network is identified in
// a fraction of them, and only the genuinely marginal cases run long.
package gate

import (
	"fmt"
	"math"
)

// SPRT is the hypothesis test applied to a match in progress.
//
// The two hypotheses are Elo differences rather than "better" and "worse",
// which matters: testing against zero would eventually promote a network that
// is a fraction of an Elo better, at the cost of running forever to prove it.
// Elo0 is the improvement considered not worth having and Elo1 the one worth
// accepting, and the test distinguishes those two rather than splitting at a
// point.
type SPRT struct {
	// Elo0 is the null hypothesis: the smallest improvement not worth taking.
	Elo0 float64
	// Elo1 is the alternative: an improvement worth taking.
	Elo1 float64
	// Alpha is the tolerated rate of promoting a network that is not better.
	Alpha float64
	// Beta is the tolerated rate of rejecting one that is.
	Beta float64
}

// DefaultSPRT returns the conventional engine-testing bounds: reject an
// improvement of 0 Elo, accept one of 5, with 5% error either way.
func DefaultSPRT() SPRT {
	return SPRT{Elo0: 0, Elo1: 5, Alpha: 0.05, Beta: 0.05}
}

func (s SPRT) validate() error {
	switch {
	case s.Elo1 <= s.Elo0:
		return fmt.Errorf("gate: elo1 (%v) must exceed elo0 (%v)", s.Elo1, s.Elo0)
	case s.Alpha <= 0 || s.Alpha >= 1:
		return fmt.Errorf("gate: alpha %v is outside 0..1", s.Alpha)
	case s.Beta <= 0 || s.Beta >= 1:
		return fmt.Errorf("gate: beta %v is outside 0..1", s.Beta)
	}
	return nil
}

// Verdict is what the test concludes from the games so far.
type Verdict int

const (
	// Inconclusive means neither bound has been crossed and more games would
	// help.
	Inconclusive Verdict = iota
	// Accept means the candidate is better by the margin that was asked for.
	Accept
	// Reject means it is not.
	Reject
)

func (v Verdict) String() string {
	switch v {
	case Accept:
		return "accepted"
	case Reject:
		return "rejected"
	default:
		return "inconclusive"
	}
}

// Bounds returns the log-likelihood-ratio thresholds. Crossing the upper bound
// accepts the candidate and the lower rejects it.
func (s SPRT) Bounds() (lower, upper float64) {
	return math.Log(s.Beta / (1 - s.Alpha)), math.Log((1 - s.Beta) / s.Alpha)
}

// EloToScore converts an Elo difference into an expected score per game, under
// the logistic model the rating system assumes.
func EloToScore(elo float64) float64 {
	return 1 / (1 + math.Pow(10, -elo/400))
}

// ScoreToElo inverts it. A clean sweep is unbounded Elo, so the extremes are
// reported as infinities rather than as some large arbitrary number that would
// read like a measurement.
func ScoreToElo(score float64) float64 {
	if score <= 0 {
		return math.Inf(-1)
	}
	if score >= 1 {
		return math.Inf(1)
	}
	return -400 * math.Log10(1/score-1)
}

// LLR returns the log-likelihood ratio of the results under the two hypotheses.
//
// This is the generalized form, which estimates the variance of the score from
// the observed results rather than assuming a fixed draw rate. That matters
// here because self-play between two closely related networks draws far more
// often than ordinary play, and a test that assumed otherwise would need many
// more games to reach the same confidence.
func (s SPRT) LLR(wins, losses, draws int) float64 {
	n := wins + losses + draws
	if n == 0 {
		return 0
	}

	s0, s1 := EloToScore(s.Elo0), EloToScore(s.Elo1)
	observed := (float64(wins) + 0.5*float64(draws)) / float64(n)

	// Per-game variance of the score, from the results themselves.
	variance := (float64(wins)*sq(1-observed) +
		float64(draws)*sq(0.5-observed) +
		float64(losses)*sq(0-observed)) / float64(n)

	// An unbroken run of one result has no observed variance, and the ratio
	// would divide by zero. The floor corresponds to the uncertainty a single
	// additional game would carry, which keeps a short sweep from reading as
	// infinite evidence.
	if minimum := 1 / (4 * float64(n)); variance < minimum {
		variance = minimum
	}

	return float64(n) * (s1 - s0) * (2*observed - s0 - s1) / (2 * variance)
}

// Test evaluates the results so far.
func (s SPRT) Test(wins, losses, draws int) (Verdict, float64) {
	llr := s.LLR(wins, losses, draws)
	lower, upper := s.Bounds()

	switch {
	case llr >= upper:
		return Accept, llr
	case llr <= lower:
		return Reject, llr
	default:
		return Inconclusive, llr
	}
}

// LOS is the likelihood that the candidate is genuinely stronger, given the
// decisive games. Draws carry no information about which side is better, so
// they do not appear.
//
// It is reported alongside the verdict rather than used to make it: LOS asks
// whether the difference is above zero, and the whole point of testing against
// Elo0 and Elo1 is that being above zero by an invisible margin is not a reason
// to promote anything.
func LOS(wins, losses int) float64 {
	decisive := wins + losses
	if decisive == 0 {
		return 0.5
	}
	return 0.5 * (1 + math.Erf(float64(wins-losses)/(2*math.Sqrt(float64(decisive)))))
}

// Elo estimates the strength difference from a match result, with the
// confidence interval that says how much to trust it.
//
// The interval is the useful half. A point estimate from a few hundred games
// carries an interval tens of Elo wide, and reporting the estimate alone
// invites reading a rounding error as progress.
func Elo(wins, losses, draws int) (estimate, margin float64) {
	n := wins + losses + draws
	if n == 0 {
		return 0, math.Inf(1)
	}

	score := (float64(wins) + 0.5*float64(draws)) / float64(n)
	estimate = ScoreToElo(score)

	variance := (float64(wins)*sq(1-score) +
		float64(draws)*sq(0.5-score) +
		float64(losses)*sq(0-score)) / float64(n)
	stddev := math.Sqrt(variance / float64(n))

	// 95%, two-sided.
	const z = 1.959963985
	lower, upper := ScoreToElo(score-z*stddev), ScoreToElo(score+z*stddev)
	return estimate, (upper - lower) / 2
}

func sq(v float64) float64 { return v * v }
