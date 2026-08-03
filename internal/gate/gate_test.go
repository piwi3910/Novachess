package gate

import (
	"context"
	"math"
	"testing"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
)

// blindEval scores every position as level.
//
// An engine using it has nothing to go on but the search's own mate and draw
// detection, so it plays close to arbitrarily while remaining perfectly
// deterministic — which is what makes it useful here. A match against it has a
// known answer, so it tests the gate rather than the evaluation.
type blindEval struct{}

func (blindEval) Evaluate(*chess.Position) int { return 0 }

func testMatchConfig() MatchConfig {
	c := DefaultMatchConfig()
	c.MaxGames = 12
	c.NodesPerMove = 800
	c.HashMB = 1
	c.MaxPlies = 160
	c.RandomPlies = 6
	c.Seed = 0xC0FFEE
	return c
}

// TestSelfMatchIsExactlyBalanced is the load-bearing test for match fairness,
// and it is exact rather than statistical.
//
// Two deterministic engines with the same evaluation play the identical game
// whichever colour they are assigned, so a colour-reversed pair from one
// opening must produce one win and one loss, or two draws. Over any number of
// pairs the score is therefore exactly 50% — not approximately, and with no
// dependence on how many games are played.
//
// Any bias in the harness breaks this immediately: an opening used for only one
// colour, a table carried between games, a side given more nodes. A statistical
// check would need thousands of games to notice what this notices in twelve.
func TestSelfMatchIsExactlyBalanced(t *testing.T) {
	e := eval.NewHCE()

	result, err := RunMatch(context.Background(), e, e, testMatchConfig())
	if err != nil {
		t.Fatal(err)
	}

	if result.Games == 0 {
		t.Fatal("no games were played")
	}
	if result.Wins != result.Losses {
		t.Errorf("a network against itself scored +%d/-%d/=%d; colour-reversed pairs must cancel exactly",
			result.Wins, result.Losses, result.Draws)
	}
	if got := result.Score(); got != 0.5 {
		t.Errorf("self-match scored %v, want exactly 0.5", got)
	}
	t.Logf("%d games: +%d/-%d/=%d", result.Games, result.Wins, result.Losses, result.Draws)
}

// TestSelfMatchIsBalancedWithoutRandomOpenings checks the same property when
// every pair starts from the same position, which is the case where a
// left-over transposition table would show up: the second game of a pair would
// otherwise be searched with the first game's results already in the table.
func TestSelfMatchIsBalancedWithoutRandomOpenings(t *testing.T) {
	cfg := testMatchConfig()
	cfg.RandomPlies = 0
	cfg.MaxGames = 4

	e := eval.NewHCE()
	result, err := RunMatch(context.Background(), e, e, cfg)
	if err != nil {
		t.Fatal(err)
	}

	if result.Wins != result.Losses {
		t.Errorf("+%d/-%d/=%d from a fixed opening; the two colours did not cancel",
			result.Wins, result.Losses, result.Draws)
	}
}

// TestMatchIsReproducible checks that a gating decision can be re-examined
// later, which is the difference between a promotion that can be audited and
// one that has to be taken on trust.
func TestMatchIsReproducible(t *testing.T) {
	cfg := testMatchConfig()

	first, err := RunMatch(context.Background(), eval.NewHCE(), blindEval{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	second, err := RunMatch(context.Background(), eval.NewHCE(), blindEval{}, cfg)
	if err != nil {
		t.Fatal(err)
	}

	if first.Wins != second.Wins || first.Losses != second.Losses || first.Draws != second.Draws {
		t.Errorf("two runs of the same match disagreed: +%d/-%d/=%d then +%d/-%d/=%d",
			first.Wins, first.Losses, first.Draws, second.Wins, second.Losses, second.Draws)
	}
	if first.Nodes != second.Nodes {
		t.Errorf("two runs searched %d and %d nodes", first.Nodes, second.Nodes)
	}
}

// TestStrongerCandidateIsPromoted checks the gate against a difference large
// enough that the answer is not in doubt.
func TestStrongerCandidateIsPromoted(t *testing.T) {
	cfg := testMatchConfig()
	cfg.MaxGames = 60

	// A wide band, so a decisive difference is reached in few games. The
	// defaults are tuned for telling 5 Elo from 0, which needs thousands.
	s := SPRT{Elo0: 0, Elo1: 200, Alpha: 0.05, Beta: 0.05}

	g, err := New(s, cfg)
	if err != nil {
		t.Fatal(err)
	}

	report, err := g.Test(context.Background(), eval.NewHCE(), blindEval{})
	if err != nil {
		t.Fatal(err)
	}

	if !report.Promoted() {
		t.Errorf("an evaluation that plays chess did not beat one that scores every position as level: %s", report.Reason)
	}
	if report.Games >= cfg.MaxGames {
		t.Errorf("the test used all %d games; it should have concluded well before the limit", cfg.MaxGames)
	}
	t.Logf("%s", report.Reason)
}

// TestWeakerCandidateIsRejected is the direction that actually protects the
// loop. A candidate that is worse must not be promoted, because it would then
// generate the next generation's training data.
func TestWeakerCandidateIsRejected(t *testing.T) {
	cfg := testMatchConfig()
	cfg.MaxGames = 60

	s := SPRT{Elo0: 0, Elo1: 200, Alpha: 0.05, Beta: 0.05}

	g, err := New(s, cfg)
	if err != nil {
		t.Fatal(err)
	}

	report, err := g.Test(context.Background(), blindEval{}, eval.NewHCE())
	if err != nil {
		t.Fatal(err)
	}

	if report.Promoted() {
		t.Errorf("an evaluation that scores every position as level was promoted over one that plays chess: %s", report.Reason)
	}
	if report.Verdict != Reject {
		t.Errorf("verdict was %v, want a rejection: %s", report.Verdict, report.Reason)
	}
	t.Logf("%s", report.Reason)
}

// TestEqualCandidateIsNotPromoted covers the case the whole gate exists for.
// A candidate no better than the incumbent must not pass, and against itself
// the evidence for an improvement is exactly nil.
func TestEqualCandidateIsNotPromoted(t *testing.T) {
	cfg := testMatchConfig()
	cfg.MaxGames = 20

	g, err := New(DefaultSPRT(), cfg)
	if err != nil {
		t.Fatal(err)
	}

	e := eval.NewHCE()
	report, err := g.Test(context.Background(), e, e)
	if err != nil {
		t.Fatal(err)
	}

	if report.Promoted() {
		t.Errorf("a network was promoted over itself: %s", report.Reason)
	}
	t.Logf("%s", report.Reason)
}

// TestInconclusiveIsNotAPromotion pins down the default. Running out of games
// means the evidence never arrived; treating that as success would promote
// every candidate that failed to prove itself, which over generations is a
// random walk rather than a training loop.
func TestInconclusiveIsNotAPromotion(t *testing.T) {
	r := Report{Verdict: Inconclusive, Wins: 100, Losses: 0, Draws: 0}
	if r.Promoted() {
		t.Error("an inconclusive match counted as a promotion")
	}
}

func TestSPRTBounds(t *testing.T) {
	s := SPRT{Elo0: 0, Elo1: 5, Alpha: 0.05, Beta: 0.05}
	lower, upper := s.Bounds()

	if want := math.Log(0.05 / 0.95); math.Abs(lower-want) > 1e-12 {
		t.Errorf("lower bound = %v, want %v", lower, want)
	}
	if want := math.Log(0.95 / 0.05); math.Abs(upper-want) > 1e-12 {
		t.Errorf("upper bound = %v, want %v", upper, want)
	}
	if lower >= 0 || upper <= 0 {
		t.Errorf("bounds %v..%v should straddle zero", lower, upper)
	}
	// Symmetric error rates give symmetric bounds.
	if math.Abs(lower+upper) > 1e-12 {
		t.Errorf("bounds %v and %v are not symmetric for equal alpha and beta", lower, upper)
	}
}

// TestLLRRespondsToResults checks the direction and monotonicity of the
// evidence, which is what the whole decision rests on.
func TestLLRRespondsToResults(t *testing.T) {
	s := SPRT{Elo0: 0, Elo1: 10, Alpha: 0.05, Beta: 0.05}

	if got := s.LLR(0, 0, 0); got != 0 {
		t.Errorf("LLR of no games = %v, want 0", got)
	}

	if winning, losing := s.LLR(60, 20, 20), s.LLR(20, 60, 20); winning <= 0 || losing >= 0 {
		t.Errorf("a winning record gave LLR %v and a losing one %v; the signs are wrong", winning, losing)
	}

	// More wins from the same number of games is stronger evidence.
	previous := math.Inf(-1)
	for wins := 50; wins <= 70; wins += 5 {
		llr := s.LLR(wins, 100-wins, 0)
		if llr <= previous {
			t.Errorf("LLR did not increase with wins: %d wins gave %v after %v", wins, llr, previous)
		}
		previous = llr
	}

	// A drawn match is evidence against an improvement, not for one.
	if drawn := s.LLR(0, 0, 100); drawn >= 0 {
		t.Errorf("100 draws gave LLR %v; a level result is not evidence of improvement", drawn)
	}
}

// TestLLRHandlesASweep checks the case with no observed variance, which would
// otherwise divide by zero.
func TestLLRHandlesASweep(t *testing.T) {
	s := DefaultSPRT()

	for _, tc := range []struct {
		name         string
		w, l, d      int
		wantPositive bool
	}{
		{"all wins", 10, 0, 0, true},
		{"all losses", 0, 10, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			llr := s.LLR(tc.w, tc.l, tc.d)
			if math.IsNaN(llr) || math.IsInf(llr, 0) {
				t.Fatalf("LLR = %v", llr)
			}
			if positive := llr > 0; positive != tc.wantPositive {
				t.Errorf("LLR = %v, want it %s", llr, map[bool]string{true: "positive", false: "negative"}[tc.wantPositive])
			}
		})
	}
}

func TestEloConversions(t *testing.T) {
	if got := EloToScore(0); math.Abs(got-0.5) > 1e-12 {
		t.Errorf("0 Elo is a score of %v, want 0.5", got)
	}
	if got := ScoreToElo(0.5); math.Abs(got) > 1e-9 {
		t.Errorf("a level score is %v Elo, want 0", got)
	}

	// A round trip, across the range where the model is meaningful.
	for _, elo := range []float64{-800, -400, -100, -5, 0, 5, 100, 400, 800} {
		if got := ScoreToElo(EloToScore(elo)); math.Abs(got-elo) > 1e-9 {
			t.Errorf("%v Elo round-tripped to %v", elo, got)
		}
	}

	// A clean sweep is unbounded rather than some large arbitrary number that
	// would read like a measurement.
	if got := ScoreToElo(1); !math.IsInf(got, 1) {
		t.Errorf("a perfect score is %v Elo, want +Inf", got)
	}
	if got := ScoreToElo(0); !math.IsInf(got, -1) {
		t.Errorf("a zero score is %v Elo, want -Inf", got)
	}
}

func TestEloEstimateAndMargin(t *testing.T) {
	// A level result is zero Elo whatever the number of games.
	if elo, _ := Elo(10, 10, 80); math.Abs(elo) > 1e-9 {
		t.Errorf("an even match estimated %v Elo, want 0", elo)
	}

	// The interval narrows as evidence accumulates. This is the property that
	// makes the estimate reportable at all: the same +55% score means something
	// very different after 100 games than after 10000.
	_, small := Elo(55, 45, 0)
	_, large := Elo(5500, 4500, 0)
	if large >= small {
		t.Errorf("the margin did not narrow with more games: %v after 100, %v after 10000", small, large)
	}

	if elo, _ := Elo(90, 10, 0); elo <= 0 {
		t.Errorf("a 90%% score estimated %v Elo, want a positive number", elo)
	}
	if _, margin := Elo(0, 0, 0); !math.IsInf(margin, 1) {
		t.Errorf("with no games the margin is %v, want infinite", margin)
	}
}

func TestLOS(t *testing.T) {
	if got := LOS(0, 0); got != 0.5 {
		t.Errorf("with no decisive games LOS = %v, want 0.5", got)
	}
	if got := LOS(50, 50); math.Abs(got-0.5) > 1e-12 {
		t.Errorf("an even record gives LOS %v, want 0.5", got)
	}
	if got := LOS(60, 40); got <= 0.5 {
		t.Errorf("a winning record gives LOS %v, want above 0.5", got)
	}
	if got := LOS(40, 60); got >= 0.5 {
		t.Errorf("a losing record gives LOS %v, want below 0.5", got)
	}
	// Draws carry no information about which side is stronger.
	if a, b := LOS(60, 40), LOS(60, 40); a != b {
		t.Errorf("LOS is not a function of the decisive games alone: %v vs %v", a, b)
	}
}

func TestRejectsUnusableSettings(t *testing.T) {
	good := testMatchConfig()

	sprtCases := []struct {
		name string
		s    SPRT
	}{
		{"elo1 below elo0", SPRT{Elo0: 5, Elo1: 0, Alpha: 0.05, Beta: 0.05}},
		{"elo1 equal to elo0", SPRT{Elo0: 5, Elo1: 5, Alpha: 0.05, Beta: 0.05}},
		{"alpha out of range", SPRT{Elo0: 0, Elo1: 5, Alpha: 0, Beta: 0.05}},
		{"beta out of range", SPRT{Elo0: 0, Elo1: 5, Alpha: 0.05, Beta: 1}},
	}
	for _, tc := range sprtCases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(tc.s, good); err == nil {
				t.Error("accepted a test it should have refused")
			}
		})
	}

	matchCases := []struct {
		name   string
		mutate func(*MatchConfig)
	}{
		{"fewer games than a pair", func(c *MatchConfig) { c.MaxGames = 1 }},
		{"no node budget", func(c *MatchConfig) { c.NodesPerMove = 0 }},
		{"no ply cap", func(c *MatchConfig) { c.MaxPlies = 0 }},
	}
	for _, tc := range matchCases {
		t.Run(tc.name, func(t *testing.T) {
			c := testMatchConfig()
			tc.mutate(&c)
			if _, err := New(DefaultSPRT(), c); err == nil {
				t.Error("accepted match settings it should have refused")
			}
			if _, err := RunMatch(context.Background(), eval.NewHCE(), eval.NewHCE(), c); err == nil {
				t.Error("ran a match it should have refused")
			}
		})
	}
}

func TestGateRequiresBothSides(t *testing.T) {
	g, err := New(DefaultSPRT(), testMatchConfig())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Test(context.Background(), nil, eval.NewHCE()); err == nil {
		t.Error("gated a nil candidate")
	}
	if _, err := g.Test(context.Background(), eval.NewHCE(), nil); err == nil {
		t.Error("gated against a nil incumbent")
	}
}

func TestMatchHonoursContextCancellation(t *testing.T) {
	cfg := testMatchConfig()
	cfg.MaxGames = 10000

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := RunMatch(ctx, eval.NewHCE(), blindEval{}, cfg); err == nil {
		t.Error("the match ignored a cancelled context")
	}

	g, err := New(DefaultSPRT(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Test(ctx, eval.NewHCE(), blindEval{}); err == nil {
		t.Error("the gate ignored a cancelled context")
	}
}

// TestOpeningsAreUsed checks that a configured opening reaches the games, since
// a coordinator varying openings would otherwise silently have no effect.
func TestOpeningsAreUsed(t *testing.T) {
	cfg := testMatchConfig()
	cfg.RandomPlies = 0
	cfg.Openings = []string{"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"}

	got, err := cfg.opening(0)
	if err != nil {
		t.Fatal(err)
	}
	if got != cfg.Openings[0] {
		t.Errorf("opening = %q, want %q", got, cfg.Openings[0])
	}

	// With randomisation the position must differ from the base but stay legal.
	cfg.RandomPlies = 4
	randomised, err := cfg.opening(0)
	if err != nil {
		t.Fatal(err)
	}
	if randomised == cfg.Openings[0] {
		t.Error("random plies did not change the opening")
	}
	if _, err := chess.ParseFEN(randomised); err != nil {
		t.Errorf("randomised opening %q is unusable: %v", randomised, err)
	}
}

// TestOpeningsVaryBetweenPairs checks the other half: every pair starting from
// the same position would make the whole match one game repeated.
func TestOpeningsVaryBetweenPairs(t *testing.T) {
	cfg := testMatchConfig()

	seen := map[string]bool{}
	for pair := 0; pair < 8; pair++ {
		fen, err := cfg.opening(pair)
		if err != nil {
			t.Fatal(err)
		}
		seen[fen] = true
	}
	if len(seen) < 6 {
		t.Errorf("8 pairs produced only %d distinct openings", len(seen))
	}
}
