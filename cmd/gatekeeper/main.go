// Command gatekeeper decides whether a candidate network is strong enough to
// replace the one in use.
//
//	gatekeeper -candidate gen2.nnue -incumbent gen1.nnue
//
// With no incumbent the candidate is matched against the hand-crafted
// evaluation, which is how the first generation is gated.
//
// The exit status is the decision: 0 promoted, 1 rejected, 2 inconclusive, and
// 3 for an error. A run that ends inconclusive is deliberately not a promotion
// — it means the evidence never arrived, and promoting on that would make the
// training loop a random walk.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/piwi3910/novachess/internal/eval"
	"github.com/piwi3910/novachess/internal/gate"
	"github.com/piwi3910/novachess/internal/nnue"
)

// Exit statuses, so a pipeline can branch on the outcome without parsing text.
const (
	exitPromoted     = 0
	exitRejected     = 1
	exitInconclusive = 2
	exitError        = 3
)

func main() {
	sprt := gate.DefaultSPRT()
	match := gate.DefaultMatchConfig()

	var (
		candidatePath = flag.String("candidate", "", "network under test (required)")
		incumbentPath = flag.String("incumbent", "", "network to beat; empty means the hand-crafted evaluation")
		elo0          = flag.Float64("elo0", sprt.Elo0, "improvement not worth taking")
		elo1          = flag.Float64("elo1", sprt.Elo1, "improvement worth taking")
		alpha         = flag.Float64("alpha", sprt.Alpha, "tolerated rate of promoting a network that is not better")
		beta          = flag.Float64("beta", sprt.Beta, "tolerated rate of rejecting one that is")
		games         = flag.Int("games", match.MaxGames, "maximum games before giving up")
		nodes         = flag.Int("nodes", match.NodesPerMove, "search nodes per move")
		hash          = flag.Int("hash", match.HashMB, "transposition table megabytes per side")
		randomPlies   = flag.Int("random-plies", match.RandomPlies, "random moves played from each opening")
		seed          = flag.Uint64("seed", match.Seed, "seed for opening selection")
	)
	flag.Parse()

	if *candidatePath == "" {
		fmt.Fprintln(os.Stderr, "gatekeeper: -candidate is required")
		flag.PrintDefaults()
		os.Exit(exitError)
	}

	sprt.Elo0, sprt.Elo1, sprt.Alpha, sprt.Beta = *elo0, *elo1, *alpha, *beta
	match.MaxGames, match.NodesPerMove, match.HashMB = *games, *nodes, *hash
	match.RandomPlies, match.Seed = *randomPlies, *seed

	candidate, err := nnue.Load(*candidatePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "gatekeeper:", err)
		os.Exit(exitError)
	}

	var incumbent eval.Evaluator = eval.NewHCE()
	incumbentName := "the hand-crafted evaluation"
	if *incumbentPath != "" {
		incumbent, err = nnue.Load(*incumbentPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "gatekeeper:", err)
			os.Exit(exitError)
		}
		incumbentName = *incumbentPath
	}

	g, err := gate.New(sprt, match)
	if err != nil {
		fmt.Fprintln(os.Stderr, "gatekeeper:", err)
		os.Exit(exitError)
	}

	// A cancelled run reports what it has rather than throwing the games away,
	// but its verdict is whatever the evidence supported at that point — which
	// for an interrupted match is normally inconclusive, and so not a
	// promotion.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Fprintf(os.Stderr, "testing %s against %s\n", *candidatePath, incumbentName)
	fmt.Fprintf(os.Stderr, "H0: %+.0f Elo, H1: %+.0f Elo, alpha %.3f, beta %.3f, at most %d games of %d nodes\n",
		sprt.Elo0, sprt.Elo1, sprt.Alpha, sprt.Beta, match.MaxGames, match.NodesPerMove)

	report, err := g.Test(ctx, candidate, incumbent)
	if err != nil && ctx.Err() == nil {
		fmt.Fprintln(os.Stderr, "gatekeeper:", err)
		os.Exit(exitError)
	}

	fmt.Fprintln(os.Stderr, report.Reason)
	fmt.Fprintf(os.Stderr, "LOS %.1f%%, %d nodes searched in %v\n",
		100*report.LOS, report.Nodes, report.Duration.Round(1e6))

	verdictName := map[gate.Verdict]string{
		gate.Accept: "promoted",
		gate.Reject: "rejected",
	}[report.Verdict]
	if verdictName == "" {
		verdictName = "inconclusive"
	}
	summary, _ := json.Marshal(map[string]any{
		"verdict":    verdictName,
		"wins":       report.Wins,
		"losses":     report.Losses,
		"draws":      report.Draws,
		"games":      report.Games,
		"elo_delta":  report.EloDelta,
		"elo_margin": report.EloMargin,
		"los":        report.LOS,
		"reason":     report.Reason,
	})
	// One machine-readable line so a dashboard or script can read the
	// outcome from the log without re-deriving it from the exit code.
	fmt.Fprintf(os.Stderr, "RESULT %s\n", summary)

	switch report.Verdict {
	case gate.Accept:
		os.Exit(exitPromoted)
	case gate.Reject:
		os.Exit(exitRejected)
	default:
		os.Exit(exitInconclusive)
	}
}
