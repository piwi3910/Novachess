// Command novabot plays chess on Lichess using the Novachess engine.
//
// It needs an API token with the bot:play scope, from an account that has been
// upgraded to a bot account. Pass the token in LICHESS_TOKEN.
//
//	export LICHESS_TOKEN=lip_xxxxxxxxxxxx
//	novabot
//
// An account is upgraded once, irreversibly, and only while it has never played
// a rated game:
//
//	novabot -upgrade
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/piwi3910/novachess/internal/lichess"
)

func main() {
	var (
		baseURL  = flag.String("url", lichess.DefaultBaseURL, "Lichess API base URL")
		upgrade  = flag.Bool("upgrade", false, "upgrade this account to a bot account (irreversible) and exit")
		games    = flag.Int("games", 4, "maximum concurrent games")
		threads  = flag.Int("threads", 1, "search threads per game")
		hash     = flag.Int("hash", 128, "transposition table size in megabytes per game")
		minTime  = flag.Int("min-initial-seconds", 60, "decline games with less initial time than this")
		variants = flag.String("variants", "standard,chess960", "comma-separated playable variants")
		greeting = flag.String("greeting", "", "message posted in the chat at the start of each game")
		rated    = flag.Bool("rated", true, "accept rated challenges")
		casual   = flag.Bool("casual", true, "accept casual challenges")
		verbose  = flag.Bool("v", false, "log debug detail")
	)
	flag.Parse()

	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	token := os.Getenv("LICHESS_TOKEN")
	if token == "" {
		fmt.Fprintln(os.Stderr, "novabot: LICHESS_TOKEN is not set")
		fmt.Fprintln(os.Stderr, "Create a token with the bot:play scope at https://lichess.org/account/oauth/token")
		os.Exit(1)
	}

	client := lichess.NewClient(token, *baseURL)

	// Signals are handled so that stopping the bot cancels its games rather
	// than abandoning them mid-move, which would leave opponents waiting out
	// the clock.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if *upgrade {
		if err := client.Upgrade(ctx); err != nil {
			fmt.Fprintln(os.Stderr, "novabot: upgrade failed:", err)
			os.Exit(1)
		}
		fmt.Println("Account upgraded to a bot account.")
		return
	}

	config := lichess.DefaultConfig()
	config.MaxConcurrentGames = *games
	config.Threads = *threads
	config.HashMB = *hash
	config.MinInitialSeconds = *minTime
	config.AcceptRated = *rated
	config.AcceptCasual = *casual
	config.GreetingMessage = *greeting
	config.AcceptVariants = splitVariants(*variants)

	bot := lichess.NewBot(client, config, log)

	if err := bot.Run(ctx); err != nil && ctx.Err() == nil {
		fmt.Fprintln(os.Stderr, "novabot:", err)
		os.Exit(1)
	}
	log.Info("shutting down")
}

func splitVariants(s string) []string {
	var out []string
	for _, v := range strings.Split(s, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}
