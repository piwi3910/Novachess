// Command coordinator drives the training loop.
//
// It issues work units, counts the batches workers return, and announces a
// dataset once a generation has enough positions. There must be exactly one:
// two coordinators issuing units for the same generation would each count half
// the batches and neither would reach its target.
//
// Configuration comes from the environment, because that is what a Kubernetes
// manifest sets:
//
//	NOVA_BUS_URLS      comma-separated NATS URLs
//	NOVA_DATA_DIR      shared volume for artifacts
//	NOVA_GENERATION    generation to run; a restart resumes here
//	NOVA_POSITIONS     positions a generation needs before training
//	NOVA_GAMES_PER_UNIT games in one work unit
//	NOVA_NODES         search nodes per move
//	NOVA_UNITS_IN_FLIGHT units to keep queued
//	NOVA_NETWORK_URI   network for workers to play with; empty bootstraps
//	NOVA_SEED          derives every unit's seed
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/coordinator"
	"github.com/piwi3910/novachess/internal/store"
)

var version = "dev"

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if err := run(log); err != nil {
		log.Error("exiting", "error", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	cfg := coordinator.DefaultConfig()

	var err error
	if cfg.Generation, err = intEnv("NOVA_GENERATION", cfg.Generation); err != nil {
		return err
	}
	if cfg.PositionsPerGeneration, err = intEnv("NOVA_POSITIONS", cfg.PositionsPerGeneration); err != nil {
		return err
	}
	if cfg.GamesPerUnit, err = intEnv("NOVA_GAMES_PER_UNIT", cfg.GamesPerUnit); err != nil {
		return err
	}
	if cfg.NodesPerMove, err = intEnv("NOVA_NODES", cfg.NodesPerMove); err != nil {
		return err
	}
	if cfg.UnitsInFlight, err = intEnv("NOVA_UNITS_IN_FLIGHT", cfg.UnitsInFlight); err != nil {
		return err
	}
	if cfg.RandomPlies, err = intEnv("NOVA_RANDOM_PLIES", cfg.RandomPlies); err != nil {
		return err
	}

	seed, err := intEnv("NOVA_SEED", int(cfg.Seed))
	if err != nil {
		return err
	}
	cfg.Seed = uint64(seed)
	cfg.NetworkURI = os.Getenv("NOVA_NETWORK_URI")

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	artifacts, err := store.NewFS(env("NOVA_DATA_DIR", "/data"))
	if err != nil {
		return err
	}
	defer artifacts.Close()

	connectCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	urls := splitList(env("NOVA_BUS_URLS", "nats://nova-nats:4222"))

	b, err := bus.Connect(connectCtx, bus.Config{
		URLs:       urls,
		ClientName: "coordinator",
		// A durable name, so a rescheduled coordinator resumes counting rather
		// than starting from the present and waiting for batches it has already
		// been sent.
		Durable: "coordinator",
	})
	if err != nil {
		return err
	}
	defer b.Close()

	c, err := coordinator.New(cfg, b, artifacts, log)
	if err != nil {
		return err
	}

	log.Info("starting",
		"version", version,
		"generation", cfg.Generation,
		"target", cfg.PositionsPerGeneration,
		"bus", urls,
		"data", artifacts.Root())

	progress, err := c.RunGeneration(ctx)
	if err != nil && ctx.Err() == nil {
		return err
	}

	log.Info("finished",
		"generation", progress.Generation,
		"positions", progress.Positions,
		"games", progress.Games,
		"batches", progress.Batches,
		"issued", progress.Issued,
		"complete", progress.Complete)
	return nil
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func intEnv(key string, fallback int) (int, error) {
	v := os.Getenv(key)
	if v == "" {
		return fallback, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", key, err)
	}
	return n, nil
}

func splitList(s string) []string {
	var out []string
	for _, v := range strings.Split(s, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}
