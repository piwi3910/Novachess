// Command selfplay-worker plays self-play games for the training pipeline.
//
// It joins a queue group on the bus, takes a work unit at a time, plays the
// games it describes, stores the positions, and announces the batch. Scaling
// the pipeline means running more of these.
//
// Configuration comes from the environment, because that is what a Kubernetes
// manifest sets:
//
//	NOVA_BUS_URLS     comma-separated NATS URLs
//	NOVA_DATA_DIR     shared volume for artifacts
//	NOVA_WORKER_ID    identity in heartbeats and batches; the pod name
//	NOVA_NODE_NAME    the board this runs on
//	NOVA_HASH_MB      transposition table size per search
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
	"github.com/piwi3910/novachess/internal/store"
	"github.com/piwi3910/novachess/internal/worker"
)

// version is stamped onto every batch so a dataset can be traced to the build
// that produced it. Set with -ldflags at build time.
var version = "dev"

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if err := run(log); err != nil {
		log.Error("exiting", "error", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	urls := splitList(env("NOVA_BUS_URLS", "nats://nova-nats:4222"))
	dataDir := env("NOVA_DATA_DIR", "/data")

	// The pod name is the useful identity: when one board produces slow
	// batches, this is what ties them back to it.
	id := env("NOVA_WORKER_ID", "")
	if id == "" {
		host, err := os.Hostname()
		if err != nil {
			return fmt.Errorf("no NOVA_WORKER_ID and no hostname: %w", err)
		}
		id = host
	}

	hashMB, err := strconv.Atoi(env("NOVA_HASH_MB", "64"))
	if err != nil {
		return fmt.Errorf("NOVA_HASH_MB: %w", err)
	}

	// Signals are handled so that a rescheduled pod returns its unit to the
	// queue rather than holding it until the acknowledgement times out.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	artifacts, err := store.NewFS(dataDir)
	if err != nil {
		return err
	}
	defer artifacts.Close()

	connectCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	b, err := bus.Connect(connectCtx, bus.Config{
		URLs:       urls,
		ClientName: id,
	})
	if err != nil {
		return err
	}
	defer b.Close()

	w, err := worker.New(worker.Config{
		ID:                id,
		NodeName:          env("NOVA_NODE_NAME", ""),
		HashMB:            hashMB,
		HeartbeatInterval: 15 * time.Second,
		EngineVersion:     version,
	}, b, artifacts, log)
	if err != nil {
		return err
	}

	log.Info("starting", "worker", id, "version", version, "bus", urls, "data", artifacts.Root())
	return w.Run(ctx)
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
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
