// Command dashboard serves the training dashboard: the boards, the
// generation, live training, history, and the operator controls.
package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/piwi3910/novachess/internal/bus"
	"github.com/piwi3910/novachess/internal/dashboard"
)

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	urls := strings.Split(env("NOVA_BUS_URLS", "nats://nova-nats:4222"), ",")
	dataDir := env("NOVA_DATA_DIR", "/data")
	namespace := env("NOVA_NAMESPACE", "novachess")
	addr := env("NOVA_HTTP_ADDR", ":8080")
	replicas, err := strconv.Atoi(env("NOVA_WORKER_REPLICAS", "8"))
	if err != nil || replicas < 0 {
		log.Error("bad NOVA_WORKER_REPLICAS")
		os.Exit(1)
	}
	positionsTarget, err := strconv.Atoi(env("NOVA_POSITIONS", "1000000"))
	if err != nil || positionsTarget < 0 {
		log.Error("bad NOVA_POSITIONS")
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	connectCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	b, err := bus.Connect(connectCtx, bus.Config{
		URLs:       urls,
		ClientName: env("HOSTNAME", "dashboard"),
		// ReplayAll rather than Durable: the dashboard's generation progress
		// lives in memory only, so a durable consumer that resumed past its
		// acked position would come back from a restart showing zero
		// progress even though the stream still holds the whole window. An
		// ephemeral, replay-everything consumer rebuilds the same state a
		// durable one would have kept, every time.
		ReplayAll: true,
	})
	cancel()
	if err != nil {
		log.Error("connecting to the bus", "error", err)
		os.Exit(1)
	}
	defer b.Close()

	cluster, err := dashboard.NewCluster(namespace)
	if err != nil {
		log.Error("kubernetes client", "error", err)
		os.Exit(1)
	}

	if err := os.MkdirAll(filepath.Join(dataDir, "dashboard"), 0o755); err != nil {
		log.Error("creating dashboard data dir", "error", err)
		os.Exit(1)
	}

	state := dashboard.NewState()
	state.SetTarget(positionsTarget)
	history := dashboard.NewHistory(filepath.Join(dataDir, "dashboard", "history.jsonl"))
	ctl := dashboard.NewController(cluster, history, dataDir, int32(replicas))

	if err := dashboard.WatchHeartbeats(ctx, b, state); err != nil {
		log.Error("heartbeat watch", "error", err)
		os.Exit(1)
	}
	if err := dashboard.WatchEvents(ctx, b, state, history); err != nil {
		log.Error("event watch", "error", err)
		os.Exit(1)
	}

	srv := &http.Server{Addr: addr, Handler: dashboard.NewServer(ctx, state, history, ctl, cluster, dashboard.WithLogger(log))}
	go func() { <-ctx.Done(); srv.Shutdown(context.Background()) }()
	log.Info("starting", "addr", addr, "namespace", namespace)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Error("serving", "error", err)
		os.Exit(1)
	}
}
