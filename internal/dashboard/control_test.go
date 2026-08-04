package dashboard

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeCluster implements Cluster with recorded calls, so tests can assert
// which methods were invoked with what rather than inspecting real cluster
// state. It deliberately does not use the k8s fake clientset - this layer's
// tests care about call sequencing and arguments, not resource manifests.
type fakeCluster struct {
	scales   map[string][]int32
	replicas map[string]int32
	coordGen []int
	coordURI []string

	// coordinatorGeneration stands in for the coordinator Deployment's
	// NOVA_GENERATION env var - what CoordinatorGeneration reads. It defaults
	// to 0, the template's starting value, and SetCoordinatorGeneration
	// updates it exactly as the real cluster's env patch would.
	coordinatorGeneration int
	jobs                  []string
	jobCommands           [][]string
	jobArgs               [][]string
	jobCronNames          []string

	// cronCommand and cronArgs stand in for the real CronJob templates in
	// deploy/jobs.yaml, tuned flags included, so tests can assert that a
	// launched Job's command carries the template's tuning verbatim and
	// only the generation-numbered paths were rewritten - mirroring what
	// k8sCluster.CreateJobFromCron does against the real template.
	cronCommand map[string][]string
	cronArgs    map[string][]string

	// calls is a single ordered log across every method, so tests can assert
	// relative order between calls that land in different fields above (e.g.
	// SetCoordinatorGeneration must happen before the coordinator is scaled
	// up, or it restarts into the wrong generation).
	calls []string

	// streamLogsErrs is consumed in order, one entry per StreamJobLogs call;
	// once exhausted, calls succeed. A nil entry means "succeed on this
	// call". Used by the follower tests to simulate the pod-not-scheduled-
	// yet error StreamJobLogs returns immediately after job creation.
	streamLogsErrs  []error
	streamLogsData  []byte
	streamLogsCalls int

	// jobResponses is consumed in order, one entry per Job call; the last
	// entry repeats once exhausted. Used to simulate a job going from
	// Active to terminal across several polls.
	jobResponses []JobState
	jobCalls     int

	// metrics and metricsErr back Metrics(); metricsErr, when set, makes
	// every call fail until cleared, so tests can simulate metrics-server
	// being briefly unavailable. Guarded by metricsMu since the metrics
	// poller reads them from its own goroutine while a test mutates them.
	metricsMu  sync.Mutex
	metrics    map[string]PodMetrics
	metricsErr error
}

// setMetrics safely updates the sample and/or error Metrics() returns, for
// use by tests exercising the concurrent metrics poller.
func (f *fakeCluster) setMetrics(m map[string]PodMetrics, err error) {
	f.metricsMu.Lock()
	defer f.metricsMu.Unlock()
	f.metrics = m
	f.metricsErr = err
}

func newFakeCluster() *fakeCluster {
	return &fakeCluster{
		scales:   make(map[string][]int32),
		replicas: make(map[string]int32),
		cronCommand: map[string][]string{
			CronTrainer: {
				"/usr/local/bin/trainer",
				"-out", "/data/networks/gen0.nnue",
				"-epochs", "15",
				"-batch", "1024",
				"-lr", "0.001",
				"-lr-drops", "8,12",
				"-init", "/data/networks/gen0.nnue",
			},
			CronGatekeeper: {
				"/usr/local/bin/gatekeeper",
				"-candidate", "/data/networks/gen0.nnue",
				"-games", "2000",
				"-nodes", "20000",
				"-concurrency", "6",
			},
		},
		cronArgs: map[string][]string{
			CronTrainer: {"/data/gen0/"},
		},
	}
}

func (f *fakeCluster) Replicas(_ context.Context, d string) (int32, error) {
	return f.replicas[d], nil
}

func (f *fakeCluster) Scale(_ context.Context, d string, n int32) error {
	f.scales[d] = append(f.scales[d], n)
	f.replicas[d] = n
	f.calls = append(f.calls, fmt.Sprintf("scale:%s:%d", d, n))
	return nil
}

func (f *fakeCluster) SetCoordinatorGeneration(_ context.Context, generation int, networkURI string) error {
	f.coordGen = append(f.coordGen, generation)
	f.coordURI = append(f.coordURI, networkURI)
	f.coordinatorGeneration = generation
	f.calls = append(f.calls, fmt.Sprintf("setgen:%d", generation))
	return nil
}

func (f *fakeCluster) CoordinatorGeneration(_ context.Context) (int, error) {
	return f.coordinatorGeneration, nil
}

func (f *fakeCluster) CreateJobFromCron(_ context.Context, cronName, jobName string, rewrite func(string) string) error {
	f.jobs = append(f.jobs, jobName)
	f.jobCronNames = append(f.jobCronNames, cronName)
	f.jobCommands = append(f.jobCommands, rewriteElements(f.cronCommand[cronName], rewrite))
	f.jobArgs = append(f.jobArgs, rewriteElements(f.cronArgs[cronName], rewrite))
	return nil
}

func (f *fakeCluster) Job(_ context.Context, name string) (JobState, error) {
	idx := f.jobCalls
	f.jobCalls++
	if len(f.jobResponses) == 0 {
		return JobState{Name: name}, nil
	}
	if idx >= len(f.jobResponses) {
		idx = len(f.jobResponses) - 1
	}
	return f.jobResponses[idx], nil
}

func (f *fakeCluster) Metrics(_ context.Context) (map[string]PodMetrics, error) {
	f.metricsMu.Lock()
	defer f.metricsMu.Unlock()
	if f.metricsErr != nil {
		return nil, f.metricsErr
	}
	return f.metrics, nil
}

func (f *fakeCluster) StreamJobLogs(_ context.Context, _ string) (io.ReadCloser, error) {
	idx := f.streamLogsCalls
	f.streamLogsCalls++
	if idx < len(f.streamLogsErrs) && f.streamLogsErrs[idx] != nil {
		return nil, f.streamLogsErrs[idx]
	}
	return io.NopCloser(bytes.NewReader(f.streamLogsData)), nil
}

func writeNetworkFile(t *testing.T, dataDir string, generation int) {
	t.Helper()
	dir := filepath.Join(dataDir, "networks")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir networks dir: %v", err)
	}
	path := filepath.Join(dir, fmt.Sprintf("gen%d.nnue", generation))
	if err := os.WriteFile(path, []byte("fake-network-bytes"), 0o644); err != nil {
		t.Fatalf("write network file: %v", err)
	}
}

func TestCoordinatorCanNeverExceedOne(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	writeNetworkFile(t, dataDir, 0)
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	if err := h.Append(Record{Type: "gate", Generation: 0, Promoted: true, At: time.Now()}); err != nil {
		t.Fatalf("append history: %v", err)
	}

	ct := NewController(fc, h, dataDir, 4)

	// Walk every public control method that touches the coordinator.
	// StartSelfplay is forced here - the history holds no dataset record for
	// generation 0 in this test, so an unforced call would also pass, but
	// forcing keeps this walk's intent (exercise the scaling path) explicit
	// rather than incidental on that empty-history detail.
	_ = ct.PauseSelfplay(ctx)
	_ = ct.ResumeSelfplay(ctx)
	_ = ct.StopSelfplay(ctx)
	_ = ct.StartSelfplay(ctx, true)
	_, _ = ct.StartTrainer(ctx, 0, true)
	_, _ = ct.StartGatekeeper(ctx, 0)
	_ = ct.AdvanceGeneration(ctx, 1)

	for _, n := range fc.scales[DeployCoordinator] {
		if n != 0 && n != 1 {
			t.Fatalf("coordinator scaled to %d, want 0 or 1", n)
		}
	}
	if len(fc.scales[DeployCoordinator]) == 0 {
		t.Fatal("expected at least one coordinator scale call")
	}
}

// TestStartRefusedWhenGenerationComplete checks that StartSelfplay refuses
// to restart the coordinator for a generation the history already holds a
// completed dataset for - a restarted coordinator would redo that generation
// from scratch. force overrides the refusal.
func TestStartRefusedWhenGenerationComplete(t *testing.T) {
	ctx := context.Background()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	if err := h.Append(Record{Type: "dataset", Generation: 0, Positions: 1000000, At: time.Now()}); err != nil {
		t.Fatalf("append history: %v", err)
	}
	fc := newFakeCluster() // coordinator env NOVA_GENERATION=0 in its template
	ct := NewController(fc, h, dataDir, 8)

	err := ct.StartSelfplay(ctx, false)
	var re *RefusedError
	if !errors.As(err, &re) {
		t.Fatalf("want RefusedError, got %v", err)
	}
	if len(fc.calls) != 0 {
		t.Fatal("a refused start must not touch the cluster")
	}
	if err := ct.StartSelfplay(ctx, true); err != nil {
		t.Fatalf("force must override: %v", err)
	}
}

// TestStartAllowedWhenGenerationOpen checks the converse: with no dataset
// record for the coordinator's configured generation, StartSelfplay succeeds
// and scales the coordinator and workers up.
func TestStartAllowedWhenGenerationOpen(t *testing.T) {
	ctx := context.Background()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	fc := newFakeCluster()
	ct := NewController(fc, h, dataDir, 8)

	if err := ct.StartSelfplay(ctx, false); err != nil {
		t.Fatalf("StartSelfplay: %v", err)
	}
	if got := fc.replicas[DeployCoordinator]; got != 1 {
		t.Fatalf("expected coordinator scaled to 1, got %d", got)
	}
	if got := fc.replicas[DeployWorkers]; got != 8 {
		t.Fatalf("expected workers scaled to 8, got %d", got)
	}
}

func TestAdvanceRefusesWithoutPromotion(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if err := ct.AdvanceGeneration(ctx, 1); err == nil {
		t.Fatal("expected AdvanceGeneration to error with empty history")
	}
	if len(fc.coordGen) != 0 {
		t.Fatalf("expected no SetCoordinatorGeneration call, got %v", fc.coordGen)
	}
	if len(fc.scales[DeployCoordinator]) != 0 {
		t.Fatalf("expected no coordinator scale call, got %v", fc.scales[DeployCoordinator])
	}
}

func TestAdvanceAfterPromotion(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	writeNetworkFile(t, dataDir, 0)
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	if err := h.Append(Record{Type: "gate", Generation: 0, Promoted: true, At: time.Now()}); err != nil {
		t.Fatalf("append history: %v", err)
	}
	ct := NewController(fc, h, dataDir, 4)

	if err := ct.AdvanceGeneration(ctx, 1); err != nil {
		t.Fatalf("AdvanceGeneration: %v", err)
	}

	if len(fc.coordGen) != 1 || fc.coordGen[0] != 1 {
		t.Fatalf("expected SetCoordinatorGeneration(1, ...), got gens=%v", fc.coordGen)
	}
	wantURI := "file:///data/networks/gen0.nnue"
	if len(fc.coordURI) != 1 || fc.coordURI[0] != wantURI {
		t.Fatalf("expected URI %q, got %v", wantURI, fc.coordURI)
	}
	if got := fc.replicas[DeployCoordinator]; got != 1 {
		t.Fatalf("expected coordinator scaled to 1, got %d", got)
	}

	// Order matters: a coordinator scaled up before its env is patched would
	// restart into the wrong generation, so setgen must precede the scale.
	setgenIdx, scaleIdx := -1, -1
	for i, c := range fc.calls {
		switch c {
		case "setgen:1":
			if setgenIdx == -1 {
				setgenIdx = i
			}
		case fmt.Sprintf("scale:%s:1", DeployCoordinator):
			if scaleIdx == -1 {
				scaleIdx = i
			}
		}
	}
	if setgenIdx == -1 || scaleIdx == -1 {
		t.Fatalf("expected both setgen:1 and scale:%s:1 in call log, got %v", DeployCoordinator, fc.calls)
	}
	if setgenIdx > scaleIdx {
		t.Fatalf("expected SetCoordinatorGeneration before coordinator scale-up, got call order %v", fc.calls)
	}
}

// TestStartAllowedAfterAdvance pins the advance-then-start seam: after
// AdvanceGeneration moves the coordinator on to generation 1, StartSelfplay
// must succeed even though gen 0 has a dataset record in history. A wrong
// CoordinatorGeneration implementation that reads the newest dataset record
// out of history - rather than the coordinator Deployment's actual
// NOVA_GENERATION env var - would still see gen 0's dataset record, refuse
// every subsequent Start, and stall the self-play loop after the very first
// generation.
func TestStartAllowedAfterAdvance(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	writeNetworkFile(t, dataDir, 0)
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	if err := h.Append(Record{Type: "dataset", Generation: 0, Positions: 1000000, At: time.Now()}); err != nil {
		t.Fatalf("append history: %v", err)
	}
	if err := h.Append(Record{Type: "gate", Generation: 0, Promoted: true, At: time.Now()}); err != nil {
		t.Fatalf("append history: %v", err)
	}
	ct := NewController(fc, h, dataDir, 4)

	if err := ct.AdvanceGeneration(ctx, 1); err != nil {
		t.Fatalf("AdvanceGeneration: %v", err)
	}

	if err := ct.StartSelfplay(ctx, false); err != nil {
		t.Fatalf("StartSelfplay after advance: %v", err)
	}
}

// TestAdvanceRefusesWithMissingNetworkFile checks Finding 5: a promotion
// record without the corresponding network file on disk is a refusal (409),
// the same as an outright missing promotion, not an unrelated 500 - both are
// the controller declining to act rather than a transport or cluster
// failure.
func TestAdvanceRefusesWithMissingNetworkFile(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	// A promotion record for generation 0 exists, but its network file was
	// never written to (or was removed from) the data volume.
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	if err := h.Append(Record{Type: "gate", Generation: 0, Promoted: true, At: time.Now()}); err != nil {
		t.Fatalf("append history: %v", err)
	}
	ct := NewController(fc, h, dataDir, 4)

	err := ct.AdvanceGeneration(ctx, 1)
	if err == nil {
		t.Fatal("expected AdvanceGeneration to error when the network file is missing")
	}
	var refused *RefusedError
	if !errors.As(err, &refused) {
		t.Fatalf("expected a *RefusedError (409), got %T: %v", err, err)
	}
	if len(fc.coordGen) != 0 {
		t.Fatalf("expected no SetCoordinatorGeneration call, got %v", fc.coordGen)
	}
	if len(fc.scales[DeployCoordinator]) != 0 {
		t.Fatalf("expected no coordinator scale call, got %v", fc.scales[DeployCoordinator])
	}
}

func TestTrainerRefusedWhileCoordinatorRuns(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	fc.replicas[DeployCoordinator] = 1
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if _, err := ct.StartTrainer(ctx, 0, false); err == nil {
		t.Fatal("expected StartTrainer to refuse while coordinator is running")
	} else if !strings.Contains(err.Error(), "coordinator") {
		t.Fatalf("expected error to mention the coordinator, got: %v", err)
	}

	jobName, err := ct.StartTrainer(ctx, 0, true)
	if err != nil {
		t.Fatalf("StartTrainer with force: %v", err)
	}
	if jobName == "" {
		t.Fatal("expected a non-empty job name")
	}
	if len(fc.jobs) != 1 || fc.jobs[0] != jobName {
		t.Fatalf("expected job %q recorded, got %v", jobName, fc.jobs)
	}
}

func TestGen0GatekeeperHasNoIncumbentFlag(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	writeNetworkFile(t, dataDir, 0)
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	jobName, err := ct.StartGatekeeper(ctx, 0)
	if err != nil {
		t.Fatalf("StartGatekeeper: %v", err)
	}
	if jobName == "" {
		t.Fatal("expected a non-empty job name")
	}
	if len(fc.jobCommands) != 1 {
		t.Fatalf("expected one CreateJobFromCron call, got %d", len(fc.jobCommands))
	}
	for _, arg := range fc.jobCommands[0] {
		if arg == "-incumbent" {
			t.Fatalf("gen 0 gatekeeper command must not contain -incumbent: %v", fc.jobCommands[0])
		}
	}
}

func TestGatekeeperIncludesIncumbentWhenPresent(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	writeNetworkFile(t, dataDir, 0)
	writeNetworkFile(t, dataDir, 1)
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if _, err := ct.StartGatekeeper(ctx, 1); err != nil {
		t.Fatalf("StartGatekeeper: %v", err)
	}
	cmd := fc.jobCommands[0]
	found := false
	for i, arg := range cmd {
		if arg == "-incumbent" {
			found = true
			if i+1 >= len(cmd) || cmd[i+1] != "/data/networks/gen0.nnue" {
				t.Fatalf("expected -incumbent to be followed by gen0 path, got %v", cmd)
			}
		}
	}
	if !found {
		t.Fatalf("expected -incumbent flag when incumbent file exists, got %v", cmd)
	}
}

// TestTrainerFlagsSurviveLaunchVerbatim is Finding 3's core assertion: tuned
// hyperparameters baked into the CronJob template must reach the launched
// Job unchanged, while only the generation-numbered paths are rewritten to
// target the requested generation. Before the fix, CreateJobFromCron
// replaced the whole Command/Args and these flags would have silently
// vanished.
func TestTrainerFlagsSurviveLaunchVerbatim(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if _, err := ct.StartTrainer(ctx, 3, true); err != nil {
		t.Fatalf("StartTrainer: %v", err)
	}

	cmd := fc.jobCommands[0]
	wantVerbatim := [][2]string{{"-epochs", "15"}, {"-batch", "1024"}, {"-lr", "0.001"}, {"-lr-drops", "8,12"}}
	for _, pair := range wantVerbatim {
		found := false
		for i, arg := range cmd {
			if arg == pair[0] {
				found = true
				if i+1 >= len(cmd) || cmd[i+1] != pair[1] {
					t.Fatalf("expected %s to be followed by %s verbatim, got %v", pair[0], pair[1], cmd)
				}
			}
		}
		if !found {
			t.Fatalf("expected tuned flag %s to survive launch, got %v", pair[0], cmd)
		}
	}

	found := false
	for i, arg := range cmd {
		if arg == "-out" {
			found = true
			if i+1 >= len(cmd) || cmd[i+1] != "/data/networks/gen3.nnue" {
				t.Fatalf("expected -out rewritten to target generation 3, got %v", cmd)
			}
		}
	}
	if !found {
		t.Fatalf("expected -out flag present, got %v", cmd)
	}

	args := fc.jobArgs[0]
	if len(args) != 1 || args[0] != "/data/gen3/" {
		t.Fatalf("expected args rewritten to /data/gen3/, got %v", args)
	}
}

// initFlagValue returns the element following -init in cmd, or "" with found
// set to false if -init is not present at all.
func initFlagValue(cmd []string) (value string, found bool) {
	for i, arg := range cmd {
		if arg == "-init" {
			if i+1 >= len(cmd) {
				return "", true
			}
			return cmd[i+1], true
		}
	}
	return "", false
}

// TestGen0TrainerKeepsInitWhenFileExists checks that generation 0's -init
// flag survives launch pointed at gen0.nnue - a warm start from an earlier
// attempt at generation 0 itself - when that file is actually present on the
// data volume.
func TestGen0TrainerKeepsInitWhenFileExists(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	writeNetworkFile(t, dataDir, 0)
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if _, err := ct.StartTrainer(ctx, 0, true); err != nil {
		t.Fatalf("StartTrainer: %v", err)
	}

	value, found := initFlagValue(fc.jobCommands[0])
	if !found {
		t.Fatalf("expected -init to survive when gen0.nnue exists, got %v", fc.jobCommands[0])
	}
	if value != "/data/networks/gen0.nnue" {
		t.Fatalf("expected -init to point at gen0.nnue, got %q", value)
	}
}

// TestGen0TrainerDropsInitWhenFileAbsent is the converse: with no gen0.nnue
// on the data volume - a generation 0's very first training attempt - -init
// must be dropped entirely rather than launched pointing at a file that was
// never written.
func TestGen0TrainerDropsInitWhenFileAbsent(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if _, err := ct.StartTrainer(ctx, 0, true); err != nil {
		t.Fatalf("StartTrainer: %v", err)
	}

	if _, found := initFlagValue(fc.jobCommands[0]); found {
		t.Fatalf("expected -init to be dropped when gen0.nnue is absent, got %v", fc.jobCommands[0])
	}
}

// TestGen2TrainerRewritesInitToPredecessor checks the fine-tune case: a
// generation 2 training launch with generation 1's promoted network present
// rewrites -init to point at gen1.nnue rather than leaving the template's
// baked-in gen0.nnue.
func TestGen2TrainerRewritesInitToPredecessor(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	writeNetworkFile(t, dataDir, 1)
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if _, err := ct.StartTrainer(ctx, 2, true); err != nil {
		t.Fatalf("StartTrainer: %v", err)
	}

	value, found := initFlagValue(fc.jobCommands[0])
	if !found {
		t.Fatalf("expected -init to survive when gen1.nnue exists, got %v", fc.jobCommands[0])
	}
	if value != "/data/networks/gen1.nnue" {
		t.Fatalf("expected -init rewritten to gen1.nnue, got %q", value)
	}
}

// TestGen2TrainerDropsInitWhenPredecessorAbsent checks that the generation-2
// fine-tune case still drops -init entirely when generation 1's network is
// not on the volume, rather than launching pointed at a file that does not
// exist.
func TestGen2TrainerDropsInitWhenPredecessorAbsent(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if _, err := ct.StartTrainer(ctx, 2, true); err != nil {
		t.Fatalf("StartTrainer: %v", err)
	}

	if _, found := initFlagValue(fc.jobCommands[0]); found {
		t.Fatalf("expected -init to be dropped when gen1.nnue is absent, got %v", fc.jobCommands[0])
	}
}

// TestGatekeeperFlagsSurviveLaunchVerbatim is the gatekeeper counterpart: its
// tuned -games/-nodes/-concurrency flags must survive while -candidate is
// rewritten to the target generation. -concurrency matters on its own: it is
// what keeps a gate run from playing one game at a time (see deploy/jobs.yaml),
// and a launch that silently dropped it would still succeed, just far slower.
func TestGatekeeperFlagsSurviveLaunchVerbatim(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	writeNetworkFile(t, dataDir, 2)
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if _, err := ct.StartGatekeeper(ctx, 2); err != nil {
		t.Fatalf("StartGatekeeper: %v", err)
	}

	cmd := fc.jobCommands[0]
	wantVerbatim := [][2]string{{"-games", "2000"}, {"-nodes", "20000"}, {"-concurrency", "6"}}
	for _, pair := range wantVerbatim {
		found := false
		for i, arg := range cmd {
			if arg == pair[0] {
				found = true
				if i+1 >= len(cmd) || cmd[i+1] != pair[1] {
					t.Fatalf("expected %s to be followed by %s verbatim, got %v", pair[0], pair[1], cmd)
				}
			}
		}
		if !found {
			t.Fatalf("expected tuned flag %s to survive launch, got %v", pair[0], cmd)
		}
	}

	found := false
	for i, arg := range cmd {
		if arg == "-candidate" {
			found = true
			if i+1 >= len(cmd) || cmd[i+1] != "/data/networks/gen2.nnue" {
				t.Fatalf("expected -candidate rewritten to target generation 2, got %v", cmd)
			}
		}
	}
	if !found {
		t.Fatalf("expected -candidate flag present, got %v", cmd)
	}
}

func TestGatekeeperRefusesWithoutCandidate(t *testing.T) {
	ctx := context.Background()
	fc := newFakeCluster()
	dataDir := t.TempDir()
	h := NewHistory(filepath.Join(dataDir, "history.jsonl"))
	ct := NewController(fc, h, dataDir, 4)

	if _, err := ct.StartGatekeeper(ctx, 0); err == nil {
		t.Fatal("expected StartGatekeeper to refuse when candidate file is missing")
	}
	if len(fc.jobs) != 0 {
		t.Fatalf("expected no job created, got %v", fc.jobs)
	}
}
