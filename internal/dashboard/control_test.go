package dashboard

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeCluster implements Cluster with recorded calls, so tests can assert
// which methods were invoked with what rather than inspecting real cluster
// state. It deliberately does not use the k8s fake clientset - this layer's
// tests care about call sequencing and arguments, not resource manifests.
type fakeCluster struct {
	scales       map[string][]int32
	replicas     map[string]int32
	coordGen     []int
	coordURI     []string
	jobs         []string
	jobCommands  [][]string
	jobArgs      [][]string
	jobCronNames []string

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
}

func newFakeCluster() *fakeCluster {
	return &fakeCluster{
		scales:   make(map[string][]int32),
		replicas: make(map[string]int32),
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
	f.calls = append(f.calls, fmt.Sprintf("setgen:%d", generation))
	return nil
}

func (f *fakeCluster) CreateJobFromCron(_ context.Context, cronName, jobName string, command, args []string) error {
	f.jobs = append(f.jobs, jobName)
	f.jobCronNames = append(f.jobCronNames, cronName)
	f.jobCommands = append(f.jobCommands, command)
	f.jobArgs = append(f.jobArgs, args)
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
	_ = ct.PauseSelfplay(ctx)
	_ = ct.ResumeSelfplay(ctx)
	_ = ct.StopSelfplay(ctx)
	_ = ct.StartSelfplay(ctx)
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
