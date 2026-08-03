package dashboard

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// RefusedError marks a guardrail refusal - a request the controller
// deliberately declines because acting on it would corrupt training state,
// as opposed to a transport or cluster error. Callers (the HTTP layer) use
// errors.As to tell the two apart and map refusals to 409 rather than 500.
type RefusedError struct{ Reason string }

func (e *RefusedError) Error() string { return e.Reason }

// Controller performs the dashboard's control actions: pausing and resuming
// self-play, launching trainer and gatekeeper jobs, and advancing the
// coordinator to a new generation. It is the single place these actions are
// implemented, so the correctness rules below are enforced structurally
// rather than trusted to every caller.
type Controller struct {
	cluster        Cluster
	history        *History
	dataDir        string
	workerReplicas int32
}

// NewController builds a Controller. dataDir is the dashboard's local view of
// the shared data volume (used for existence checks); the cluster-side path
// for the same volume is /data, used when constructing URIs and job args.
func NewController(c Cluster, h *History, dataDir string, workerReplicas int32) *Controller {
	return &Controller{cluster: c, history: h, dataDir: dataDir, workerReplicas: workerReplicas}
}

// scaleCoordinator is the only path that touches the coordinator's replica
// count. Two coordinators would each count half the batches and neither
// would reach its target - a deadlock with no error anywhere - so the clamp
// lives here structurally rather than in each caller's judgment.
func (ct *Controller) scaleCoordinator(ctx context.Context, n int32) error {
	if n != 0 && n != 1 {
		return fmt.Errorf("dashboard: coordinator replicas must be 0 or 1, got %d", n)
	}
	return ct.cluster.Scale(ctx, DeployCoordinator, n)
}

// PauseSelfplay stops the self-play workers without touching the
// coordinator, so resuming does not require replaying generation setup.
func (ct *Controller) PauseSelfplay(ctx context.Context) error {
	return ct.cluster.Scale(ctx, DeployWorkers, 0)
}

// ResumeSelfplay brings the self-play workers back to their configured
// replica count.
func (ct *Controller) ResumeSelfplay(ctx context.Context) error {
	return ct.cluster.Scale(ctx, DeployWorkers, ct.workerReplicas)
}

// StopSelfplay halts both the workers and the coordinator.
func (ct *Controller) StopSelfplay(ctx context.Context) error {
	if err := ct.cluster.Scale(ctx, DeployWorkers, 0); err != nil {
		return err
	}
	return ct.scaleCoordinator(ctx, 0)
}

// StartSelfplay brings the coordinator up first, then the workers, so
// workers never find themselves without a coordinator to report to.
func (ct *Controller) StartSelfplay(ctx context.Context) error {
	if err := ct.scaleCoordinator(ctx, 1); err != nil {
		return err
	}
	return ct.cluster.Scale(ctx, DeployWorkers, ct.workerReplicas)
}

// StartTrainer launches a training job for generation. Training reads the
// self-play data the coordinator is currently producing, so it refuses to
// start while the coordinator is running unless force is set - otherwise the
// dataset it trains on could be mutated mid-run.
func (ct *Controller) StartTrainer(ctx context.Context, generation int, force bool) (string, error) {
	if !force {
		replicas, err := ct.cluster.Replicas(ctx, DeployCoordinator)
		if err != nil {
			return "", err
		}
		if replicas > 0 {
			return "", &RefusedError{Reason: fmt.Sprintf("dashboard: coordinator is still running for generation %d; pass force to train anyway", generation)}
		}
	}
	jobName := fmt.Sprintf("train-gen%d-%d", generation, time.Now().Unix())
	command := []string{"/usr/local/bin/trainer", "-out", fmt.Sprintf("/data/networks/gen%d.nnue", generation)}
	args := []string{fmt.Sprintf("/data/gen%d/", generation)}
	if err := ct.cluster.CreateJobFromCron(ctx, CronTrainer, jobName, command, args); err != nil {
		return "", err
	}
	return jobName, nil
}

// StartGatekeeper launches a gatekeeper job that judges the candidate network
// for generation against the previous incumbent, if one exists. Generation 0
// has no incumbent network file - it gates against the hand-crafted
// evaluation baked into the gatekeeper binary - so the flag is omitted
// entirely rather than pointed at a file that does not exist.
func (ct *Controller) StartGatekeeper(ctx context.Context, generation int) (string, error) {
	candidatePath := filepath.Join(ct.dataDir, "networks", fmt.Sprintf("gen%d.nnue", generation))
	if _, err := os.Stat(candidatePath); err != nil {
		return "", &RefusedError{Reason: fmt.Sprintf("dashboard: candidate network for generation %d not found: %v", generation, err)}
	}
	command := []string{"/usr/local/bin/gatekeeper", "-candidate", fmt.Sprintf("/data/networks/gen%d.nnue", generation)}
	if generation > 0 {
		incumbentPath := filepath.Join(ct.dataDir, "networks", fmt.Sprintf("gen%d.nnue", generation-1))
		if _, err := os.Stat(incumbentPath); err == nil {
			command = append(command, "-incumbent", fmt.Sprintf("/data/networks/gen%d.nnue", generation-1))
		}
	}
	jobName := fmt.Sprintf("gate-gen%d-%d", generation, time.Now().Unix())
	if err := ct.cluster.CreateJobFromCron(ctx, CronGatekeeper, jobName, command, nil); err != nil {
		return "", err
	}
	return jobName, nil
}

// AdvanceGeneration moves the coordinator on to toGeneration. It refuses
// unless the previous generation was actually promoted and its network file
// is present on disk - advancing on an unpromoted or missing network would
// point the coordinator at self-play data or a file that was never produced.
func (ct *Controller) AdvanceGeneration(ctx context.Context, toGeneration int) error {
	from := toGeneration - 1
	if !ct.history.HasPromotion(from) {
		return &RefusedError{Reason: fmt.Sprintf("dashboard: generation %d has no recorded promotion, refusing to advance to %d", from, toGeneration)}
	}
	networkFile := fmt.Sprintf("gen%d.nnue", from)
	if _, err := os.Stat(filepath.Join(ct.dataDir, "networks", networkFile)); err != nil {
		return fmt.Errorf("dashboard: network file for generation %d not found: %w", from, err)
	}
	networkURI := fmt.Sprintf("file:///data/networks/%s", networkFile)
	if err := ct.cluster.SetCoordinatorGeneration(ctx, toGeneration, networkURI); err != nil {
		return err
	}
	return ct.scaleCoordinator(ctx, 1)
}
