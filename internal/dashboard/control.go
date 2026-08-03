package dashboard

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"time"
)

// genDirPattern matches a per-generation data directory path, e.g.
// /data/gen3/, wherever it appears in a Command or Args element.
var genDirPattern = regexp.MustCompile(`/data/gen\d+/`)

// networkFilePattern matches a network artifact path, e.g.
// /data/networks/gen3.nnue, as a whole element.
var networkFilePattern = regexp.MustCompile(`^/data/networks/gen\d+\.nnue$`)

// generationPathRewriter rewrites the per-generation data directory and any
// network path following -out or -candidate to target generation, leaving
// every other element - including tuned hyperparameters like -epochs or
// -games baked into the CronJob template - untouched. This is what lets the
// trainer and gatekeeper Jobs launch with the template's real tuning instead
// of the dashboard's own hardcoded flags.
func generationPathRewriter(generation int) func(string) string {
	var prev string
	return func(elem string) string {
		out := elem
		switch {
		case genDirPattern.MatchString(elem):
			out = genDirPattern.ReplaceAllString(elem, fmt.Sprintf("/data/gen%d/", generation))
		case networkFilePattern.MatchString(elem) && (prev == "-out" || prev == "-candidate"):
			out = fmt.Sprintf("/data/networks/gen%d.nnue", generation)
		}
		prev = elem
		return out
	}
}

// gatekeeperRewriter extends generationPathRewriter with the -incumbent
// flag's generation-aware handling: pointed at generation-1 when it should
// be present, removed entirely when it should not (generation 0, or no
// incumbent file on disk), and inserted right after the candidate path when
// the template does not carry the flag at all - which is the shape of the
// gatekeeper CronJob template today (see deploy/jobs.yaml): generation 0 has
// no -incumbent baked in because there is nothing to gate against yet, and
// later generations need it added rather than merely rewritten.
//
// The insertion point is the candidate path because it is guaranteed to
// appear exactly once, immediately after -candidate, in every gatekeeper
// invocation; if a future template ever bakes an explicit -incumbent flag in
// as well, this would append a second one; nothing in the current template
// does.
func gatekeeperRewriter(generation int, includeIncumbent bool) func(string) string {
	base := generationPathRewriter(generation)
	incumbentPath := fmt.Sprintf("/data/networks/gen%d.nnue", generation-1)

	var prev string
	removeNext := false
	return func(elem string) string {
		if removeNext {
			removeNext = false
			prev = elem
			return ""
		}
		if elem == "-incumbent" {
			prev = elem
			if !includeIncumbent {
				removeNext = true
				return ""
			}
			return elem
		}

		out := base(elem)
		switch {
		case prev == "-incumbent" && includeIncumbent:
			out = incumbentPath
		case prev == "-candidate" && includeIncumbent:
			out = out + " -incumbent " + incumbentPath
		}
		prev = elem
		return out
	}
}

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
// workers never find themselves without a coordinator to report to. A
// restarted coordinator re-runs its configured generation from scratch, so
// starting on a generation whose dataset the history already holds complete
// would redo finished work - exactly what burned cluster time before this
// guardrail existed. Unless force is set, that case is refused with a
// RefusedError (409) rather than acted on.
func (ct *Controller) StartSelfplay(ctx context.Context, force bool) error {
	if !force {
		gen, err := ct.cluster.CoordinatorGeneration(ctx)
		if err != nil {
			return err
		}
		if ct.history.HasDataset(gen) {
			return &RefusedError{Reason: fmt.Sprintf("dashboard: generation %d's dataset is already complete; starting now would redo finished work, pass force to restart anyway", gen)}
		}
	}
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
	if err := ct.cluster.CreateJobFromCron(ctx, CronTrainer, jobName, generationPathRewriter(generation)); err != nil {
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
	includeIncumbent := false
	if generation > 0 {
		incumbentPath := filepath.Join(ct.dataDir, "networks", fmt.Sprintf("gen%d.nnue", generation-1))
		if _, err := os.Stat(incumbentPath); err == nil {
			includeIncumbent = true
		}
	}
	jobName := fmt.Sprintf("gate-gen%d-%d", generation, time.Now().Unix())
	if err := ct.cluster.CreateJobFromCron(ctx, CronGatekeeper, jobName, gatekeeperRewriter(generation, includeIncumbent)); err != nil {
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
		return &RefusedError{Reason: fmt.Sprintf("dashboard: network file for generation %d not found: %v", from, err)}
	}
	networkURI := fmt.Sprintf("file:///data/networks/%s", networkFile)
	if err := ct.cluster.SetCoordinatorGeneration(ctx, toGeneration, networkURI); err != nil {
		return err
	}
	return ct.scaleCoordinator(ctx, 1)
}
