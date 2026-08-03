package dashboard

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	metricsversioned "k8s.io/metrics/pkg/client/clientset/versioned"
)

const (
	DeployWorkers     = "novachess-worker"
	DeployCoordinator = "novachess-coordinator"
	CronTrainer       = "novachess-trainer"
	CronGatekeeper    = "novachess-gatekeeper"
)

// JobState is the dashboard's view of a launched Job: whether it is still
// active, whether it finished (and how), and when it started. ExitCode is
// the gatekeeper's verdict on a training generation and is nil until the
// job's pod has actually terminated.
type JobState struct {
	Name      string     `json:"name"`
	Active    bool       `json:"active"`
	Succeeded bool       `json:"succeeded"`
	Failed    bool       `json:"failed"`
	ExitCode  *int32     `json:"exit_code,omitempty"` // gatekeeper: the verdict
	StartedAt *time.Time `json:"started_at,omitempty"`
}

// Cluster is the dashboard's narrow view onto Kubernetes: scaling the
// self-play worker deployment, advancing the coordinator to a new
// generation, launching one-shot Jobs from the suspended CronJob templates,
// reading back job status, and streaming logs. Kept small and behind an
// interface so the control layer can be tested against the fake clientset.
type Cluster interface {
	Replicas(ctx context.Context, deployment string) (int32, error)
	Scale(ctx context.Context, deployment string, replicas int32) error
	SetCoordinatorGeneration(ctx context.Context, generation int, networkURI string) error
	// CoordinatorGeneration reads the coordinator Deployment's currently
	// configured NOVA_GENERATION env var - the generation it will (re)run
	// self-play for the next time it starts. A coordinator template with no
	// such env defaults to 0, matching the binary's own default.
	CoordinatorGeneration(ctx context.Context) (int, error)
	// CreateJobFromCron launches a one-shot Job from the named suspended
	// CronJob's template. rewrite, if non-nil, is applied to every element
	// of the template's Command and Args in place - not a wholesale
	// replacement - so tuned flags the template carries (-epochs, -games,
	// and the like) survive untouched and only the elements a caller
	// chooses to change (generation-numbered paths) are rewritten. See
	// rewriteElements for how a rewrite can also remove or insert elements.
	CreateJobFromCron(ctx context.Context, cronName, jobName string, rewrite func(element string) string) error
	Job(ctx context.Context, name string) (JobState, error)
	StreamJobLogs(ctx context.Context, jobName string) (io.ReadCloser, error)
	// Metrics returns current CPU/memory usage for every pod in the
	// namespace, keyed by pod name, as reported by metrics-server. A cluster
	// without metrics-server installed returns an error every call - callers
	// must degrade gracefully (cards without graphs), not treat it as fatal.
	Metrics(ctx context.Context) (map[string]PodMetrics, error)
}

type k8sCluster struct {
	cs        kubernetes.Interface
	metrics   metricsversioned.Interface
	namespace string
}

// NewCluster builds a Cluster from in-cluster config. It only works when
// running as a pod inside the target Kubernetes cluster.
func NewCluster(namespace string) (Cluster, error) {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("dashboard: not running in a cluster: %w", err)
	}
	cs, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}
	mc, err := metricsversioned.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}
	c := newClusterWith(cs, namespace)
	c.metrics = mc
	return c, nil
}

// newClusterWith is the unexported seam tests use with the fake clientset.
// The metrics client is not part of this seam - it rides only the real
// NewCluster path, since metrics-server has no clientset fake worth using
// here; tests exercise Metrics through the Cluster interface instead.
func newClusterWith(cs kubernetes.Interface, namespace string) *k8sCluster {
	return &k8sCluster{cs: cs, namespace: namespace}
}

func (c *k8sCluster) Replicas(ctx context.Context, deployment string) (int32, error) {
	d, err := c.cs.AppsV1().Deployments(c.namespace).Get(ctx, deployment, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}
	if d.Spec.Replicas == nil {
		return 0, nil
	}
	return *d.Spec.Replicas, nil
}

// Scale updates the deployment's desired replica count directly on its spec.
// The scale subresource (GetScale/UpdateScale) would be the idiomatic path
// against a real API server, but the fake clientset's default subresource
// reactor hands back the stored Deployment cast to *autoscalingv1.Scale,
// which panics - so this goes through the plain Deployments Get/Update,
// which is exact for a Deployment (scale is just spec.replicas) and is what
// the fake clientset actually supports.
func (c *k8sCluster) Scale(ctx context.Context, deployment string, replicas int32) error {
	deps := c.cs.AppsV1().Deployments(c.namespace)
	d, err := deps.Get(ctx, deployment, metav1.GetOptions{})
	if err != nil {
		return err
	}
	d.Spec.Replicas = &replicas
	_, err = deps.Update(ctx, d, metav1.UpdateOptions{})
	return err
}

// SetCoordinatorGeneration rewrites exactly NOVA_GENERATION and
// NOVA_NETWORK_URI on the coordinator, touching nothing else. Search
// configuration is deliberately unreachable from here.
func (c *k8sCluster) SetCoordinatorGeneration(ctx context.Context, generation int, networkURI string) error {
	deps := c.cs.AppsV1().Deployments(c.namespace)
	d, err := deps.Get(ctx, DeployCoordinator, metav1.GetOptions{})
	if err != nil {
		return err
	}
	env := d.Spec.Template.Spec.Containers[0].Env
	for i, e := range env {
		switch e.Name {
		case "NOVA_GENERATION":
			env[i].Value = fmt.Sprintf("%d", generation)
		case "NOVA_NETWORK_URI":
			env[i].Value = networkURI
		}
	}
	d.Spec.Template.Spec.Containers[0].Env = env
	_, err = deps.Update(ctx, d, metav1.UpdateOptions{})
	return err
}

// CoordinatorGeneration reads the coordinator Deployment's NOVA_GENERATION
// env var. Missing env or an unparsable value both default to 0, which is
// the generation the coordinator binary itself defaults to when the env is
// absent.
func (c *k8sCluster) CoordinatorGeneration(ctx context.Context) (int, error) {
	d, err := c.cs.AppsV1().Deployments(c.namespace).Get(ctx, DeployCoordinator, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}
	for _, e := range d.Spec.Template.Spec.Containers[0].Env {
		if e.Name == "NOVA_GENERATION" {
			n, err := strconv.Atoi(e.Value)
			if err != nil {
				return 0, nil
			}
			return n, nil
		}
	}
	return 0, nil
}

func (c *k8sCluster) CreateJobFromCron(ctx context.Context, cronName, jobName string, rewrite func(string) string) error {
	cron, err := c.cs.BatchV1().CronJobs(c.namespace).Get(ctx, cronName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	spec := *cron.Spec.JobTemplate.Spec.DeepCopy()
	if len(spec.Template.Spec.Containers) != 1 {
		return fmt.Errorf("dashboard: cron %s has %d containers, expected 1", cronName, len(spec.Template.Spec.Containers))
	}
	cont := &spec.Template.Spec.Containers[0]
	cont.Command = rewriteElements(cont.Command, rewrite)
	cont.Args = rewriteElements(cont.Args, rewrite)
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: jobName, Namespace: c.namespace,
			Labels: map[string]string{"app": "novachess-" + cronName, "novachess.dev/launched-by": "dashboard"}},
		Spec: spec,
	}
	_, err = c.cs.BatchV1().Jobs(c.namespace).Create(ctx, job, metav1.CreateOptions{})
	return err
}

// rewriteElements applies rewrite to every element of tokens in order. A nil
// rewrite leaves tokens unchanged. An element rewritten to the empty string
// is dropped, and a result containing spaces expands into several tokens on
// return - which is how a caller adds or removes elements (for example the
// gatekeeper's -incumbent flag and its value) despite the mapping being
// expressed one element at a time.
func rewriteElements(tokens []string, rewrite func(string) string) []string {
	if rewrite == nil {
		return tokens
	}
	out := make([]string, 0, len(tokens))
	for _, t := range tokens {
		r := rewrite(t)
		if r == "" {
			continue
		}
		out = append(out, strings.Fields(r)...)
	}
	return out
}

func (c *k8sCluster) Job(ctx context.Context, name string) (JobState, error) {
	j, err := c.cs.BatchV1().Jobs(c.namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return JobState{}, err
	}
	out := JobState{Name: name, Active: j.Status.Active > 0, Succeeded: j.Status.Succeeded > 0, Failed: j.Status.Failed > 0}
	if j.Status.StartTime != nil {
		t := j.Status.StartTime.Time
		out.StartedAt = &t
	}
	// The gatekeeper's exit code is the verdict, so a failure to list its
	// pods must be surfaced rather than silently read as "not decided yet".
	pods, err := c.cs.CoreV1().Pods(c.namespace).List(ctx, metav1.ListOptions{LabelSelector: "job-name=" + name})
	if err != nil {
		return out, err
	}
	// Under retries (backoffLimit > 0) more than one pod can exist for the
	// job. The verdict is whichever pod ran most recently, not whichever
	// happens to sort first in the list - so track the newest terminated
	// container seen, using pod start time (falling back to creation time
	// for pods that never got scheduled) to break ties.
	var newest *metav1.Time
	for _, p := range pods.Items {
		podTime := p.Status.StartTime
		if podTime == nil {
			podTime = &p.CreationTimestamp
		}
		for _, st := range p.Status.ContainerStatuses {
			if st.State.Terminated == nil {
				continue
			}
			if newest == nil || podTime.After(newest.Time) {
				code := st.State.Terminated.ExitCode
				out.ExitCode = &code
				newest = podTime
			}
		}
	}
	return out, nil
}

func (c *k8sCluster) StreamJobLogs(ctx context.Context, jobName string) (io.ReadCloser, error) {
	pods, err := c.cs.CoreV1().Pods(c.namespace).List(ctx, metav1.ListOptions{LabelSelector: "job-name=" + jobName})
	if err != nil {
		return nil, err
	}
	if len(pods.Items) == 0 {
		return nil, fmt.Errorf("dashboard: no pods for job %s yet", jobName)
	}
	req := c.cs.CoreV1().Pods(c.namespace).GetLogs(pods.Items[0].Name, &corev1.PodLogOptions{Follow: true})
	return req.Stream(ctx)
}

// Metrics lists the namespace's current PodMetricses and sums usage across
// each pod's containers. A pod with no metrics recorded yet (just started)
// is simply absent from the result rather than reported as zero.
func (c *k8sCluster) Metrics(ctx context.Context) (map[string]PodMetrics, error) {
	list, err := c.metrics.MetricsV1beta1().PodMetricses(c.namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	out := make(map[string]PodMetrics, len(list.Items))
	for _, pm := range list.Items {
		var cpu, mem int64
		for _, cont := range pm.Containers {
			cpu += cont.Usage.Cpu().MilliValue()
			mem += cont.Usage.Memory().Value()
		}
		out[pm.Name] = PodMetrics{CPUMillicores: cpu, MemoryBytes: mem}
	}
	return out, nil
}
