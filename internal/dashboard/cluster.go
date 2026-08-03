package dashboard

import (
	"context"
	"fmt"
	"io"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
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
	CreateJobFromCron(ctx context.Context, cronName, jobName string, command []string, args []string) error
	Job(ctx context.Context, name string) (JobState, error)
	StreamJobLogs(ctx context.Context, jobName string) (io.ReadCloser, error)
}

type k8sCluster struct {
	cs        kubernetes.Interface
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
	return newClusterWith(cs, namespace), nil
}

// newClusterWith is the unexported seam tests use with the fake clientset.
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

func (c *k8sCluster) CreateJobFromCron(ctx context.Context, cronName, jobName string, command, args []string) error {
	cron, err := c.cs.BatchV1().CronJobs(c.namespace).Get(ctx, cronName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	spec := *cron.Spec.JobTemplate.Spec.DeepCopy()
	if len(spec.Template.Spec.Containers) != 1 {
		return fmt.Errorf("dashboard: cron %s has %d containers, expected 1", cronName, len(spec.Template.Spec.Containers))
	}
	if command != nil {
		spec.Template.Spec.Containers[0].Command = command
	}
	if args != nil {
		spec.Template.Spec.Containers[0].Args = args
	}
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: jobName, Namespace: c.namespace,
			Labels: map[string]string{"app": "novachess-" + cronName, "novachess.dev/launched-by": "dashboard"}},
		Spec: spec,
	}
	_, err = c.cs.BatchV1().Jobs(c.namespace).Create(ctx, job, metav1.CreateOptions{})
	return err
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
	// The gatekeeper's exit code is the verdict; dig it out of the pod.
	pods, err := c.cs.CoreV1().Pods(c.namespace).List(ctx, metav1.ListOptions{LabelSelector: "job-name=" + name})
	if err == nil {
		for _, p := range pods.Items {
			for _, st := range p.Status.ContainerStatuses {
				if st.State.Terminated != nil {
					code := st.State.Terminated.ExitCode
					out.ExitCode = &code
				}
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
