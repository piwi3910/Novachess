package dashboard

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func int32p(v int32) *int32 { return &v }

func testDeployment(name string, replicas int32) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "novachess"},
		Spec: appsv1.DeploymentSpec{
			Replicas: int32p(replicas),
			Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
				Name: "coordinator",
				Env: []corev1.EnvVar{
					{Name: "NOVA_GENERATION", Value: "0"},
					{Name: "NOVA_NETWORK_URI", Value: ""},
				},
			}}}},
		},
	}
}

func TestScaleAndReplicas(t *testing.T) {
	cs := fake.NewClientset(testDeployment(DeployWorkers, 8))
	c := newClusterWith(cs, "novachess")
	if err := c.Scale(context.Background(), DeployWorkers, 0); err != nil {
		t.Fatal(err)
	}
	n, err := c.Replicas(context.Background(), DeployWorkers)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("replicas = %d after scaling to 0", n)
	}
}

func TestSetCoordinatorGenerationPatchesOnlyItsEnv(t *testing.T) {
	cs := fake.NewClientset(testDeployment(DeployCoordinator, 0))
	c := newClusterWith(cs, "novachess")
	if err := c.SetCoordinatorGeneration(context.Background(), 1, "file:///data/networks/gen0.nnue"); err != nil {
		t.Fatal(err)
	}
	d, _ := cs.AppsV1().Deployments("novachess").Get(context.Background(), DeployCoordinator, metav1.GetOptions{})
	env := map[string]string{}
	for _, e := range d.Spec.Template.Spec.Containers[0].Env {
		env[e.Name] = e.Value
	}
	if env["NOVA_GENERATION"] != "1" || env["NOVA_NETWORK_URI"] != "file:///data/networks/gen0.nnue" {
		t.Fatalf("env not advanced: %v", env)
	}
}

func TestCreateJobFromCronCopiesTemplateAndOverridesCommand(t *testing.T) {
	cron := &batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{Name: CronTrainer, Namespace: "novachess"},
		Spec: batchv1.CronJobSpec{JobTemplate: batchv1.JobTemplateSpec{
			Spec: batchv1.JobSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				RestartPolicy: corev1.RestartPolicyNever,
				Containers:    []corev1.Container{{Name: "trainer", Image: "novachess:dev", Command: []string{"/usr/local/bin/trainer", "-out", "/data/networks/gen0.nnue"}, Args: []string{"/data/gen0/"}}},
			}}},
		}},
	}
	cs := fake.NewClientset(cron)
	c := newClusterWith(cs, "novachess")
	err := c.CreateJobFromCron(context.Background(), CronTrainer, "train-gen1",
		[]string{"/usr/local/bin/trainer", "-out", "/data/networks/gen1.nnue"}, []string{"/data/gen1/"})
	if err != nil {
		t.Fatal(err)
	}
	j, err := cs.BatchV1().Jobs("novachess").Get(context.Background(), "train-gen1", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got := j.Spec.Template.Spec.Containers[0]
	if got.Command[2] != "/data/networks/gen1.nnue" || got.Args[0] != "/data/gen1/" {
		t.Fatalf("template not overridden for the generation: %v %v", got.Command, got.Args)
	}
	if got.Image != "novachess:dev" {
		t.Fatal("image must come from the cron template, not be invented")
	}
}
