// Command trainer fits an evaluation network to self-play data.
//
// It reads one or more packed sample files, trains a network and writes it out:
//
//	trainer -out gen1.nnue data/*.novadata
//
// The run is deterministic. The same inputs in the same order with the same
// seed produce byte-identical weights, which is what lets a generation be
// reproduced or audited later rather than merely trusted.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"syscall"

	"github.com/piwi3910/novachess/internal/train"
)

func main() {
	params := train.DefaultParams()

	var (
		out     = flag.String("out", "network.nnue", "where to write the trained network")
		epochs  = flag.Int("epochs", params.Epochs, "passes over the data")
		batch   = flag.Int("batch", params.BatchSize, "samples per weight update")
		lr      = flag.Float64("lr", params.LearningRate, "learning rate")
		lambda  = flag.Float64("result-weight", params.ResultWeight, "blend between search score (0) and game result (1)")
		holdout = flag.Float64("validation", params.ValidationFraction, "fraction of samples held out for validation")
		seed    = flag.Uint64("seed", params.Seed, "seed for initialization and shuffling")
	)
	flag.Parse()

	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "trainer: no input files")
		fmt.Fprintln(os.Stderr, "usage: trainer [flags] <data files...>")
		flag.PrintDefaults()
		os.Exit(1)
	}

	params.Epochs = *epochs
	params.BatchSize = *batch
	params.LearningRate = *lr
	params.ResultWeight = *lambda
	params.ValidationFraction = *holdout
	params.Seed = *seed
	params.Progress = func(epoch int, loss float64) {
		fmt.Fprintf(os.Stderr, "  epoch %2d  loss %.6f\n", epoch, loss)
	}

	trainer, err := train.NewTrainer(params)
	if err != nil {
		fmt.Fprintln(os.Stderr, "trainer:", err)
		os.Exit(1)
	}

	// Inputs are sorted so that a shell glob's order, which depends on the
	// locale, cannot change the result. Determinism is the point of the seed
	// and it would be undone by reading the same files in a different order.
	inputs, err := expand(flag.Args())
	if err != nil {
		fmt.Fprintln(os.Stderr, "trainer:", err)
		os.Exit(1)
	}

	samples, err := readAll(inputs)
	if err != nil {
		fmt.Fprintln(os.Stderr, "trainer:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "read %d samples from %d files\n", len(samples), len(inputs))

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	net, stats, err := trainer.Train(ctx, samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, "trainer:", err)
		os.Exit(1)
	}

	report(stats)

	if err := write(*out, net); err != nil {
		fmt.Fprintln(os.Stderr, "trainer:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", *out)
}

// expand turns the given paths into a sorted list of data files.
//
// A directory becomes every .novadata file inside it. That is not a
// convenience: the runtime image has no shell, so a Kubernetes Job cannot
// expand a glob, and a generation's batches are one file per work unit — a
// command line naming them individually would be thousands of arguments long.
//
// The result is sorted so that the order does not depend on the filesystem's
// directory ordering, which would otherwise make an otherwise-reproducible run
// differ between machines.
func expand(paths []string) ([]string, error) {
	var files []string

	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			return nil, err
		}

		if !info.IsDir() {
			files = append(files, path)
			continue
		}

		matches, err := filepath.Glob(filepath.Join(path, "*"+train.FileExtension))
		if err != nil {
			return nil, err
		}
		if len(matches) == 0 {
			return nil, fmt.Errorf("%s holds no %s files", path, train.FileExtension)
		}
		files = append(files, matches...)
	}

	sort.Strings(files)
	return files, nil
}

// readAll loads every input file.
func readAll(paths []string) ([]train.Sample, error) {
	var all []train.Sample

	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}

		r, err := train.NewReader(f)
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("%s: %w", path, err)
		}

		samples, err := r.ReadAll()
		f.Close()
		if err != nil && err != io.EOF {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if len(samples) == 0 {
			return nil, fmt.Errorf("%s: holds no samples", path)
		}

		all = append(all, samples...)
	}

	if len(all) == 0 {
		return nil, fmt.Errorf("no samples in any input file")
	}
	return all, nil
}

// report prints what the run did.
//
// The losses are diagnostics, not a verdict. A network can improve its loss
// against the training objective and still play worse, which is exactly what
// the gatekeeper's match exists to catch — so nothing here decides whether the
// network is any good.
func report(s train.Stats) {
	fmt.Fprintf(os.Stderr, "trained on %d samples, held out %d, in %v\n",
		s.TrainingSamples, s.ValidationSamples, s.Duration.Round(1e6))

	fmt.Fprintf(os.Stderr, "final training loss %.6f, validation loss %.6f\n",
		s.TrainingLoss, s.ValidationLoss)

	if s.EvaluationScale > 0 {
		fmt.Fprintf(os.Stderr, "quantization moved the evaluation by %.2f cp against a %.0f cp range (%.2f%%)\n",
			s.QuantizationError, s.EvaluationScale, 100*s.QuantizationError/s.EvaluationScale)
	}
	if s.ClippedWeights > 0 {
		// Worth shouting about: the network that was written is not the model
		// whose loss is printed above.
		fmt.Fprintf(os.Stderr, "WARNING: %d weights did not fit the integer range and were clipped;\n"+
			"         the saved network is not the model these losses describe\n", s.ClippedWeights)
	}
}

// write saves the network, via a temporary file so that an interrupted write
// cannot leave a half-written network where a valid one used to be.
func write(path string, net interface {
	WriteTo(io.Writer) (int64, error)
},
) error {
	// The directory may not exist yet: on a fresh volume the first generation
	// is the first thing to write a network, and a Job failing on a missing
	// parent directory is a poor way to find that out.
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dir, ".network-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	if _, err := net.WriteTo(tmp); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
