package main

import (
	"path/filepath"
	"testing"
)

func TestParseEpochsRoundTripsFormatEpochs(t *testing.T) {
	cases := [][]int{nil, {8, 12}, {1}, {3, 5, 9}}
	for _, epochs := range cases {
		got, err := parseEpochs(formatEpochs(epochs))
		if err != nil {
			t.Fatalf("parseEpochs(%q): %v", formatEpochs(epochs), err)
		}
		if len(got) != len(epochs) {
			t.Fatalf("round trip of %v produced %v", epochs, got)
		}
		for i := range epochs {
			if got[i] != epochs[i] {
				t.Fatalf("round trip of %v produced %v", epochs, got)
			}
		}
	}
}

func TestParseEpochsEmptyMeansNone(t *testing.T) {
	got, err := parseEpochs("")
	if err != nil {
		t.Fatalf("parseEpochs(\"\"): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("parseEpochs(\"\") = %v, want none", got)
	}
}

func TestParseEpochsRejectsGarbage(t *testing.T) {
	if _, err := parseEpochs("8,twelve"); err == nil {
		t.Fatal("expected an error for a non-numeric entry")
	}
}

// TestLoadInitNetworkMissingFileIsNotAnError checks the fallback the trainer
// relies on for a generation's first attempt: -init pointed at a file that
// does not exist yet must not fail the run, since there is no predecessor to
// warm-start from.
func TestLoadInitNetworkMissingFileIsNotAnError(t *testing.T) {
	net, err := loadInitNetwork(filepath.Join(t.TempDir(), "does-not-exist.nnue"))
	if err != nil {
		t.Fatalf("loadInitNetwork on a missing file: %v", err)
	}
	if net != nil {
		t.Fatalf("expected a nil network for a missing file, got %v", net)
	}
}
