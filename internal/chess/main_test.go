package chess

import (
	"flag"
	"os"
	"testing"
)

// perftFull enables the deep perft counts, which run into hundreds of millions
// of nodes and take minutes rather than seconds. CI runs with it; `go test` on
// a laptop does not.
var perftFull = flag.Bool("perft-full", false, "run the deep perft node counts")

func TestMain(m *testing.M) {
	flag.Parse()
	os.Exit(m.Run())
}
