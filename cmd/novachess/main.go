// Command novachess is a UCI chess engine.
//
// It speaks the Universal Chess Interface on stdin and stdout, so it can be
// loaded by any UCI-compatible GUI or match runner — Cutechess, Arena, Banksia
// — and is what the self-play workers and the Lichess bot drive.
//
// Run it with no arguments and type UCI commands, or point a GUI at the binary.
package main

import (
	"fmt"
	"os"
	"runtime/debug"

	"github.com/piwi3910/novachess/internal/uci"
)

// version is stamped at build time with -ldflags "-X main.version=...". It
// falls back to the module's VCS revision so that a plain `go build` still
// produces something traceable — which matters because every self-play game is
// tagged with the engine version that generated it.
var version = ""

func main() {
	engine := uci.NewEngine(os.Stdout, resolveVersion())

	if err := engine.Run(os.Stdin); err != nil {
		fmt.Fprintln(os.Stderr, "novachess:", err)
		os.Exit(1)
	}
}

func resolveVersion() string {
	if version != "" {
		return version
	}

	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "dev"
	}
	revision, modified := "", false
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			revision = s.Value
		case "vcs.modified":
			modified = s.Value == "true"
		}
	}
	if revision == "" {
		return "dev"
	}
	if len(revision) > 12 {
		revision = revision[:12]
	}
	if modified {
		return revision + "-dirty"
	}
	return revision
}
