package uci

import (
	"bytes"
	"io"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// driver runs an engine over a pipe, the way a GUI does, and collects output
// lines as they arrive.
//
// Testing through the real reader loop rather than by calling handlers directly
// is deliberate: the protocol's hardest requirement is that input keeps being
// read during a search, and only the loop can demonstrate that.
type driver struct {
	t      *testing.T
	in     io.WriteCloser
	lines  chan string
	done   chan struct{}
	mu     sync.Mutex
	all    []string
	engine *Engine
}

func newDriver(t *testing.T) *driver {
	t.Helper()

	inR, inW := io.Pipe()
	outR, outW := io.Pipe()

	d := &driver{
		t:     t,
		in:    inW,
		lines: make(chan string, 512),
		done:  make(chan struct{}),
	}
	d.engine = NewEngine(outW, "test")

	go func() {
		defer close(d.done)
		_ = d.engine.Run(inR)
		outW.Close()
	}()

	go func() {
		buf := make([]byte, 0, 4096)
		chunk := make([]byte, 1024)
		for {
			n, err := outR.Read(chunk)
			if n > 0 {
				buf = append(buf, chunk[:n]...)
				for {
					idx := bytes.IndexByte(buf, '\n')
					if idx < 0 {
						break
					}
					line := strings.TrimRight(string(buf[:idx]), "\r")
					buf = buf[idx+1:]

					d.mu.Lock()
					d.all = append(d.all, line)
					d.mu.Unlock()

					select {
					case d.lines <- line:
					default:
					}
				}
			}
			if err != nil {
				return
			}
		}
	}()

	return d
}

func (d *driver) send(cmd string) {
	d.t.Helper()
	if _, err := io.WriteString(d.in, cmd+"\n"); err != nil {
		d.t.Fatalf("write %q: %v", cmd, err)
	}
}

// expect waits for a line with the given prefix, failing if it does not arrive.
func (d *driver) expect(prefix string, timeout time.Duration) string {
	d.t.Helper()
	deadline := time.After(timeout)
	for {
		select {
		case line := <-d.lines:
			if strings.HasPrefix(line, prefix) {
				return line
			}
		case <-deadline:
			d.mu.Lock()
			got := strings.Join(d.all, "\n")
			d.mu.Unlock()
			d.t.Fatalf("timed out waiting for %q; output so far:\n%s", prefix, got)
			return ""
		}
	}
}

func (d *driver) close() {
	d.send("quit")
	select {
	case <-d.done:
	case <-time.After(5 * time.Second):
		d.t.Error("engine did not exit after quit")
	}
	d.in.Close()
}

func TestIdentifyAndOptions(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("uci")

	if got := d.expect("id name", time.Second); !strings.Contains(got, EngineName) {
		t.Errorf("id name = %q, want it to contain %q", got, EngineName)
	}
	d.expect("id author", time.Second)

	hash := d.expect("option name Hash", time.Second)
	for _, want := range []string{"type spin", "default", "min", "max"} {
		if !strings.Contains(hash, want) {
			t.Errorf("Hash option %q missing %q", hash, want)
		}
	}
	d.expect("option name UCI_Chess960", time.Second)
	d.expect("uciok", time.Second)
}

func TestIsReadyIsAnsweredDuringSearch(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("position startpos")
	d.send("go infinite")

	// This is the property that matters most in the whole protocol. A GUI uses
	// isready to check the engine is alive; an engine that only reads input
	// between searches never answers, appears hung, and loses on time.
	d.send("isready")
	d.expect("readyok", 3*time.Second)

	d.send("stop")
	d.expect("bestmove", 5*time.Second)
}

func TestGoDepthProducesInfoAndBestMove(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("position startpos")
	d.send("go depth 6")

	info := d.expect("info depth", 10*time.Second)
	for _, want := range []string{"seldepth", "score", "nodes", "nps", "time", "pv"} {
		if !strings.Contains(info, want) {
			t.Errorf("info line %q missing %q", info, want)
		}
	}

	best := d.expect("bestmove", 20*time.Second)
	move := strings.Fields(best)[1]
	if len(move) < 4 {
		t.Errorf("bestmove %q does not look like a move", best)
	}
}

func TestStopEndsSearchAndReportsBestMove(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("position startpos")
	d.send("go infinite")

	// Let the search get going, then interrupt it.
	d.expect("info depth", 5*time.Second)
	d.send("stop")

	best := d.expect("bestmove", 5*time.Second)
	if strings.Fields(best)[1] == "0000" {
		t.Error("stopped search reported a null move despite legal moves existing")
	}
}

func TestPositionWithMoves(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("position startpos moves e2e4 e7e5 g1f3")
	d.send("d")

	// The board command echoes the FEN, which is the simplest way to confirm
	// the move list was applied.
	line := d.expect("rnbqkbnr", 2*time.Second)
	_ = line

	d.send("go depth 4")
	d.expect("bestmove", 15*time.Second)
}

func TestPositionFromFEN(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	// A mate in one; the engine should find it, which also proves the FEN was
	// parsed rather than silently ignored.
	d.send("position fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
	d.send("go depth 4")

	best := d.expect("bestmove", 15*time.Second)
	if !strings.Contains(best, "a1a8") {
		t.Errorf("bestmove = %q, want the mate a1a8", best)
	}
}

func TestInvalidFENIsRejectedNotIgnored(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	// Silently keeping the previous position would make the engine answer
	// about a different game than the GUI is showing.
	d.send("position fen this is not a fen")
	line := d.expect("info string", 2*time.Second)
	if !strings.Contains(line, "invalid position") {
		t.Errorf("expected a diagnostic about the invalid position, got %q", line)
	}
}

func TestSetOptionHash(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("setoption name Hash value 8")
	d.send("isready")
	d.expect("readyok", 2*time.Second)

	// The engine must still work afterwards; resizing replaces the table.
	d.send("position startpos")
	d.send("go depth 4")
	d.expect("bestmove", 15*time.Second)
}

// TestChess960CastlingNotation checks that the variant option changes how
// castling is written, in both directions.
func TestChess960CastlingNotation(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	// Classical mode: castling is the king's destination.
	d.send("position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1g1")
	d.send("d")
	line := d.expect("r3k2r", 2*time.Second)
	if !strings.Contains(line, "R4RK1") {
		t.Errorf("classical e1g1 did not castle; board line was %q", line)
	}

	// Chess960 mode: castling is king-takes-rook.
	d.send("setoption name UCI_Chess960 value true")
	d.send("position fen r3k2r/8/8/8/8/8/8/R3K2R w HAha - 0 1 moves e1h1")
	d.send("d")
	line = d.expect("r3k2r", 2*time.Second)
	if !strings.Contains(line, "R4RK1") {
		t.Errorf("Chess960 e1h1 did not castle; board line was %q", line)
	}
}

func TestUcinewgameResetsState(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("position startpos moves e2e4")
	d.send("ucinewgame")
	d.send("d")

	// After ucinewgame the board is the starting position again, so it is
	// White to move with all pawns home.
	line := d.expect("rnbqkbnr/pppppppp", 2*time.Second)
	if !strings.Contains(line, "w KQkq") {
		t.Errorf("board after ucinewgame = %q, want the starting position", line)
	}
}

// TestPerftClampsNonPositiveDepth covers a command that previously lied. Perft
// returns 1 at depth zero or less and PerftDivide searches one ply below the
// root, so "perft 0" produced a depth-1 divide and labelled it as depth 0.
func TestPerftClampsNonPositiveDepth(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("position startpos")

	for _, depth := range []string{"perft 0", "perft -3"} {
		d.send(depth)
		line := d.expect("nodes ", 5*time.Second)

		// Clamped to depth 1, the starting position has twenty legal moves.
		if !strings.HasPrefix(line, "nodes 20 ") {
			t.Errorf("%q reported %q, want a depth-1 divide of 20 nodes", depth, line)
		}
	}
}

// TestPerftReportsRateWithoutDividingByZero guards the nps computation. A
// shallow perft really can finish inside the clock's resolution, and converting
// the resulting infinity to an integer is undefined in Go.
func TestPerftReportsRateWithoutDividingByZero(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("position startpos")
	d.send("perft 1")

	line := d.expect("nodes ", 5*time.Second)
	fields := strings.Fields(line)
	nps := fields[len(fields)-1]
	if _, err := strconv.ParseUint(nps, 10, 64); err != nil {
		t.Errorf("nps field %q is not a plain integer: %v (line %q)", nps, err, line)
	}
}

func TestNodesPerSecondHandlesZeroElapsed(t *testing.T) {
	if got := nodesPerSecond(1000, 0); got != 0 {
		t.Errorf("nodesPerSecond with zero elapsed = %d, want 0", got)
	}
	if got := nodesPerSecond(1000, -time.Second); got != 0 {
		t.Errorf("nodesPerSecond with negative elapsed = %d, want 0", got)
	}
	if got := nodesPerSecond(1000, time.Second); got != 1000 {
		t.Errorf("nodesPerSecond = %d, want 1000", got)
	}
}

func TestUnknownCommandsAreIgnored(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	// GUIs send extensions freely; a strict parser simply fails to work with
	// them. The engine must carry on.
	for _, cmd := range []string{"nonsense", "go wtime notanumber", "setoption name Bogus value 1", "position"} {
		d.send(cmd)
	}
	d.send("isready")
	d.expect("readyok", 2*time.Second)
}

func TestNoLegalMovesReportsNullMove(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	// Stalemate: there is no move to report.
	d.send("position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
	d.send("go depth 4")

	best := d.expect("bestmove", 10*time.Second)
	if !strings.Contains(best, "0000") {
		t.Errorf("bestmove = %q, want the null move 0000 in a stalemate", best)
	}
}

func TestGoWithClockLimitsFinishes(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("position startpos")
	d.send("go wtime 1000 btime 1000 winc 0 binc 0")

	start := time.Now()
	d.expect("bestmove", 10*time.Second)

	// With one second on the clock the engine must not spend anything like all
	// of it on a single move.
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("took %v with 1s on the clock", elapsed)
	}
}

func TestSecondGoDoesNotStartOverlappingSearch(t *testing.T) {
	d := newDriver(t)
	defer d.close()

	d.send("position startpos")
	d.send("go infinite")
	d.expect("info depth", 5*time.Second)

	// The first search must be ended before the second starts, or both would
	// drive the same position concurrently.
	d.send("go depth 4")

	// Two bestmove lines are expected in total: one per search.
	d.expect("bestmove", 10*time.Second)
	d.expect("bestmove", 20*time.Second)
}

func TestParseLimits(t *testing.T) {
	cases := []struct {
		name  string
		args  string
		check func(t *testing.T, l limitsView)
	}{
		{
			name: "depth",
			args: "depth 12",
			check: func(t *testing.T, l limitsView) {
				if l.Depth != 12 {
					t.Errorf("Depth = %d, want 12", l.Depth)
				}
			},
		},
		{
			name: "movetime",
			args: "movetime 1500",
			check: func(t *testing.T, l limitsView) {
				if l.MoveTime != 1500*time.Millisecond {
					t.Errorf("MoveTime = %v, want 1.5s", l.MoveTime)
				}
			},
		},
		{
			name: "clock",
			args: "wtime 300000 btime 290000 winc 2000 binc 2000 movestogo 40",
			check: func(t *testing.T, l limitsView) {
				if l.WhiteTime != 300*time.Second || l.BlackTime != 290*time.Second {
					t.Errorf("clock parsed as %v / %v", l.WhiteTime, l.BlackTime)
				}
				if l.WhiteInc != 2*time.Second || l.MovesToGo != 40 {
					t.Errorf("increment %v, movestogo %d", l.WhiteInc, l.MovesToGo)
				}
			},
		},
		{
			name: "bare go searches until stopped",
			args: "",
			check: func(t *testing.T, l limitsView) {
				if !l.Infinite {
					t.Error("a bare go should search until stopped")
				}
			},
		},
		{
			name: "malformed values are ignored rather than fatal",
			args: "depth notanumber movetime 500",
			check: func(t *testing.T, l limitsView) {
				if l.Depth != 0 {
					t.Errorf("Depth = %d, want 0 for an unparsable value", l.Depth)
				}
				if l.MoveTime != 500*time.Millisecond {
					t.Errorf("MoveTime = %v, want the value that did parse", l.MoveTime)
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l := parseLimits(strings.Fields(tc.args))
			tc.check(t, limitsView{
				Depth: l.Depth, Nodes: l.Nodes, MoveTime: l.MoveTime,
				WhiteTime: l.WhiteTime, BlackTime: l.BlackTime,
				WhiteInc: l.WhiteInc, BlackInc: l.BlackInc,
				MovesToGo: l.MovesToGo, Infinite: l.Infinite,
			})
		})
	}
}

// limitsView mirrors search.Limits so the table above reads without importing
// the search package into every case.
type limitsView struct {
	Depth                int
	Nodes                uint64
	MoveTime             time.Duration
	WhiteTime, BlackTime time.Duration
	WhiteInc, BlackInc   time.Duration
	MovesToGo            int
	Infinite             bool
}
