// Package uci implements the Universal Chess Interface.
//
// UCI is a line protocol over stdin and stdout, and its one hard rule is that
// the engine must never stop reading input. A GUI sends "stop" while the engine
// is thinking and expects a reply; an engine that only reads between searches
// appears to hang and loses on time. So the reader loop owns the goroutine that
// reads, and searches run elsewhere.
//
// The protocol is also permissive by convention: unknown commands and unknown
// tokens within a command are ignored rather than rejected, because GUIs send
// extensions freely and a strict parser simply fails to work with them.
package uci

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
	"github.com/piwi3910/novachess/internal/search"
)

// Engine identification, reported in response to "uci".
const (
	EngineName   = "Novachess"
	EngineAuthor = "Novachess contributors"
)

// Default and permitted option values.
const (
	defaultHashMB = 64
	minHashMB     = 1
	maxHashMB     = 32768
)

// Engine wires the UCI protocol to a searcher.
type Engine struct {
	out *bufio.Writer
	mu  sync.Mutex

	searcher *search.Searcher
	position *chess.Position

	// options
	hashMB   int
	chess960 bool

	// searching guards the running search so that "stop" and "quit" can reach
	// it and so that a second "go" cannot start one on top of another.
	searchMu   sync.Mutex
	searching  bool
	cancelFunc context.CancelFunc
	searchDone chan struct{}

	version string
}

// NewEngine returns an engine writing protocol output to w.
func NewEngine(w io.Writer, version string) *Engine {
	e := &Engine{
		out:      bufio.NewWriter(w),
		hashMB:   defaultHashMB,
		position: chess.NewPosition(),
		version:  version,
	}
	e.searcher = search.New(eval.NewHCE(), e.hashMB)
	return e
}

// Run reads commands until the input ends or "quit" is received.
func (e *Engine) Run(r io.Reader) error {
	scanner := bufio.NewScanner(r)
	// Long "position startpos moves ..." lines in a long game can exceed the
	// default token size.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if quit := e.execute(line); quit {
			break
		}
	}

	e.stopSearch()
	e.flush()
	return scanner.Err()
}

// execute handles one command, reporting whether the engine should quit.
func (e *Engine) execute(line string) bool {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return false
	}

	switch fields[0] {
	case "uci":
		e.identify()
	case "isready":
		// "isready" must be answered even mid-search; it is the GUI's way of
		// checking the engine is alive.
		e.send("readyok")
	case "ucinewgame":
		e.newGame()
	case "setoption":
		e.setOption(fields[1:])
	case "position":
		e.setPosition(fields[1:])
	case "go":
		e.goCommand(fields[1:])
	case "stop":
		e.stopSearch()
	case "ponderhit":
		// Pondering is not implemented; the search simply continues.
	case "quit":
		return true

	// Non-standard conveniences that cost nothing and make manual testing far
	// easier than driving the engine through a GUI.
	case "d", "board":
		e.send(e.currentPosition().String())
	case "eval":
		p := e.currentPosition()
		e.sendf("static evaluation: %d", eval.NewHCE().Evaluate(p))
	case "perft":
		e.perft(fields[1:])
	}
	return false
}

func (e *Engine) identify() {
	e.sendf("id name %s %s", EngineName, e.version)
	e.sendf("id author %s", EngineAuthor)
	e.sendf("option name Hash type spin default %d min %d max %d", defaultHashMB, minHashMB, maxHashMB)
	e.sendf("option name UCI_Chess960 type check default false")
	e.send("uciok")
}

func (e *Engine) newGame() {
	e.stopSearch()
	e.searcher.Clear()
	e.setPositionTo(chess.NewPosition())
}

// setOption parses "setoption name <name> value <value>". Option names may
// contain spaces, so the name runs up to the "value" keyword.
func (e *Engine) setOption(args []string) {
	nameParts, valueParts := []string{}, []string{}
	target := &nameParts

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "name":
			target = &nameParts
		case "value":
			target = &valueParts
		default:
			*target = append(*target, args[i])
		}
	}

	name := strings.ToLower(strings.Join(nameParts, " "))
	value := strings.Join(valueParts, " ")

	switch name {
	case "hash":
		mb, err := strconv.Atoi(value)
		if err != nil {
			return
		}
		mb = clamp(mb, minHashMB, maxHashMB)
		e.stopSearch()
		e.hashMB = mb
		e.searcher.SetHashSize(mb)

	case "uci_chess960":
		e.chess960 = strings.EqualFold(value, "true")
		// Re-parse the current position so its castling notation follows the
		// new setting; otherwise moves would be echoed in the wrong form.
		e.setPositionTo(reinterpret(e.currentPosition(), e.chess960))
	}
}

// setPosition handles "position [startpos | fen <fen>] [moves <m>...]".
func (e *Engine) setPosition(args []string) {
	if len(args) == 0 {
		return
	}

	var p *chess.Position
	rest := args

	switch args[0] {
	case "startpos":
		var err error
		p, err = chess.ParseFENVariant(chess.StartingFEN, e.chess960)
		if err != nil {
			return
		}
		rest = args[1:]

	case "fen":
		// The FEN runs until "moves" or the end of the line. Its field count
		// varies because the halfmove and fullmove numbers are optional.
		end := len(args)
		for i, f := range args {
			if f == "moves" {
				end = i
				break
			}
		}
		fen := strings.Join(args[1:end], " ")
		parsed, err := chess.ParseFENVariant(fen, e.chess960)
		if err != nil {
			// A malformed position is dropped rather than accepted: continuing
			// with a stale board would make the engine answer about a
			// different game entirely.
			e.sendf("info string ignoring invalid position: %v", err)
			return
		}
		p = parsed
		rest = args[end:]

	default:
		return
	}

	// Apply the move list, resolving each against the legal moves of the
	// position it is played in. That is what makes castling unambiguous in
	// both notations.
	if len(rest) > 0 && rest[0] == "moves" {
		for _, token := range rest[1:] {
			m, ok := p.ParseUCIMove(token)
			if !ok {
				e.sendf("info string ignoring illegal move %q", token)
				break
			}
			p.MakeMove(m)
		}
	}

	e.setPositionTo(p)
}

// goCommand parses the search limits and starts a search.
func (e *Engine) goCommand(args []string) {
	// A second "go" while already searching would corrupt the shared position,
	// so the previous search is ended first.
	e.stopSearch()

	limits := parseLimits(args)
	pos := e.currentPosition()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})

	e.searchMu.Lock()
	e.searching = true
	e.cancelFunc = cancel
	e.searchDone = done
	e.searchMu.Unlock()

	// The search runs off the reader goroutine so that "stop" and "isready"
	// continue to be answered while it thinks.
	go func() {
		defer close(done)
		defer cancel()

		result := e.searcher.Search(ctx, pos, limits, func(info search.Info) {
			e.sendInfo(pos, info)
		})

		e.sendBestMove(pos, result)

		e.searchMu.Lock()
		e.searching = false
		e.searchMu.Unlock()
	}()
}

// stopSearch ends any running search and waits for it to report, so that
// bestmove is always emitted before the next command is processed.
func (e *Engine) stopSearch() {
	e.searchMu.Lock()
	cancel, done, running := e.cancelFunc, e.searchDone, e.searching
	e.searchMu.Unlock()

	if !running {
		return
	}

	e.searcher.Stop()
	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
}

func (e *Engine) sendInfo(pos *chess.Position, info search.Info) {
	var b strings.Builder

	fmt.Fprintf(&b, "info depth %d seldepth %d", info.Depth, info.SelDepth)

	if info.Mate {
		fmt.Fprintf(&b, " score mate %d", info.MateIn)
	} else {
		fmt.Fprintf(&b, " score cp %d", info.Score)
	}

	fmt.Fprintf(&b, " nodes %d nps %d time %d hashfull %d",
		info.Nodes, info.NodesPerSecond(), info.Elapsed.Milliseconds(), info.HashFull)

	if len(info.PV) > 0 {
		b.WriteString(" pv")
		// Each move must be rendered in the notation the position it is played
		// in would use, which for castling depends on the variant.
		walk := pos.Clone()
		for _, m := range info.PV {
			b.WriteByte(' ')
			b.WriteString(walk.MoveToUCI(m))
			walk.MakeMove(m)
		}
	}

	e.send(b.String())
}

func (e *Engine) sendBestMove(pos *chess.Position, result search.Result) {
	if result.Best.IsNone() {
		// There is no legal move. UCI has no clean way to say so; "0000" is
		// the conventional null move and is what GUIs expect here.
		e.send("bestmove 0000")
		return
	}

	best := pos.MoveToUCI(result.Best)
	if !result.Ponder.IsNone() {
		after := pos.Clone()
		after.MakeMove(result.Best)
		e.sendf("bestmove %s ponder %s", best, after.MoveToUCI(result.Ponder))
		return
	}
	e.sendf("bestmove %s", best)
}

// perft is a debugging command, not part of UCI. It is worth having because
// verifying move generation against a GUI is otherwise painful.
func (e *Engine) perft(args []string) {
	depth := 1
	if len(args) > 0 {
		if d, err := strconv.Atoi(args[0]); err == nil {
			depth = d
		}
	}

	p := e.currentPosition()
	start := time.Now()
	divide, total := chess.PerftDivide(p, depth)

	var ml chess.MoveList
	p.GenerateLegalMoves(&ml)
	for _, m := range ml.Slice() {
		e.sendf("%s: %d", p.MoveToUCI(m), divide[m.String()])
	}

	elapsed := time.Since(start)
	e.sendf("nodes %d time %d nps %d", total, elapsed.Milliseconds(),
		int64(float64(total)/elapsed.Seconds()))
}

// currentPosition returns a copy of the engine's position, so that a running
// search cannot be affected by a command that changes it.
func (e *Engine) currentPosition() *chess.Position {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.position.Clone()
}

func (e *Engine) setPositionTo(p *chess.Position) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.position = p
}

func (e *Engine) send(s string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	fmt.Fprintln(e.out, s)
	// Output is flushed on every line. Buffering protocol replies would be a
	// deadlock: the GUI waits for a response the engine is holding in memory.
	e.out.Flush()
}

func (e *Engine) sendf(format string, args ...any) { e.send(fmt.Sprintf(format, args...)) }

func (e *Engine) flush() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.out.Flush()
}

// parseLimits reads the arguments of "go".
func parseLimits(args []string) search.Limits {
	var l search.Limits

	intAt := func(i int) (int, bool) {
		if i >= len(args) {
			return 0, false
		}
		v, err := strconv.Atoi(args[i])
		if err != nil {
			return 0, false
		}
		return v, true
	}
	msAt := func(i int) (time.Duration, bool) {
		v, ok := intAt(i)
		if !ok || v < 0 {
			return 0, false
		}
		return time.Duration(v) * time.Millisecond, true
	}

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "infinite":
			l.Infinite = true
		case "ponder":
			// Treated as an infinite search until "ponderhit" or "stop".
			l.Infinite = true
		case "depth":
			if v, ok := intAt(i + 1); ok {
				l.Depth = v
				i++
			}
		case "nodes":
			if v, ok := intAt(i + 1); ok && v >= 0 {
				l.Nodes = uint64(v)
				i++
			}
		case "movetime":
			if v, ok := msAt(i + 1); ok {
				l.MoveTime = v
				i++
			}
		case "wtime":
			if v, ok := msAt(i + 1); ok {
				l.WhiteTime = v
				i++
			}
		case "btime":
			if v, ok := msAt(i + 1); ok {
				l.BlackTime = v
				i++
			}
		case "winc":
			if v, ok := msAt(i + 1); ok {
				l.WhiteInc = v
				i++
			}
		case "binc":
			if v, ok := msAt(i + 1); ok {
				l.BlackInc = v
				i++
			}
		case "movestogo":
			if v, ok := intAt(i + 1); ok {
				l.MovesToGo = v
				i++
			}
		}
	}

	// "go" with nothing else means search until told to stop.
	if l.Depth == 0 && l.Nodes == 0 && l.MoveTime == 0 &&
		l.WhiteTime == 0 && l.BlackTime == 0 {
		l.Infinite = true
	}
	return l
}

// reinterpret re-parses a position under a different variant setting, so that
// castling is written the way the new setting expects.
func reinterpret(p *chess.Position, chess960 bool) *chess.Position {
	q, err := chess.ParseFENVariant(p.FEN(), chess960)
	if err != nil {
		return p
	}
	return q
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
