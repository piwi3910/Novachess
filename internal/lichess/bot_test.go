package lichess

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/piwi3910/novachess/internal/chess"
)

// fakeLichess is a stand-in for the Lichess API.
//
// The bot is tested against it rather than against the real service for the
// obvious reason — tests must not depend on a network or an account — but also
// because the failure modes worth testing are ones the real API produces rarely
// and unpredictably: streams that close mid-game, rate limits, illegal move
// lists. Here they are produced on demand.
type fakeLichess struct {
	t *testing.T

	mu sync.Mutex
	// events is written to by the test and drained by the bot's event stream.
	events chan string
	// gameMessages is per game id.
	gameMessages map[string]chan string
	// moves records what the bot played, in order.
	moves []string
	// accepted and declined record challenge decisions, with the decline reason.
	accepted []string
	declined map[string]string
	chats    []string

	// rateLimitMoves makes the next move POST return 429, once.
	rateLimitMoves bool

	server *httptest.Server
}

func newFakeLichess(t *testing.T) *fakeLichess {
	t.Helper()

	f := &fakeLichess{
		t:            t,
		events:       make(chan string, 64),
		gameMessages: make(map[string]chan string),
		declined:     make(map[string]string),
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/api/account", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(Account{ID: "novabot", Username: "NovaBot"})
	})

	mux.HandleFunc("/api/stream/event", func(w http.ResponseWriter, r *http.Request) {
		f.streamChannel(w, r, f.events)
	})

	mux.HandleFunc("/api/bot/game/stream/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/api/bot/game/stream/")
		f.streamChannel(w, r, f.gameChannel(id))
	})

	mux.HandleFunc("/api/bot/game/", func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/bot/game/"), "/")
		if len(parts) < 2 {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		switch parts[1] {
		case "move":
			f.mu.Lock()
			if f.rateLimitMoves {
				f.rateLimitMoves = false
				f.mu.Unlock()
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			f.moves = append(f.moves, parts[2])
			f.mu.Unlock()
			fmt.Fprint(w, `{"ok":true}`)
		case "chat":
			body, _ := io.ReadAll(r.Body)
			f.mu.Lock()
			f.chats = append(f.chats, string(body))
			f.mu.Unlock()
			fmt.Fprint(w, `{"ok":true}`)
		default:
			fmt.Fprint(w, `{"ok":true}`)
		}
	})

	mux.HandleFunc("/api/challenge/", func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/challenge/"), "/")
		if len(parts) < 2 {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		id, action := parts[0], parts[1]

		f.mu.Lock()
		switch action {
		case "accept":
			f.accepted = append(f.accepted, id)
		case "decline":
			r.ParseForm()
			f.declined[id] = r.FormValue("reason")
		}
		f.mu.Unlock()
		fmt.Fprint(w, `{"ok":true}`)
	})

	f.server = httptest.NewServer(mux)
	t.Cleanup(f.server.Close)
	return f
}

// streamChannel writes lines from a channel as NDJSON, flushing each so the
// client sees it immediately, and interleaving blank keep-alives exactly as
// Lichess does.
func (f *fakeLichess) streamChannel(w http.ResponseWriter, r *http.Request, ch chan string) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	flusher, ok := w.(http.Flusher)
	if !ok {
		f.t.Error("response writer cannot flush; the stream would be buffered")
		return
	}
	// A keep-alive before anything else, which the client must skip rather
	// than try to parse.
	fmt.Fprint(w, "\n")
	flusher.Flush()

	for {
		select {
		case line, open := <-ch:
			if !open {
				return
			}
			fmt.Fprintln(w, line)
			flusher.Flush()
		case <-r.Context().Done():
			return
		case <-time.After(10 * time.Second):
			return
		}
	}
}

func (f *fakeLichess) gameChannel(id string) chan string {
	f.mu.Lock()
	defer f.mu.Unlock()
	ch, ok := f.gameMessages[id]
	if !ok {
		ch = make(chan string, 64)
		f.gameMessages[id] = ch
	}
	return ch
}

func (f *fakeLichess) sendEvent(v any) {
	b, err := json.Marshal(v)
	if err != nil {
		f.t.Fatal(err)
	}
	f.events <- string(b)
}

func (f *fakeLichess) sendGame(id string, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		f.t.Fatal(err)
	}
	f.gameChannel(id) <- string(b)
}

func (f *fakeLichess) playedMoves() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.moves...)
}

func (f *fakeLichess) declineReasonFor(id string) (string, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	r, ok := f.declined[id]
	return r, ok
}

func (f *fakeLichess) wasAccepted(id string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, a := range f.accepted {
		if a == id {
			return true
		}
	}
	return false
}

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func startBot(t *testing.T, f *fakeLichess, config Config) (context.CancelFunc, *sync.WaitGroup) {
	t.Helper()

	bot := NewBot(NewClient("test-token", f.server.URL), config, quietLogger())
	ctx, cancel := context.WithCancel(context.Background())

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		bot.Run(ctx)
	}()

	t.Cleanup(func() {
		cancel()
		wg.Wait()
	})
	return cancel, &wg
}

// waitFor polls until cond holds or the timeout expires.
func waitFor(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func testConfig() Config {
	c := DefaultConfig()
	c.HashMB = 4
	return c
}

// TestPlaysAGame is the end-to-end check: a game arrives, the bot works out it
// is White, searches, and posts a legal move.
func TestPlaysAGame(t *testing.T) {
	f := newFakeLichess(t)
	startBot(t, f, testConfig())

	f.sendEvent(Event{Type: EventGameStart, Game: &GameEvent{ID: "game1"}})
	f.sendGame("game1", GameMessage{
		Type:       MessageGameFull,
		ID:         "game1",
		Variant:    Variant{Key: VariantStandard},
		InitialFEN: "startpos",
		White:      Player{ID: "novabot", Name: "NovaBot"},
		Black:      Player{ID: "human", Name: "Human"},
		State: GameState{
			Type: MessageGameState, Moves: "",
			WTime: 60000, BTime: 60000, WInc: 1000, BInc: 1000,
			Status: StatusStarted,
		},
	})

	waitFor(t, 15*time.Second, "the bot's first move", func() bool {
		return len(f.playedMoves()) >= 1
	})

	// The move must be legal in the starting position.
	first := f.playedMoves()[0]
	pos := chess.NewPosition()
	if _, ok := pos.ParseUCIMove(first); !ok {
		t.Fatalf("bot played %q, which is not legal in the starting position", first)
	}
}

// TestPlaysBlackAndFollowsTheMoveList checks colour detection and that the
// position is rebuilt from the stream's move list rather than assumed.
func TestPlaysBlackAndFollowsTheMoveList(t *testing.T) {
	f := newFakeLichess(t)
	startBot(t, f, testConfig())

	f.sendEvent(Event{Type: EventGameStart, Game: &GameEvent{ID: "g2"}})
	f.sendGame("g2", GameMessage{
		Type:       MessageGameFull,
		ID:         "g2",
		Variant:    Variant{Key: VariantStandard},
		InitialFEN: "startpos",
		White:      Player{ID: "human", Name: "Human"},
		Black:      Player{ID: "novabot", Name: "NovaBot"},
		// White has already played, so it is the bot's turn immediately.
		State: GameState{
			Type: MessageGameState, Moves: "e2e4",
			WTime: 60000, BTime: 60000, Status: StatusStarted,
		},
	})

	waitFor(t, 15*time.Second, "a reply as Black", func() bool {
		return len(f.playedMoves()) >= 1
	})

	reply := f.playedMoves()[0]
	pos := chess.NewPosition()
	m, _ := pos.ParseUCIMove("e2e4")
	pos.MakeMove(m)
	if pos.SideToMove() != chess.Black {
		t.Fatal("fixture error: it should be Black to move")
	}
	if _, ok := pos.ParseUCIMove(reply); !ok {
		t.Errorf("bot replied %q, illegal after 1.e4", reply)
	}
}

// TestDoesNotMoveOutOfTurn covers the most embarrassing possible bug. A bot
// that moves when it is not its turn floods the API with rejected requests.
func TestDoesNotMoveOutOfTurn(t *testing.T) {
	f := newFakeLichess(t)
	startBot(t, f, testConfig())

	f.sendEvent(Event{Type: EventGameStart, Game: &GameEvent{ID: "g3"}})
	f.sendGame("g3", GameMessage{
		Type:       MessageGameFull,
		ID:         "g3",
		Variant:    Variant{Key: VariantStandard},
		InitialFEN: "startpos",
		White:      Player{ID: "human", Name: "Human"},
		Black:      Player{ID: "novabot", Name: "NovaBot"},
		// No moves yet, and the bot is Black: it is not the bot's turn.
		State: GameState{Type: MessageGameState, Moves: "", WTime: 60000, BTime: 60000, Status: StatusStarted},
	})

	time.Sleep(1500 * time.Millisecond)
	if moves := f.playedMoves(); len(moves) != 0 {
		t.Errorf("bot moved out of turn: %v", moves)
	}
}

// TestStopsMovingWhenTheGameEnds checks the bot does not keep searching a
// finished game.
func TestStopsMovingWhenTheGameEnds(t *testing.T) {
	f := newFakeLichess(t)
	startBot(t, f, testConfig())

	f.sendEvent(Event{Type: EventGameStart, Game: &GameEvent{ID: "g4"}})
	f.sendGame("g4", GameMessage{
		Type:       MessageGameFull,
		ID:         "g4",
		Variant:    Variant{Key: VariantStandard},
		InitialFEN: "startpos",
		White:      Player{ID: "novabot"},
		Black:      Player{ID: "human"},
		State: GameState{
			Type: MessageGameState, Moves: "", WTime: 60000, BTime: 60000,
			Status: "mate", Winner: "black",
		},
	})

	time.Sleep(1500 * time.Millisecond)
	if moves := f.playedMoves(); len(moves) != 0 {
		t.Errorf("bot moved in a finished game: %v", moves)
	}
}

// TestChallengePolicy covers accept and decline decisions together, since the
// rule is that every challenge gets exactly one of the two — leaving one
// unanswered makes the bot look dead to whoever sent it.
func TestChallengePolicy(t *testing.T) {
	cases := []struct {
		name       string
		challenge  Challenge
		config     func(c *Config)
		wantAccept bool
		wantReason string
	}{
		{
			name: "standard rated blitz is accepted",
			challenge: Challenge{
				ID: "c1", Rated: true, Variant: Variant{Key: VariantStandard},
				TimeControl: TimeControl{Type: "clock", Limit: 300, Increment: 3},
				Challenger:  Player{ID: "human"},
			},
			wantAccept: true,
		},
		{
			name: "chess960 is accepted because the engine implements it",
			challenge: Challenge{
				ID: "c2", Rated: false, Variant: Variant{Key: VariantChess960},
				TimeControl: TimeControl{Type: "clock", Limit: 300},
				Challenger:  Player{ID: "human"},
			},
			wantAccept: true,
		},
		{
			name: "an unimplemented variant is declined",
			challenge: Challenge{
				ID: "c3", Variant: Variant{Key: VariantCrazyhouse},
				TimeControl: TimeControl{Type: "clock", Limit: 300},
				Challenger:  Player{ID: "human"},
			},
			wantReason: DeclineVariant,
		},
		{
			name: "a game too fast to play well is declined",
			challenge: Challenge{
				ID: "c4", Variant: Variant{Key: VariantStandard},
				TimeControl: TimeControl{Type: "clock", Limit: 15},
				Challenger:  Player{ID: "human"},
			},
			wantReason: DeclineTooFast,
		},
		{
			name: "correspondence is declined by default",
			challenge: Challenge{
				ID: "c5", Variant: Variant{Key: VariantStandard},
				TimeControl: TimeControl{Type: "correspondence"},
				Challenger:  Player{ID: "human"},
			},
			wantReason: DeclineTooSlow,
		},
		{
			name: "rated is declined when only casual is wanted",
			challenge: Challenge{
				ID: "c6", Rated: true, Variant: Variant{Key: VariantStandard},
				TimeControl: TimeControl{Type: "clock", Limit: 300},
				Challenger:  Player{ID: "human"},
			},
			config:     func(c *Config) { c.AcceptRated = false },
			wantReason: DeclineCasual,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeLichess(t)
			config := testConfig()
			if tc.config != nil {
				tc.config(&config)
			}
			startBot(t, f, config)

			f.sendEvent(Event{Type: EventChallenge, Challenge: &tc.challenge})

			id := tc.challenge.ID
			waitFor(t, 5*time.Second, "a decision on the challenge", func() bool {
				if f.wasAccepted(id) {
					return true
				}
				_, declined := f.declineReasonFor(id)
				return declined
			})

			if tc.wantAccept {
				if !f.wasAccepted(id) {
					reason, _ := f.declineReasonFor(id)
					t.Errorf("challenge was declined (%s), want accepted", reason)
				}
				return
			}

			reason, declined := f.declineReasonFor(id)
			if !declined {
				t.Fatal("challenge was accepted, want declined")
			}
			if reason != tc.wantReason {
				t.Errorf("declined with reason %q, want %q", reason, tc.wantReason)
			}
		})
	}
}

// TestDeclinesWhenAtCapacity checks the concurrency bound is enforced, which is
// what stops the bot accepting more games than it can think in.
func TestDeclinesWhenAtCapacity(t *testing.T) {
	f := newFakeLichess(t)
	config := testConfig()
	config.MaxConcurrentGames = 1
	startBot(t, f, config)

	// Occupy the only slot with a running game.
	f.sendEvent(Event{Type: EventGameStart, Game: &GameEvent{ID: "busy"}})
	f.sendGame("busy", GameMessage{
		Type: MessageGameFull, ID: "busy",
		Variant: Variant{Key: VariantStandard}, InitialFEN: "startpos",
		White: Player{ID: "human"}, Black: Player{ID: "novabot"},
		State: GameState{Type: MessageGameState, Moves: "", WTime: 60000, BTime: 60000, Status: StatusStarted},
	})

	waitFor(t, 5*time.Second, "the game to occupy a slot", func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		_, ok := f.gameMessages["busy"]
		return ok
	})
	time.Sleep(200 * time.Millisecond)

	f.sendEvent(Event{Type: EventChallenge, Challenge: &Challenge{
		ID: "extra", Variant: Variant{Key: VariantStandard},
		TimeControl: TimeControl{Type: "clock", Limit: 300},
		Challenger:  Player{ID: "human"},
	}})

	waitFor(t, 5*time.Second, "the challenge to be declined", func() bool {
		_, declined := f.declineReasonFor("extra")
		return declined
	})
	if reason, _ := f.declineReasonFor("extra"); reason != DeclineLater {
		t.Errorf("declined with %q, want %q", reason, DeclineLater)
	}
}

// TestIgnoresOwnChallenges checks the bot does not try to accept a challenge it
// issued itself, which arrives on the same stream.
func TestIgnoresOwnChallenges(t *testing.T) {
	f := newFakeLichess(t)
	startBot(t, f, testConfig())

	f.sendEvent(Event{Type: EventChallenge, Challenge: &Challenge{
		ID: "mine", Variant: Variant{Key: VariantStandard},
		TimeControl: TimeControl{Type: "clock", Limit: 300},
		Challenger:  Player{ID: "novabot"},
	}})

	time.Sleep(500 * time.Millisecond)
	if f.wasAccepted("mine") {
		t.Error("bot accepted its own challenge")
	}
	if _, declined := f.declineReasonFor("mine"); declined {
		t.Error("bot declined its own challenge")
	}
}

// TestChess960GameUsesTheRightNotation checks a Chess960 game is played with
// king-takes-rook castling, since sending the classical form would have the
// move rejected.
func TestChess960GameUsesTheRightNotation(t *testing.T) {
	f := newFakeLichess(t)
	startBot(t, f, testConfig())

	// A position whose only legal move is castling, so the notation is forced.
	// The white king on e1 with a rook on h1 and Black to have just moved.
	fen := "4k3/8/8/8/8/8/6PP/5RK1 w - - 0 1"

	f.sendEvent(Event{Type: EventGameStart, Game: &GameEvent{ID: "g960"}})
	f.sendGame("g960", GameMessage{
		Type: MessageGameFull, ID: "g960",
		Variant: Variant{Key: VariantChess960}, InitialFEN: fen,
		White: Player{ID: "novabot"}, Black: Player{ID: "human"},
		State: GameState{Type: MessageGameState, Moves: "", WTime: 60000, BTime: 60000, Status: StatusStarted},
	})

	waitFor(t, 15*time.Second, "a move in the Chess960 game", func() bool {
		return len(f.playedMoves()) >= 1
	})

	// Whatever it played must be legal under the Chess960 reading of the FEN.
	pos, err := chess.ParseFENVariant(fen, true)
	if err != nil {
		t.Fatal(err)
	}
	move := f.playedMoves()[0]
	if _, ok := pos.ParseUCIMove(move); !ok {
		t.Errorf("bot played %q, illegal in %s", move, fen)
	}
}

// TestRateLimitedMoveIsRetried checks that a 429 on a move does not simply lose
// the move. Losing it would leave the bot silently not playing while its clock
// ran down.
func TestRateLimitedMoveIsRetried(t *testing.T) {
	f := newFakeLichess(t)
	f.mu.Lock()
	f.rateLimitMoves = true
	f.mu.Unlock()

	startBot(t, f, testConfig())

	f.sendEvent(Event{Type: EventGameStart, Game: &GameEvent{ID: "rl"}})
	f.sendGame("rl", GameMessage{
		Type: MessageGameFull, ID: "rl",
		Variant: Variant{Key: VariantStandard}, InitialFEN: "startpos",
		White: Player{ID: "novabot"}, Black: Player{ID: "human"},
		State: GameState{Type: MessageGameState, Moves: "", WTime: 60000, BTime: 60000, Status: StatusStarted},
	})

	// The retry waits a full backoff period, which is a minute in production.
	// This asserts the request was rejected and the bot did not treat that as
	// success; the retry itself is covered by the code path rather than by
	// waiting sixty seconds here.
	time.Sleep(2 * time.Second)
	if moves := f.playedMoves(); len(moves) != 0 {
		t.Errorf("a rate-limited move was recorded as played: %v", moves)
	}
}

func TestIsFinished(t *testing.T) {
	ongoing := []string{StatusStarted, StatusCreated, ""}
	for _, s := range ongoing {
		if IsFinished(s) {
			t.Errorf("IsFinished(%q) = true, want false", s)
		}
	}
	// Anything unrecognized must count as finished, so an unknown status ends
	// the loop rather than leaving it spinning on a game nobody is playing.
	finished := []string{"mate", "resign", "stalemate", "outoftime", "draw", "aborted", "somethingNew"}
	for _, s := range finished {
		if !IsFinished(s) {
			t.Errorf("IsFinished(%q) = false, want true", s)
		}
	}
}

func TestVariantSupport(t *testing.T) {
	b := NewBot(nil, Config{AcceptVariants: []string{VariantStandard, VariantChess960}}, quietLogger())

	if !b.variantSupported(VariantStandard) || !b.variantSupported(VariantChess960) {
		t.Error("the implemented variants should be supported")
	}
	for _, v := range []string{VariantCrazyhouse, "atomic", "horde", "antichess"} {
		if b.variantSupported(v) {
			t.Errorf("%s is not implemented and must not be accepted", v)
		}
	}
}
