package lichess

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
	"github.com/piwi3910/novachess/internal/search"
)

// Config controls which challenges the bot accepts and how hard it thinks.
type Config struct {
	// MaxConcurrentGames bounds how many games run at once. Each game holds a
	// searcher with its own transposition table, so this is a memory decision
	// as much as a strength one: a bot playing eight games at once thinks with
	// an eighth of the CPU in each.
	MaxConcurrentGames int

	// Threads and HashMB configure each game's searcher. Threads may exceed one
	// here — unlike self-play, nothing downstream needs these games reproduced.
	Threads int
	HashMB  int

	// AcceptRated and AcceptCasual select which kinds of game to play.
	AcceptRated  bool
	AcceptCasual bool

	// AcceptVariants lists playable variant keys. Anything else is declined,
	// because the engine implements no other variant's rules and accepting one
	// would mean playing illegal moves rather than merely playing badly.
	AcceptVariants []string

	// MinInitialSeconds declines games too fast to play well. A bot that
	// accepts bullet it cannot handle just loses on time repeatedly.
	MinInitialSeconds int

	// AcceptCorrespondence allows games with no clock. These last days, so the
	// default is to decline rather than hold a game slot indefinitely.
	AcceptCorrespondence bool

	// GreetingMessage is posted in the chat at the start of each game. Empty
	// posts nothing.
	GreetingMessage string
}

// DefaultConfig returns a sensible starting configuration.
func DefaultConfig() Config {
	return Config{
		MaxConcurrentGames: 4,
		Threads:            1,
		HashMB:             128,
		AcceptRated:        true,
		AcceptCasual:       true,
		AcceptVariants:     []string{VariantStandard, VariantChess960},
		MinInitialSeconds:  60,
	}
}

// Bot plays games on Lichess.
type Bot struct {
	client  *Client
	config  Config
	log     *slog.Logger
	account *Account

	mu      sync.Mutex
	playing map[string]context.CancelFunc
}

// NewBot returns a bot using the given client.
func NewBot(client *Client, config Config, log *slog.Logger) *Bot {
	if log == nil {
		log = slog.Default()
	}
	if config.MaxConcurrentGames < 1 {
		config.MaxConcurrentGames = 1
	}
	return &Bot{
		client:  client,
		config:  config,
		log:     log,
		playing: make(map[string]context.CancelFunc),
	}
}

// Run connects and plays until the context is cancelled.
//
// The event stream is reconnected whenever it ends, because Lichess closes it
// routinely and an ended stream is the normal case rather than a failure.
func (b *Bot) Run(ctx context.Context) error {
	account, err := b.client.Account(ctx)
	if err != nil {
		return fmt.Errorf("lichess: cannot identify account: %w", err)
	}
	b.account = account
	b.log.Info("connected", "account", account.Username, "id", account.ID)

	backoff := time.Second
	for ctx.Err() == nil {
		err := b.client.StreamEvents(ctx, func(e Event) error {
			b.handleEvent(ctx, e)
			return nil
		})

		if ctx.Err() != nil {
			break
		}

		switch {
		case err == nil:
			// A clean close is expected; reconnect immediately.
			backoff = time.Second
		case errors.Is(err, ErrRateLimited):
			b.log.Warn("rate limited on the event stream", "backoff", RateLimitBackoff)
			if !sleepCtx(ctx, RateLimitBackoff) {
				return ctx.Err()
			}
			continue
		default:
			b.log.Warn("event stream ended", "error", err, "backoff", backoff)
			if !sleepCtx(ctx, backoff) {
				return ctx.Err()
			}
			// Grow the delay so a persistently broken connection does not turn
			// into a request flood, which is how bots get banned.
			if backoff < time.Minute {
				backoff *= 2
			}
			continue
		}
	}

	b.stopAllGames()
	return ctx.Err()
}

func (b *Bot) handleEvent(ctx context.Context, e Event) {
	switch e.Type {
	case EventChallenge:
		if e.Challenge != nil {
			b.handleChallenge(ctx, *e.Challenge)
		}
	case EventGameStart:
		if e.Game != nil {
			b.startGame(ctx, e.Game.ID)
		}
	case EventGameFinish:
		if e.Game != nil {
			b.finishGame(e.Game.ID)
		}
	}
}

// handleChallenge accepts or declines, always doing one or the other: leaving a
// challenge unanswered makes the bot look dead to whoever sent it.
func (b *Bot) handleChallenge(ctx context.Context, ch Challenge) {
	// A challenge the bot itself issued arrives here too; there is nothing to
	// accept.
	if b.account != nil && ch.Challenger.ID == b.account.ID {
		return
	}

	if reason, ok := b.declineReason(ch); !ok {
		b.log.Info("declining challenge", "id", ch.ID, "from", ch.Challenger.Name,
			"variant", ch.Variant.Key, "reason", reason)
		if err := b.client.DeclineChallenge(ctx, ch.ID, reason); err != nil {
			b.log.Warn("decline failed", "id", ch.ID, "error", err)
		}
		return
	}

	b.log.Info("accepting challenge", "id", ch.ID, "from", ch.Challenger.Name,
		"variant", ch.Variant.Key, "rated", ch.Rated, "speed", ch.Speed)
	if err := b.client.AcceptChallenge(ctx, ch.ID); err != nil {
		b.log.Warn("accept failed", "id", ch.ID, "error", err)
	}
}

// declineReason reports whether a challenge is playable, and if not, which of
// Lichess's standard reasons to send.
func (b *Bot) declineReason(ch Challenge) (string, bool) {
	if !b.variantSupported(ch.Variant.Key) {
		return DeclineVariant, false
	}
	if ch.Rated && !b.config.AcceptRated {
		return DeclineCasual, false
	}
	if !ch.Rated && !b.config.AcceptCasual {
		return DeclineRated, false
	}

	switch ch.TimeControl.Type {
	case "clock":
		if b.config.MinInitialSeconds > 0 && ch.TimeControl.Limit < b.config.MinInitialSeconds {
			return DeclineTooFast, false
		}
	case "correspondence", "unlimited":
		if !b.config.AcceptCorrespondence {
			return DeclineTooSlow, false
		}
	}

	b.mu.Lock()
	atCapacity := len(b.playing) >= b.config.MaxConcurrentGames
	b.mu.Unlock()
	if atCapacity {
		return DeclineLater, false
	}

	return "", true
}

func (b *Bot) variantSupported(key string) bool {
	for _, v := range b.config.AcceptVariants {
		if v == key {
			return true
		}
	}
	return false
}

// startGame begins playing a game in its own goroutine.
//
// Games must not run on the event-stream goroutine: a search takes seconds, and
// blocking there would stop the bot noticing anything else, including moves in
// its other games.
func (b *Bot) startGame(ctx context.Context, gameID string) {
	b.mu.Lock()
	if _, already := b.playing[gameID]; already {
		b.mu.Unlock()
		return
	}
	gameCtx, cancel := context.WithCancel(ctx)
	b.playing[gameID] = cancel
	b.mu.Unlock()

	go func() {
		defer b.finishGame(gameID)
		if err := b.playGame(gameCtx, gameID); err != nil && gameCtx.Err() == nil {
			b.log.Warn("game ended with an error", "game", gameID, "error", err)
		}
	}()
}

func (b *Bot) finishGame(gameID string) {
	b.mu.Lock()
	cancel, ok := b.playing[gameID]
	delete(b.playing, gameID)
	b.mu.Unlock()

	if ok {
		cancel()
	}
}

func (b *Bot) stopAllGames() {
	b.mu.Lock()
	cancels := make([]context.CancelFunc, 0, len(b.playing))
	for _, cancel := range b.playing {
		cancels = append(cancels, cancel)
	}
	b.playing = make(map[string]context.CancelFunc)
	b.mu.Unlock()

	for _, cancel := range cancels {
		cancel()
	}
}

// playGame drives one game to completion.
func (b *Bot) playGame(ctx context.Context, gameID string) error {
	g := &game{
		bot:      b,
		id:       gameID,
		searcher: search.New(eval.NewHCE(), b.config.HashMB),
	}
	g.searcher.SetThreads(b.config.Threads)

	return b.client.StreamGame(ctx, gameID, func(m GameMessage) error {
		return g.handle(ctx, m)
	})
}

// game is the per-game state.
type game struct {
	bot      *Bot
	id       string
	searcher *search.Searcher

	// Fixed for the game's lifetime, set from the opening gameFull message.
	initialFEN string
	chess960   bool
	ourColor   chess.Color
	started    bool
	greeted    bool
}

func (g *game) handle(ctx context.Context, m GameMessage) error {
	switch m.Type {
	case MessageGameFull:
		if err := g.begin(ctx, m); err != nil {
			return err
		}
		return g.maybeMove(ctx, m.State)

	case MessageGameState:
		return g.maybeMove(ctx, GameState{
			Moves: m.Moves, WTime: m.WTime, BTime: m.BTime,
			WInc: m.WInc, BInc: m.BInc, Status: m.Status, Winner: m.Winner,
		})

	case MessageOpponentGone:
		// Lichess awards the win automatically after a delay; nothing to do.
		return nil
	}
	return nil
}

// begin records the immutable facts about the game.
func (g *game) begin(ctx context.Context, m GameMessage) error {
	g.chess960 = m.Variant.Key == VariantChess960
	g.initialFEN = m.InitialFEN
	if g.initialFEN == "" || g.initialFEN == "startpos" {
		g.initialFEN = chess.StartingFEN
	}

	// Which side we are is decided by matching the account id, not by
	// assuming; a bot plays both colors.
	switch {
	case g.bot.account != nil && m.White.ID == g.bot.account.ID:
		g.ourColor = chess.White
	case g.bot.account != nil && m.Black.ID == g.bot.account.ID:
		g.ourColor = chess.Black
	default:
		return fmt.Errorf("lichess: game %s does not include this account", g.id)
	}

	g.started = true
	g.bot.log.Info("game started", "game", g.id, "color", g.ourColor,
		"variant", m.Variant.Key, "opponent", g.opponentName(m))

	if msg := g.bot.config.GreetingMessage; msg != "" && !g.greeted {
		g.greeted = true
		if err := g.bot.client.Chat(ctx, g.id, "player", msg); err != nil {
			// A chat failure must never take down a game.
			g.bot.log.Debug("greeting failed", "game", g.id, "error", err)
		}
	}
	return nil
}

func (g *game) opponentName(m GameMessage) string {
	if g.ourColor == chess.White {
		return m.Black.Name
	}
	return m.White.Name
}

// maybeMove replays the game and moves if it is our turn.
//
// Replaying from the initial position on every update, rather than tracking
// state incrementally, is deliberate: the stream is the authority, it can be
// re-delivered after a reconnect, and a position rebuilt from the move list is
// correct by construction. Games are a few hundred plies at most, so the cost
// is nothing next to a single search.
func (g *game) maybeMove(ctx context.Context, state GameState) error {
	if !g.started {
		return nil
	}
	if IsFinished(state.Status) {
		g.bot.log.Info("game finished", "game", g.id, "status", state.Status, "winner", state.Winner)
		return nil
	}

	pos, err := g.positionAfter(state.Moves)
	if err != nil {
		return err
	}
	if pos.SideToMove() != g.ourColor {
		return nil
	}
	if outcome, _ := pos.Outcome(); outcome != chess.Ongoing {
		return nil
	}

	limits := search.Limits{
		WhiteTime: time.Duration(state.WTime) * time.Millisecond,
		BlackTime: time.Duration(state.BTime) * time.Millisecond,
		WhiteInc:  time.Duration(state.WInc) * time.Millisecond,
		BlackInc:  time.Duration(state.BInc) * time.Millisecond,
	}
	// A game with no clock still needs a bound, or the search would run until
	// the context ended.
	if limits.WhiteTime == 0 && limits.BlackTime == 0 {
		limits.MoveTime = 5 * time.Second
	}

	result := g.searcher.Search(ctx, pos, limits, nil)
	if result.Best.IsNone() {
		return nil
	}

	move := pos.MoveToUCI(result.Best)
	if err := g.bot.client.MakeMove(ctx, g.id, move); err != nil {
		if errors.Is(err, ErrRateLimited) {
			g.bot.log.Warn("rate limited posting a move", "game", g.id)
			if !sleepCtx(ctx, RateLimitBackoff) {
				return ctx.Err()
			}
			return g.bot.client.MakeMove(ctx, g.id, move)
		}
		return fmt.Errorf("lichess: play %s in %s: %w", move, g.id, err)
	}

	g.bot.log.Info("moved", "game", g.id, "move", move,
		"score", result.Info.Score, "depth", result.Info.Depth, "nodes", result.Info.Nodes)
	return nil
}

// positionAfter rebuilds the position from the initial FEN and the move list.
func (g *game) positionAfter(moves string) (*chess.Position, error) {
	pos, err := chess.ParseFENVariant(g.initialFEN, g.chess960)
	if err != nil {
		return nil, fmt.Errorf("lichess: game %s has an unusable initial position: %w", g.id, err)
	}

	for _, token := range strings.Fields(moves) {
		m, ok := pos.ParseUCIMove(token)
		if !ok {
			return nil, fmt.Errorf("lichess: game %s sent move %q, illegal in %s", g.id, token, pos.FEN())
		}
		pos.MakeMove(m)
	}
	return pos, nil
}

// sleepCtx waits for d, reporting false if the context ended first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}
