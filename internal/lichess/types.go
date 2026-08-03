// Package lichess implements a client for the Lichess Bot API and a bot that
// plays games with the engine.
//
// The API is HTTP with two long-lived NDJSON streams: one carrying account
// events (challenges arriving, games starting) and one per game carrying its
// state. Both stay open for hours and both will be dropped by intermediaries
// without warning, so reconnection is part of normal operation rather than
// error handling.
//
// A bot account is required: the token needs the bot:play scope, and the
// account must be upgraded via /api/bot/account/upgrade, which only works on an
// account that has never played a rated game and cannot be undone.
package lichess

// Event is one message from the account event stream.
type Event struct {
	Type      string     `json:"type"`
	Challenge *Challenge `json:"challenge,omitempty"`
	Game      *GameEvent `json:"game,omitempty"`
}

// Event types on the account stream.
const (
	EventChallenge         = "challenge"
	EventChallengeCanceled = "challengeCanceled"
	EventChallengeDeclined = "challengeDeclined"
	EventGameStart         = "gameStart"
	EventGameFinish        = "gameFinish"
)

// Challenge is an invitation to play.
type Challenge struct {
	ID          string      `json:"id"`
	Rated       bool        `json:"rated"`
	Variant     Variant     `json:"variant"`
	TimeControl TimeControl `json:"timeControl"`
	Speed       string      `json:"speed"`
	Challenger  Player      `json:"challenger"`
	DestUser    Player      `json:"destUser"`
	Color       string      `json:"color"`
}

// GameEvent identifies a game on the account stream.
type GameEvent struct {
	ID     string `json:"id"`
	Source string `json:"source"`
}

// Variant names the rules in force. Only "standard" and "chess960" are
// playable here; the engine implements no other variant's rules, and accepting
// one would mean playing illegal moves rather than playing badly.
type Variant struct {
	Key  string `json:"key"`
	Name string `json:"name"`
}

// Variant keys this bot understands.
const (
	VariantStandard   = "standard"
	VariantChess960   = "chess960"
	VariantFromPos    = "fromPosition"
	VariantCrazyhouse = "crazyhouse"
)

// TimeControl describes the clock. Type is "clock", "correspondence" or
// "unlimited"; only "clock" carries Limit and Increment.
type TimeControl struct {
	Type      string `json:"type"`
	Limit     int    `json:"limit"`     // initial seconds
	Increment int    `json:"increment"` // seconds added per move
	Show      string `json:"show"`
}

// Player identifies an account.
type Player struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Title  string `json:"title"`
	Rating int    `json:"rating"`
}

// GameMessage is one message from a game stream. The first is always a
// gameFull; the rest are gameState, chatLine or opponentGone.
type GameMessage struct {
	Type string `json:"type"`

	// gameFull fields.
	ID         string    `json:"id"`
	Variant    Variant   `json:"variant"`
	InitialFEN string    `json:"initialFen"`
	White      Player    `json:"white"`
	Black      Player    `json:"black"`
	State      GameState `json:"state"`

	// gameState fields, present when Type is gameState.
	Moves  string `json:"moves"`
	WTime  int64  `json:"wtime"`
	BTime  int64  `json:"btime"`
	WInc   int64  `json:"winc"`
	BInc   int64  `json:"binc"`
	Status string `json:"status"`
	Winner string `json:"winner"`

	// chatLine fields.
	Username string `json:"username"`
	Text     string `json:"text"`
	Room     string `json:"room"`
}

// Game stream message types.
const (
	MessageGameFull     = "gameFull"
	MessageGameState    = "gameState"
	MessageChatLine     = "chatLine"
	MessageOpponentGone = "opponentGone"
)

// GameState is the mutable part of a game: the moves played so far and the
// clocks.
type GameState struct {
	Type   string `json:"type"`
	Moves  string `json:"moves"`
	WTime  int64  `json:"wtime"`
	BTime  int64  `json:"btime"`
	WInc   int64  `json:"winc"`
	BInc   int64  `json:"binc"`
	Status string `json:"status"`
	Winner string `json:"winner"`
}

// Game statuses. Only "started" and "created" mean play continues.
const (
	StatusCreated = "created"
	StatusStarted = "started"
)

// IsFinished reports whether a status means the game is over. Anything that is
// not explicitly still running is treated as finished, so an unrecognized
// status ends the game loop rather than leaving it spinning against a game
// nobody is playing.
func IsFinished(status string) bool {
	return status != StatusStarted && status != StatusCreated && status != ""
}

// Account is the authenticated user.
type Account struct {
	ID       string `json:"id"`
	Username string `json:"username"`
	Title    string `json:"title"`
}
