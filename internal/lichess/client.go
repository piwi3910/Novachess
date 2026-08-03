package lichess

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// DefaultBaseURL is the public Lichess API.
const DefaultBaseURL = "https://lichess.org"

// Client talks to the Lichess Bot API.
//
// Two HTTP clients are kept, and the distinction matters: requests have a
// timeout, streams must not. A stream is expected to stay open for the length
// of a game and produce nothing at all while the opponent thinks, so a client
// timeout would tear it down mid-game.
type Client struct {
	baseURL string
	token   string

	requests *http.Client
	streams  *http.Client
}

// NewClient returns a client authenticating with the given token. An empty
// baseURL selects the public API.
func NewClient(token, baseURL string) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Client{
		baseURL:  strings.TrimSuffix(baseURL, "/"),
		token:    token,
		requests: &http.Client{Timeout: 30 * time.Second},
		streams:  &http.Client{}, // no timeout; cancellation is by context
	}
}

// ErrRateLimited is returned when Lichess asks the client to slow down.
//
// It is worth a distinct error because the correct response is specific: back
// off for a good while rather than retry promptly. Lichess bans bots that
// hammer a rate limit, so treating 429 as an ordinary failure risks losing the
// account rather than losing a request.
var ErrRateLimited = errors.New("lichess: rate limited")

// RateLimitBackoff is how long to wait after a 429. The API documentation asks
// for a full minute.
const RateLimitBackoff = time.Minute

// Account returns the authenticated account.
func (c *Client) Account(ctx context.Context) (*Account, error) {
	body, err := c.do(ctx, http.MethodGet, "/api/account", nil)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	var a Account
	if err := json.NewDecoder(body).Decode(&a); err != nil {
		return nil, fmt.Errorf("lichess: decode account: %w", err)
	}
	return &a, nil
}

// Upgrade converts the account to a bot account.
//
// This is irreversible and only succeeds on an account that has never played a
// rated game, which is why it is a separate call rather than something the bot
// does automatically at startup.
func (c *Client) Upgrade(ctx context.Context) error {
	body, err := c.do(ctx, http.MethodPost, "/api/bot/account/upgrade", nil)
	if err != nil {
		return err
	}
	body.Close()
	return nil
}

// AcceptChallenge accepts a challenge by id.
func (c *Client) AcceptChallenge(ctx context.Context, id string) error {
	body, err := c.do(ctx, http.MethodPost, "/api/challenge/"+url.PathEscape(id)+"/accept", nil)
	if err != nil {
		return err
	}
	body.Close()
	return nil
}

// DeclineChallenge declines a challenge, optionally giving one of Lichess's
// standard reasons so the challenger sees something useful.
func (c *Client) DeclineChallenge(ctx context.Context, id, reason string) error {
	var form io.Reader
	if reason != "" {
		form = strings.NewReader(url.Values{"reason": {reason}}.Encode())
	}
	body, err := c.do(ctx, http.MethodPost, "/api/challenge/"+url.PathEscape(id)+"/decline", form)
	if err != nil {
		return err
	}
	body.Close()
	return nil
}

// Decline reasons Lichess recognizes.
const (
	DeclineGeneric     = "generic"
	DeclineLater       = "later"
	DeclineTooFast     = "tooFast"
	DeclineTooSlow     = "tooSlow"
	DeclineTimeControl = "timeControl"
	DeclineRated       = "rated"
	DeclineCasual      = "casual"
	DeclineStandard    = "standard"
	DeclineVariant     = "variant"
	DeclineNoBot       = "noBot"
	DeclineOnlyBot     = "onlyBot"
)

// MakeMove plays a move in a game. The move is in UCI notation, which for
// castling means whatever form the game's variant expects.
func (c *Client) MakeMove(ctx context.Context, gameID, move string) error {
	path := "/api/bot/game/" + url.PathEscape(gameID) + "/move/" + url.PathEscape(move)
	body, err := c.do(ctx, http.MethodPost, path, nil)
	if err != nil {
		return err
	}
	body.Close()
	return nil
}

// Chat posts a message to a game's chat.
func (c *Client) Chat(ctx context.Context, gameID, room, text string) error {
	form := url.Values{"room": {room}, "text": {text}}.Encode()
	path := "/api/bot/game/" + url.PathEscape(gameID) + "/chat"
	body, err := c.do(ctx, http.MethodPost, path, strings.NewReader(form))
	if err != nil {
		return err
	}
	body.Close()
	return nil
}

// Resign resigns a game.
func (c *Client) Resign(ctx context.Context, gameID string) error {
	body, err := c.do(ctx, http.MethodPost, "/api/bot/game/"+url.PathEscape(gameID)+"/resign", nil)
	if err != nil {
		return err
	}
	body.Close()
	return nil
}

// StreamEvents delivers account events until the context ends or the stream
// closes. Returning nil means the server closed the stream cleanly, which
// happens routinely and means "reconnect", not "stop".
func (c *Client) StreamEvents(ctx context.Context, fn func(Event) error) error {
	return c.stream(ctx, "/api/stream/event", func(line []byte) error {
		var e Event
		if err := json.Unmarshal(line, &e); err != nil {
			// A malformed line is not worth ending the stream over; the next
			// one is very likely fine.
			return nil
		}
		return fn(e)
	})
}

// StreamGame delivers a game's messages until the context ends or the stream
// closes.
func (c *Client) StreamGame(ctx context.Context, gameID string, fn func(GameMessage) error) error {
	path := "/api/bot/game/stream/" + url.PathEscape(gameID)
	return c.stream(ctx, path, func(line []byte) error {
		var m GameMessage
		if err := json.Unmarshal(line, &m); err != nil {
			return nil
		}
		return fn(m)
	})
}

// stream reads an NDJSON response line by line.
//
// Lichess sends a blank line every few seconds as a keep-alive, which is how
// the connection is kept from being reaped by intermediaries. Those lines carry
// no data and must be skipped rather than parsed.
func (c *Client) stream(ctx context.Context, path string, fn func([]byte) error) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return err
	}
	c.authorize(req)

	resp, err := c.streams.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := statusError(resp); err != nil {
		return err
	}

	scanner := bufio.NewScanner(resp.Body)
	// Game streams can carry a long move list in a single line.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue // keep-alive
		}
		if err := fn(line); err != nil {
			return err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}

	if err := scanner.Err(); err != nil {
		// A cancelled context surfaces here as a read error; report the
		// cancellation instead, which callers distinguish from a dropped
		// stream that should be retried.
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	return nil
}

func (c *Client) do(ctx context.Context, method, path string, body io.Reader) (io.ReadCloser, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	c.authorize(req)
	if body != nil {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}

	resp, err := c.requests.Do(req)
	if err != nil {
		return nil, err
	}
	if err := statusError(resp); err != nil {
		resp.Body.Close()
		return nil, err
	}
	return resp.Body, nil
}

func (c *Client) authorize(req *http.Request) {
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	req.Header.Set("Accept", "application/x-ndjson")
}

// statusError converts a non-2xx response into an error, singling out rate
// limiting so callers can back off rather than retry.
func statusError(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		return ErrRateLimited
	}

	// Lichess explains failures in the body, and those explanations are the
	// difference between "your token lacks bot:play" and an opaque 401.
	snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	msg := strings.TrimSpace(string(snippet))
	if msg == "" {
		msg = http.StatusText(resp.StatusCode)
	}
	return fmt.Errorf("lichess: %s: %s", strconv.Itoa(resp.StatusCode), msg)
}
