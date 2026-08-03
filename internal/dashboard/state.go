// Package dashboard collects the training loop's state and serves it to a
// browser, with guarded controls for the steps an operator does by hand.
package dashboard

import (
	"sort"
	"sync"
	"time"

	"github.com/piwi3910/novachess/internal/events"
)

// StaleAfter is how long a board can be silent before it is flagged. Workers
// heartbeat every 15 seconds; three misses means the pod or the board is gone.
const StaleAfter = 45 * time.Second

// EvictAfter is how long a silent board stays visible before it is dropped
// entirely. The gap between StaleAfter and EvictAfter is the alarm window: a
// board that stopped heartbeating shows red long enough to be noticed, but a
// pod replaced by a rollout or a scale-down does not haunt the page forever —
// heartbeats are keyed by pod name, and every rollout mints new ones.
const EvictAfter = 10 * time.Minute

type Board struct {
	WorkerID       string    `json:"worker_id"`
	NodeName       string    `json:"node_name"`
	CurrentUnit    string    `json:"current_unit,omitempty"`
	EngineVersion  string    `json:"engine_version"`
	NodesPerSecond float64   `json:"nodes_per_second"`
	GamesCompleted int       `json:"games_completed"`
	LastSeen       time.Time `json:"last_seen"`
	Stale          bool      `json:"stale"`
}

type GenProgress struct {
	Generation int       `json:"generation"`
	Positions  int       `json:"positions"`
	Batches    int       `json:"batches"`
	Target     int       `json:"target,omitempty"`
	UpdatedAt  time.Time `json:"updated_at"`
}

type Snapshot struct {
	Boards     []Board     `json:"boards"`
	Generation GenProgress `json:"generation"`
}

type State struct {
	mu       sync.Mutex
	boards   map[string]Board
	progress GenProgress
	target   int
	watchers map[chan struct{}]struct{}
}

func NewState() *State {
	return &State{
		boards:   make(map[string]Board),
		watchers: make(map[chan struct{}]struct{}),
	}
}

// SetTarget records the positions target for the current generation (from
// NOVA_POSITIONS), independent of the batch-driven progress counters. It is
// set once at startup rather than folded into GenProgress, because progress
// is rebuilt from scratch on every generation rollover (see WatchEvents) and
// would otherwise drop the target the moment the first batch of a new
// generation arrived.
func (s *State) SetTarget(target int) {
	s.mu.Lock()
	s.target = target
	s.mu.Unlock()
	s.notify()
}

func (s *State) NoteHeartbeat(hb events.WorkerHeartbeat, at time.Time) {
	s.mu.Lock()
	s.boards[hb.WorkerID] = Board{
		WorkerID:       hb.WorkerID,
		NodeName:       hb.NodeName,
		CurrentUnit:    hb.CurrentWorkUnitID,
		EngineVersion:  hb.EngineVersion,
		NodesPerSecond: hb.NodesPerSecond,
		GamesCompleted: hb.GamesCompleted,
		LastSeen:       at,
	}
	s.mu.Unlock()
	s.notify()
}

func (s *State) NoteProgress(p GenProgress) {
	s.mu.Lock()
	s.progress = p
	s.mu.Unlock()
	s.notify()
}

func (s *State) Snapshot(now time.Time) Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	gen := s.progress
	gen.Target = s.target
	out := Snapshot{Generation: gen, Boards: []Board{}}
	for id, b := range s.boards {
		if now.Sub(b.LastSeen) > EvictAfter {
			delete(s.boards, id)
			continue
		}
		b.Stale = now.Sub(b.LastSeen) > StaleAfter
		out.Boards = append(out.Boards, b)
	}
	sort.Slice(out.Boards, func(i, j int) bool { return out.Boards[i].WorkerID < out.Boards[j].WorkerID })
	return out
}

// Touch notifies subscribers without changing any state. Control actions use
// it so a scale change reaches open SSE streams immediately — with workers
// paused there are no heartbeats, and without this nudge nothing else would
// ever push a fresh frame.
func (s *State) Touch() {
	s.notify()
}

// Subscribe returns a channel that receives a tick whenever state changes,
// and a cancel that must be called when the subscriber goes away.
func (s *State) Subscribe() (<-chan struct{}, func()) {
	ch := make(chan struct{}, 1)
	s.mu.Lock()
	s.watchers[ch] = struct{}{}
	s.mu.Unlock()
	return ch, func() {
		s.mu.Lock()
		delete(s.watchers, ch)
		s.mu.Unlock()
	}
}

func (s *State) notify() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for ch := range s.watchers {
		select {
		case ch <- struct{}{}:
		default: // a slow subscriber keeps its pending tick; never block here
		}
	}
}
