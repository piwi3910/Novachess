// Package events defines the messages services exchange over the bus.
//
// These are cross-service contracts. Two rules govern everything here:
//
// Messages carry references, not payloads. A self-play batch is megabytes of
// packed positions; it goes to object storage and the message carries its URI.
// The bus coordinates, the artifact store moves data. Keeping messages small
// enough to log and replay verbatim is worth more during debugging than the
// round trip it saves.
//
// Messages are versioned from the first commit. A pipeline that runs for weeks
// will have producers and consumers at different versions during any rollout,
// and adding a version field afterwards means a flag day.
//
// This package deliberately imports nothing from the engine. A service should
// be able to consume pipeline messages without linking the search.
package events

import "time"

// SchemaVersion is the current contract version. Bump it on any change that
// existing consumers cannot ignore; additive optional fields do not require it.
const SchemaVersion = 1

// Subjects. NATS subjects are dot-separated and hierarchical, so a wildcard
// like "nova.net.*" subscribes to every network lifecycle event.
const (
	// SubjectWorkAssign carries WorkUnit to the self-play worker queue group.
	// Workers subscribe with a queue group so each unit goes to exactly one.
	SubjectWorkAssign = "nova.work.assign"

	// SubjectGamesProduced carries GameBatch back to the coordinator once a
	// worker has durably stored its output.
	SubjectGamesProduced = "nova.games.produced"

	// SubjectDatasetReady carries DatasetReady to the trainer.
	SubjectDatasetReady = "nova.dataset.ready"

	// SubjectNetCandidate carries NetworkCandidate to the gatekeeper.
	SubjectNetCandidate = "nova.net.candidate"

	// SubjectNetPromoted carries NetworkVerdict when a candidate wins its SPRT
	// and becomes the network subsequent self-play uses.
	SubjectNetPromoted = "nova.net.promoted"

	// SubjectNetRejected carries NetworkVerdict when a candidate fails.
	SubjectNetRejected = "nova.net.rejected"

	// SubjectWorkerHeartbeat carries WorkerHeartbeat for liveness and for
	// throughput telemetry.
	SubjectWorkerHeartbeat = "nova.worker.heartbeat"
)

// Envelope wraps every message on the bus. It exists so that consumers can
// route, deduplicate and reject messages without understanding the payload.
type Envelope struct {
	// ID uniquely identifies this message. Delivery is at-least-once, so
	// consumers that perform non-idempotent work deduplicate on it.
	ID string `json:"id"`

	// Type is the subject the message was published on.
	Type string `json:"type"`

	// SchemaVersion is the contract version the producer used.
	SchemaVersion int `json:"schema_version"`

	// Producer identifies the sending service instance, for tracing a message
	// back to the pod that emitted it.
	Producer string `json:"producer"`

	// PublishedAt is the producer's clock. It is for diagnostics only: nothing
	// in the pipeline orders events by wall-clock time, because clocks across
	// eight boards do not agree closely enough to be worth trusting.
	PublishedAt time.Time `json:"published_at"`

	// Payload is the message body, encoded according to Type.
	Payload []byte `json:"payload"`
}

// WorkUnit instructs a self-play worker to generate games.
//
// A unit is self-describing and deterministic: given the same WorkUnit, any
// worker on any node must produce byte-identical games. That is what makes the
// work safely redeliverable when a pod dies, and it is why the search
// parameters here are node counts rather than time limits.
type WorkUnit struct {
	// ID identifies this unit. It is stamped onto the resulting artifact so
	// the coordinator can deduplicate redelivered work.
	ID string `json:"id"`

	// Generation is the training generation this work belongs to.
	Generation int `json:"generation"`

	// NetworkURI locates the evaluation network to play with, or is empty to
	// use the built-in hand-crafted evaluation (which is how generation zero
	// bootstraps, before any network exists).
	NetworkURI string `json:"network_uri,omitempty"`

	// Games is how many games to play.
	Games int `json:"games"`

	// OpeningFENs seeds the games. Self-play from a single starting position
	// produces near-identical games and worthless training data, so openings
	// are varied deliberately.
	OpeningFENs []string `json:"opening_fens,omitempty"`

	// RandomPlies is how many random moves to play from the opening before
	// the engine takes over, as a second source of diversity.
	RandomPlies int `json:"random_plies"`

	// NodesPerMove fixes the search effort per move. This must be a node count
	// and never a time limit: a time-limited search returns different moves on
	// a thermally throttled board than on an idle one, which would make the
	// training data depend on which node happened to run it.
	NodesPerMove int `json:"nodes_per_move"`

	// Seed initializes the worker's RNG. All randomness in the unit derives
	// from it, so the unit is reproducible.
	Seed uint64 `json:"seed"`
}

// GameBatch reports completed self-play games. It is sent only after the
// artifact is durably stored, so that a worker crash replays the unit rather
// than losing games while claiming success.
type GameBatch struct {
	// WorkUnitID is the unit that produced this batch, for deduplication.
	WorkUnitID string `json:"work_unit_id"`

	// WorkerID identifies the pod, for attributing throughput and failures.
	WorkerID string `json:"worker_id"`

	// Generation echoes the work unit's generation.
	Generation int `json:"generation"`

	// ArtifactURI locates the packed positions in object storage. The data
	// itself never travels on the bus.
	ArtifactURI string `json:"artifact_uri"`

	// SizeBytes and Positions describe the artifact without fetching it.
	SizeBytes int64 `json:"size_bytes"`
	Positions int   `json:"positions"`
	Games     int   `json:"games"`

	// Checksum is the SHA-256 of the artifact, hex encoded, so the coordinator
	// can prove it is counting the data the worker actually wrote. Corruption
	// between the two is the failure nothing downstream can detect: a dataset
	// assembled from unverified batches trains a network from whatever the
	// disks happened to return.
	Checksum string `json:"checksum,omitempty"`

	// NodesPerSecond is the worker's measured search throughput. Reporting it
	// per batch is what makes thermal throttling visible in the data rather
	// than a thing to infer from wall-clock times.
	NodesPerSecond float64 `json:"nodes_per_second"`

	// Duration is how long the unit took.
	Duration time.Duration `json:"duration"`

	// EngineVersion is the build that produced the games, so a dataset can be
	// traced to the code that generated it.
	EngineVersion string `json:"engine_version"`
}

// DatasetReady announces an assembled training set.
type DatasetReady struct {
	// Generation is the generation being trained.
	Generation int `json:"generation"`

	// ArtifactURIs are the batches making up the set. Training mixes several
	// recent generations rather than only the newest data, which is more
	// stable, so this typically spans more than one generation's output.
	ArtifactURIs []string `json:"artifact_uris"`

	// Positions is the total across all artifacts.
	Positions int `json:"positions"`

	// BaselineNetworkURI is the network to initialize from, or empty to train
	// from scratch.
	BaselineNetworkURI string `json:"baseline_network_uri,omitempty"`
}

// NetworkCandidate announces a freshly trained network awaiting gating. A
// candidate is never used for play until the gatekeeper promotes it.
type NetworkCandidate struct {
	Generation int `json:"generation"`

	// ArtifactURI locates the candidate weights.
	ArtifactURI string `json:"artifact_uri"`

	// IncumbentURI is the network it must beat, or empty when the incumbent is
	// the hand-crafted evaluation.
	IncumbentURI string `json:"incumbent_uri,omitempty"`

	// TrainingLoss and ValidationLoss are diagnostics. They are deliberately
	// not promotion criteria: a network can improve its loss against the
	// training objective and still play worse, which is exactly what the SPRT
	// gate exists to catch.
	TrainingLoss   float64 `json:"training_loss"`
	ValidationLoss float64 `json:"validation_loss"`

	Positions int           `json:"positions"`
	Duration  time.Duration `json:"duration"`
}

// NetworkVerdict reports the outcome of gating a candidate.
type NetworkVerdict struct {
	Generation  int    `json:"generation"`
	ArtifactURI string `json:"artifact_uri"`

	// Promoted is the decision. When true, subsequent self-play uses this
	// network and the bot picks it up.
	Promoted bool `json:"promoted"`

	// Match results from the SPRT.
	Wins   int `json:"wins"`
	Losses int `json:"losses"`
	Draws  int `json:"draws"`

	// EloDelta is the point estimate of the strength change, and LOS the
	// likelihood of superiority. Both are reported whether or not the
	// candidate was promoted, since a rejected candidate's margin is useful
	// signal for the next round.
	EloDelta float64 `json:"elo_delta"`
	LOS      float64 `json:"los"`

	// Reason explains the decision in human terms, for the dashboards.
	Reason string `json:"reason,omitempty"`
}

// WorkerHeartbeat reports liveness and throughput.
type WorkerHeartbeat struct {
	WorkerID string `json:"worker_id"`
	NodeName string `json:"node_name"`

	// CurrentWorkUnitID is empty when the worker is idle.
	CurrentWorkUnitID string `json:"current_work_unit_id,omitempty"`

	// NodesPerSecond is the current measured throughput. Watching this per
	// node is the practical way to spot a board that is throttling or has lost
	// its fan.
	NodesPerSecond float64 `json:"nodes_per_second"`

	GamesCompleted int           `json:"games_completed"`
	Uptime         time.Duration `json:"uptime"`
	EngineVersion  string        `json:"engine_version"`
}
