// Package store holds the pipeline's artifacts: batches of self-play positions
// and trained networks.
//
// It exists because messages carry references rather than payloads. A batch of
// positions is megabytes; putting it on the bus would make every message
// unloggable and every replay expensive, so the data goes here and the message
// carries a URI.
//
// Two properties are non-negotiable and shape the whole interface.
//
// Writes are atomic. A worker evicted mid-write must not leave a half-written
// artifact where a complete one is expected — the trainer would read it, the
// record boundaries would be wrong, and the resulting network would be quietly
// bad with nothing to point at. Everything is written to a temporary name and
// renamed into place, so an artifact either exists in full or does not exist.
//
// And every artifact carries a checksum. Corruption between the worker that
// wrote a batch and the trainer that reads it is exactly the failure this
// pipeline cannot detect downstream, so the checksum travels in the message and
// the reader can prove it got what was written.
package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"strings"
)

// ErrNotFound is returned when an artifact does not exist.
var ErrNotFound = errors.New("store: artifact not found")

// Artifact describes something stored.
type Artifact struct {
	// URI locates the artifact. It is what travels in pipeline messages.
	URI string

	// Size is the length in bytes.
	Size int64

	// Checksum is the SHA-256 of the contents, hex encoded.
	Checksum string
}

// Store reads and writes artifacts.
type Store interface {
	// Put stores the contents under a key and returns where it went.
	//
	// The write is atomic: a Put that fails or is interrupted leaves no
	// artifact at the key rather than a partial one.
	Put(ctx context.Context, key string, r io.Reader) (Artifact, error)

	// Get opens an artifact by URI.
	Get(ctx context.Context, uri string) (io.ReadCloser, error)

	// Stat reports on an artifact without reading it, which is how a
	// coordinator checks that a worker's claimed output is really there before
	// counting it towards a dataset.
	Stat(ctx context.Context, uri string) (Artifact, error)

	// Delete removes an artifact. Missing artifacts are not an error, so that
	// cleaning up after a failed generation is idempotent.
	Delete(ctx context.Context, uri string) error
}

// ValidateKey checks that a key is safe to use as a storage location.
//
// Keys are derived from work unit IDs, which arrive over the bus. A filesystem
// store turns a key into a path, so a key of "../../etc/cron.d/x" would write
// outside the store entirely — and the pipeline's whole shape is that a service
// accepts instructions from a message it did not author. Validating here rather
// than in the filesystem backend means every backend gets the same rule and a
// new one cannot forget it.
func ValidateKey(key string) error {
	switch {
	case key == "":
		return fmt.Errorf("store: empty key")
	case len(key) > 512:
		return fmt.Errorf("store: key is %d bytes, the limit is 512", len(key))
	case strings.HasPrefix(key, "/"):
		return fmt.Errorf("store: key %q is absolute", key)
	case strings.Contains(key, `\`):
		return fmt.Errorf("store: key %q contains a backslash", key)
	}

	for _, element := range strings.Split(key, "/") {
		switch element {
		case "":
			return fmt.Errorf("store: key %q has an empty path element", key)
		case ".", "..":
			return fmt.Errorf("store: key %q navigates the directory tree", key)
		}
	}

	// Control characters have no legitimate use in a key and several have
	// surprising effects on the tools that will eventually list these files.
	for _, r := range key {
		if r < 0x20 || r == 0x7f {
			return fmt.Errorf("store: key %q contains a control character", key)
		}
	}

	return nil
}

// readerWithContext stops a read when its context is done.
//
// io.Copy has no notion of cancellation, so a long read is otherwise
// uninterruptible however carefully the context was threaded to it. Checking
// once per Read costs an atomic load against buffers of tens of kilobytes,
// which is nothing next to the disk.
type readerWithContext struct {
	ctx context.Context
	r   io.Reader
}

func (c readerWithContext) Read(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.Read(p)
}

// checksummer wraps a reader, accumulating the hash and length as it is read.
type checksummer struct {
	r    io.Reader
	h    hash.Hash
	size int64
}

func newChecksummer(r io.Reader) *checksummer {
	return &checksummer{r: r, h: sha256.New()}
}

func (c *checksummer) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	if n > 0 {
		c.h.Write(p[:n])
		c.size += int64(n)
	}
	return n, err
}

func (c *checksummer) sum() string { return hex.EncodeToString(c.h.Sum(nil)) }

// Verify reads an artifact and checks it against an expected checksum.
//
// This is the read side of the integrity guarantee. A trainer that assembled a
// dataset from batches it never verified would produce a network from whatever
// the disks happened to return, and no amount of care in the training code
// would notice.
// Cancellation is honoured throughout the read, not merely before it. An
// artifact is the whole point of this package being separate from the bus —
// it is large — so a Verify that only checked its context up front would run
// to completion on a gigabyte of data after the process had been told to stop,
// which is the difference between a pod shutting down and a pod being killed.
func Verify(ctx context.Context, s Store, want Artifact) error {
	rc, err := s.Get(ctx, want.URI)
	if err != nil {
		return err
	}
	defer rc.Close()

	c := newChecksummer(readerWithContext{ctx: ctx, r: rc})
	if _, err := io.Copy(io.Discard, c); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return fmt.Errorf("store: reading %s: %w", want.URI, err)
	}

	if got := c.sum(); got != want.Checksum {
		return fmt.Errorf("store: %s is corrupt: checksum %s, expected %s", want.URI, got, want.Checksum)
	}
	if c.size != want.Size {
		return fmt.Errorf("store: %s is %d bytes, expected %d", want.URI, c.size, want.Size)
	}
	return nil
}
