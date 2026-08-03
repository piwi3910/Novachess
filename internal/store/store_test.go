package store

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newTestStore(t *testing.T) *FS {
	t.Helper()
	return openStore(t, t.TempDir())
}

// openStore builds a store and closes it when the test ends. A store holds a
// directory descriptor for as long as it lives — that is what confines it — so
// a test that never closes one leaks a descriptor per store and, on platforms
// where an open handle blocks deletion, leaves the temporary directory behind
// too.
func openStore(t *testing.T, dir string) *FS {
	t.Helper()
	s, err := NewFS(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("closing the store: %v", err)
		}
	})
	return s
}

func TestPutAndGet(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	want := []byte("NOVADATA and then some records")

	artifact, err := s.Put(ctx, "gen1/unit-7.novadata", bytes.NewReader(want))
	if err != nil {
		t.Fatal(err)
	}

	if artifact.Size != int64(len(want)) {
		t.Errorf("size = %d, want %d", artifact.Size, len(want))
	}
	if artifact.Checksum == "" {
		t.Error("no checksum was recorded")
	}
	if !strings.HasPrefix(artifact.URI, Scheme) {
		t.Errorf("URI %q does not start with %q", artifact.URI, Scheme)
	}

	rc, err := s.Get(ctx, artifact.URI)
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()

	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("read back %q, want %q", got, want)
	}

	// The artifact the worker describes must be the one the trainer verifies.
	if err := Verify(ctx, s, artifact); err != nil {
		t.Errorf("the artifact did not verify against its own description: %v", err)
	}
}

// TestKeysCannotEscapeTheStore is the security-relevant test in this package.
//
// Keys are derived from work unit IDs, which arrive over the bus from a
// coordinator this process did not authenticate. A key that resolves outside
// the store would let a message dictate where a worker writes — into a cron
// directory, over an SSH key, anywhere the process can reach. This is the one
// failure here that is not merely a corrupt dataset.
func TestKeysCannotEscapeTheStore(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	// A file outside the store, to prove none of these reach it.
	outside := filepath.Join(filepath.Dir(s.Root()), "outside.txt")
	if err := os.WriteFile(outside, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}

	hostile := []string{
		"../outside.txt",
		"../../outside.txt",
		"gen1/../../outside.txt",
		"./../outside.txt",
		"/etc/passwd",
		"/tmp/absolute",
		"..",
		".",
		"a/./b",
		"a//b",
		`a\..\..\b`,
		"",
		"gen1/",
		"/",
		"nul\x00byte",
		"line\nbreak",
	}

	for _, key := range hostile {
		t.Run(fmt.Sprintf("%q", key), func(t *testing.T) {
			if _, err := s.Put(ctx, key, strings.NewReader("hostile")); err == nil {
				t.Errorf("Put accepted %q", key)
			}
		})
	}

	// Nothing outside the store was touched.
	after, err := os.ReadFile(outside)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != "original" {
		t.Errorf("a file outside the store was modified: %q", after)
	}
}

// TestURIsCannotEscapeTheStore covers the read side. A URI naming a file
// outside the store would turn the trainer into a way to read arbitrary files.
func TestURIsCannotEscapeTheStore(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	outside := filepath.Join(filepath.Dir(s.Root()), "secret.txt")
	if err := os.WriteFile(outside, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}

	hostile := []string{
		Scheme + outside,
		Scheme + s.Root() + "/../secret.txt",
		Scheme + "/etc/passwd",
		"http://example.com/artifact",
		"/no/scheme/at/all",
		s.Root() + "/relative.novadata",
		Scheme + "relative/not/absolute",
		"",
	}

	for _, uri := range hostile {
		t.Run(fmt.Sprintf("%q", uri), func(t *testing.T) {
			if rc, err := s.Get(ctx, uri); err == nil {
				rc.Close()
				t.Errorf("Get accepted %q", uri)
			}
			if _, err := s.Stat(ctx, uri); err == nil {
				t.Errorf("Stat accepted %q", uri)
			}
			if err := s.Delete(ctx, uri); err == nil {
				t.Errorf("Delete accepted %q", uri)
			}
		})
	}

	if _, err := os.Stat(outside); err != nil {
		t.Errorf("a file outside the store was deleted: %v", err)
	}
}

// TestSymlinkedRootIsResolvedOnce checks that a store reached through a symlink
// still confines its keys. The root is resolved at construction, so later
// comparisons are against a real path rather than one that could be re-pointed.
func TestSymlinkedRootIsResolvedOnce(t *testing.T) {
	base := t.TempDir()
	real := filepath.Join(base, "real")
	if err := os.MkdirAll(real, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(base, "link")
	if err := os.Symlink(real, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	s := openStore(t, link)
	if s.Root() == link {
		t.Error("the root was not resolved through the symlink")
	}

	ctx := context.Background()
	artifact, err := s.Put(ctx, "batch.novadata", strings.NewReader("data"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(strings.TrimPrefix(artifact.URI, Scheme), s.Root()) {
		t.Errorf("artifact %q is not under the resolved root %q", artifact.URI, s.Root())
	}
	if _, err := s.Get(ctx, artifact.URI); err != nil {
		t.Errorf("an artifact written through a symlinked root could not be read: %v", err)
	}
}

// failingReader fails part way through, standing in for a worker evicted
// mid-write.
type failingReader struct {
	data  []byte
	after int
	read  int
}

func (r *failingReader) Read(p []byte) (int, error) {
	if r.read >= r.after {
		return 0, errors.New("the worker went away")
	}
	n := copy(p, r.data[r.read:min(r.read+64, len(r.data))])
	r.read += n
	return n, nil
}

// TestFailedWriteLeavesNothing is the atomicity guarantee.
//
// A half-written batch is worse than a missing one: the trainer would read it,
// the final record would be truncated, and the resulting network would be
// quietly bad. So a failed write must leave nothing at the key rather than a
// prefix of the data.
func TestFailedWriteLeavesNothing(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	const key = "gen1/interrupted.novadata"
	r := &failingReader{data: bytes.Repeat([]byte("x"), 4096), after: 512}

	if _, err := s.Put(ctx, key, r); err == nil {
		t.Fatal("an interrupted write reported success")
	}

	if _, err := s.Get(ctx, s.URI(key)); !errors.Is(err, ErrNotFound) {
		t.Errorf("something was left at the key after a failed write: %v", err)
	}

	// And no debris either.
	entries, err := os.ReadDir(filepath.Join(s.Root(), "gen1"))
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		t.Errorf("a failed write left %q behind", e.Name())
	}
}

// TestCancelledWriteLeavesNothing covers the same guarantee for a context
// cancelled while the write is in flight, which is how a pod eviction arrives.
func TestCancelledWriteLeavesNothing(t *testing.T) {
	s := newTestStore(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	const key = "cancelled.novadata"
	if _, err := s.Put(ctx, key, strings.NewReader("data")); err == nil {
		t.Fatal("a cancelled write reported success")
	}

	if _, err := s.Get(context.Background(), s.URI(key)); !errors.Is(err, ErrNotFound) {
		t.Errorf("a cancelled write left something behind: %v", err)
	}
	entries, err := os.ReadDir(s.Root())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		t.Errorf("a cancelled write left %q behind", e.Name())
	}
}

// TestVerifyDetectsCorruption is the read side of the integrity guarantee.
// Corruption between the worker that wrote a batch and the trainer that reads
// it is the failure this pipeline cannot detect downstream.
func TestVerifyDetectsCorruption(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	artifact, err := s.Put(ctx, "batch.novadata", strings.NewReader("the original contents"))
	if err != nil {
		t.Fatal(err)
	}
	if err := Verify(ctx, s, artifact); err != nil {
		t.Fatalf("a freshly written artifact did not verify: %v", err)
	}

	path := strings.TrimPrefix(artifact.URI, Scheme)

	t.Run("contents changed", func(t *testing.T) {
		if err := os.WriteFile(path, []byte("the tampered contents"), 0o644); err != nil {
			t.Fatal(err)
		}
		err := Verify(ctx, s, artifact)
		if err == nil {
			t.Fatal("verification passed on altered contents")
		}
		if !strings.Contains(err.Error(), "corrupt") {
			t.Errorf("error %q does not say the artifact is corrupt", err)
		}
	})

	t.Run("length changed", func(t *testing.T) {
		if err := os.WriteFile(path, []byte("short"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := Verify(ctx, s, artifact); err == nil {
			t.Fatal("verification passed on a truncated artifact")
		}
	})

	t.Run("missing entirely", func(t *testing.T) {
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := Verify(ctx, s, artifact); !errors.Is(err, ErrNotFound) {
			t.Errorf("verifying a missing artifact gave %v, want a not-found error", err)
		}
	})
}

// TestChecksumIsContentAddressed checks that the checksum actually distinguishes
// contents, rather than being a constant that would verify anything.
func TestChecksumIsContentAddressed(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	a, err := s.Put(ctx, "a", strings.NewReader("contents one"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := s.Put(ctx, "b", strings.NewReader("contents two"))
	if err != nil {
		t.Fatal(err)
	}
	same, err := s.Put(ctx, "c", strings.NewReader("contents one"))
	if err != nil {
		t.Fatal(err)
	}

	if a.Checksum == b.Checksum {
		t.Error("different contents produced the same checksum")
	}
	if a.Checksum != same.Checksum {
		t.Error("identical contents produced different checksums")
	}
	if len(a.Checksum) != 64 {
		t.Errorf("checksum %q is %d characters, want 64 for hex SHA-256", a.Checksum, len(a.Checksum))
	}
}

// TestRewriteIsSafe covers a redelivered work unit. Generation is
// deterministic, so replaying a unit produces byte-identical data and
// overwriting is harmless — but the overwrite must still be atomic, and the
// artifact must still be readable throughout.
func TestRewriteIsSafe(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	const key = "gen1/unit-3.novadata"
	data := "identical output from a replayed unit"

	first, err := s.Put(ctx, key, strings.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.Put(ctx, key, strings.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}

	if first.Checksum != second.Checksum || first.URI != second.URI {
		t.Errorf("replaying a unit produced a different artifact: %+v then %+v", first, second)
	}
	if err := Verify(ctx, s, second); err != nil {
		t.Errorf("the rewritten artifact does not verify: %v", err)
	}

	// One file, not two.
	entries, err := os.ReadDir(filepath.Join(s.Root(), "gen1"))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("after two writes the directory holds %v", names)
	}
}

func TestStatDoesNotReadTheArtifact(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	want := strings.Repeat("data", 1000)
	artifact, err := s.Put(ctx, "big.novadata", strings.NewReader(want))
	if err != nil {
		t.Fatal(err)
	}

	info, err := s.Stat(ctx, artifact.URI)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size != int64(len(want)) {
		t.Errorf("Stat reports %d bytes, want %d", info.Size, len(want))
	}
	// Deliberately empty: computing it would mean reading the whole artifact,
	// which is what Stat exists to avoid.
	if info.Checksum != "" {
		t.Errorf("Stat returned a checksum %q; it should not read the contents", info.Checksum)
	}
}

func TestMissingArtifacts(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	uri := s.URI("never/written.novadata")

	if _, err := s.Get(ctx, uri); !errors.Is(err, ErrNotFound) {
		t.Errorf("Get of a missing artifact gave %v, want a not-found error", err)
	}
	if _, err := s.Stat(ctx, uri); !errors.Is(err, ErrNotFound) {
		t.Errorf("Stat of a missing artifact gave %v, want a not-found error", err)
	}
	// Deleting what is not there is how cleanup after a failed generation
	// stays repeatable.
	if err := s.Delete(ctx, uri); err != nil {
		t.Errorf("Delete of a missing artifact failed: %v", err)
	}
}

func TestDeleteRemoves(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	artifact, err := s.Put(ctx, "temporary.novadata", strings.NewReader("data"))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Delete(ctx, artifact.URI); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Get(ctx, artifact.URI); !errors.Is(err, ErrNotFound) {
		t.Errorf("the artifact survived deletion: %v", err)
	}
}

func TestValidateKey(t *testing.T) {
	valid := []string{
		"a",
		"batch.novadata",
		"gen1/unit-7.novadata",
		"deeply/nested/but/fine/x",
		"unit_with-punctuation.2026",
		strings.Repeat("a", 512),
	}
	for _, key := range valid {
		if err := ValidateKey(key); err != nil {
			t.Errorf("ValidateKey(%q) = %v, want it accepted", truncate(key), err)
		}
	}

	invalid := []string{"", "..", ".", "/absolute", "a/../b", "a//b", "a/", `a\b`, strings.Repeat("a", 513), "\x00", "\x1f", "\x7f"}
	for _, key := range invalid {
		if err := ValidateKey(key); err == nil {
			t.Errorf("ValidateKey(%q) accepted it", truncate(key))
		}
	}
}

func truncate(s string) string {
	if len(s) > 30 {
		return s[:30] + "..."
	}
	return s
}

func TestNewFSRejectsAnEmptyDirectory(t *testing.T) {
	if _, err := NewFS(""); err == nil {
		t.Error("NewFS accepted an empty directory")
	}
}

func TestDirectoriesAreNotArtifacts(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	if _, err := s.Put(ctx, "gen1/unit.novadata", strings.NewReader("data")); err != nil {
		t.Fatal(err)
	}

	// "gen1" exists but is a directory, not an artifact.
	if _, err := s.Stat(ctx, s.URI("gen1")); err == nil {
		t.Error("Stat treated a directory as an artifact")
	}
}

// TestSymlinksInsideTheStoreCannotEscape is why every operation goes through an
// os.Root rather than through paths joined onto a prefix.
//
// The store lives on a volume several pods share. A lexical check that the
// joined path sits under the root passes happily for a key like
// "escape/passwd" when "escape" is a symlink to /etc — the string comparison
// happens long before the kernel resolves the link, so the check is looking at
// a path that is not the one the write will reach. Nothing in this package
// creates symlinks, but anything else with access to the volume can.
func TestSymlinksInsideTheStoreCannotEscape(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	// Somewhere outside the store, with a file to protect.
	outside := t.TempDir()
	victim := filepath.Join(outside, "victim.txt")
	if err := os.WriteFile(victim, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A symlink planted inside the store, pointing out of it. Written directly
	// rather than through the store, which offers no way to create one.
	if err := os.Symlink(outside, filepath.Join(s.Root(), "escape")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	t.Run("write through the link", func(t *testing.T) {
		if _, err := s.Put(ctx, "escape/victim.txt", strings.NewReader("overwritten")); err == nil {
			t.Error("Put followed a symlink out of the store")
		}
		if got, err := os.ReadFile(victim); err != nil {
			t.Fatal(err)
		} else if string(got) != "original" {
			t.Errorf("a file outside the store was overwritten: %q", got)
		}
	})

	t.Run("read through the link", func(t *testing.T) {
		if rc, err := s.Get(ctx, s.URI("escape/victim.txt")); err == nil {
			rc.Close()
			t.Error("Get followed a symlink out of the store")
		}
	})

	t.Run("delete through the link", func(t *testing.T) {
		if err := s.Delete(ctx, s.URI("escape/victim.txt")); err == nil {
			t.Error("Delete followed a symlink out of the store")
		}
		if _, err := os.Stat(victim); err != nil {
			t.Errorf("a file outside the store was deleted: %v", err)
		}
	})

	t.Run("stat through the link", func(t *testing.T) {
		if _, err := s.Stat(ctx, s.URI("escape/victim.txt")); err == nil {
			t.Error("Stat followed a symlink out of the store")
		}
	})
}

// TestConcurrentWritesToOneKeyAreSafe covers a redelivered work unit landing on
// two workers at once, which at-least-once delivery makes possible. Both write
// identical bytes, since generation is deterministic, so the requirement is
// that neither observes the other's partial file and the result verifies.
func TestConcurrentWritesToOneKeyAreSafe(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	const key = "gen1/contended.novadata"
	data := strings.Repeat("deterministic output ", 500)

	results := make(chan Artifact, 8)
	errs := make(chan error, 8)
	for i := 0; i < 8; i++ {
		go func() {
			a, err := s.Put(ctx, key, strings.NewReader(data))
			if err != nil {
				errs <- err
				return
			}
			results <- a
		}()
	}

	var first Artifact
	for i := 0; i < 8; i++ {
		select {
		case err := <-errs:
			t.Fatalf("a concurrent write failed: %v", err)
		case a := <-results:
			if first.URI == "" {
				first = a
			} else if a.Checksum != first.Checksum || a.Size != first.Size {
				t.Errorf("concurrent writes of identical data disagreed: %+v vs %+v", a, first)
			}
		}
	}

	if err := Verify(ctx, s, first); err != nil {
		t.Errorf("the contended artifact does not verify: %v", err)
	}

	// Exactly one artifact, with no temporary files left over.
	entries, err := os.ReadDir(filepath.Join(s.Root(), "gen1"))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("eight concurrent writes left %v", names)
	}
}

// cancellingStore hands out a reader that cancels the context once reading has
// begun, so a test can prove Verify stops part way rather than only before it
// starts.
type cancellingStore struct {
	cancel context.CancelFunc
	reads  int
}

func (c *cancellingStore) Get(context.Context, string) (io.ReadCloser, error) {
	return io.NopCloser(&cancellingReader{owner: c}), nil
}

func (c *cancellingStore) Put(context.Context, string, io.Reader) (Artifact, error) {
	return Artifact{}, errors.New("not used")
}
func (c *cancellingStore) Stat(context.Context, string) (Artifact, error) {
	return Artifact{}, errors.New("not used")
}
func (c *cancellingStore) Delete(context.Context, string) error { return errors.New("not used") }

// cancellingReader never ends. If Verify ignored its context this would hang
// rather than fail, which is the honest way to represent "reads a very large
// artifact to the end".
type cancellingReader struct{ owner *cancellingStore }

func (r *cancellingReader) Read(p []byte) (int, error) {
	r.owner.reads++
	if r.owner.reads == 1 {
		// Cancelled once reading is under way, not before it.
		r.owner.cancel()
	}
	for i := range p {
		p[i] = 'x'
	}
	return len(p), nil
}

// TestVerifyStopsWhenCancelled checks that cancellation reaches inside the
// read. An artifact is large by definition — that is why it is not on the bus —
// so a Verify that only checked its context up front would keep reading a
// gigabyte after the process was told to stop.
func TestVerifyStopsWhenCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s := &cancellingStore{cancel: cancel}

	done := make(chan error, 1)
	go func() {
		done <- Verify(ctx, s, Artifact{URI: "file:///anything", Size: 1 << 40, Checksum: "unused"})
	}()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("Verify returned %v, want the cancellation", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Verify did not stop when its context was cancelled")
	}

	// It stopped early rather than reading the whole endless stream.
	if s.reads > 100 {
		t.Errorf("Verify made %d reads after cancellation; it should have stopped at the next one", s.reads)
	}
	t.Logf("stopped after %d reads", s.reads)
}
