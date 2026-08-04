package store

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// Scheme is the URI scheme a filesystem store issues.
const Scheme = "file://"

// FS stores artifacts under a directory.
//
// On the cluster that directory is a shared volume — the boards have NVMe
// drives and a mount they can all see, which is enough for a pipeline of this
// size and avoids running an object store to hold a few hundred gigabytes.
// Swapping in S3 later means another implementation of Store, not a change to
// anything that uses one.
//
// Every operation goes through an os.Root rather than through paths joined onto
// a prefix. The difference matters because the volume is shared: a lexical
// check that a joined path sits under the root is defeated by a symlink planted
// inside the store, since resolving it happens in the kernel long after the
// string comparison said the path was fine. os.Root holds a descriptor for the
// directory and refuses to traverse out of it, symlinks included, so
// confinement is a property of the file operations rather than of the arithmetic
// leading up to them.
type FS struct {
	root *os.Root
	// dir is the resolved absolute path, used only to build and parse URIs.
	dir string
}

// NewFS returns a store rooted at dir, creating it if needed.
func NewFS(dir string) (*FS, error) {
	if dir == "" {
		return nil, fmt.Errorf("store: no directory given")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("store: creating %s: %w", dir, err)
	}

	// Resolved once, at construction, so URIs name a stable location even if
	// the store was reached through a symlink.
	resolved, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return nil, fmt.Errorf("store: resolving %s: %w", dir, err)
	}
	absolute, err := filepath.Abs(resolved)
	if err != nil {
		return nil, fmt.Errorf("store: resolving %s: %w", dir, err)
	}

	root, err := os.OpenRoot(absolute)
	if err != nil {
		return nil, fmt.Errorf("store: opening %s: %w", absolute, err)
	}

	return &FS{root: root, dir: absolute}, nil
}

// Close releases the directory handle.
func (f *FS) Close() error { return f.root.Close() }

// Root returns the directory artifacts are stored under.
func (f *FS) Root() string { return f.dir }

// URI returns where a key would be stored.
func (f *FS) URI(key string) string { return Scheme + filepath.Join(f.dir, key) }

// keyFromURI converts a stored URI back to a key within this store.
func (f *FS) keyFromURI(uri string) (string, error) {
	if !strings.HasPrefix(uri, Scheme) {
		return "", fmt.Errorf("store: %q is not a %s URI", uri, Scheme)
	}

	full := filepath.Clean(strings.TrimPrefix(uri, Scheme))
	if !filepath.IsAbs(full) {
		return "", fmt.Errorf("store: %q is not absolute", uri)
	}

	rel, err := filepath.Rel(f.dir, full)
	if err != nil {
		return "", fmt.Errorf("store: %q is outside the store at %s", uri, f.dir)
	}

	key := filepath.ToSlash(rel)
	if err := ValidateKey(key); err != nil {
		return "", fmt.Errorf("store: %q does not name an artifact in this store: %w", uri, err)
	}
	return key, nil
}

// Put writes an artifact atomically.
//
// The contents go to a temporary name in the same directory and are renamed
// into place once fully written and flushed. A rename within a directory is
// atomic, so a reader sees either the whole artifact or none of it — never the
// half a worker managed to write before it was evicted, which would parse as a
// truncated final record and quietly corrupt a dataset.
//
// The file is flushed before the rename. Without that the rename can be durable
// while the contents are not, and a machine that loses power leaves a correctly
// named file full of zeroes.
func (f *FS) Put(ctx context.Context, key string, r io.Reader) (Artifact, error) {
	var artifact Artifact

	if err := ValidateKey(key); err != nil {
		return artifact, err
	}
	if err := ctx.Err(); err != nil {
		return artifact, err
	}

	if dir := path.Dir(key); dir != "." {
		if err := f.root.MkdirAll(dir, 0o755); err != nil {
			return artifact, fmt.Errorf("store: creating %s: %w", dir, err)
		}
	}

	tmpKey, tmp, err := f.createTemp(key)
	if err != nil {
		return artifact, err
	}

	// Removed on every failure path. A leftover partial file is harmless to
	// readers, which only open names they were given, but it would accumulate
	// across every eviction.
	defer f.root.Remove(tmpKey)

	c := newChecksummer(r)
	if _, err := io.Copy(tmp, c); err != nil {
		tmp.Close()
		return artifact, fmt.Errorf("store: writing %s: %w", key, err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return artifact, fmt.Errorf("store: flushing %s: %w", key, err)
	}
	if err := tmp.Close(); err != nil {
		return artifact, fmt.Errorf("store: closing %s: %w", key, err)
	}

	// Checked after the write rather than before: a cancellation part way
	// through leaves the temporary file, which the deferred remove clears, and
	// nothing at the real name.
	if err := ctx.Err(); err != nil {
		return artifact, err
	}

	if err := f.root.Rename(tmpKey, key); err != nil {
		return artifact, fmt.Errorf("store: publishing %s: %w", key, err)
	}

	return Artifact{
		URI:      f.URI(key),
		Size:     c.size,
		Checksum: c.sum(),
	}, nil
}

// createTemp opens a uniquely named file beside where the artifact will land.
//
// It is written here rather than taken from os.CreateTemp because that works on
// paths and this has to work through the root. O_EXCL makes the uniqueness a
// guarantee rather than a hope: two workers replaying the same unit at once
// would otherwise be able to pick the same name and interleave their writes.
func (f *FS) createTemp(key string) (string, *os.File, error) {
	dir := path.Dir(key)

	for attempt := 0; attempt < 10; attempt++ {
		var suffix [8]byte
		if _, err := rand.Read(suffix[:]); err != nil {
			return "", nil, fmt.Errorf("store: generating a temporary name: %w", err)
		}

		name := ".partial-" + hex.EncodeToString(suffix[:])
		if dir != "." {
			name = dir + "/" + name
		}

		file, err := f.root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if errors.Is(err, fs.ErrExist) {
			continue
		}
		if err != nil {
			return "", nil, fmt.Errorf("store: creating a temporary file for %s: %w", key, err)
		}
		return name, file, nil
	}

	return "", nil, fmt.Errorf("store: could not find an unused temporary name for %s", key)
}

// Get opens an artifact.
func (f *FS) Get(ctx context.Context, uri string) (io.ReadCloser, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	key, err := f.keyFromURI(uri)
	if err != nil {
		return nil, err
	}

	file, err := f.root.Open(key)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("%w: %s", ErrNotFound, uri)
	}
	if err != nil {
		return nil, fmt.Errorf("store: opening %s: %w", uri, err)
	}
	return file, nil
}

// Stat reports on an artifact without reading it.
//
// The checksum is deliberately left empty: computing it would mean reading the
// whole artifact, which is exactly what Stat exists to avoid. A caller that
// needs the checksum verified wants Verify, which is explicit about the cost.
func (f *FS) Stat(ctx context.Context, uri string) (Artifact, error) {
	var artifact Artifact

	if err := ctx.Err(); err != nil {
		return artifact, err
	}

	key, err := f.keyFromURI(uri)
	if err != nil {
		return artifact, err
	}

	info, err := f.root.Stat(key)
	if errors.Is(err, fs.ErrNotExist) {
		return artifact, fmt.Errorf("%w: %s", ErrNotFound, uri)
	}
	if err != nil {
		return artifact, fmt.Errorf("store: inspecting %s: %w", uri, err)
	}
	if info.IsDir() {
		return artifact, fmt.Errorf("store: %s is a directory", uri)
	}

	return Artifact{URI: uri, Size: info.Size()}, nil
}

// List returns the URIs of every artifact directly under a key prefix,
// sorted.
//
// Only files are returned, not subdirectories: a prefix names a generation's
// directory, and nothing in this store nests one artifact directory inside
// another.
func (f *FS) List(ctx context.Context, prefix string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := ValidateKey(prefix); err != nil {
		return nil, err
	}

	dir, err := f.root.Open(prefix)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("store: listing %s: %w", prefix, err)
	}
	defer dir.Close()

	entries, err := dir.ReadDir(-1)
	if err != nil {
		return nil, fmt.Errorf("store: listing %s: %w", prefix, err)
	}

	keys := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		keys = append(keys, path.Join(prefix, e.Name()))
	}
	sort.Strings(keys)

	uris := make([]string, len(keys))
	for i, key := range keys {
		uris[i] = f.URI(key)
	}
	return uris, nil
}

// Delete removes an artifact. A missing one is not an error, so cleaning up
// after a failed generation can be repeated safely.
func (f *FS) Delete(ctx context.Context, uri string) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	key, err := f.keyFromURI(uri)
	if err != nil {
		return err
	}

	if err := f.root.Remove(key); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("store: removing %s: %w", uri, err)
	}
	return nil
}
