// Package webdist embeds the built dashboard SPA. The committed
// dist/index.html is a placeholder so `go build ./...` works without Node;
// the Docker build overwrites it with the real Vite output.
package webdist

import "embed"

//go:embed all:dist
var Dist embed.FS
