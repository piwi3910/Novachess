# Novachess pipeline images.
#
# One image holds every binary. They are a few megabytes each and share the
# whole engine, so building six images would multiply the layers, the pushes and
# the version skew for nothing. The manifest picks a binary per Deployment.
#
# Build for the cluster:
#   docker buildx build --platform linux/arm64 -t novachess:dev --load .

# The SPA builds first so the Go stage can embed it. package.json is copied
# alone first for layer caching, same reasoning as go.mod below.
FROM --platform=$BUILDPLATFORM node:22 AS web
WORKDIR /web
COPY web/dashboard/package.json web/dashboard/package-lock.json ./
RUN npm ci
COPY web/dashboard/ ./
RUN npm run build

# The build stage runs on the builder's native platform and cross-compiles via
# TARGETARCH; without this an amd64 CI runner would emulate the arm64 compiler
# under QEMU.
FROM --platform=$BUILDPLATFORM golang:1.26 AS build

WORKDIR /src

# Dependencies first, so a source change does not re-download the module cache.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

COPY --from=web /web/dist/ web/dashboard/dist/

ARG VERSION=dev
ARG TARGETARCH

# CGO off is what makes the result a static binary that runs on a scratch image.
# It is also why this project avoids cgo dependencies: adding one would mean
# either a libc in the runtime image or a dynamically linked binary that has to
# match the host.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" \
    -o /out/ ./cmd/...

# Fail the build rather than the deployment if a binary is missing.
RUN test -x /out/selfplay-worker && test -x /out/coordinator && \
    test -x /out/trainer && test -x /out/gatekeeper && \
    test -x /out/novachess && test -x /out/novabot && test -x /out/dashboard

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/ /usr/local/bin/

# Nonroot, and the data volume is mounted writable for this user. The binaries
# need no shell, no package manager and no writable root filesystem, so nothing
# is given one.
USER nonroot:nonroot

# Overridden per Deployment.
ENTRYPOINT ["/usr/local/bin/selfplay-worker"]
