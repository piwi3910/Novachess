# Working agreements for Novachess

## Ship each chunk of work end to end, without being asked

When a piece of work is finished, carry it all the way through. Do not stop at
"pushed" and wait to be told to open a pull request, and do not ask whether to
merge. The full sequence is:

1. **Commit and push** to the working branch.
2. **Open a pull request** against `main`, as a draft.
3. **Wait for CI** and confirm every check is green. Check the job logs when a
   result is surprising rather than trusting the tick — a suite that silently
   matched no tests also reports success.
4. **Address every review comment**, including automated ones. Fix what is
   genuinely wrong, and reply explaining why if something should not change.
   Resolve the threads.
5. **Mark ready and merge**, with a merge commit rather than a squash, so each
   commit's reasoning survives.
6. **Clean up**: restart the working branch from the merged `main` so follow-up
   work does not stack on merged history, confirm the tree is clean, and empty
   the scratchpad.

Report what happened. Do not narrate each step as a question.

## Before pushing anything

- `gofmt -l .` must be silent, `go vet ./...` clean.
- `go test -race ./...` must pass.
- `GOOS=linux GOARCH=arm64 go build ./...` must succeed — the self-play
  workers run on arm64 and a break there is only found at deploy time
  otherwise.

## Measurement discipline

This container's timings vary by more than 4x run to run, and so do the CI
runners. Wall-clock comparisons across runs mean nothing here.

- Prove a change is behaviour-neutral with **node counts**, not milliseconds.
- Compare performance only **back to back on the same host**, with a cold
  transposition table on both sides.
- State plainly when a number is indicative rather than measured.

## Correctness properties that must not regress

- **Fixed-depth and fixed-node search is deterministic at one thread.** The
  distributed pipeline replays work units on different cluster nodes; training
  data must not depend on which machine ran them. Lazy SMP is non-deterministic
  by construction, so self-play scales by running more worker pods, never more
  threads per worker.
- **The principal variation is legal move by move.** This is what catches
  corruption in the PV table, the transposition table or make/unmake, all of
  which otherwise show up only as mysteriously bad play.
- **Perft counts are exact.** Any change touching move generation re-runs the
  full-depth suite, classical and Chess960.

## Testing style

Assert invariants, not node counts or exact scores — those change whenever a
heuristic is touched, and a test that breaks on every tuning change gets
deleted rather than fixed.

When a test fails, check the fixture before the code. A large share of failures
in this repo have been bad test positions: a FEN with the wrong side already in
check, a "pinned" piece that was not pinned, a square that did not attack what
it was supposed to.
