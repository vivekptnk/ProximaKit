# How ProximaKit Is Built

ProximaKit is developed by a team of AI coding agents working under adversarial
review, directed by a human maintainer who owns direction, priorities, and every
release decision. Nothing ships on an agent's say-so.

This page documents the engineering methods behind the library. Not as a
novelty — because they are the reason the release notes, the benchmark numbers,
and the format guarantees can be taken at face value. If a claim appears in the
[CHANGELOG](../CHANGELOG.md), the process below is what stands behind it.

## Direction and ownership

A human owns what gets built, in what order, and whether it is good enough to
release. AI agents propose designs, write the code, and review each other's
work; the maintainer sets the goals, resolves the trade-offs, and cuts every
tag. Multi-model orchestration is a means, not the point — the point is that
each release is a decision a person made and stands behind.

## Generator ≠ evaluator

The agent that writes a change never grades it. Every change is reviewed by an
independent evaluator that did not produce it, and a claim is not accepted until
someone other than its author has re-derived it. This shows up most visibly in
the measured decisions below: the GPU-vs-vDSP benchmark behind the Metal NO-GO
was re-run independently before it was accepted
([ADR-009 addendum](adr/ADR-009-metal-backend.md)), and the paged-access
overhead behind the v1.8.0 zero-copy verdict was independently re-measured on a
second random seed before the GO/NO-GO was recorded. Separating the builder from
the grader is the single discipline most of the rest of this page depends on.

## Every regression test proves itself (red-green)

A test written for a bug must **fail against the unfixed code and pass after the
fix** — the bug is reproduced before it is patched and re-verified after. A
green test that never went red proves nothing, so we don't count it.

- Format changes codify this as policy:
  [ADR-010](adr/ADR-010-format-evolution.md) rule 5 requires corruption tests
  plus an N-1 backward-compatibility test for **every** on-disk format bump.
- The critical tombstone-liveness fix in v1.5.0 was reproduced 20/20 times
  before patching and locked in by `TombstoneLivenessTests`.
- The v1.6.0 PQHW loader guard ships with a corruption test proven to load
  (wrongly) before the fix and throw after it.
- The v1.8.0 empty-originals fix ships a byte-patch test
  (`testEmptyOriginalsEntryWithNonzeroOffsetThrows`) that pins the exact error
  a crafted trailer must raise.

## Measure before optimize

ProximaKit does not ship speculative performance code. When a plausible
optimization is proposed, it is instrumented and measured against a
pre-declared threshold, and the measurement is allowed to say no. Three recent
decisions were resolved this way — and shipped **zero** new optimization code
between them:

- **GPU distance in the HNSW insert loop — NO-GO.** vDSP (AMX) beat Metal at
  every measured N from 32 to 1,000,000, roughly 215× faster at the real build
  shape (N=32). Build stays on vDSP.
  ([ADR-009 addendum](adr/ADR-009-metal-backend.md).)
- **Neural Engine embedding — NO-GO.** Measured 0.9986× against a pre-declared
  ≥1.5× bar needed to justify a public `computeUnits` knob. CoreML's defaults
  stay. (v1.8.0.)
- **Zero-copy paged search — scoped GO on the design, implementation deferred.**
  The overhead was measured but the code is gated on consumer-hardware
  re-measurement before it is written. (v1.8.0.)

Where a win is real, it is measured the same way: paged PQ originals cost a
measured 8.0 MB versus 43.1 MB resident at 100,000 × 384d
([ADR-014](adr/ADR-014-paged-originals.md), v1.7.0). No number in the changelog
is estimated; each traces to a committed benchmark or test.

## Bugs caught before release

A pre-release audit is part of every release cycle, and it is expected to find
things. In the v1.8.0 cycle it found a HIGH-severity corruption bug: a retaining
`QuantizedHNSWIndex` with every node removed saved a v3 trailer shape the
library's own reader rejected, so save-then-reload broke permanently at the
empty edge. It was found and fixed before the release was cut — v1.8.0 ships the
fix, not the bug — and a byte-patch regression test now guards the boundary.

## Built for real consumers

Format and API decisions are pressure-tested against a real downstream project,
not hypothetical users. **tinybrain** — an on-device agent-memory consumer, the
consumer [ADR-015](adr/ADR-015-agent-memory-integration.md) was designed
around — persists its RAG index on ProximaKit's raw journaled surface (in-index
chunk metadata, fingerprint-keyed cache acceptance) on its main branch. Friction
surfaced by that integration drives documentation and API rounds; the resulting
guidance is captured, CI-verified, in
[the RAG wrapper recipe](RAG-WRAPPER-RECIPE.md).

## CI runs the same commands you would

Every gate is a plain, reproducible command — no bespoke tooling stands between
a green check and a build you can run yourself. On every push and pull request
([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)):

- `swift build -Xswiftc -warnings-as-errors` (Debug) and `swift build -c release`
- `swift test --skip RecallBenchmarkTests` (the long recall sweeps are opt-in
  locally via `PROXIMA_RECALL_BENCH=1`)
- the demo app and the DocC documentation both build
- a version / changelog consistency check
- SwiftLint `--strict` (pinned) plus an import-policy guard
  (`scripts/check-imports.sh`) that enforces the module dependency allowlist
- a separate iOS-SDK compile, since a macOS `swift build` only exercises the
  macOS SDK

Releases ([`.github/workflows/release.yml`](../.github/workflows/release.yml))
verify that the `ProximaKit.version` constant matches the tag and publish notes
extracted verbatim from [`CHANGELOG.md`](../CHANGELOG.md) — so the public release
notes and this repository's changelog are, by construction, the same text.

## By the numbers

Verified against the repository at the time of writing:

- **8 tagged releases** (v1.0.0 → v1.8.0)
- **621 test functions** across **67 XCTestCase suites** (62 test files)
- **16 Architecture Decision Records** (ADR-001 → ADR-016) recording the
  designs, the trade-offs, and the decisions that were deferred or declined
