# Changelog

All notable changes to ProximaKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [1.9.0] — 2026-07-15

### Added
- **`HNSWGraphSnapshot` + `HNSWIndex.liveGraphSnapshot()`: read-only
  live-graph inspection.** A read-only, value-typed view of the live HNSW
  graph — live nodes (UUID, level, live-filtered layer-0 neighbor UUIDs,
  metadata), `liveCount`, `maxLevel`, and per-layer live-node counts
  (`nodesPerLayer`). Unlike `persistenceSnapshot()`, it never compacts and
  never materializes vectors, reading the layer arrays directly and applying
  the same liveness/tombstone filter search and compaction use — keeping
  inspection side-effect-free. Runtime is linear in stored slots (including
  tombstones), represented-layer accounting, and the inspected live layer-0
  adjacency; it is not described as O(live graph).
- **Public persistence file-extension constants.**
  `ProximaKit.FileExtension.index`, `.writeAheadLog`, and `.sparseIndex` expose
  the canonical `pxkt`, `pxwal`, and `pxbm` suffixes so consumer wrappers do
  not duplicate persistence filenames as string literals.
- **Canonical index layout names, with source-compatible deprecations
  ([ADR-015](docs/adr/ADR-015-agent-memory-integration.md), Stage C).**
  `IndexSaveLayout` (`.resident` / `.pagedV3`) is now the canonical quantized
  save-layout type alongside `IndexResidency` (`.resident` / `.paged`).
  `HNSWOpenMode`, `PQHWOpenMode`, and `PQHWSaveLayout` remain deprecated
  typealiases, so existing source keeps compiling and persisted files require
  no migration.

### Changed
- **Demo Index Inspector now reads through `liveGraphSnapshot()`
  (`Examples/ProximaDemoApp/Sources/SearchEngine.swift`).** Closes the
  v1.8.0 release-time followup: the Inspector previously reused the public
  `persistenceSnapshot()` save/compaction path (safe at the time
  because the demo Search index is append-only); it now calls the additive,
  non-mutating accessor added above.
- **PQ benchmark-class tests are explicitly acceptance/release validation, not
  normal PR functional CI.** `PQBenchmarkTests` remains opt-in through
  `PROXIMA_PQ_BENCH=1` and is run as an acceptance/release gate; normal PR CI
  continues to skip both long benchmark classes. Core-index PRs retain the
  separate SIFT smoke regression gate.

### Documentation
- **Agent Memory DocC guide.** New
  [`AgentMemory.md`](Sources/ProximaKit/Documentation.docc/AgentMemory.md)
  documents the recommended one journaled `HybridVectorStore` (or dense-only
  `VectorStore`) with automatic checkpointing and `.paged` residency, hybrid
  recall, UUID-to-domain metadata filtering, raw-HNSW first-base bootstrap,
  the optional consumer-composed hot/cold architecture, and the explicit
  mechanism/policy boundary: consumers own summarization, salience,
  forgetting, promotion, tier cadence, and distillation. It links to the raw
  wrapper recipe rather than duplicating that implementation.
- **RAG wrapper recipe — initial consumer guide.** New
  [`docs/RAG-WRAPPER-RECIPE.md`](docs/RAG-WRAPPER-RECIPE.md) documents the
  first-class raw-`HNSWIndex` wrapper path: in-index chunk records, WAL-backed
  crash recovery, checkpoint lifecycle, cache acceptance, and sidecar
  reconciliation from `liveEntries()`, with a compiled companion test as the
  source of truth. The guide now bridges to the Agent Memory DocC article for
  store-level and policy guidance.
- **Benchmark card, capability/competition pages, and README conversion
  rewrite.** Added the reproducible 10K/100K release-mode benchmark card,
  corrected cold-open scaling to O(file size), refreshed the capability matrix
  and Swift-ecosystem comparison, and rewrote the README's landing flow,
  measured-performance summary, API references, and roadmap links without
  changing benchmark methodology or values.
- **Community surface.** Added issue and pull-request templates,
  `SECURITY.md`, and `CODE_OF_CONDUCT.md` so bug reports, security disclosures,
  and contributions have explicit public paths.
- **Engineering-process page + release-notes editorial pass.** New
  [`docs/ENGINEERING-PROCESS.md`](docs/ENGINEERING-PROCESS.md) documents how
  ProximaKit is designed, built, and verified — the measure-before-optimize
  GO/NO-GO decisions, the red-green proof required of every regression test,
  independent review, and the CI gates behind each release. The changelog and
  the public GitHub release notes for v1.6.0–v1.8.0 were given an editorial
  pass so every entry reads for users — what shipped, why it matters, how to
  migrate — with build-process detail moved to the engineering-process page.
  No shipped facts changed.
- **RAG wrapper recipe — journaled-surface round 2 (consumer friction #2).**
  [`docs/RAG-WRAPPER-RECIPE.md`](docs/RAG-WRAPPER-RECIPE.md) gains five
  source-verified additions for consumers who adopted the raw journaled surface
  (`HNSWIndex.open` / WAL add-remove / `syncJournal()` / `needsCheckpoint` /
  `checkpoint`) for incremental folder indexing. (1) **First open** — `open`
  requires the base `.pxkt` to already exist and throws Foundation's
  `NSFileReadNoSuchFileError` (not a typed `PersistenceError`) on a missing one,
  so a first launch must `checkpoint` once to establish the base; the recipe now
  shows the open-if-present-else-establish shape. (2) **Remove durability** —
  `remove(id:)` is non-throwing, so a failed WAL append is deferred into the
  journal and surfaced only by the next `add` / `syncJournal()` / `checkpoint()`;
  remove-driven replacement should `syncJournal()` after removals before trusting
  the delete. (3) **Testing against the WAL** — because `checkpoint` writes a
  16 KiB-page-padded v3 base, the default `walBytesFractionOfBase: 0.10` arm
  trips within one or two 384d adds, so a test that watches WAL growth must
  pass a custom policy with the byte arm disabled (`.infinity`);
  `journalRecordCount` / `journalByteCount` are documented as the observability
  hooks. (4) **Incremental-vs-rebuild order caveat** — a WAL-grown graph and a
  from-scratch rebuild agree on the top hit and result set but can differ at
  secondary ranks, so equivalence gates should assert top-result + set, not
  byte-order. (5) **Optional-cost note** — a per-add `O(liveEntries)` dedup scan
  is largely redundant when open-time `liveEntries()` validation already forces a
  rebuild on any manifest divergence. Four additive `CustomRAGWrapperRecipeTests`
  pin items 1–3 (`testOpenJournaledWithoutBaseFileThrowsFoundationFileNotFound`,
  `testFirstOpenEstablishesBaseViaCheckpointThenReopens`,
  `testRemoveThenSyncJournalIsCleanAndRemovalSurvivesReopen`,
  `testCustomPolicyDisablingByteFractionExposesWALRecordGrowth`), and the
  `RecipeRAGIndex` template gains additive `remove` / `syncJournal` /
  `journalRecordCount` / `journalByteCount` / `needsCheckpoint` passthroughs.
  Existing tests unchanged.

---

## [1.8.0] — 2026-07-04

### Fixed
- **PQHW v3 empty-retaining index round-trips — writer and reader now agree
  on `(0, 0)` originals.** A retaining
  `QuantizedHNSWIndex` with every node removed saved successfully as
  `.pagedV3`, but wrote a trailer shape the library's own reader rejected (a
  zero-length originals section at a nonzero padded offset) — save-then-reload
  permanently broke at the empty edge, while v2 round-tripped fine. The writer
  now emits the parser-legal `(0, 0)` entry, unpadded, exactly when the
  retained originals are empty; the flag stays truthful (`1` = retains
  originals, zero of them). The reader accepts that shape only when the
  expected originals byte count is zero — the existing exact-count guard
  backstops the relaxation — and a new byte-patch test
  (`testEmptyOriginalsEntryWithNonzeroOffsetThrows`) proves a crafted
  zero-length-at-nonzero-offset trailer is still rejected, with
  `PersistenceError.corruptedData` naming "empty originals section must have
  offset 0". `upgradeToV3(at:)` now succeeds on a valid v2 empty-retaining
  base (same legal shape, idempotent, inheriting the temp-verify-atomic-replace
  crash safety), and a paged open of the empty shape is safe by
  construction — the mapped region special-cases zero length and never calls
  `mmap`.

### Added
- **Agent-memory ergonomics — `checkpointAutomatically:`,
  `HNSWIndex.load(from:mode:)`, store-level dense residency, `IndexResidency`
  ([ADR-015](docs/adr/ADR-015-agent-memory-integration.md), Stages A+B).**
  `VectorStore.open(...)` and `HybridVectorStore.open(...)` gain an optional
  `checkpointAutomatically: WALCheckpointPolicy? = nil`: when set, the WAL
  folds automatically inside the actor's serialized mutation chain the
  moment `needsCheckpoint(policy:)` trips after `addChunks`/`removeDocument`,
  so a concurrent batch cannot land between apply and fold (the default
  `nil` preserves today's manual `needsCheckpoint`/`checkpoint` lifecycle,
  byte-identical). The error contract is stated on both factories:
  "A failed automatic fold is rethrown by the mutation call that triggered
  it, but the triggering mutation has already been applied and made
  durable. Do not retry that mutation" — `addChunks` assigns fresh UUIDs, so
  retrying would duplicate chunks; the store remains consistent, and the
  next mutation, `save()`, or `checkpoint()` re-attempts the fold. The tests'
  fixtures pin this contract for the one fold-failure family they actually
  produce — a throw *after* the WAL-truncation commit point, during the
  post-fold `docmap.json` / `hybrid.json` cache-refresh write; the other
  family (a throw *before* that commit point, inside the fold itself) is not
  fixture-tested but holds by the append-fsync argument — under the default
  `.everyBatch` dial the triggering mutation's own WAL record is already
  fsynced at append, so its durability never depended on the fold's outcome.
  Per-turn ingest ceremony drops from 4 awaited calls to 2
  (manual folds 2 → 0); the README store example now leads with the 2-call
  loop, with manual checkpointing demoted to an advanced option. Two more
  additive pieces round out the surface: `HNSWIndex.load(from:mode:)`
  mirrors the quantized family's WAL-free paged loader, closing the
  asymmetry where only `QuantizedHNSWIndex` had one; and both stores gain
  `dense: IndexResidency = .resident`, plumbing `.paged` through to the
  dense leg so a journaled store's base vectors can be served from a
  read-only mapping instead of held resident. `IndexResidency`
  (`.resident` / `.paged`) is now the single canonical residency enum;
  `HNSWOpenMode` and `PQHWOpenMode` become zero-breakage `public
  typealias`es of it (both spellings compile-tested), with the
  `PQHWSaveLayout` → `IndexSaveLayout` rename explicitly deferred to a Stage
  C. 8 new `StoreAutoCheckpointTests` plus a 55-test store
  regression run, all passing, cover the fold-under-mutation interleaving, the
  fold-failure family the fixtures produce (a throw during the
  post-commit-point cache-refresh write; the other family holds by the
  append-fsync argument, not a fixture), and the paged/resident dense-leg
  branches on fresh-store and reopen paths.
- **Demo phase 2: Index Inspector, custom corpus import, results export
  (`Examples/ProximaDemoApp`).** Three new screens, taking the demo from
  three to six (Search, Persistence, Benchmark, Inspector, Import, Export).
  The **Index Inspector** renders a seeded, pausable, force-directed
  visualization of the live HNSW graph in SwiftUI `Canvas`/`TimelineView` —
  46 live vectors, 892 layer-0 links (the Inspector's own on-screen stat
  reports an average degree of 32, consistent with `m = 16`), a layer
  histogram, and tap-to-select showing a node's
  stored document text and metadata. **Custom corpus import** adds
  `.txt`/`.md` files or folders through a verified-balanced security-scoped
  resource flow (guarded start, deferred release, reads complete before
  scope exit), chunking → embedding → indexing with progress UI and
  per-file skip errors, confirmed to persist across relaunch (46 → 48
  vectors on-disk). **Results export** writes CSV and JSON of the live
  search results (query, ids, full-precision scores, titles, categories,
  text; fixed field order via `.sortedKeys`) via `ShareLink` on
  iOS/visionOS and `NSSavePanel` on macOS. Verified end-to-end:
  **BUILD SUCCEEDED** on macOS, the iPhone 16 simulator, and the Apple
  Vision Pro 4K simulator, with all three new screens exercised live on
  iPhone and iPad simulators. Launch hooks extend to `-demoScreen
  inspector|import|export` (`index` kept as a backward-compatible alias for
  `inspector`). Non-blocking followups recorded: an additive read-only
  `liveGraphSnapshot()` library accessor as the proper long-term replacement
  for the Inspector's current reuse of the public `persistenceSnapshot()`
  save/compaction path (safe here because the demo Search index is
  append-only and the Persistence lab uses a separate index instance), and
  iPhone canvas centering.
- **`ProximaBench` gains `paged-access-bench` and `embed-bench`.**
  `paged-access-bench` quantifies the copy-on-access overhead
  [ADR-013](docs/adr/ADR-013-streaming-persistence.md)/[ADR-014](docs/adr/ADR-014-paged-originals.md)
  deferred as a future optimization — paged-vs-resident HNSW search and PQ
  rerank, plus a local unsafe-mmap-read isolation diagnostic — against a
  pre-declared threshold (default `--threshold-percent 5.0`). `embed-bench`
  compares Core ML embedding throughput across `cpuOnly`/`cpuAndGPU`/
  `cpuAndNeuralEngine` compute units against a pre-declared speedup bar
  (default `--threshold-speedup 1.5`), seeded, with per-unit first-load
  latency recorded, and writes a NEEDS-MODEL report when no local model is
  available. Both join the existing `insert-shape`/`distance-kernel`/
  `migrate` subcommands; findings and decisions below.

### Changed
- **Paged-access overhead measured: zero-copy scoped GO, gate corrected
  ([ADR-013](docs/adr/ADR-013-streaming-persistence.md)/[ADR-014](docs/adr/ADR-014-paged-originals.md)).**
  Verdict from an independent re-measurement (2 seeds, reps=11, an
  8-measurement run with an observed floor
  of 8.75% and a mean of ~12%, against the pre-declared 5% threshold):
  **scoped GO** for a zero-copy design pass on the paged HNSW search path
  only — the rerank path stays copy-on-access (≤3.3% overhead at realistic
  rerank depths) and is out of scope. Zero-copy *implementation* itself is
  deferred to future work gated on consumer-hardware re-measurement.
  Three artifact corrections are carried in this
  commit: the automated gate now keys only on the **observed**
  paged-vs-resident delta (it previously could print GO on an
  isolation-extrapolation number that was noise-prone); the metric is
  relabeled "paged-overhead fraction" with an honest framing that the delta
  is copy **+ mmap page locality**, so a zero-copy change recovers only the
  alloc/memcpy component — **realizable benefit ≤ the observed 9–17%
  warm-best-case**, not the full delta; and the original two runs are now
  disclosed as one seed-42 realization with an independent seed-7
  series as confirmation. Ride-along: the platform probe now reports the
  compiler version with `compiler(>=)` instead of the incorrect `swift(>=)`.
- **ANE embedding measured: NO-GO, CoreML defaults stay.** Measured (local
  MiniLM `.mlpackage`, batch 512, 5 reps, seeded, release): `cpuOnly` 117.64
  texts/s, `cpuAndGPU` 117.93, `cpuAndNeuralEngine` 117.48 — **0.9986×**,
  dead even against the pre-declared ≥1.5× bar needed to justify a public
  `computeUnits` knob; ANE first-load additionally pays 213.88 ms versus
  88.03 ms for CPU (the classic compile tax). Decision: **NO-GO** — no
  public `computeUnits` knob ships; CoreML's defaults stay. Honest caveat
  recorded: the local model emits **NaN embeddings** through the provider
  path, so these are CoreML runtime-dispatch throughput numbers only, not a
  semantic-quality validation; the reopen condition names a finite-output
  sentence-embedding artifact re-measured on consumer hardware. Completes
  the measure-first triptych: Metal NO-GO, zero-copy scoped-GO (above,
  deferred), ANE NO-GO — zero speculative code shipped on any of the three.
- **Two new design-only ADRs.**
  [ADR-015](docs/adr/ADR-015-agent-memory-integration.md) (Proposed) maps
  tinybrain's agent-memory lifecycle onto the existing ProximaKit surface
  and recommends the single-journaled-store-with-a-paged-dense-leg shape
  over a two-tier hot/cold split, quantified by labeled arithmetic: a
  50-turns/day, 2-chunks/turn agent profile holds **~17.5 MB** resident at
  1 year on a paged dense leg versus **~73.6 MB** full-precision resident
  (**~87.6 MB** vs **~367.9 MB** at 5 years) — comfortably in an iPhone
  app's budget either way, so the two-tier shape is endorsed as a
  *consumer-composed pattern* over already-shipped primitives (journaled
  HNSW + PQHW `.paged`), not new store API; `distill()` is named an explicit
  non-goal (it needs an LLM ProximaKit does not have). The Stages A+B
  surface above ships ahead of the ADR's own formal acceptance; Stage C
  (the `PQHWSaveLayout` → `IndexSaveLayout` rename and broader agent-memory
  docs) remains pending.
  [ADR-016](docs/adr/ADR-016-dynamic-m.md) (Proposed) recommends **DEFER,
  leaning NO-GO** on hierarchical-NSW dynamic-`M`: the paper's `mMax0 = 2m`
  layer-0 schedule already ships, so "dynamic M" can only mean an
  *upper-layer-only* schedule, and since only ~6.25% of nodes reach layer
  ≥ 1 (arithmetic: `exp(-1/mL) = 1/m` at `m = 16`), the residual lever is
  worth **< 0.25%** of resident memory against ADR-011/013's ≈200 B/node
  adjacency baseline. No format change would be needed (adjacency is
  already count-prefixed, never padded to `mMax0`) and no named consumer
  asks for it; a pre-committed measurement gate (≥ +2.0 pp recall@10 at
  `ef = 16`, no regression past −0.5 pp at `ef ≥ 64`, a Pareto gate against
  simply raising uniform `m` at equal memory, and `Q ≥ 1000` queries to
  clear the noise floor) is declared to reopen the question.
- **Five documentation-accuracy fixes.**
  `VectorStore`/`HybridVectorStore` type docs now steer continuous-mutation
  consumers to `open(...)` (journaled-vs-non-journaled Discussion and
  Topics blocks, per-initializer steering) — the journaled surface was
  previously invisible from the page most adopters land on. A stale DocC
  link for `open(baseURL:walURL:durability:mode:)` is fixed (a 21-link
  catalog sweep found no other drift). The README/ROADMAP contradiction
  over the demo Benchmark tab's status is reconciled to **Shipped**
  (CoreML-download UI and result export stayed **Planned** at doc time;
  export has since shipped, see Added above). Test-suite timing claims are
  made honest: the old "~30s (fast)" claim had survived three releases
  while being **~60–70× off** measured CI reality; docs now state **~600
  tests / 33–38 min on CI**, name the seconds-scale `--filter` inner loop
  for local iteration, and document the `PROXIMA_RECALL_BENCH=1` local-skip
  gate; two stale "400+ tests" mentions are reconciled to ~600. The README
  Quick Start is reordered: a 12-line, signature-verified add-and-search
  snippet now leads, with the demo-clone detour following.

---

## [1.7.0] — 2026-07-03

### Added
- **PQHW v3 format + v2-to-v3 migration rewriter ([ADR-014](docs/adr/ADR-014-paged-originals.md)
  Stage 1).** The format-and-migration half of paged PQ originals: a 56-byte
  header and body byte-identical to v2, plus a new 128-byte `"PQH3"` trailer
  appending a `UInt64` seven-section offset/length table and a reserved
  snapshot generation; the originals section start pads to 16 KiB. Writer
  policy is conservative and additive — the default `save(to:)` still writes
  v2 byte-for-byte; the new `save(to:layout:)` with `.pagedV3` writes v3 only
  when originals are retained, falling back to v2 otherwise. **Migration:**
  `QuantizedHNSWIndex.upgradeToV3(at:)` (PQHW) and
  `PersistenceEngine.upgradeToV3(at:)` (`.pxkt`) are pure section-copy
  rewriters — payload bytes bit-identical, no graph decode — with a
  full-checksum read-back verification, temp-file + atomic replace, and a
  no-op on an already-migrated file; both let an iPhone app self-upgrade an
  existing base to a paging-capable v3 without a rebuild. `ProximaBench`
  gains a `migrate --path PATH [--family pxkt|pqhw]` subcommand wrapping both
  upgraders (family auto-detected from the file magic). The `metadataOffset`
  4 GiB ceiling is lifted with **no `.pxkt` version bump**: the v3 trailer's
  `UInt64` offset is authoritative, and the legacy `UInt32` header field
  carries a documented `0xFFFF_FFFF` sentinel only when the true offset
  exceeds `UInt32.max`. 28 new tests: format byte-identity/alignment/rerank
  parity (7), trailer corruption matrix — truncation, magic, counts,
  overflow, contiguity, flag mismatches (10), migration round-trip fidelity
  including per-section bit-identity and fingerprint equality (9), and
  crash-safety truncation sweeps proving every torn output is rejected typed
  while the source stays untouched (2, 432 assertions).
- **Paged PQ originals — [ADR-014](docs/adr/ADR-014-paged-originals.md)
  complete (Stage 2, status now Accepted).** A retaining `QuantizedHNSWIndex`
  can serve rerank originals from a read-only mmap of its v3 file instead of
  holding them resident, closing the arc [ADR-012](docs/adr/ADR-012-pq-reranking.md)
  opened. `OriginalsStore` (`.resident([Vector])` / `.paged(MappedVectorRegion)`)
  replaces the resident originals array, copying on access inside one
  synchronous scope so no pointer ever crosses an actor suspension —
  deliberately not `VectorProvider`, since a quantized index has no
  `add()`/resident tail. `MappedVectorRegion` is generalized behind a layout
  resolver; the new PQHW resolver reads only the header + trailer to bind the
  padded originals section (the `.pxkt` path is untouched). The new opt-in
  `load(from:mode:)` — `.resident` (default, byte-identical to before) /
  `.paged` — rejects a paged open of a v2, flag-0-v3, or unaligned-v3 base
  with a typed, actionable `PersistenceError` (the same file still loads
  `.resident`), closing the Stage-1-deferred corruption items. Accounting is
  now honest: `originalStorageBytes` reports 0 when paged (the originals live
  on flash), `mappedOriginalStorageBytes` and `originalsArePaged` are new,
  and `memorySavingsRatio` rises back to the pure-PQ (32×) ratio for a paged
  index instead of counting mapped bytes as resident cost. **Measured, Apple
  M4 Max, release:** a 100,000 × 384d retaining fixture (146.5 MB originals
  payload) shows a paged open costing **8.0 MB** — 18× less than the 146.5 MB
  payload — versus **43.1 MB** resident (5.4× more than paged), and stays
  flat (8.2 MB) after 50 warm reranks (`PagedOriginalsMemoryTests`); a
  resident rerank A/B on the same benchmark
  shows **no regression** (median ~3.6% faster, though re-measurement puts
  the true delta near 0–2% with sign varying run-to-run — noise at this
  scale; the ±2% bail-out bound applies to regressions only and was never
  approached — `ResidentRerankBenchTests`). `PagedOriginalsParityTests`
  proves paged results are bit-identical to resident across rerank on/off,
  filtered, post-remove, and save/reload. Ride-along: migration failures are
  now wrapped in a typed `PersistenceError.migrationFailed(String)` preserving
  the underlying error, with new fixtures for v1 PQHW, v1 `.pxkt`, and
  unpadded-v3 migration paths.
- **Journaled stores — `VectorStore`/`HybridVectorStore` WAL wiring
  ([ADR-013](docs/adr/ADR-013-streaming-persistence.md), closes deviation
  5).** New async `VectorStore.open(name:embedder:storageDirectory:metric:config:durability:)`
  and `HybridVectorStore.open(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:durability:)`
  factories stream mutations to the dense leg's WAL, so `save()` becomes an
  **O(1) durability flush** (`syncJournal()`) instead of a full rewrite, and
  `checkpoint()` / `needsCheckpoint(policy:)` provide the periodic O(corpus)
  fold. The multi-file crash-consistency problem — a replayed dense index
  running ahead of its sidecars (`docmap.json`; for hybrid, the whole
  WAL-less sparse `index.pxbm` leg) — is solved by **derivation, not
  ordering**: the dense index + its WAL are the single source of truth, and
  a new additive `HNSWIndex.liveEntries()` hook lets a journaled `open`
  rebuild the document map (and, for hybrid, the entire BM25 sparse leg)
  from the recovered index's live entries. A doc-map or sparse entry exists
  iff a live dense vector exists, so orphaned vectors and phantom mappings
  are structurally impossible after any crash, and any stale or corrupt
  sidecar cache on disk is simply ignored rather than reconciled. 9 recovery
  tests (`StoreJournalRecoveryTests`) prove it for both store types:
  index-ahead-of-stale-sidecar, phantom-sidecar-ignored, torn-WAL-tail
  bijection, sparse-leg-rebuilt-from-dense-WAL, remove-then-recover, and
  typed `walGenerationMismatch` rejection. The historical initializers,
  `save()`, `query()`, `addChunks`, `removeDocument`, and
  `loadDocumentMap()` are unchanged for non-journaled stores — byte-identical
  behavior, `CHA-107` contract untouched, existing store test suites pass
  unmodified.

### Changed
- **Memory-acceptance benches recalibrated to measured-baseline ratio gates
  (macOS 26 compressor reality).** Both `PagedVectorMemoryTests`
  ([ADR-013](docs/adr/ADR-013-streaming-persistence.md) Stage 2, HNSW
  vectors) and the new `PagedOriginalsMemoryTests`
  ([ADR-014](docs/adr/ADR-014-paged-originals.md) Stage 2, PQ originals)
  moved off an absolute "≥60% of payload recovered" gate. macOS 26's memory
  compressor counts freshly-copied anonymous pages at their **compressed**
  size, so `phys_footprint` captures only a fraction of the theoretical
  payload on the resident side — an OS accounting reality, not a residency
  leak — and the old gate demonstrably flaked (measured 57.7%/60.5% across
  two runs against its 60% threshold, Apple M4 Max, release). Both benches
  now gate on the measured baseline with margin: the paged-open delta must
  be a small fraction of the payload (HNSW: < payload/3, measured 22.6 MB of
  146.5 MB; PQHW: < payload/8, measured 8.0 MB of 146.5 MB), the resident
  open must cost materially more (> 2.5× paged, both), the recovered slice
  must clear a floor (HNSW: > payload/4; PQHW: > payload/8), and a warm
  search/rerank sweep must stay bounded (both: < paged open + payload/4) —
  with the compressor rationale documented in both test file headers and in
  the ADR-013 Stage 2 and ADR-014 Stage 2 notes.
- **`docs.yml` deploy-pages step gains an if-failure retry.**
  `actions/deploy-pages@v4` flaked transiently 4 times in 24h with a
  "Deployment failed, try again later" error even though the artifact was
  always fine — a rerun healed every one — so the workflow now retries the
  deploy step automatically (`if: failure()`) instead of requiring a human
  to run `gh run rerun --failed`.

---

## [1.6.1] — 2026-07-03

### Added
- **Demo app Persistence lab + Benchmark tab (`Examples/ProximaDemoApp/`).**
  `ProximaDemoApp` grows from a single Search screen into a genuinely
  multiplatform SwiftUI target (iPhone/iPad/macOS/visionOS) that dogfoods the
  Stage 1/Stage 2 persistence surfaces from 1.6.0 on-device.
  - **Persistence tab** (`PersistenceLab.swift` controller + `PersistencePanel.swift`
    view): builds a reproducible synthetic corpus (3,000 / 6,000 / 12,000 × 384d
    via the deterministic `SyntheticCorpus` / `DemoRNG` generator — no embedder,
    no network), saves a v2 base, opens it journaled, and surfaces live WAL-state
    readouts (generation, WAL bytes on disk, ops since checkpoint, needs-checkpoint,
    base format v2/v3) with a one-tap checkpoint. It dogfoods the real typed-error
    path rather than faking success: a `.paged` open on an unpadded v2 base is
    refused with `PersistenceError`, surfaced as an honest "Paged open blocked"
    banner with one-tap checkpoint-to-recover (a checkpoint writes the page-aligned
    v3 base that `.paged` requires). It also measures LIVE resident-vs-paged
    process memory via `task_vm_info.phys_footprint` — the same probe
    `PagedVectorMemoryTests` uses — including the incremental cost of warm searches
    against the paged mapping.
  - **Benchmark tab** (`BenchmarkEngine.swift` + `BenchmarkView.swift`): runs a
    seeded `efSearch` sweep (16 / 32 / 64 / 128 / 256) over a reproducible synthetic
    corpus (3,000 × 128d), measuring recall@10 against an exact `BruteForceIndex`
    ground truth alongside live per-query latency (median + p90), visualized with
    SwiftUI Charts (recall-vs-latency) plus a results table.
  - **Shared support** (`DemoLabSupport.swift`): `MemoryProbe`, `SyntheticCorpus`,
    and `DemoRNG` — dependency-free and reproducible, with no embedder or network.
  - **Layout**: compact width uses a four-tab `TabView`
    (Search / Benchmark / Persistence / Index); regular width keeps the
    `NavigationSplitView` with a segmented switcher across the three feature screens
    in the detail pane.
  - **Automation hooks**: screenshot/UI-test capture reads launch-arg
    `UserDefaults` — `-demoScreen <search|persistence|benchmark|index>`,
    `-demoQuery "<text>"`, `-demoAutorun 1`, `-demoFlow memory` — so every screen
    can be captured non-interactively in a known state.
- **`scripts/check-imports.sh` dependency-policy guard, wired into CI lint.**
  A POSIX-sh linter enforces the `CONTRIBUTING.md` "Module Rules" allowlist —
  `ProximaKit` may import only Foundation/Accelerate/Metal/Darwin/Glibc (each
  justified against real usage: vDSP per ADR-001, the GPU utility per ADR-009,
  `fcntl`/`F_FULLFSYNC` + `mmap` per ADR-013); `ProximaEmbeddings` additionally
  CoreML/NaturalLanguage/Vision/CoreGraphics. Handles attributed imports
  (`@preconcurrency import X`), excludes the DocC catalog's snippet imports
  from scope, and runs as a new "Check import policy" step in the `lint` CI
  job (`sh scripts/check-imports.sh`) — proven to fail on an injected
  violation. This backstops what was previously enforced only by manual PR
  review (`import CoreML` inside `ProximaKit` still compiles: Apple's system
  frameworks link against any target, and `Package.swift` declares no
  per-target linkage that would reject it).
- **`RecallBenchmarkTests` local skip gate.** A class-level `setUpWithError`
  now requires `PROXIMA_RECALL_BENCH=1`, so a bare local `swift test` (without
  `--skip`) drops from 20+ minutes of recall sweeps to an instant skip; set
  `PROXIMA_RECALL_BENCH=1` to run them locally, matching the established
  `PROXIMA_*` benchmark-gate idiom. CI behavior is unchanged — its explicit
  `swift test --skip RecallBenchmarkTests` stays, so this weakens nothing CI
  verifies.

### Changed
- **`QuantizedHNSWIndex` and `ScalarQuantizedHNSWIndex` `remove(id:)` repair
  dangling incoming edges in O(in-degree)**, porting the `HNSWIndex`
  reverse-adjacency map (1.6.0, below) onto both quantized indexes: a
  maintained `inEdges` transpose map replaces the previous per-layer sweep of
  every adjacency list. No public API change, no on-disk format change — the
  map is derived state, lazily materialized (`ensureInEdges()`) on first use,
  strictly after the build/load validation gates rather than eagerly in the
  memberwise initializer (an eager cut tripped `PersistenceCorruptionTests`'
  out-of-bounds-neighbor fixture, which builds corrupt in-memory indexes
  through that initializer — a path `HNSWIndex` never exposes, since its
  `init(restoring:)` only ever receives loader-validated layers). Equivalence
  is proven differentially against the retired O(E_l) full sweep (kept
  internal, test-only, as the control — retained because without
  reconnection there is no reachability oracle and only the differential
  catches over-removal): graph fingerprints asserted equal after every one of
  270 seeded churn ops (140 on PQHW, 130 on SQHW), plus save/load rebuild
  consistency and interleaved search equality (`QuantizedReverseAdjacencyTests`).
- **Corruption-contract hardening rode along.** `ScalarQuantizedHNSWIndex` had
  the identical latent out-of-bounds-neighbor gap the `QuantizedHNSWIndex` fix
  above closes, invisible only because no fixture exercised it for SQHW. A
  new memberwise-init corrupt-fixture test
  (`testOutOfBoundsNeighborViaMemberwiseInitSaveThenLoadThrows` in
  `ScalarQuantizationPersistenceTests.swift`, mirroring
  `PersistenceCorruptionTests`' `testQuantizedOutOfBoundsNeighborThrows` for
  PQHW) is proven to trap RED against the unfixed code (ADR-010 rule 5) and
  passes now, bringing SQHW's corruption coverage to parity with PQHW.

### Fixed
- **WAL record counter no longer resets to 0 across a reopen, restoring the
  documented "since the last checkpoint" contract.** `journalByteCount` and
  `journalRecordCount` are both documented as counting activity since the last
  checkpoint, but `HNSWIndex.open()` broke the pair: after replaying a non-empty
  WAL, `WALJournal.init(appendingTo:…)` seeded `byteCount` from the replayed
  valid-prefix byte count yet hardcoded `recordCount = 0`. Immediately after a
  reopen the two counters disagreed — the op-count arm of `needsCheckpoint(policy:)`
  (`journal.recordCount > policy.maxOps`) under-counted, believing zero ops had
  accrued since the checkpoint when the N replayed records actually had, so a WAL
  already past its `WALCheckpointPolicy.maxOps` budget went unflagged until enough
  *new* appends re-crossed it. The fix threads `existingRecordCount` through
  `WALJournal.init(appendingTo:parentGeneration:dimension:existingByteCount:existingRecordCount:durability:)`,
  passed from `HNSWIndex.open()` as `replay.records.count` — the exact record count
  the decoder recovered from the same valid prefix that produced `existingByteCount`
  — so both counters agree from the very first append after reopen (a torn tail
  drops its partial record and its bytes together). Proven by two new
  `WALRecoveryTests`: `testJournalRecordCountSurvivesReopen` (reopen carries the
  count in unchanged, a subsequent append advances it N → N+1, and a checkpoint
  resets it to 0) and `testNeedsCheckpointCountsCarriedInOpsAfterReopen` (with the
  byte-fraction rule disabled via `walBytesFractionOfBase: .infinity` and `maxOps`
  set below the carried-in count, `needsCheckpoint()` must report `true` immediately
  after reopen, before any new append — exactly the case that returned `false`
  before the fix).

---

## [1.6.0] — 2026-07-03

### Added
- **WAL incremental saves for `HNSWIndex` ([ADR-013](docs/adr/ADR-013-streaming-persistence.md),
  Stage 1).** Opt-in journaling via
  `HNSWIndex.open(baseURL:walURL:durability:)` turns saves from O(corpus) into
  O(change): each `add`/`remove` appends a framed `PXWL` v1 record
  (`[length][crc32][payload]`, little-endian) instead of rewriting the whole
  snapshot — ≈1.6 KB per `add` at 384d versus the full-file rewrite `save(to:)`
  still does. `add` records journal the assigned HNSW level so replay is
  **deterministic end-to-end**, not merely valid — `WALRecoveryTests` asserts
  exact structural equality (adjacency, levels, entry point, tombstones,
  vectors, metadata) against the producing index, not just search validity.
  The base format bumps to **`.pxkt` v3** (64-byte legacy header + section
  table + `snapshotGeneration: UInt64`); `minSupportedVersion` stays 1, so v1/v2
  files still load through the unchanged resident path, and the header binds
  the WAL to its parent generation *and* dimension/metric so a stale or
  mispaired sidecar is rejected with typed errors
  (`PersistenceError.walGenerationMismatch` /
  `.walDimensionMismatch` / `.walMetricMismatch`) rather than replayed into the
  wrong base. `checkpoint(baseURL:walURL:durability:)` folds the WAL back into
  a fresh base (atomic write, then `F_FULLFSYNC`) and resets the journal;
  `needsCheckpoint(policy:)` reports when the configurable
  `WALCheckpointPolicy` (WAL > 10% of base size, or > 10,000 ops by default) is
  exceeded. Durability is a three-way dial —
  `WALDurability.everyRecord` / `.everyBatch` (default) / `.manual` — with doc
  comments stating the Darwin truth plainly: a plain `fsync(2)` reaches the
  drive cache only, `F_FULLFSYNC` (always used at checkpoint commits) is what
  forces media. Recovery is proven, not claimed: an in-process truncation
  sweep across every byte boundary of the final record and every record
  boundary of the WAL (`WALTruncationSweepTests`) plus an out-of-process
  `SIGKILL` rig (`WALKillRecoveryTests`, `WALKillWriter` helper) assert
  typed-errors-only, longest-valid-prefix recovery — 100/100 randomized-kill
  recoveries opt-in via `PROXIMA_RUN_KILL_RIG`, with a 5-iteration smoke run in
  every CI run. `save(to:)`/`load(from:)` are byte-identical to before and
  still read/write v2; journaling is a strictly additive, opt-in surface.
  **Store-level wiring (`VectorStore`/`HybridVectorStore`) is deferred, not
  shipped** — `HybridVectorStore` froze the v1 store contract (CHA-107) and the
  sparse leg has no WAL codec yet; this release is index-level only.
- **Paged vector region for `HNSWIndex` ([ADR-013](docs/adr/ADR-013-streaming-persistence.md),
  Stage 2).** A new opt-in `.paged` open mode
  (`HNSWOpenMode`, via `HNSWIndex.open(baseURL:walURL:durability:mode:)`)
  serves the vector section directly from a read-only file mapping
  (`MappedVectorRegion`) instead of decoding it resident, keeping only the
  graph, ids, levels, metadata, and a resident tail of post-snapshot adds in
  memory. Paging rides the `.pxkt` v3 format Stage 1 already shipped:
  `checkpoint(...)` now zero-pads the vector section to a 16 KiB
  (Apple-Silicon page) boundary so it can be mapped independently, and the v3
  section table records the padded offset — both padded and unpadded v3 bases
  still decode identically through the unchanged resident path. Paged search
  is byte-identical to resident — same ids, bit-equal Float32 distances —
  across seeded queries, graph-aware filtered search, and post-WAL-replay
  state, asserted by `PagedVectorParityTests`. `.resident` stays the default,
  byte-identical to before; `.paged` requires a padded v3 base (any
  `checkpoint` writes one). Measured, Apple M4 Max, release: a
  100,000 × 384d fixture with a 146.5 MB vector payload shows a paged open
  resident at 18.1 MB versus 112.3 MB for the same base opened `.resident` —
  94.1 MB (64%) of the payload not resident. Resident-mode search is
  unaffected by the change: the ADR's before/after regression benchmark
  (identical seed, 9 reps + 2 warmup) puts the worst-case comparison at
  +0.5%, far under the project's 2% bail-out threshold (measured, Apple M4
  Max, release).
- **Graph-aware filtered search extended to `QuantizedHNSWIndex` and
  `ScalarQuantizedHNSWIndex` ([ADR-008 second
  addendum](docs/adr/ADR-008-filtered-search.md)).** Both quantized indexes now
  apply the search predicate inside the layer-0 beam — the strategy
  `HNSWIndex` adopted in the first addendum — instead of post-filtering, ported
  onto each index's own scoring path (ADC for PQ, dequantize for SQ) with the
  same adaptive `ef` widening and `efCap` bound. `search(query:k:efSearch:filter:)`
  (and PQ's `rerankDepth:` overload) is unchanged in shape, and unfiltered
  queries are structurally untouched. On the PQ path with `retainOriginals`,
  the filtered beam's target becomes `rerankDepth` rather than `k`, composing
  with reranking exactly as post-filter did (only filter-passing candidates
  count toward the depth). Measured recall floors from
  `FilteredSearchSelectivityTests` (seeded 2000-vector/32d corpus, 20 seeded
  queries) are honest about ADC error: pure-ADC recall@10 sits below the
  full-precision 0.9 target (0.745 at ~10% selectivity, floor 0.65; 0.870 at
  ~1%, floor 0.78), while rerank-enabled PQ and dequantized SQ both clear
  ≥0.95 at both selectivities. A post-filter under-fill control
  (`testQuantizedOnePercentControlPostFilterUnderfillsK`) proves the upgrade:
  the retired post-filter pipeline under-fills `k` on every seeded query at
  ~1% selectivity, while the graph-aware beam fills all 10. `SparseIndex`
  keeps post-filter deliberately — it is a BM25 postings scan with no
  `ef`-bounded beam to route through.
- **Full-precision reranking for `QuantizedHNSWIndex`
  ([ADR-012](docs/adr/ADR-012-pq-reranking.md)).** `build(...)` gains an
  opt-in `retainOriginals: Bool = false` that keeps the Float32 vectors
  alongside the PQ codes; a new throwing overload
  `search(query:k:efSearch:rerankDepth:filter:)` overscans the ADC beam and
  re-scores the top `rerankDepth` candidates with exact Euclidean distance
  before truncating to `k`. When originals are retained, the existing
  non-throwing `search` reranks by default at depth `4·k`; indexes without
  originals produce search results byte-identical to before. Requesting `rerankDepth > 0`
  without retained originals throws the new typed
  `QuantizedIndexError.originalsNotRetained` (fail-fast, never a silent
  ~30%-recall fallback). On the seeded clustered fixture, reranked recall@10
  is asserted ≥ 0.90 — vs the 0.667–0.717 pure-ADC band — in
  `PQRerankTests`. Honest cost: retention pays the full `4·d` bytes/vector
  again, so a retaining index has no compression story
  (`memorySavingsRatio` drops below 1.0; the new `originalStorageBytes`
  reports the cost). Reranking trades PQ's 32× memory win for recall.
- **Graph-aware filtered search for `HNSWIndex`
  ([ADR-008 addendum](docs/adr/ADR-008-filtered-search.md)).** Filtered
  queries now apply the predicate *during* the layer-0 beam with adaptive
  `ef` widening — rejected nodes still route the beam but never occupy
  result slots — so selective filters fill `k` instead of under-filling.
  API shape is unchanged (`search(query:k:efSearch:filter:)`); unfiltered
  queries run the original code path untouched. The selectivity acceptance
  suite (`FilteredSearchSelectivityTests`, seeded corpus) asserts
  recall@10 ≥ 0.9 with a full `k` results at ~10% and ~1% predicate pass
  rates and exact set-and-order equality at ~0.1%, plus a control showing
  the retired post-filter pipeline returning 0–1 results at 1% selectivity.
  (`QuantizedHNSWIndex` and `ScalarQuantizedHNSWIndex` have since adopted the
  same strategy — see the second addendum entry above; `SparseIndex` keeps
  post-filter.)
- **`PQConfiguration.seed`** — seeds the PQ k-means centroid-initialization
  draws so training is reproducible: same seed + same vectors →
  byte-identical codebooks and codes (`PQDeterminismTests`). A
  training-time knob like `HNSWConfiguration.levelSeed`: deliberately not
  persisted by the codecs and excluded from `Codable`.
- **`JensenShannonDistance`** — a true distance metric computing `sqrt` of
  the base-2 Jensen-Shannon divergence: unlike raw JSD (a dissimilarity
  only), the square root is symmetric *and* satisfies the triangle
  inequality. Range `[0, 1]` — 0 for identical distributions, 1 for disjoint
  support — with inputs treated as unnormalized finite non-negative
  distributions and L1-normalized internally before comparison. Domain
  handling is bounded-sentinel, never a trap: negative or non-finite input
  components, and non-finite intermediate results (overflow/NaN during
  normalization or the divergence sum), both return `1` (maximal
  dissimilarity) rather than a precondition failure. Serializable as
  `DistanceMetricType.jensenShannon` (raw value 7); 13 tests in
  `NewMetricsTests.swift` exercise it — identity/symmetry, triangle
  inequality, and scalar/batch parity through the shared metric-testing
  harness (`assertIdentityAndSymmetry`, `assertTriangleInequality`,
  `assertBatchMatchesScalar`), plus disjoint-support, zero/negative/
  non-finite/denormal edge cases and `DistanceMetricType` + `BruteForceIndex`
  round-trips.
- **`MetalBatchDistance` ([ADR-009](docs/adr/ADR-009-metal-backend.md), v1
  scope).** A standalone GPU utility for one-query-to-N batch distances
  (squared L2 and cosine) over the same flat row-major layout as the vDSP
  batch paths. Inline-MSL compute kernel, lazily compiled and cached; vDSP
  numerical parity asserted to 1e-4 in tests; automatic CPU (vDSP) fallback
  on any runtime Metal failure; `init?` returns `nil` where no GPU exists
  and a same-API stub compiles on non-Metal platforms. **Scope honesty: v1
  is a batch utility only — it is not wired into `HNSWIndex` build or
  search, and no speedup numbers are claimed until measured** (since
  measured — see the Metal NO-GO entry under Changed, below).
- **On-device RAG example + tutorial.** `swift run OnDeviceRAG`
  ([`Examples/OnDeviceRAG/`](Examples/OnDeviceRAG/)) answers questions over
  20 built-in notes entirely on-device: NLEmbedding embeddings →
  `HNSWIndex` retrieval → a 2-requirement `LanguageModel` seam with a
  deterministic `TemplateLLM` everywhere and `FoundationModelsLLM` (Apple's
  on-device LLM) where the OS provides one. Supports interactive and
  scripted (`-question`, `-llm template`) modes. The walkthrough lives in
  [`docs/RAG-TUTORIAL.md`](docs/RAG-TUTORIAL.md).
- **Interactive DocC tutorial.** A `@Tutorials` catalog ("Meet ProximaKit")
  with the step-by-step *Build On-Device Semantic Search* tutorial — create
  an `HNSWIndex`, embed text with `NLEmbeddingProvider`, persist and reload
  — linked from the DocC landing page and Getting Started.
- **Apple-grade visual system** for all README/docs assets: SF Pro outline
  logo and restyled animated diagram family per `docs/assets/DESIGN.md`.

### Changed
- **Metal insert-loop integration: measured NO-GO ([ADR-009
  addendum](docs/adr/ADR-009-metal-backend.md)).** ADR-009 shipped
  `MetalBatchDistance` as a standalone utility and deferred wiring it into the
  `HNSWIndex` build path until someone measured it against vDSP at realistic
  build shapes — that measurement now exists, and the answer is no. A new
  `ProximaBench` subcommand, `insert-shape`, instruments the real `add()` path:
  the insert loop contains **zero** batch-distance calls — 86–93% of build-time
  distance evaluations are pairwise (heuristic neighbor selection and
  pruning), and the largest batchable one-query-to-N unit is a single node
  expansion (≤ `mMax0 = 32`), never a shape a GPU dispatch can amortize. A
  `distance-kernel` GPU-vs-vDSP sweep (release build, N from 32 to 1M, d ∈
  {384, 768}, cosine + euclidean, per-cell parity-checked) found vDSP (AMX)
  winning at **every** measured N — ~215× faster at the real build shape
  (N=32), still ~4.8× faster at N=1M, no crossover anywhere. `HNSWIndex` build
  stays on vDSP unchanged; the "gated on measurement" deferral in ADR-009 is
  now a settled decision, not open follow-up — reopenable only under the
  concrete conditions the addendum lists (a batched-build design plus
  zero-copy GPU dispatch, or hardware without AMX-class CPU matrix
  acceleration). `docs/ROADMAP.md`'s GPU row is updated to match.
- **`HNSWIndex.remove(id:)` repairs dangling incoming edges in
  O(in-degree).** A maintained reverse-adjacency map replaces the previous
  sweep of every layer's edge lists. The map is derived state: rebuilt from
  the snapshot on load, never persisted (on-disk format unchanged), and
  equivalence-tested against brute force through
  add/remove/compact/save/load churn (`ReverseAdjacencyTests`).
- **`PQHW` on-disk format v2 ([ADR-010](docs/adr/ADR-010-format-evolution.md)
  rules).** A previously reserved header field becomes an `originalsPresent`
  flag; when set, a slot-aligned Float32 originals section follows metadata
  (compacted on save like every other per-slot section, corruption-tested).
  v1 files load unchanged (`retainOriginals = false`). **Migration:**
  writers always write v2 now, so files saved by this version — even
  without originals — are rejected by v1.5.0-and-older readers with
  `unsupportedVersion`.
- **CI:** SIFT1M dataset verification now pins SHA-256 digests (recorded
  from a trusted CI run) in addition to byte-size and record-header checks;
  the benchmark smoke and nightly-full jobs are deduplicated through a
  reusable `workflow_call` workflow
  ([`benchmark-core.yml`](.github/workflows/benchmark-core.yml)).

### Fixed
- **`QuantizedHNSWIndex` (PQHW) loader rejects a metadata-count/node-count
  mismatch** with `PersistenceError.corruptedData`, mirroring the existing
  SQHW guard. A crafted or corrupted file whose metadata array length
  disagreed with `nodeCount` previously loaded without error and trapped on
  the first `search()` or save-after-remove; a corruption regression test
  proves the load returns before the fix and throws after
  (`PersistenceCorruptionTests`).
- **`ScalarQuantizer.encode` no longer spins on a non-finite vector
  component.** A ±infinite component drove the overflow guard's
  `scale.nextDown` step ~58.6M times (measured: 0.73s in a debug test run)
  trying to step down from infinity toward a finite scale it could never
  reach. Non-finite `maxAbs` now takes the existing zero-vector fallback
  immediately, same as the zero and subnormal-underflow cases; regression-
  pinned in `ScalarQuantizerTests`.

---

## [1.5.0] — 2026-06-10

Correctness fixes — every fix reproduced before patching and re-verified after — plus INT8 scalar quantization, three new distance metrics, reproducible graph construction, and a CI overhaul.

### Added
- **Multiplatform demo app**: `ProximaDemoApp` now targets iPhone, iPad,
  macOS, and visionOS from one SwiftUI target (compact widths get a
  search-first tab layout; AppKit image loading replaced with ImageIO).
  The persisted demo index is validated against the current embedder's
  dimension before reuse — NLEmbedding can pin sentence (512d) or
  word-averaging (300d) mode depending on which language assets the OS
  has, and a stale-dimension index made every search silently empty.
  Dimension mismatches now surface as an actionable error instead.
  A `-demoQuery` launch argument supports screenshot automation.
- **INT8 scalar quantization (ADR-007).** `ScalarQuantizer` — symmetric
  per-vector scaling (`scale = maxAbs / 127`, explicit zero-vector handling)
  — plus the `ScalarQuantizedHNSWIndex` actor. ~4× vector-storage reduction
  (384d: 1,536 B → 388 B per vector), **no training phase**, and search runs
  through the configured `DistanceMetric`, so any serialisable metric works
  (contrast with PQ's L2-only ADC). Two-phase `build` (full-precision graph
  construction, then encode), binary persistence, memory introspection
  (`codeStorageBytes` / `memorySavingsRatio`), and acceptance-tested recall
  floors: Recall@10 ≥ 0.95 (euclidean) / ≥ 0.93 (cosine) against brute-force
  ground truth. Design rationale in
  [ADR-007](docs/adr/ADR-007-int8-scalar-quantization.md).
- **Three new distance metrics:** `ChebyshevDistance` (L∞),
  `BrayCurtisDistance`, and `MahalanobisDistance` (constructible from a
  covariance or inverse-covariance matrix). Chebyshev and Bray-Curtis join
  `DistanceMetricType` and persist with any index; Mahalanobis is search-only
  (not serialisable), and `persistenceSnapshot()` reports it as
  `PersistenceError.unserializableMetric` rather than guessing.
- **`HNSWConfiguration.levelSeed`** — seeds the layer-assignment RNG so graph
  construction is reproducible: the same insertion sequence yields the same
  topology. Build-time knob only; deliberately not persisted.
- **Persistence corruption-hardening test matrix** — 42 tests across all four
  binary codecs, covering truncated sections, out-of-range graph indices,
  invalid entry points, and bad configuration values.
- **DocC published to GitHub Pages** on every push to `main` (`docs.yml`),
  and **automatic GitHub Releases** with CHANGELOG-extracted notes on version
  tags (`release.yml`).
- **CI overhaul:** SwiftLint job (pinned 0.63.2, strict config), iOS Simulator
  build job for `ProximaKit` + `ProximaEmbeddings`, release tag/version/
  changelog consistency check, benchmark regression gate wired to
  `compare.py`, SIFT1M SHA-256 verification, and fixed SwiftPM caching.
- **ADRs:** [ADR-007](docs/adr/ADR-007-int8-scalar-quantization.md) (INT8
  scalar quantization — accepted),
  [ADR-008](docs/adr/ADR-008-filtered-search.md) (filtered search —
  retrospective), [ADR-010](docs/adr/ADR-010-format-evolution.md) (format
  evolution policy), [ADR-011](docs/adr/ADR-011-pq-codec.md) (PQ codec —
  retrospective). ADR-006 moved into `docs/adr/` with its siblings.

### Changed

- **`NLEmbeddingProvider` sentence embeddings are now L2-normalized**, matching
  the word-averaging fallback path (previously only the fallback normalized).
  Every vector the provider returns now has unit magnitude. **Migration:**
  indexes persisted from pre-1.5 *unnormalized* sentence vectors will rank
  differently under `DotProductDistance`/`EuclideanDistance` when queried with
  the new unit-length vectors — re-embed and rebuild those indexes, or pin to
  v1.4.x until you can. (`CosineDistance` users are unaffected.)

- **On-disk format v2.** `autoCompactionThreshold` now survives a save/load
  round-trip. Format v1 files still load — see
  [ADR-010](docs/adr/ADR-010-format-evolution.md) for the evolution policy.
- `HNSWConfiguration` rejects `m < 2` (`m == 1` yields an infinite level
  multiplier and trapped on the first `add`).
- `ProximaKit.version` now reports the actual release (was stuck at `1.0.0`);
  a consistency test and a release-workflow check keep it that way.

### Fixed
- **Critical: tombstone liveness is now identity-based.** Liveness was
  presence-based (`uuidToNode[uuid] != nil`), which breaks after re-adding an
  existing UUID: the old tombstoned slot looked live because the UUID resolves
  to the *new* node. Search could return stale vectors/metadata, entry-point
  recovery could select a disconnected tombstone (collapsing the graph), and
  `compact()` resurrected deleted vector bodies. Affected `HNSWIndex`,
  `QuantizedHNSWIndex`, and `SparseIndex`; reproduced 20/20 pre-fix and locked
  in by `TombstoneLivenessTests`.
- **Batch cosine zero-vector parity.** The batch fast path returned distance
  `0` (perfect match) for zero-magnitude vectors where scalar `CosineDistance`
  returns `1.0` (neutral) — degenerate embeddings ranked as top hits in batch
  paths. Both zero-query and zero-stored-vector now return `1.0`.
- **Store reentrancy.** `VectorStore.save()` no longer loses concurrent
  `addChunks` dirty-flag updates across its suspension point;
  `HybridVectorStore` two-leg saves can no longer persist diverged
  dense/sparse files; `removeDocument()` closed its orphan window; document-map
  writes are atomic.
- **Persistence loaders validate before trusting.** Graph indices, entry
  points, levels, and configuration ranges are checked on load, throwing typed
  `PersistenceError` instead of crashing on corrupt or hostile files.
- `QuantizedHNSWIndex.build` no longer misaligns PQ codes/metadata when the
  input contains duplicate ids; HNSW `remove()` now repairs dangling incoming
  edges; `.weightedSum` fusion validates `alpha ∈ [0, 1]`.
- `DefaultBM25Tokenizer` dropped locale-sensitive lowercasing — tokenization
  is now deterministic regardless of device locale, per its contract.
- `CoreMLEmbeddingProvider` now conforms to `EmbeddingProvider` /
  `TextEmbedder` as documented, so it plugs into `VectorStore` directly.

---

## [1.4.0] — 2026-04-19

Hybrid BM25 + dense retrieval, product quantization, the `VectorStore` document layer, two new distance metrics, and a cross-library benchmark harness (FAISS + ScaNN). The core `ProximaKit` target remains Foundation + Accelerate only — no new external dependencies.

> No v1.2/v1.3 tags were cut — all work merged to `main` between v1.1.0 and v1.4.0 first shipped in this release.

### Added
- **Product quantization (PQ).** `ProductQuantizer` — k-means-trained
  codebooks (K = 256 centroids per sub-quantizer) with asymmetric distance
  computation (ADC) — plus `QuantizedHNSWIndex`, which searches the HNSW
  graph over PQ codes instead of full vectors. Memory per vector drops from
  `d × 4` bytes to `M` bytes (e.g. 384d at M = 48: 1,536 B → 48 B, 32×).
  Codebook + index persistence via `ProductQuantizerPersistence`. Codec
  format documented retrospectively in
  [ADR-011](docs/adr/ADR-011-pq-codec.md).
- **`VectorStore` actor (ADR-006 Phase 1).** Document-level layer over
  `HNSWIndex` + `TextEmbedder`: `addChunks` / `query` / `removeDocument` /
  `save`, `ChunkMetadata` (documentId, chunkIndex, text), typed
  `VectorStoreError`, document → chunk-UUID map persisted as JSON alongside
  the index, and a dirty flag to skip redundant saves. `TextEmbedder` lives
  in core so `ProximaKit` gains no dependency on `ProximaEmbeddings`.
- **Manhattan (L1) and Hamming distance metrics**, both with
  Accelerate-optimised paths and `DistanceMetricType` serialization.
- **Actor-based `CoreMLEmbeddingProvider`** and flat-array batch-distance
  overloads, with 10K-scale batch benchmarks comparing flat-array vs
  `Vector`-array layouts across all metrics.
- **Cross-library benchmark harness (`Benchmarks/`).** Standalone SPM package
  `ProximaBench` that compares ProximaKit HNSW against FAISS HNSW and ScaNN
  on identical datasets and identical brute-force ground truth. The core
  `ProximaKit` target stays dependency-free — baselines run in Python and
  all harnesses write a flat JSON schema (see `Benchmarks/JSON_SCHEMA.md`).
  - Swift subcommands: `ground-truth` (exact k-NN via `BruteForceIndex`)
    and `hnsw` (build + timed search + recall@k against GT).
  - Python baselines under `Benchmarks/python/`: `faiss_hnsw.py`,
    `scann_hnsw.py` (auto-skips on unsupported platforms), `compare.py`
    aggregator that emits a Markdown table.
  - Datasets: SIFT1M 100K subset + MS MARCO passages 50K (MiniLM-L6-v2
    embeddings). Idempotent download scripts under `Benchmarks/datasets/`.
  - Metrics: recall@10 vs exact GT, p50/p95 query latency, QPS, build time,
    resident memory (`mach_task_basic_info` on Swift, `psutil` on Python).
- **`docs/BENCHMARKS.md` — "Cross-Library Comparison" section** with
  design rules, dataset table, metrics table, and end-to-end reproduction
  steps that call the harness binaries directly.
- **`docs/adr/ADR-005-benchmark-methodology.md`** documenting why the
  baselines live out-of-process and why `Benchmarks/` is a separate SPM
  package rather than a target of `Package.swift`.
- **CI: `.github/workflows/benchmark.yml`.** Smoke slice (SIFT1M 10K) runs
  on every PR that touches `Sources/ProximaKit/**` or the harness. Full
  slice (100K) runs nightly. Results (per-library JSON + aggregated
  `compare.md`) are uploaded as workflow artifacts.

- **Hybrid retrieval (BM25 + dense).** Three new public types in the core
  `ProximaKit` target, sibling to the existing dense-only stack:
  - `SparseIndex` — BM25 actor (`SparseVectorIndex` protocol), Okapi scoring
    with Lucene-style `log(1 + (N − df + 0.5) / (df + 0.5))` IDF, configurable
    `k1` / `b`, tombstoning + auto-compaction matching `HNSWIndex`.
  - `HybridIndex` — concurrent fan-out over a dense `VectorIndex` and a
    `SparseVectorIndex`, with `HybridFusionStrategy` = `.rrf(k:)` (default,
    `k = 60`) or `.weightedSum(alpha:)`.
  - `HybridVectorStore` — sibling of `VectorStore` with the same
    `addChunks` / `query` / `removeDocument` / `save` shape. Persists both
    legs side-by-side (`index.pxkt` + `index.pxbm`).
- `BM25Tokenizer` protocol with `DefaultBM25Tokenizer` — Unicode word-break
  segmentation + lowercasing, no NaturalLanguage dependency. Bring-your-own
  tokenizer for language-aware tokenization (e.g. Lumen's `NLTokenizer`).
- `BM25Configuration` with `k1`, `b`, `autoCompactionThreshold` knobs.
- `.pxbm` binary persistence for `SparseIndex` via an extension on
  `PersistenceEngine`. Same header / offset layout conventions as
  `.pxkt`; compacts tombstones on save.
- `docs/HYBRID.md` — hybrid retrieval design, fusion-strategy rationale,
  Lumen opt-in snippet.
- 40 new tests across `SparseIndexTests`, `DefaultBM25TokenizerTests`,
  `HybridIndexTests`, and `HybridVectorStoreTests`, including a 1K-doc BM25
  parity check against an oracle implementation and the RRF
  `top-k ⊇ (dense ∩ sparse)` invariant on constructed cases.

### Changed
- `.gitignore` now tracks `Benchmarks/` sources but ignores the on-demand
  `Benchmarks/datasets/` payloads and `Benchmarks/out/` run artifacts.
- [`docs/adr/ADR-006-lumen-integration.md`](docs/adr/ADR-006-lumen-integration.md)
  — new addendum covering the hybrid opt-in path. The v1.1 `VectorStore`
  contract is unchanged.

### Fixed
- `SparseIndexTests.testBM25ParityAgainstOracle` no longer flakes when BM25
  score ties straddle the top-k truncation boundary. Both the oracle and
  `SparseIndex` are queried with `k + 50` and the assertion walks fully
  realized score buckets until it covers the top-k window — BM25 makes no
  tie-break guarantee, so the test now verifies only what parity actually
  demands (score agreement across the top-k window).

---

## [1.1.0] — 2026-03-17

### Added
- **SIMD-accelerated batch vector operations** (`batchDotProducts`, `batchL2Distances`)
- SIMD benchmark tests comparing vDSP vs naive loop performance
- **WordPiece tokenizer** for BERT-compatible CoreML model input
- **Image search** in demo app via `VisionEmbeddingProvider`
- **Index persistence** in demo app — index survives app restart
- Xcode demo app (`Examples/ProximaDemoApp`) with SwiftUI interface
- `efSearch` slider in demo for live tuning
- User note and image input in demo app
- CONTRIBUTING.md, CHANGELOG.md, BENCHMARKS.md

### Changed
- README rewritten with ASCII architecture diagrams, feature comparison table, and performance dashboard
- Distance color thresholds adjusted for NLEmbedding quality range

---

## [1.0.0] — 2026-03-16

Initial public release of ProximaKit — pure-Swift vector search for Apple platforms.

> No tags predate v1.0.0. Pre-release development (tickets `PK-001`–`PK-013`:
> package scaffolding, the `Vector` type, distance metrics, `BruteForceIndex`,
> single- then multi-layer HNSW, compaction and recall benchmarks,
> persistence, NLEmbedding/Vision/CoreML embedding providers, and the demo
> app) merged straight to `main` with no version tags cut along the way —
> v1.0.0 is both the first tag and this changelog's earliest entry.

### Core Library (`ProximaKit`)

- **`Vector`** value type with Accelerate/vDSP-backed math (dot product, L2 distance, magnitude, normalization)
- **`DistanceMetric`** protocol with three implementations:
  - `CosineDistance` — direction-based similarity (best for text)
  - `EuclideanDistance` — straight-line distance
  - `DotProductDistance` — alignment-based (for normalized vectors)
- **`VectorIndex`** actor protocol with two implementations:
  - `HNSWIndex` — multi-layer graph search, O(log n) queries, heuristic neighbor selection (ADR-004)
  - `BruteForceIndex` — exact linear scan, O(n) queries
- **`PersistenceEngine`** — compact binary format with memory-mapped loading (50ms cold start for 10K vectors)
- **`SearchResult`** — result type with `id`, `distance`, and optional `metadata`
- **`HNSWConfiguration`** — tuning knobs: `m`, `efConstruction`, `efSearch`
- HNSW compaction: remove tombstoned vectors and reclaim memory
- Full actor isolation for thread safety (ADR-002)

### Embedding Providers (`ProximaEmbeddings`)

- **`NLEmbeddingProvider`** — Apple NaturalLanguage framework, zero setup
- **`VisionEmbeddingProvider`** — Apple Vision framework for image embeddings
- **`CoreMLEmbeddingProvider`** — bring-your-own sentence-transformer model
- **`EmbeddingProvider`** protocol for custom implementations

### Quality

- 149 tests passing across unit, integration, recall, and SIMD benchmarks
- Recall@10: 98–99% at 1K vectors, 87%+ at 10K vectors (Euclidean, random data)
- Query latency: ~104ms at 1K/384d, 50ms cold start with mmap
- GitHub Actions CI workflow
- DocC documentation catalog
- 4 Architecture Decision Records

### Platforms

- macOS 14+, iOS 17+, visionOS 1.0+
- Swift 5.9+, Apple Silicon (M1/M2/M3/M4)
- Zero external dependencies (Foundation + Accelerate only)
