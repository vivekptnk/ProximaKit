# Changelog

All notable changes to ProximaKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

_Nothing yet._

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
  internal, test-only, as the control — judge-ruled KEEP, since without
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

Correctness fixes from a multi-agent audit (every fix reproduced before patching, re-verified after), INT8 scalar quantization, three new distance metrics, reproducible graph construction, and a CI overhaul.

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
