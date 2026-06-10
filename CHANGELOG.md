# Changelog

All notable changes to ProximaKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
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
  `QuantizedHNSWIndex`, `ScalarQuantizedHNSWIndex`, and `SparseIndex` keep
  the post-filter strategy for now.
- **`PQConfiguration.seed`** — seeds the PQ k-means centroid-initialization
  draws so training is reproducible: same seed + same vectors →
  byte-identical codebooks and codes (`PQDeterminismTests`). A
  training-time knob like `HNSWConfiguration.levelSeed`: deliberately not
  persisted by the codecs and excluded from `Codable`.
- **`MetalBatchDistance` ([ADR-009](docs/adr/ADR-009-metal-backend.md), v1
  scope).** A standalone GPU utility for one-query-to-N batch distances
  (squared L2 and cosine) over the same flat row-major layout as the vDSP
  batch paths. Inline-MSL compute kernel, lazily compiled and cached; vDSP
  numerical parity asserted to 1e-4 in tests; automatic CPU (vDSP) fallback
  on any runtime Metal failure; `init?` returns `nil` where no GPU exists
  and a same-API stub compiles on non-Metal platforms. **Scope honesty: v1
  is a batch utility only — it is not wired into `HNSWIndex` build or
  search, and no speedup numbers are claimed until measured.**
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
- **[ADR-013](docs/adr/ADR-013-streaming-persistence.md) (Proposed):
  streaming persistence.** A worked design — WAL incremental saves plus a
  memory-mapped, demand-paged vector region — for saves proportional to the
  change and corpora larger than RAM. Design only; **not implemented**, and
  it makes no performance claims.

### Changed
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
