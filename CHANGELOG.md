# Changelog

All notable changes to ProximaKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [1.4.0] — 2026-04-19

Hybrid BM25 + dense retrieval and a cross-library benchmark harness (FAISS + ScaNN). The core `ProximaKit` target remains Foundation + Accelerate only — no new external dependencies.

### Added
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
- `docs/ADR-006-lumen-integration.md` — new addendum covering the hybrid
  opt-in path. The v1.1 `VectorStore` contract is unchanged.

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
