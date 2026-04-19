# Changelog

All notable changes to ProximaKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

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

### Changed
- `.gitignore` now tracks `Benchmarks/` sources but ignores the on-demand
  `Benchmarks/datasets/` payloads and `Benchmarks/out/` run artifacts.

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
