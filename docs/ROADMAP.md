# ProximaKit Roadmap

**Updated:** 2026-04-23
**Current release:** v1.4.0

This document tracks planned improvements across the library, benchmarking harness, and demo experience. Items are grouped by theme, not by release, because ordering depends on dependencies and measured impact.

---

## Distance Metrics

### Completed
- Cosine distance (vDSP dot product + magnitude)
- Euclidean / L2 distance (vDSP)
- Dot product similarity
- Manhattan / L1 distance
- Hamming distance (binary vectors)

### Planned

| Metric | Use Case | Status |
|--------|----------|--------|
| Mahalanobis | Covariate-aware similarity; useful when embedding dimensions have different scales | Planned |
| Chebyshev (L∞) | Grid/game-AI pathfinding over embedded state spaces | Planned |
| Bray-Curtis | Ecological / compositional similarity (count vectors) | Planned |
| Jensen-Shannon divergence | Probability distribution comparison | Considering |

All new metrics must satisfy the `DistanceMetric` protocol and pass the existing symmetry + triangle-inequality tests before merge.

---

## Quantization & Memory Efficiency

ProximaKit currently stores all vectors as `Float32`. The next major memory reduction comes from quantization.

### INT8 Scalar Quantization

Store each vector component as a signed 8-bit integer with a per-vector scale factor. Reduces index memory by ~4×; query accuracy degrades by ~1–2% Recall@10 at typical efSearch values. An ADR will document the chosen dequantization point (query-time vs. compare-time) and the tradeoff against vDSP batch alignment requirements.

### Product Quantization (PQ)

Divide each vector into `M` sub-vectors, each quantized to a `K`-centroid codebook. Memory footprint: `N × M × log₂(K)` bits vs. `N × d × 32` bits. Enables 32–128× compression at moderate recall cost.

**Complexity note:** PQ requires a training phase (k-means over the vector population) and adds a codebook persistence format. An ADR is needed before starting implementation to agree on the API surface and codec versioning.

### Status

- INT8 scalar quantization: ADR draft in progress (ADR-007)
- PQ: design phase, no ADR yet

---

## GPU Acceleration

### Batch Index Build

Building a 100K-vector HNSW index on CPU (M-series) currently takes ~30–60 s. The bottleneck is repeated pairwise distance computation during insertion. Porting the distance kernel to a Metal compute shader would parallelize across GPU cores.

**Plan:**
1. Instrument and confirm the distance kernel is the dominant cost via `Instruments.app`.
2. Write a Metal `.metal` shader implementing the same vDSP-equivalent distance ops.
3. Add a `MetalDistanceBackend` conforming to `DistanceBackend` (new internal protocol).
4. Gate behind `#if canImport(Metal)` so the package remains buildable on Linux and CI.
5. Benchmark: target 5–10× build speedup on M2 for 100K × 128d vectors.

**Dependencies:** ADR for backend abstraction layer must be accepted first.

### Batch Embedding

`NLEmbeddingProvider` and `CoreMLEmbeddingProvider` already run in `TaskGroup` for concurrency, but embedding is still CPU-bound via CoreML. Exploring `MLComputeUnits.cpuAndNeuralEngine` to offload to the ANE for large batch inserts.

---

## Filtered Search

Support a metadata predicate in `VectorIndex.search(query:k:filter:)` that narrows the candidate set before or during ANN traversal. Two strategies:

| Strategy | Recall | Latency | Notes |
|----------|--------|---------|-------|
| Post-filter | Lower (may return < k) | Fast | Simple; degrades with high selectivity |
| Graph-aware filter | Higher | Slower build | Requires filter-aware neighbour selection in HNSW |

An ADR will evaluate both strategies against a selectivity benchmark (10%, 1%, 0.1% pass rate) before committing to the API.

---

## HNSW Graph Improvements

- **Incremental delete:** current `remove(id:)` marks nodes as deleted (tombstone). A background compaction pass to physically remove tombstoned nodes and relink the graph is deferred; it requires an ADR on compaction policy.
- **Hierarchical NSW variant with dynamic `M`:** vary the number of connections per layer based on layer height to improve recall at low `efSearch` values.
- **Serialisation versioning:** add a format version byte to `.proxima` files so future changes to the binary layout are detectable and recoverable.

---

## ADR Backlog

| ADR | Topic | Status |
|-----|-------|--------|
| ADR-006 | Lumen integration (ProximaKit as KV-store backend) | Draft (in `docs/`) |
| ADR-007 | INT8 scalar quantization: dequantization policy + codec format | In progress |
| ADR-008 | Filtered search: post-filter vs. graph-aware strategy | Not started |
| ADR-009 | Metal backend abstraction layer | Not started |
| ADR-010 | Serialisation versioning and format evolution | Not started |

---

## Demo App Evolution

The `ProximaDemoApp` (macOS SwiftUI) ships with the repo and demonstrates semantic search on 48 sample documents. Planned improvements:

| Item | Priority |
|------|----------|
| iOS / iPadOS target | High |
| CoreML model download UI — browse HuggingFace Hub, download `.mlpackage`, hot-swap embedding provider | High |
| Benchmark tab — run efSearch sweep in-app and display a recall vs. latency chart | Medium |
| Export results to CSV / JSON | Medium |
| Custom corpus loading — import a folder of `.txt` / `.md` files into the index | Medium |
| Index inspector — visualise the HNSW layer graph as a force-directed diagram | Low |

---

## Documentation & Developer Experience

Flagged in the [documentation audit](../docs/DOCUMENTATION-AUDIT.md) as out of scope for the initial documentation push but tracked here for completeness:

- CONTRIBUTING.md — polish onboarding flow, add `scripts/check-imports.sh` usage note
- CHANGELOG.md — backfill pre-v1.0 history; switch to Keep-a-Changelog format
- Demo app README — expand with CoreML model install instructions
- DocC Getting Started tutorial — interactive tutorial linked from the docc catalog

---

## Contributing

If you want to work on any of these, open an issue first to discuss the design. For quantization, GPU, and filtered-search items, an ADR with an accepted decision is a prerequisite for a PR. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).
