# ProximaKit Roadmap

**Updated:** 2026-06-09
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
- Chebyshev (L∞) distance
- Bray-Curtis dissimilarity
- Mahalanobis distance (covariance- or inverse-covariance-initialised; search-only — not serialisable via `DistanceMetricType`, so indices built with it cannot be persisted)

### Planned

| Metric | Use Case | Status |
|--------|----------|--------|
| Jensen-Shannon divergence | Probability distribution comparison | Considering |

All new metrics must satisfy the `DistanceMetric` protocol and pass the existing symmetry + triangle-inequality tests before merge.

---

## Quantization & Memory Efficiency

Full-precision indexes store vectors as `Float32`; both quantization tiers below ship as alternatives when memory is the constraint.

### INT8 Scalar Quantization — Shipped

Store each vector component as a signed 8-bit integer with a per-vector scale factor. Reduces vector storage by ~4× (e.g. 384d: 1,536 B → 388 B per vector) with no training phase, and — unlike PQ's L2-only ADC — works with any serialisable distance metric.

Implemented in `ScalarQuantizer` + `ScalarQuantizedHNSWIndex` (query-time reconstruction through the configured metric, binary persistence, acceptance-tested recall floors of ≥ 0.95 Recall@10 euclidean / ≥ 0.93 cosine). The dequantization-point decision (query-time vs. compare-time) and codec format are documented in [ADR-007](adr/ADR-007-int8-scalar-quantization.md).

### Product Quantization (PQ) — Shipped (v1.4.0)

Divide each vector into `M` sub-vectors, each quantized to a `K`-centroid codebook. Memory footprint: `N × M × log₂(K)` bits vs. `N × d × 32` bits. Enables 32–128× compression at moderate recall cost.

Implemented in `ProductQuantizer` + `QuantizedHNSWIndex` (asymmetric distance computation, codebook persistence in `ProductQuantizerPersistence`). The codec format is documented in [ADR-011](adr/ADR-011-pq-codec.md).

### Status

- PQ: **shipped** — `QuantizedHNSWIndex` with ADC and persistence; retrospective ADR accepted ([ADR-011](adr/ADR-011-pq-codec.md))
- INT8 scalar quantization: **shipped** — `ScalarQuantizer` + `ScalarQuantizedHNSWIndex` with persistence and acceptance tests; ADR accepted ([ADR-007](adr/ADR-007-int8-scalar-quantization.md))

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

`VectorIndex.search(query:k:efSearch:filter:)` ships with a **post-filter** strategy across all index types (HNSW, BruteForce, QuantizedHNSW, Sparse, Hybrid, and the stores). Two strategies were considered:

| Strategy | Recall | Latency | Status |
|----------|--------|---------|--------|
| Post-filter | Lower (may return < k) | Fast | **Shipped** — predicate applied during candidate traversal |
| Graph-aware filter | Higher | Slower build | Planned — requires filter-aware neighbour selection in HNSW |

The post-filter decision and the graph-aware upgrade path (with the selectivity benchmark at 10%, 1%, 0.1% pass rates as acceptance criteria) are documented in [ADR-008](adr/ADR-008-filtered-search.md).

---

## HNSW Graph Improvements

- **Incremental delete:** current `remove(id:)` marks nodes as deleted (tombstone). A background compaction pass to physically remove tombstoned nodes and relink the graph is deferred; it requires an ADR on compaction policy.
- **Hierarchical NSW variant with dynamic `M`:** vary the number of connections per layer based on layer height to improve recall at low `efSearch` values.
- **Serialisation versioning:** a magic number (`PXKT`) and format version field are already written and validated on load (`PersistenceError.unsupportedVersion`). The format-evolution policy (monotonic version bumps, N-1 reads, documented defaults, mandatory corruption tests) is settled in [ADR-010](adr/ADR-010-format-evolution.md); format v2 shipped under it.

---

## ADR Backlog

| ADR | Topic | Status |
|-----|-------|--------|
| ADR-006 | Lumen integration (ProximaKit as KV-store backend) | Draft (in `docs/adr/`) |
| ADR-007 | INT8 scalar quantization: dequantization policy + codec format | Accepted |
| ADR-008 | Filtered search: post-filter shipped; document decision + graph-aware upgrade path | Accepted (retrospective) |
| ADR-009 | Metal backend abstraction layer | Not started |
| ADR-010 | Serialisation format evolution policy (version field already shipped) | Accepted |
| ADR-011 | Product quantization codec format (`PQTT` / `PQHW`, ADC, K=256) | Accepted (retrospective) |

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

Flagged during the documentation audit as out of scope for the initial documentation push but tracked here for completeness:

- CONTRIBUTING.md — polish onboarding flow, add `scripts/check-imports.sh` usage note
- CHANGELOG.md — backfill pre-v1.0 history (Keep-a-Changelog format already adopted)
- Demo app README — expand with CoreML model install instructions
- DocC Getting Started tutorial — interactive tutorial linked from the docc catalog

---

## Contributing

If you want to work on any of these, open an issue first to discuss the design. For quantization, GPU, and filtered-search items, an ADR with an accepted decision is a prerequisite for a PR. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).
