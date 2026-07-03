# ProximaKit Roadmap

**Updated:** 2026-07-02
**Current release:** v1.5.0

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
- Jensen-Shannon distance (sqrt of base-2 JSD; serializable, DistanceMetricType raw value 7)

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

**Reranking option (post-1.5.0):** ADC quantization error costs ~30% recall@10 on the clustered acceptance fixture. Building with `retainOriginals: true` keeps the Float32 vectors alongside the codes, and search re-scores the top quantized candidates with exact distances (`rerankDepth`, default `4·k` when originals are retained) — reranked recall@10 is asserted ≥ 0.90 in `PQRerankTests`. The trade is explicit: retention pays the full `4·d` bytes/vector again, so a retaining index has no compression story. Training is also seedable now (`PQConfiguration.seed` → byte-identical codebooks). Design in [ADR-012](adr/ADR-012-pq-reranking.md); [ADR-013](adr/ADR-013-streaming-persistence.md) sketches the on-disk-originals follow-up that would restore the memory story.

### Status

- PQ: **shipped** — `QuantizedHNSWIndex` with ADC and persistence; retrospective ADR accepted ([ADR-011](adr/ADR-011-pq-codec.md))
- PQ reranking: **shipped** — opt-in `retainOriginals` + `rerankDepth` with PQHW v2 persistence; ADR accepted ([ADR-012](adr/ADR-012-pq-reranking.md))
- INT8 scalar quantization: **shipped** — `ScalarQuantizer` + `ScalarQuantizedHNSWIndex` with persistence and acceptance tests; ADR accepted ([ADR-007](adr/ADR-007-int8-scalar-quantization.md))

---

## GPU Acceleration

### Batch Distance Kernel — v1 Shipped ([ADR-009](adr/ADR-009-metal-backend.md))

Building a 100K-vector HNSW index on CPU (M-series) currently takes ~30–60 s. The bottleneck is repeated one-query-to-N distance computation during insertion — exactly the shape where a GPU dispatch amortizes its launch overhead.

**What v1 ships:** `MetalBatchDistance`, a standalone utility computing one-query-to-N squared-L2 and cosine distances over the same flat row-major layout as the vDSP batch paths. Inline-MSL kernel (SwiftPM can't build `.metal` files portably), lazily compiled and cached; vDSP numerical parity asserted to 1e-4; automatic CPU fallback on any runtime Metal failure; clean `XCTSkip` on GPU-less CI runners.

**Insert-loop integration — benchmarked, decision NO-GO** ([ADR-009 addendum](adr/ADR-009-metal-backend.md#addendum-insert-loop-integration-benchmarked--no-go-2026-07)):

The deferred insert-loop integration was benchmarked (the "instrument first" step) via the `Benchmarks` package's new `insert-shape` and `distance-kernel` subcommands, and **rejected**. Two measured findings killed the premise:
- **The one-query-to-N batch ADR-009 bet on does not exist in the real build.** `HNSWIndex.add()` has zero batch-distance calls; distance work is serial scalar `metric.distance`, of which only 7–14 % is even one-query-to-N shaped (`searchLayer` traversal) and 86–93 % is pairwise (heuristic selection + pruning). The largest batchable unit is a single node expansion of `≤ mMax0 = 32` candidates.
- **No GPU crossover exists at any realistic N.** On an Apple M4 Max (release build), vDSP beats `MetalBatchDistance` at every N from 32 to 1,000,000 for both metrics and both dimensions (384/768); the GPU stays ~4.8× slower even at N = 1M (vDSP_mmul on the AMX coprocessor vs the GPU's per-call copy + dispatch floor). Full measured table in the ADR addendum.

Consequently `MetalBatchDistance` remains a standalone, parity-tested utility and is **not** integrated; the `DistanceBackend` protocol is **not** extracted (still no second consumer). The ADR addendum states exactly what measurement would reopen the decision (a batched-build algorithm plus a zero-copy dispatch path, or hardware without AMX-class CPU matrix acceleration).

**Still out of scope for the GPU path:**
- Per-query search latency — at efSearch-scale candidate counts, kernel-launch overhead dominates and vDSP wins (ADR-001's verdict stands for search).
- Zero-copy buffers (`makeBuffer(bytesNoCopy:)`) — would only matter if the NO-GO above were reopened.

Build-phase GPU latency is recorded in the ADR-009 addendum / the `distance-kernel` JSON, never asserted in CI (it is hardware-dependent). No GPU speedup is published in `docs/BENCHMARKS.md` because, per the measurement above, there is none to publish for this workload.

### Batch Embedding

`NLEmbeddingProvider` and `CoreMLEmbeddingProvider` already run in `TaskGroup` for concurrency, but embedding is still CPU-bound via CoreML. Exploring `MLComputeUnits.cpuAndNeuralEngine` to offload to the ANE for large batch inserts.

---

## Filtered Search

`VectorIndex.search(query:k:efSearch:filter:)` takes a `@Sendable` predicate on every index type. Strategy by index:

| Strategy | Recall | Status |
|----------|--------|--------|
| Graph-aware filter | Higher — selective filters still fill `k` | **Shipped** for `HNSWIndex`, `QuantizedHNSWIndex`, and `ScalarQuantizedHNSWIndex` — predicate applied during the layer-0 beam with the same adaptive `ef` widening formula on all three; acceptance gated by the selectivity suite (`FilteredSearchSelectivityTests`: HNSW recall@10 ≥ 0.9 with full `k` at ~10%/~1% pass rates and exact set-and-order match at ~0.1%; quantized-index recall floors at each selectivity published in `docs/BENCHMARKS.md`; a post-filter under-fill control on every index) |
| Post-filter | Lower (may return < k under selective filters) | **Shipped, and now a deliberate choice rather than a gap** — the only index still using it is `SparseIndex` (a BM25 postings scan has no `ef`-bounded beam to route through, so the graph-aware mechanism doesn't structurally apply; rationale in the ADR-008 second addendum); `BruteForceIndex` is exact under any filter; `HybridIndex` inherits graph-aware behavior on its dense leg regardless of which HNSW-family index it wraps |

Graph-aware filtering now covers every HNSW-graph index — the "extend to the quantized indexes" gap this section used to call out is closed. The post-filter decision, the `HNSWIndex` upgrade, and the quantized-index upgrade (with its measured recall table) are documented in [ADR-008](adr/ADR-008-filtered-search.md) (see both addenda).

---

## Persistence

### Streaming Persistence — WAL Incremental Saves + Paged Vectors, Both Stages Shipped ([ADR-013](adr/ADR-013-streaming-persistence.md))

Every save through the default API still rewrites the entire index snapshot — ADR-013 works out the arithmetic at ≈1.76 GB per save for a 1M × 384d index, regardless of how many vectors actually changed. **Stage 1 (the write-ahead log, "Option A") is shipped**: `HNSWIndex` now has an additive, opt-in journaled path — `open(baseURL:walURL:durability:)` / `checkpoint(baseURL:walURL:durability:)` — that appends a `.pxwal` mutation record (file-format arithmetic: ~1.6 KB per `add` at 384d, per ADR-013 — see `docs/BENCHMARKS.md`) instead of a full rewrite. The existing `save(to:)`/`load(from:)` API is untouched and byte-identical; journaling changes nothing for callers who don't opt in.

Delivered: the `.pxwal` v1 sidecar (CRC-framed records, deterministic replay via journaled HNSW levels so recovered state is asserted byte-exact, not merely valid), `.pxkt` v3 (a section table plus a snapshot-generation binding; `minSupportedVersion` stays 1), a configurable checkpoint policy and fsync dial (`.everyRecord` / `.everyBatch` / `.manual`, with the Darwin `fsync`-vs-`F_FULLFSYNC` distinction documented rather than glossed over), and recovery proven rather than merely designed — an in-process truncation sweep across every WAL byte/record boundary plus an out-of-process `SIGKILL` rig (100/100 recoveries opt-in via `PROXIMA_RUN_KILL_RIG`, a 5-iteration smoke in every CI run). Full design in `docs/ARCHITECTURE.md`; the ADR's "Stage 1 implementation notes" addendum records the built format bytes and every documented deviation (store-level `VectorStore`/`HybridVectorStore` wiring deferred; auto-compaction suppressed while a journal is attached; one narrow checkpoint crash window that surfaces as a typed error, never silent data loss).

**Stage 2 (paged, on-demand vector loading over a memory-mapped region, "Option C") is shipped.** An additive `.paged` open mode (`HNSWOpenMode`, via `HNSWIndex.open(baseURL:walURL:durability:mode:)`) serves the vector section from a read-only file mapping (`MappedVectorRegion`) instead of decoding it resident, riding the same `.pxkt` v3 bump Stage 1 already forced — `checkpoint(...)` now pads the vector section to a 16 KiB boundary so it can be mapped independently. Measured, Apple M4 Max, release: a 100,000 × 384d fixture with a 146.5 MB vector payload shows a paged open resident at 18.1 MB versus 112.3 MB for the same base opened `.resident` — 94.1 MB (64%) of the payload not resident — with paged search byte-identical to resident and no measurable resident-mode search regression (worst case +0.5%, well under the 2% bail-out threshold). `.resident` stays the default, byte-identical to before. The graph adjacency stays resident and unpaged — variable-length encoding, in-place mutation, and the traversal hot path rule it out, per the ADR's Option C analysis. Recorded follow-up: `SQHW`/`PQHW` paged originals, the fix ADR-012 deferred for PQ's 32× memory story.

---

## HNSW Graph Improvements

- **Incremental delete:** current `remove(id:)` marks nodes as deleted (tombstone). Dangling-incoming-edge repair is now O(in-degree) via a maintained reverse-adjacency map (post-1.5.0; map rebuilt on load, format unchanged).
- **Compaction — Shipped:** `compact()` is a public, synchronous API that snapshots every live node, resets storage, and re-inserts each one — physically reclaiming tombstoned slots (`count` becomes `== liveCount`) and fully relinking the graph, in O(n log n). It also runs automatically: `remove(id:)` invokes it whenever `liveCount / count` drops below `HNSWConfiguration.autoCompactionThreshold` (persisted in the format header, default `0.7`); covered by `CompactionTests`. What remains open is scheduling it off the hot path — today's compaction always runs inline on the calling task, blocking the triggering `remove(id:)` through the full rebuild, and an incremental or asynchronous/background-thread pass that avoids that stall has no design yet.
- **Hierarchical NSW variant with dynamic `M`:** vary the number of connections per layer based on layer height to improve recall at low `efSearch` values.
- **Serialisation versioning:** a magic number (`PXKT`) and format version field are already written and validated on load (`PersistenceError.unsupportedVersion`). The format-evolution policy (monotonic version bumps, N-1 reads, documented defaults, mandatory corruption tests) is settled in [ADR-010](adr/ADR-010-format-evolution.md); format v2 shipped under it.

---

## ADR Backlog

| ADR | Topic | Status |
|-----|-------|--------|
| ADR-006 | Lumen integration (ProximaKit as KV-store backend) | Draft (in `docs/adr/`) |
| ADR-007 | INT8 scalar quantization: dequantization policy + codec format | Accepted |
| ADR-008 | Filtered search: post-filter decision + graph-aware addenda, now implemented for `HNSWIndex`, `QuantizedHNSWIndex`, and `ScalarQuantizedHNSWIndex` (`SparseIndex` stays post-filter by design) | Accepted (retrospective + two addenda; first amended by a correction) |
| ADR-009 | Metal batch distance — v1 shipped a standalone build-phase utility (`MetalBatchDistance`); insert-loop integration was benchmarked and settled **NO-GO** (vDSP wins at every measured N, no crossover — see the ADR-009 addendum), so the `DistanceBackend` protocol stays unextracted | Accepted (amended) |
| ADR-010 | Serialisation format evolution policy (version field already shipped) | Accepted |
| ADR-011 | Product quantization codec format (`PQTT` / `PQHW`, ADC, K=256) | Accepted (retrospective) |
| ADR-012 | Full-precision reranking for quantized HNSW (`retainOriginals` + `rerankDepth`, PQHW v2) | Accepted |
| ADR-013 | Streaming persistence: WAL incremental saves (Stage 1) + paged vector region (Stage 2) | Accepted — both stages shipped |
| ADR-014 | Paged originals for quantized reranking — PQHW v3 section table + 16 KiB-padded originals, mapped rerank reads, and the v2→v3 upgrade path for both format families ([ADR-014](adr/ADR-014-paged-originals.md)) | Proposed — design only, not implemented |

---

## Demo App Evolution

The `ProximaDemoApp` (macOS SwiftUI) ships with the repo and demonstrates semantic search on 46 sample documents. Planned improvements:

| Item | Priority |
|------|----------|
| iOS / iPadOS / visionOS target | **Shipped** — single multiplatform SwiftUI target (compact tab layout on iPhone; split view on iPad; spatial panel on Vision Pro) |
| CoreML model download UI — browse HuggingFace Hub, download `.mlpackage`, hot-swap embedding provider | High |
| Benchmark tab — run efSearch sweep in-app and display a recall vs. latency chart | Medium |
| Export results to CSV / JSON | Medium |
| Custom corpus loading — import a folder of `.txt` / `.md` files into the index | Medium |
| Index inspector — visualise the HNSW layer graph as a force-directed diagram | Low |

---

## Documentation & Developer Experience

Flagged during the documentation audit as out of scope for the initial documentation push but tracked here for completeness:

- CONTRIBUTING.md — **shipped** (post-1.5.0): onboarding flow polish
- `scripts/check-imports.sh` guard — **shipped** (post-1.5.0): POSIX-sh import-boundary linter enforcing ProximaKit → Foundation/Accelerate/Metal/Darwin/Glibc (DocC catalog snippets excluded) and ProximaEmbeddings → additionally ProximaKit/CoreML/NaturalLanguage/Vision/CoreGraphics; wired into the `lint` CI job, with the authoritative allowlist + per-import justification in the script header
- CHANGELOG.md — backfill pre-v1.0 history (Keep-a-Changelog format already adopted)
- Demo app README — expand with CoreML model install instructions
- DocC Getting Started tutorial — **shipped** (post-1.5.0): interactive "Build On-Device Semantic Search" tutorial in the docc catalog, linked from the landing page and Getting Started
- On-device RAG example + tutorial — **shipped** (post-1.5.0): `swift run OnDeviceRAG` (`Examples/OnDeviceRAG/`) with the walkthrough in `docs/RAG-TUTORIAL.md`

---

## Contributing

If you want to work on any of these, open an issue first to discuss the design. For quantization, GPU, and filtered-search items, an ADR with an accepted decision is a prerequisite for a PR. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).
