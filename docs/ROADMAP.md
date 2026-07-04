# ProximaKit Roadmap

**Updated:** 2026-07-04
**Current release:** v1.8.0

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

**Reranking option (post-1.5.0):** ADC quantization error costs ~30% recall@10 on the clustered acceptance fixture. Building with `retainOriginals: true` keeps the Float32 vectors alongside the codes, and search re-scores the top quantized candidates with exact distances (`rerankDepth`, default `4·k` when originals are retained) — reranked recall@10 is asserted ≥ 0.90 in `PQRerankTests`. The trade used to be unconditional: **resident** retention pays the full `4·d` bytes/vector again, so a resident retaining index has no compression story. Training is also seedable now (`PQConfiguration.seed` → byte-identical codebooks). Design in [ADR-012](adr/ADR-012-pq-reranking.md).

**The "retention pays full memory" caveat is now resolved (post-1.6.1, [ADR-014](adr/ADR-014-paged-originals.md)):** an opt-in **paged** open — `QuantizedHNSWIndex.load(from:mode: .paged)` on a `PQHW` v3 base written `save(to:layout: .pagedV3)` — serves the retained originals from a read-only file mapping instead of the resident heap, restoring the 32× compression story on the vector payload while keeping rerank exact by construction (copy-on-access parity; the fault ceiling is bounded by `rerankDepth` and lands after the ADC beam, never on its critical path). `.resident` stays the default and is byte-identical to before. An existing v2 base can enable paging **without a rebuild** via the in-library `upgradeToV3(at:)` section-copy rewriter (both `.pxkt` and `PQHW` families; bit-identical payloads, checksum-verified, temp+atomic-replace). Measured (Apple M4 Max, release, 100K × 384d, 146.5 MB originals payload): paged open resident at 8.0 MB vs. 43.1 MB for the same base opened resident, flat after 50 warm reranks — see `docs/BENCHMARKS.md` for the full table and the memory-accounting caveat (macOS's compressor counts fresh anonymous pages at compressed size, so the resident-side number undercounts the true payload).

### Status

- PQ: **shipped** — `QuantizedHNSWIndex` with ADC and persistence; retrospective ADR accepted ([ADR-011](adr/ADR-011-pq-codec.md))
- PQ reranking: **shipped** — opt-in `retainOriginals` + `rerankDepth` with PQHW v2 persistence; ADR accepted ([ADR-012](adr/ADR-012-pq-reranking.md))
- PQ reranking memory story (paged originals): **shipped** — opt-in `PQHW` v3 + `.paged` open + v2→v3 migration for both format families, restoring the 32× compression story while keeping rerank exact; ADR accepted, both stages ([ADR-014](adr/ADR-014-paged-originals.md))
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
- Metal zero-copy buffers (`makeBuffer(bytesNoCopy:)`) — unrelated to the paged-access zero-copy design pass under "Persistence" below; this one would only matter if the GPU NO-GO above were reopened.

Build-phase GPU latency is recorded in the ADR-009 addendum / the `distance-kernel` JSON, never asserted in CI (it is hardware-dependent). No GPU speedup is published in `docs/BENCHMARKS.md` because, per the measurement above, there is none to publish for this workload.

### Batch Embedding — Measured, NO-GO

`NLEmbeddingProvider` and `CoreMLEmbeddingProvider` already run in `TaskGroup` for concurrency; embedding was CPU-bound via CoreML, and this section used to say "exploring `MLComputeUnits.cpuAndNeuralEngine`." That exploration is now closed with data via the new `embed-bench` subcommand (pre-declared threshold: `cpuAndNeuralEngine` must beat `cpuOnly` by ≥ 1.5× on batch throughput to justify a public `computeUnits` knob on `CoreMLEmbeddingProvider`, which today exposes none). Measured (local MiniLM `.mlpackage`, batch 512, 5 seeded reps, release): cpuOnly 117.64 texts/s vs. cpuAndNeuralEngine 117.48 — 0.9986×, dead even; ANE additionally pays ~126 ms more first-load latency than CPU (213.88 ms vs. 88.03 ms, the classic compile tax). **Decision: NO-GO** — no public `computeUnits` knob ships; `CoreMLEmbeddingProvider` keeps Core ML's own dispatch default. Honest caveat: the local model emits non-finite output through the full provider path, so this is a CoreML dispatch-throughput measurement, not a semantic-correctness one — the reopen condition names a finite-output sentence-embedding artifact re-measured on consumer hardware. Full table and caveat in `docs/BENCHMARKS.md`.

Measure-first triptych complete for this mission: Metal insert-loop integration NO-GO (above), paged-access zero-copy scoped-GO but implementation-deferred (see "Persistence" below), ANE offload NO-GO (here) — every acceleration path this mission touched shipped or deferred against a measured gate, never speculatively.

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

### Streaming Persistence — WAL Incremental Saves + Paged Vectors + Store Journaling, All Shipped ([ADR-013](adr/ADR-013-streaming-persistence.md))

Every save through the default API still rewrites the entire index snapshot — ADR-013 works out the arithmetic at ≈1.76 GB per save for a 1M × 384d index, regardless of how many vectors actually changed. **Stage 1 (the write-ahead log, "Option A") is shipped**: `HNSWIndex` now has an additive, opt-in journaled path — `open(baseURL:walURL:durability:)` / `checkpoint(baseURL:walURL:durability:)` — that appends a `.pxwal` mutation record (file-format arithmetic: ~1.6 KB per `add` at 384d, per ADR-013 — see `docs/BENCHMARKS.md`) instead of a full rewrite. The existing `save(to:)`/`load(from:)` API is untouched and byte-identical; journaling changes nothing for callers who don't opt in.

Delivered: the `.pxwal` v1 sidecar (CRC-framed records, deterministic replay via journaled HNSW levels so recovered state is asserted byte-exact, not merely valid), `.pxkt` v3 (a section table plus a snapshot-generation binding; `minSupportedVersion` stays 1), a configurable checkpoint policy and fsync dial (`.everyRecord` / `.everyBatch` / `.manual`, with the Darwin `fsync`-vs-`F_FULLFSYNC` distinction documented rather than glossed over), and recovery proven rather than merely designed — an in-process truncation sweep across every WAL byte/record boundary plus an out-of-process `SIGKILL` rig (100/100 recoveries opt-in via `PROXIMA_RUN_KILL_RIG`, a 5-iteration smoke in every CI run). Full design in `docs/ARCHITECTURE.md`; the ADR's "Stage 1 implementation notes" addendum records the built format bytes and every documented deviation (auto-compaction suppressed while a journal is attached; one narrow checkpoint crash window that surfaces as a typed error, never silent data loss). **Deviation 5 (store-level `VectorStore`/`HybridVectorStore` wiring, deferred at Stage 1) has since closed** — see "Store journaling" below.

**Stage 2 (paged, on-demand vector loading over a memory-mapped region, "Option C") is shipped.** An additive `.paged` open mode (`HNSWOpenMode`, via `HNSWIndex.open(baseURL:walURL:durability:mode:)`) serves the vector section from a read-only file mapping (`MappedVectorRegion`) instead of decoding it resident, riding the same `.pxkt` v3 bump Stage 1 already forced — `checkpoint(...)` now pads the vector section to a 16 KiB boundary so it can be mapped independently. Measured, Apple M4 Max, release: a 100,000 × 384d fixture with a 146.5 MB vector payload shows a paged open resident at 18.1 MB versus 112.3 MB for the same base opened `.resident` — 94.1 MB (64%) of the payload not resident — with paged search byte-identical to resident and no measurable resident-mode search regression (worst case +0.5%, well under the 2% bail-out threshold). `.resident` stays the default, byte-identical to before. The graph adjacency stays resident and unpaged — variable-length encoding, in-place mutation, and the traversal hot path rule it out, per the ADR's Option C analysis. Recorded follow-up **`PQHW` paged originals — shipped**: see [ADR-014](adr/ADR-014-paged-originals.md) under Quantization & Memory Efficiency above (`SQHW` was scoped out of that ADR — it retains no originals to page).

**Zero-copy paged access — measured, scoped GO, deferred to a future mission.** Both ADR-013 Stage 2's copy-on-access HNSW vector reads and ADR-014's copy-on-access PQHW rerank reads were left as a possible future optimization at design time ("Zero-copy scoped access remains a possible future optimization under ADR-005 measurement" / "zero-copy stays on the deferred list"). The new `paged-access-bench` subcommand has now measured it: the paged HNSW search path's observed paged-vs-resident overhead clears the pre-declared 5% gate (8.75–17.01% across eight warm measurements over two fixture seeds, mean ~12%), so a scoped zero-copy design pass is **scoped GO** — but the implementation itself is deferred to a future mission, gated on re-measurement on target consumer hardware. The PQHW rerank path does not clear the gate (0.00–3.28% observed) and stays copy-on-access with no planned change. Full methodology, the corrected-gate story, and the non-transferable isolation-microbench caveat are in `docs/BENCHMARKS.md`.

**Store journaling (opt-in, closes deviation 5) is shipped.** `VectorStore.open(...)` / `HybridVectorStore.open(...)` establish WAL journaling at the store level: `save()` becomes an O(1) WAL flush, `checkpoint()` is the periodic O(corpus) fold, and the multi-file crash-consistency problem (a replayed dense index racing ahead of stale sidecars) is solved by **derivation, not ordering** — a journaled `open` always rebuilds the document map, and for `HybridVectorStore` the entire WAL-less sparse BM25 leg, from the recovered index's live entries (new `HNSWIndex.liveEntries()` hook), so a doc-map or sparse entry exists iff a live dense vector exists and any on-disk sidecar cache is simply ignored rather than trusted. The historical (non-journaled) initializers, `save()`, `query()`, `addChunks`, `removeDocument`, and `loadDocumentMap()` are unchanged — CHA-107 is untouched, and PXWL v1 / `.pxkt` v3 are untouched too (this is store-layer wiring over the already-shipped index-level WAL, not a new format). Full design and the crash-semantics table in `docs/ARCHITECTURE.md` and the ADR's "Store-level journaling" addendum.

---

## HNSW Graph Improvements

- **Incremental delete:** current `remove(id:)` marks nodes as deleted (tombstone). Dangling-incoming-edge repair is now O(in-degree) via a maintained reverse-adjacency map (post-1.5.0; map rebuilt on load, format unchanged).
- **Compaction — Shipped:** `compact()` is a public, synchronous API that snapshots every live node, resets storage, and re-inserts each one — physically reclaiming tombstoned slots (`count` becomes `== liveCount`) and fully relinking the graph, in O(n log n). It also runs automatically: `remove(id:)` invokes it whenever `liveCount / count` drops below `HNSWConfiguration.autoCompactionThreshold` (persisted in the format header, default `0.7`); covered by `CompactionTests`. What remains open is scheduling it off the hot path — today's compaction always runs inline on the calling task, blocking the triggering `remove(id:)` through the full rebuild, and an incremental or asynchronous/background-thread pass that avoids that stall has no design yet.
- **Hierarchical NSW variant with dynamic `M`:** vary the number of connections per layer based on layer height to improve recall at low `efSearch` values. **Deferred, measurement-gated ([ADR-016](adr/ADR-016-dynamic-m.md)):** the paper's `mMax0 = 2m` layer-0 heuristic is already shipped, so "dynamic M" means an *upper-layer-only* schedule (adjacency is count-prefixed, so **no format change**; config travels with the file, so **no replay-determinism change**). No named consumer needs it and the shipped levers (`m`, `efSearch`) already trade for low-ef recall, so the ADR recommends **not building now** — with the exact offline harness + GO threshold (≥ +2 pp recall@10 at `ef = 16`, and a Pareto gate vs. raising uniform `m` at equal memory) that would reopen it. Leaning NO-GO, the ADR-009 register.
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
| ADR-014 | Paged originals for quantized reranking — PQHW v3 section table + 16 KiB-padded originals, paged rerank reads, and the v2→v3 upgrade path for both format families ([ADR-014](adr/ADR-014-paged-originals.md)) | Accepted — both stages shipped |
| ADR-015 | Agent-memory integration — ProximaKit as tinybrain's on-device memory substrate: store-level auto-checkpoint hook (`checkpointAutomatically:`, closes `api-ergo-01`), paged dense leg in journaled stores (M5-F49 `dense:` param), `HNSWIndex.load(from:mode:)` mirror (`api-ergo-02`), and a unified `IndexResidency` open-mode name (`api-ergo-03`); one journaled paged store meets the memory bound at the stated agent scale, with the two-tier hot/cold shape endorsed as a consumer-composed pattern over journaled HNSW + PQHW-paged (ADR-014), not new store API ([ADR-015](adr/ADR-015-agent-memory-integration.md)) | Proposed — design only; **Stages A+B shipped** (`checkpointAutomatically`, `HNSWIndex.load(from:mode:)`, store-level `dense:` residency, and the `IndexResidency` naming foundation with `HNSWOpenMode`/`PQHWOpenMode` as source-compatible typealiases); **Stage C** (the `PQHWSaveLayout` → `IndexSaveLayout` rename and broader agent-memory docs/pattern work) **pending** |
| ADR-016 | Dynamic-`M` HNSW — per-layer-height connection schedules. The paper's `mMax0 = 2m` is already shipped (`HNSWIndex.swift:71`), so the only new lever is an *upper-layer-only* schedule; count-prefixed adjacency needs no format change and file-borne config needs no replay-determinism change. **Deferred, measurement-gated:** no named consumer, a proven alternative lever (`m`/`efSearch`), and a weak literature prior — recommend not building now, with a declared offline recall threshold + Pareto-vs-uniform-`m` gate that would reopen it ([ADR-016](adr/ADR-016-dynamic-m.md)) | Proposed — design only (DEFER, leaning NO-GO) |

---

## Demo App Evolution

The `ProximaDemoApp` (macOS SwiftUI) ships with the repo and demonstrates semantic search on 46 sample documents. Planned improvements:

| Item | Priority |
|------|----------|
| iOS / iPadOS / visionOS target | **Shipped** — single multiplatform SwiftUI target (compact tab layout on iPhone; split view on iPad; spatial panel on Vision Pro) |
| CoreML model download UI — browse HuggingFace Hub, download `.mlpackage`, hot-swap embedding provider | High |
| Benchmark tab — run efSearch sweep in-app and display a recall vs. latency chart | **Shipped** — seeded `efSearch` sweep (16–256) charting recall@10 against query latency with SwiftUI Charts, recall measured against an exact `BruteForceIndex` ground truth |
| Export results to CSV / JSON | **Shipped** — CSV + JSON export of live search results (query, ids, full-precision scores, titles, categories, text; fixed field order, `.sortedKeys`), `ShareLink` on iOS/visionOS, `NSSavePanel` on macOS |
| Custom corpus loading — import a folder of `.txt` / `.md` files into the index | **Shipped** — `.txt`/`.md` files or folders, balanced security-scoped resource handling, chunk → embed → index with progress UI and per-file skip errors; imports persist across relaunch |
| Index inspector — visualise the HNSW layer graph as a force-directed diagram | **Shipped** — force-directed visualization of the real HNSW graph (Canvas/TimelineView, seeded deterministic layout, pausable), layer histogram, tap-to-select showing the node's document text and metadata, driven by the public `persistenceSnapshot()` |

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
