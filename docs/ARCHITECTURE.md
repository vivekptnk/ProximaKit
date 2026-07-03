# ProximaKit Architecture

> Read this before modifying any core component. For decision rationale, see ADRs in `docs/adr/`.

## Module Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│            Consumer App  /  ProximaDemo (terminal demo)            │
├────────────────────────────────────────────────────────────────────┤
│  ProximaEmbeddings                                                 │
│    NLEmbeddingProvider · CoreMLEmbeddingProvider (+ WordPiece)     │
│    VisionEmbeddingProvider (image — standalone)                    │
│    EmbeddingProvider (protocol) ── refines ProximaKit.TextEmbedder │
├────────────────────────────────────────────────────────────────────┤
│  ProximaKit (core)                                                 │
│                                                                    │
│  Store/         VectorStore · HybridVectorStore · TextEmbedder     │
│  Index/         HNSWIndex · BruteForceIndex · SparseIndex (BM25)   │
│                 HybridIndex (RRF / weighted-sum fusion)            │
│                 QuantizedHNSWIndex · ScalarQuantizedHNSWIndex      │
│  Quantization/  ProductQuantizer · ScalarQuantizer                 │
│  Distance/      8 metrics + BatchDistance vDSP fast paths          │
│  Persistence/   PersistenceEngine + per-format binary codecs       │
│  Query/         SearchResult        Vector (vDSP value type)       │
│                                                                    │
│  Imports: Foundation, Accelerate ONLY                              │
└────────────────────────────────────────────────────────────────────┘
```

## Source Layout

| Directory | Contents |
|-----------|----------|
| `Sources/ProximaKit/Distance/` | `DistanceMetric` protocol, 8 metric implementations, `DistanceMetricType` (serialization), `BatchDistance` vDSP batch paths |
| `Sources/ProximaKit/Index/` | Six index types, `VectorIndex` / `SparseVectorIndex` protocols, BM25 tokenizers, internal `Heap`, quantized-index codecs |
| `Sources/ProximaKit/Quantization/` | `ProductQuantizer` (+ codec, errors), `ScalarQuantizer` |
| `Sources/ProximaKit/Persistence/` | `PersistenceEngine` (`.pxkt`), `SparseIndexPersistence` (`.pxbm`), `PersistenceError` |
| `Sources/ProximaKit/Store/` | `VectorStore`, `HybridVectorStore`, `TextEmbedder`, `ChunkMetadata`, store errors |
| `Sources/ProximaKit/Query/` | `SearchResult` |
| `Sources/ProximaEmbeddings/` | `EmbeddingProvider` protocol, NL / CoreML / Vision providers, `WordPieceTokenizer` |
| `Sources/ProximaDemo/` | Interactive terminal demo (`swift run ProximaDemo`) |

## Distance Layer

Eight `DistanceMetric` implementations: Cosine, Euclidean, DotProduct, Manhattan,
Hamming, Chebyshev, BrayCurtis, Mahalanobis.

- Seven are stateless and serializable via `DistanceMetricType` (a `UInt32` enum
  stored in index file headers; `makeMetric()` reconstructs the instance on load).
- `MahalanobisDistance` carries a `dimension × dimension` inverse-covariance
  matrix, so it has no `DistanceMetricType` case — saving an index configured
  with it throws `PersistenceError.unserializableMetric`, same as custom metrics.

`BatchDistance` computes query-vs-N distances against a flat row-major matrix
(ADR-001). Fast paths by metric:

| Metric | Path |
|--------|------|
| Cosine, DotProduct | One `vDSP_mmul` for all dot products, then post-processing |
| Euclidean | `batchL2Distances` (`vDSP_svesq` norms + `vDSP_mmul` dots) |
| Manhattan | `batchL1Distances` (`vDSP_vsub`/`vabs`/`sve` per row) |
| All others | Scalar fallback through `metric.distance(_:_:)` |

Batch and scalar paths are required to agree element-wise — including the
zero-vector convention (cosine distance to a zero vector is `1.0`, neutral, on
both paths).

## Index Layer

<p align="center">
  <img src="assets/hnsw-search.svg" alt="Animated HNSW search: greedy descent across layers, then beam search on layer 0" width="720" />
</p>

| Index | Protocol | Search | Persists as | Notes |
|-------|----------|--------|-------------|-------|
| `BruteForceIndex` | `VectorIndex` | Exact, O(n) batch scan | `.pxkt` | Ground truth / small corpora |
| `HNSWIndex` | `VectorIndex` | Approximate, O(log n) | `.pxkt` | Tombstone deletes + auto-compaction |
| `SparseIndex` | `SparseVectorIndex` | BM25 (Lucene-style IDF) | `.pxbm` | Pluggable `BM25Tokenizer`; same tombstone/compaction model |
| `HybridIndex` | — (actor) | Both legs concurrently, then fusion | per leg | RRF (default, k=60) or weighted-sum fusion |
| `QuantizedHNSWIndex` | — (actor) | HNSW graph over PQ codes, ADC | `PQHW` codec | L2-only by construction (ADR-011) |
| `ScalarQuantizedHNSWIndex` | — (actor) | HNSW graph over INT8 codes, dequantize-per-candidate | `SQHW` codec | Metric-general except Hamming (ADR-007) |

All dense indexes support **filtered search** (ADR-008): an optional
`@Sendable (UUID) -> Bool` predicate, with default overloads supplied by
protocol extension. Strategy differs by index: `HNSWIndex` applies the
predicate **during** the layer-0 beam with adaptive `ef` widening (ADR-008
addendum) — rejected nodes still route the beam toward matching regions but
never occupy result slots, so selective filters still fill `k` (acceptance
gated by `FilteredSearchSelectivityTests`). `QuantizedHNSWIndex`,
`ScalarQuantizedHNSWIndex`, and `SparseIndex` remain **post-filter** — the
beam traversal itself is filter-blind, and the predicate is applied only as
results are materialized, so a selective filter can return fewer than `k`.
`BruteForceIndex` is exact under any filter (full scan). `HybridIndex.search`
applies the filter to both legs before fusion, inheriting each leg's
strategy — graph-aware on its dense leg if that leg is an `HNSWIndex`.

`Heap` is an internal support type (beam search priority queues), not public API.

### Hybrid retrieval

`HybridIndex` fans writes out to a dense leg (`any VectorIndex`) and a sparse
leg (`any SparseVectorIndex`) under one shared UUID, queries both with
`async let`, and fuses the two rankings with a `HybridFusionStrategy`:

- `.rrf(k: 60)` — Reciprocal Rank Fusion (Cormack et al., SIGIR 2009). Scale-free; the robust default.
- `.weightedSum(alpha:)` — min-max-normalized weighted sum; `alpha` weights the dense leg, validated to `[0, 1]`.

Each leg contributes a candidate pool of `candidatePoolK` (default
`max(k * 5, 50)`) before fusion. See `docs/HYBRID.md` for the full design.

### The build-then-quantize pattern

Both quantized indexes share one construction pipeline:

1. Insert all vectors into a **full-precision** `HNSWIndex` — graph edges are
   chosen using exact distances, so graph quality is not degraded by
   quantization error.
2. Encode every vector (train a `ProductQuantizer` first for PQ; the scalar
   quantizer is stateless and needs no training).
3. `build(...)` extracts the finished graph + codes; the Float32 vectors are
   discarded by default. Search runs on compressed codes.

The two differ in how search computes distances:

- **PQ / ADC** (`QuantizedHNSWIndex`): a per-query distance table over the
  codebooks; asymmetric distance = full-precision query vs. compressed codes.
  ~48-byte codes for 384d (M=48) ≈ **32× memory reduction**, L2 only.
  **Opt-in reranking (ADR-012):** `build(retainOriginals: true)` keeps the
  Float32 vectors alongside the codes instead of discarding them. The layer-0
  beam still runs on ADC/compressed distances, but `search(rerankDepth:)`
  re-scores the top `rerankDepth` live, filter-passing candidates with exact
  Euclidean distance against those retained originals before the final sort
  — reranked recall@10 is asserted ≥ 0.90 in `PQRerankTests`. Retention
  forfeits the compression story above (originals cost the full `4·d`
  bytes/vector again); the default (`retainOriginals: false`) behaves exactly
  as described in step 3.
- **INT8** (`ScalarQuantizedHNSWIndex`): each candidate is dequantized on the
  fly and fed to the configured metric, so cosine, euclidean, dot product,
  Manhattan, Chebyshev, and Bray-Curtis all work. 384d: 1536 B → 388 B ≈
  **3.96× memory reduction**. Hamming is excluded — lossy reconstruction
  destroys bit-equality semantics.

## Quantization Layer

- `ProductQuantizer` (ADR-011): k-means-trained codebooks over `M` subspaces,
  `PQConfiguration` for shape/training knobs, ADC via per-query distance
  tables. Has its own codec (`PQTT`) so trained codebooks can be reused.
- `ScalarQuantizer` (ADR-007): stateless symmetric INT8 — one signed byte per
  component plus a per-vector Float32 scale (`scale = maxAbs / 127`). No
  training phase, no codebook to persist; handles zero vectors and subnormal
  scales explicitly.

## Persistence Layer

One engine plus per-format codecs, all little-endian binary with
magic + version headers (ADR-003, ADR-010):

| Format | Magic | Current / min version | Codec | Used by |
|--------|-------|-----------------------|-------|---------|
| `.pxkt` | `PXKT` | 2 / 1 | `PersistenceEngine` | `HNSWIndex`, `BruteForceIndex` |
| `.pxbm` | `PXBM` | 2 / 1 | `SparseIndexPersistence` | `SparseIndex` |
| quantized HNSW | `PQHW` | 2 / 1 | `QuantizedHNSWIndexPersistence` | `QuantizedHNSWIndex` |
| scalar-quantized HNSW | `SQHW` | 1 | `ScalarQuantizedHNSWIndexPersistence` | `ScalarQuantizedHNSWIndex` |
| PQ codebooks | `PQTT` | 1 | `ProductQuantizerPersistence` | `ProductQuantizer` |

Shared invariants:

- **Format versioning** (ADR-010): every header carries a version; loaders
  accept `minSupportedVersion...formatVersion` and throw
  `PersistenceError.unsupportedVersion` otherwise. v1 `.pxkt` files (which
  predate threshold serialization) still load with documented defaults.
- **Corruption hardening**: bounds-checked little-endian readers throw
  `PersistenceError.corruptedData` instead of trapping — section truncation,
  out-of-range graph indices, implausible header fields, and invalid metric
  types are all validated before any state is constructed.
- **Atomic writes**: every save goes through `Data.write(options: .atomic)`,
  so a crash mid-save never leaves a torn file.
- **Loading**: files are read with `.mappedIfSafe` to make the decode pass
  cheap, but loaded indexes are *fully resident* — floats are copied into
  Swift arrays during decode. See the 2026-06 correction in ADR-003.

## Store Layer

Document-level API over the indexes, designed for RAG pipelines (ADR-006):

- `VectorStore` — wraps an `HNSWIndex` + any `TextEmbedder`. Add text chunks
  (auto-embedded), query by text, remove by document ID. Persists to a named
  directory: `index.pxkt` + `docmap.json` (document → UUID map).
- `HybridVectorStore` — same API shape over a `HybridIndex`
  (`index.pxkt` + `index.pxbm` + `hybrid.json`). Deliberately a sibling, not a
  subclass: the `VectorStore` contract stays frozen and hybrid is opt-in.
- `TextEmbedder` — minimal embedding protocol defined in ProximaKit so stores
  have no dependency on ProximaEmbeddings; `EmbeddingProvider` refines it, so
  every provider plugs in directly.

Compound operations (`addChunks`, `removeDocument`, `save`) are **serialized
through an internal operation chain**, and `save()` snapshots a monotonic
mutation-generation counter — the store is only marked clean if no mutation
landed while the write was in flight. This closes the actor-reentrancy
lost-update window without blocking reads.

## Data Flow

```
Index   : text ─▶ TextEmbedder.embed() ─▶ Vector ─▶ index.add(vector, id:)
Query   : text ─▶ embed() ─▶ Vector ─▶ index.search(query:k:) ─▶ [SearchResult]
Hybrid  : add/search fan out to dense + sparse legs ─▶ fuse (RRF | weighted)
Quantize: vectors ─▶ full-precision HNSW build ─▶ encode ─▶ drop Float32s (or retain, ADR-012)
Persist : actor snapshot ─▶ codec ─▶ atomic write ─▶ load() ─▶ new actor
```

## Module Rules

| Module | Can Import | Cannot Import |
|--------|-----------|---------------|
| ProximaKit | Foundation, Accelerate | ProximaEmbeddings, UIKit, SwiftUI |
| ProximaEmbeddings | ProximaKit, CoreML, NaturalLanguage, Vision | UIKit, SwiftUI |
| ProximaDemo | Everything | — |

## Concurrency Model (ADR-002)

- Every index and store is an `actor`. Reads queue, writes serialize.
- Immutable configuration is readable without data races; `HNSWIndex` and
  the quantized indexes mark `dimension`/`configuration` (and, on
  `ScalarQuantizedHNSWIndex`, `metricType`) as `nonisolated` for
  `await`-free access.
- `Vector`, `SearchResult`, snapshots, configurations, and both quantizers are
  `Sendable` value types; search filters are `@Sendable`.
- Actor *reentrancy* is handled explicitly where it matters: stores serialize
  compound operations and gate `save()` on a mutation-generation counter
  (see Store Layer above).
- Batch embedding uses `TaskGroup`; `HybridIndex` queries its legs with
  `async let` so neither leg waits on the other's queue.

## Reproducibility

`HNSWConfiguration.levelSeed: UInt64?` seeds layer assignment (SplitMix64).
Set it for bit-reproducible graph construction in tests and benchmarks; leave
`nil` for production. The seed is a build-time knob and is not persisted.

## Performance-Critical Paths

1. Distance computation → vDSP batch ops, single `vDSP_mmul` for dot products (ADR-001)
2. HNSW neighbor selection → heuristic algorithm (ADR-004)
3. PQ search → per-query ADC distance tables, no decode in the inner loop (ADR-011)
4. Beam search → internal min/max `Heap` on layer 0
5. Cold start → `.mappedIfSafe` read + bounds-checked single-pass decode (ADR-003)

## Extension Points

| Add a... | Location | Conform to |
|----------|----------|-----------|
| Distance metric | `Sources/ProximaKit/Distance/` | `DistanceMetric` (add a `DistanceMetricType` case if stateless, so it persists) |
| Dense index | `Sources/ProximaKit/Index/` | `VectorIndex` (actor) |
| Sparse index | `Sources/ProximaKit/Index/` | `SparseVectorIndex` (actor) |
| BM25 tokenizer | `Sources/ProximaKit/Index/` | `BM25Tokenizer` |
| Embedding provider | `Sources/ProximaEmbeddings/` | `EmbeddingProvider` |
| Store embedder | anywhere (no ProximaEmbeddings needed) | `TextEmbedder` |
