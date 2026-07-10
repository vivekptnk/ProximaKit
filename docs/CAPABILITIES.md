# ProximaKit Capabilities

The full capability matrix for ProximaKit, with a link to the Architecture Decision Record (ADR) behind each feature. For the quick tour, start with the [README](../README.md); for measured performance, see [BENCHMARKS.md](BENCHMARKS.md).

## Capability matrix

| Capability | Details |
|------------|---------|
| **HNSW graph search** | From-scratch multi-layer implementation — heuristic neighbour selection, tombstone deletes, auto-compaction, reproducible builds via `levelSeed` |
| **Hybrid retrieval** | BM25 + dense fusion (`HybridIndex`, `HybridVectorStore`) with Reciprocal Rank Fusion or weighted sum |
| **Product quantization** | 32× vector compression with asymmetric distance computation, plus opt-in exact reranking — retain the originals and `search` re-scores the top ADC candidates at full precision; *(new)* paged originals restore the full 32× story even with reranking on (`load(from:mode: .paged)` maps originals from disk instead of residenting them), and existing bases self-upgrade in place with `upgradeToV3(at:)` / `ProximaBench migrate` (`QuantizedHNSWIndex`, [ADR-011](adr/ADR-011-pq-codec.md), [ADR-012](adr/ADR-012-pq-reranking.md), [ADR-014](adr/ADR-014-paged-originals.md)) |
| **INT8 scalar quantization** | ~4× less vector memory, **works with any metric**, no training phase (`ScalarQuantizedHNSWIndex`, [ADR-007](adr/ADR-007-int8-scalar-quantization.md)) |
| **Filtered search** | `@Sendable` predicate on every index and store; graph-aware on `HNSWIndex`, `QuantizedHNSWIndex`, and `ScalarQuantizedHNSWIndex` — the layer-0 beam applies the filter during traversal with adaptive widening, so selective filters still fill `k` (`SparseIndex` keeps post-filter — no beam to route through) ([ADR-008](adr/ADR-008-filtered-search.md)) |
| **GPU batch distance (v1)** | `MetalBatchDistance` — standalone one-query-to-N squared-L2/cosine utility with automatic vDSP fallback. Measured **NO-GO** on wiring it into `HNSWIndex` build/search — vDSP (AMX) wins at every tested scale, no crossover ([ADR-009 addendum](adr/ADR-009-metal-backend.md)) |
| **9 distance metrics** | Cosine, Euclidean, dot product, Manhattan, Hamming, Chebyshev, Bray-Curtis, Mahalanobis, Jensen-Shannon — all vDSP-accelerated where it pays |
| **Persistence** | Versioned binary format, fast bulk loads, corruption-hardened loaders ([ADR-003](adr/ADR-003-binary-persistence.md), [ADR-010](adr/ADR-010-format-evolution.md)); opt-in WAL incremental saves + paged vector region make index mutations O(change) instead of O(corpus), *(new)* now wired all the way to `VectorStore`/`HybridVectorStore.open(...)` with derivation-based crash consistency ([ADR-013](adr/ADR-013-streaming-persistence.md)) |
| **Embedding providers** | Apple NaturalLanguage, Vision, and bring-your-own CoreML (BERT/MiniLM via WordPiece tokenizer) |
| **Concurrency** | Every index is a Swift `actor`; `Sendable` API surface, built with `StrictConcurrency` |
| **Proof** | ~600 tests, recall floors enforced in CI, cross-library benchmark harness vs FAISS/ScaNN running nightly |

## At a glance

<table>
<tr>
<td align="center" width="33%">

```
  ┌─────────────┐
  │      ◆      │
  │    ╱   ╲    │
  │   ◆─────◆   │
  │  ON-DEVICE   │
  └─────────────┘
```

**No Cloud Required**<br/>
Runs entirely on Apple Silicon.<br/>
No server, no API key, no internet.

</td>
<td align="center" width="33%">

```
  ┌─────────────┐
  │    ┌───┐    │
  │    │ 0 │    │
  │    └───┘    │
  │  PURE SWIFT │
  └─────────────┘
```

**Pure Swift**<br/>
Foundation + Accelerate only.<br/>
No C++ wrappers. No bridging.

</td>
<td align="center" width="33%">

```
  ┌─────────────┐
  │   L2: ·──·  │
  │   L1: ·─·─· │
  │   L0: ····· │
  │  HNSW BUILT │
  └─────────────┘
```

**From Scratch**<br/>
Full HNSW implementation.<br/>
Not a wrapper. Not a port.

</td>
</tr>
</table>
