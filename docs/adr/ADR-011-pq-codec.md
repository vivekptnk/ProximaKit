# ADR-011: Product Quantization Codec Format

## Status
Accepted (retrospective)

## Context
PQ shipped in v1.4 (`ProductQuantizer` + `QuantizedHNSWIndex`): split each D-dimensional vector into M subspaces, cluster each subspace with k-means, store one centroid index per subspace instead of Float32 components. Decisions to record: on-disk codec, search-time distance scheme, codebook size, and training behavior.

## Decision

### On-disk layout
Two little-endian formats, both following the ADR-003/ADR-010 magic + version convention:
- **`PQTT`** — standalone trained quantizer (`ProductQuantizerPersistence`). 24-byte header: magic `0x50515454`, version (1), dimension, subspaceCount M, centroidsPerSubspace K (always 256), trainingIterations. Then M contiguous codebooks of K × ds Float32 (ds = dimension / M; centroid j of subspace m starts at `codebooks[m][j * ds]` — flat layout enables vDSP batch passes).
- **`PQHW`** — quantized index (`QuantizedHNSWIndexPersistence`). 56-byte header: magic `0x50514857`, version (1), dimension, nodeCount, subspaceCount, HNSW m / efConstruction / efSearch, maxLevel, entryPoint (-1 if nil), layerCount, trainingIterations, 8 reserved bytes. Then PQ codebooks (M × 256 × ds Float32), PQ codes (nodeCount × M UInt8, row-major), UUIDs (16 bytes each), node levels (Int32), per-layer adjacency lists, JSON-encoded metadata.

### Asymmetric distance computation (ADC)
The query stays full-precision; only database vectors are quantized. `buildDistanceTable(query:)` precomputes an M × K table of squared-L2 distances from each query subvector to every centroid — O(M·K·ds) once per query — then each candidate costs M table lookups (`asymmetricDistance`), O(M) with no decoding. Symmetric (code-vs-code) distance would quantize the query too, doubling quantization error; decode-then-compare would cost O(D) per candidate. This is the standard scheme from Jégou, Douze, Schmid (2011).

### K fixed at 256
`PQConfiguration` hard-codes `centroidsPerSubspace = 256` so each code is exactly one `UInt8` per subspace: no bit packing, byte-addressable codes, and a 256-entry (1 KB) table row per subspace. Loaders reject K ≠ 256 as `corruptedData`.

### Training
Per-subspace k-means with random initialization (k distinct sampled vectors, not k-means++ — converges well within the default 25 iterations at K=256). If the training set is smaller than 256 vectors, effective K = vectorCount and the last centroid is duplicated to pad the codebook. Empty clusters keep their previous centroid (no re-initialization).

## Consequences
- Memory: D × 4 bytes → M bytes per vector (384d: 1536 B → 48 B at M=48, 32×). Codebook overhead is fixed at M · K · ds · 4 = D · 1024 bytes (~384 KB at 384d), amortized across the index; HNSW graph overhead (~200 B/node at M=16) is unchanged.
- Recall tradeoff: quantization error lowers recall vs full-precision HNSW. The acceptance test (`testRecallAtClusteredData`) requires > 50% recall@10 on clustered data, with > 80% typically observed. No PQ recall numbers are published in `docs/BENCHMARKS.md` yet — that gap should close before claiming "moderate recall cost" externally.
  - *Pointer (post-1.5.0):* [ADR-012](ADR-012-pq-reranking.md) addresses this consequence — opt-in `retainOriginals` + rerank recovers the recall (asserted ≥ 0.90 reranked recall@10 vs the 0.667–0.717 pure-ADC band) at the cost of storing the originals again, and the asserted pure-ADC and reranked floors are now published in `docs/BENCHMARKS.md` ("Reranked PQ Recall").
- The codec is L2-only: encoding and distance tables use squared L2; cosine/dot ADC is not supported as shipped.
- `PQHW` removal is tombstone-only with no neighbor reconnection (the full vectors needed for the diversity heuristic were discarded at build time); heavy removal workloads should rebuild.
- `PQHW` does not persist `autoCompactionThreshold` (restored as `nil`), unlike `.pxkt` v2.
