# ADR-007: INT8 Scalar Quantization

## Status
Accepted

## Context
The roadmap's "Quantization & Memory Efficiency" track needs a quantization option between full-precision Float32 and product quantization (ADR-011). PQ delivers 32× compression but pays for it: a k-means training phase, an L2-only ADC search path, and a visible recall cost. Many workloads want a near-lossless ~4× memory reduction that works with *any* configured metric and requires no training. INT8 scalar quantization — one signed byte per component plus one Float32 scale per vector — is the industry-standard answer (FAISS `SQ8`, hnswlib, most vector databases).

## Decision

### Symmetric per-vector scaling
Each vector is quantized independently: `scale = maxAbs / 127` where `maxAbs` is the largest absolute component (vDSP `maxmgv`), then `code = round(component / scale)` clamped to `[-127, 127]`. Dequantization is `Float(code) * scale`. Symmetric scaling (no zero-point offset) keeps the codec branch-free and sign-preserving; per-vector scales adapt to each vector's dynamic range instead of a single global scale that would be dominated by outliers. The **zero vector is handled explicitly**: `maxAbs == 0` stores `scale = 0` with all-zero codes, and decoding reproduces the exact zero vector (no division by zero ever occurs).

### Query-time reconstruction, not integer arithmetic
Search dequantizes each candidate's codes back to Float32 on the fly and applies the configured `DistanceMetric`. Compare-time integer arithmetic (int8 dot products via SIMD) was rejected for v1:
- **Metric generality**: cosine needs per-vector magnitudes, Manhattan needs absolute differences — an integer kernel per metric multiplies the code paths, while reconstruction makes every metric correct by construction.
- **The vDSP float pipeline is already optimized** (ADR-001); reconstruction is two vectorized passes (`vflt8` + `vsmul`) per candidate.
- **The ~4× memory win is unaffected**: codes dominate the footprint either way; integer compare only changes compute, and HNSW visits O(ef · M) candidates per query, not the whole index.

### Metric support — the selling point vs PQ
`ScalarQuantizedHNSWIndex.build` takes a `DistanceMetricType`, so the index supports **six of the seven serializable metrics** (cosine, euclidean, dot product, Manhattan, Chebyshev, Bray-Curtis). This is the key contrast with `QuantizedHNSWIndex`, whose ADC tables are squared-L2 only (ADR-011). The graph is built at full precision with the same metric used at search time.

`HammingDistance` is the deliberate exception: it compares exact float bit-equality, which lossy reconstruction destroys (a component stored as 1.0 dequantizes to e.g. 1.0079 and counts as a mismatch). Use the full-precision `HNSWIndex` for Hamming workloads.

### No training phase
The quantizer is stateless — its only parameter is the dimension. There is nothing to train (contrast with PQ's per-subspace k-means codebooks) and therefore **no standalone `SQTT` codec**: a `ScalarQuantizer` carries no learned state worth persisting. Only the index format exists.

### Memory math
Per vector: `d` bytes of Int8 codes + 4 bytes of Float32 scale, vs `4d` bytes of Float32. At d = 384: 388 B vs 1536 B ≈ **3.96×**. Graph overhead (~200 B/node at m = 16) is identical to the full-precision index.

### Accuracy expectation
Per-component reconstruction error is bounded by `scale / 2`. Expected search impact is **~1–2% Recall@10 degradation** vs full precision (the roadmap claim). The acceptance tests assert a bound rather than print: Recall@10 vs `BruteForceIndex` ≥ 0.95 with euclidean and ≥ 0.93 with cosine on 1000 clustered synthetic vectors, with queries striding across all clusters. Both the dataset (seeded `SeededRandom`) and the graph topology (`HNSWConfiguration.levelSeed`) are seeded, so the measurement is a reproducible constant, not a per-run sample.

### Codec
New magic **`SQHW`** (`0x53514857`), version 1, little-endian, per the ADR-010 evolution policy. 64-byte header: magic, version, dimension, nodeCount, metric (`DistanceMetricType` raw value), HNSW m / efConstruction / efSearch, maxLevel, entryPoint (−1 if nil), layerCount, `autoCompactionThreshold` as Float64 bit pattern (all-zero bits encode `nil`, following the `.pxkt` v2 precedent — closing the gap ADR-011 noted for `PQHW`), 12 reserved bytes. Then scales (nodeCount × Float32), codes (nodeCount × d Int8), UUIDs, node levels, per-layer adjacency lists, JSON-encoded metadata. Loading performs full corruption validation: bounds-checked reads, header sanity before any type precondition can trap (`m >= 2`, threshold in (0, 1), entry point and node levels in range), out-of-bounds neighbor rejection, and finite non-negative scales.

## Rationale
- ~4× memory at near-lossless recall fills the gap between Float32 and PQ's 32×/lower-recall point.
- Zero training keeps `build` a single pass — no sampling, no k-means iterations, no codebook persistence.
- Any-metric support makes it the drop-in memory upgrade for existing `HNSWIndex` users, including the cosine-heavy text-embedding default.
- Per-vector scale + round-to-nearest gives the tight `scale / 2` per-component error bound that the tests verify directly.

## Consequences
- Search reconstructs one candidate vector per visited node (O(d) and one short-lived allocation each). Acceptable for v1; an integer fast path for L2/dot can be added later without a format change.
- Like `PQHW`, removal is tombstone-only with a full reverse-edge sweep and no neighbor reconnection (full-precision vectors are discarded at build time); heavy removal workloads should rebuild. Reconnection over *dequantized* vectors is a possible upgrade — unlike PQ, the reconstruction is metric-faithful.
- A vector with one extreme outlier component squashes the rest of its components toward zero (per-vector symmetric scaling has no outlier clipping). Real embedding distributions rarely trigger this; percentile-based scale selection is a possible v2.
- `SQHW` v1 persists the metric and `autoCompactionThreshold` from day one, so the first format bump will start with no known field gaps.
