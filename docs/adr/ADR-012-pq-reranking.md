# ADR-012: Full-Precision Reranking for Quantized HNSW

## Status
Accepted

## Context
`QuantizedHNSWIndex` discards the original vectors at build time and searches with asymmetric distance computation only (ADR-011). ADC quantization error compounds in graph search — `searchLayerADC` both navigates and scores with quantized distances — and the measured end-to-end cost on the CHA-91 clustered fixture is ~30% recall@10 loss vs full-precision HNSW (`PQBenchmarkTests.testQuantizedHNSWMemoryVsRecall`, observed PQ band 0.667–0.717 against ~0.97–1.0 full-precision). The standard recovery, used by every production PQ system, is reranking: overscan the quantized candidate list, then re-score the top N with exact distances before truncating to k. Reranking requires the original vectors, which v1 of the index deliberately does not keep.

## Decision

### Opt-in original-vector retention
`build(vectors:ids:metadata:dimension:hnswConfig:pqConfig:retainOriginals:)` gains a `retainOriginals: Bool = false` parameter. When `true`, the index keeps the snapshot's `[Vector]` originals alongside the PQ codes, slot-aligned with `codes`/`nodeToUUID` (both come from the same compacted `persistenceSnapshot()`, so duplicate-id replacement cannot misalign them). `remove(id:)` stays tombstone-based with identity liveness (`uuidToNode[uuid] == node`); originals for dead slots are simply never read and are compacted out at save time together with codes, UUIDs, levels, and metadata.

### Search: overscan + exact rerank
A new throwing overload `search(query:k:efSearch:rerankDepth:filter:)` is added (`rerankDepth: Int?` has no default, so existing call sites keep resolving to the unchanged non-throwing overload):

- `rerankDepth` of `nil` or `<= 0` disables reranking — the result is byte-identical to the pure-ADC path.
- `rerankDepth > 0` widens the layer-0 beam to `ef = max(efSearch ?? config.efSearch, k, rerankDepth)`, takes the top `rerankDepth` live, filter-passing ADC candidates, re-scores them with exact Euclidean distance against the retained originals, then sorts and truncates to k. Reranked results carry exact L2 distances (same scale as `HNSWIndex`); pure-ADC results keep squared-L2 ADC approximations as before.
- The existing non-throwing `search(query:k:efSearch:filter:)` reranks **by default at depth `4·k`** when originals are retained, and is unchanged (pure ADC) otherwise. Retention is an explicit opt-in to recall recovery — having paid 4·d bytes/vector for it, the default search should use it. No previously constructible index changes behavior. `4·k` is the conventional overscan starting point: deep enough that the exact ranking, not ADC ordering, decides the top k, while keeping the exact-distance work at O(k·d).
- The filter is applied before candidates count toward `rerankDepth` (post-filter per ADR-008), so selective filters don't starve the rerank pool below what the beam surfaced.

### Rerank without retention: throw
Reranking without retained originals is impossible — there is nothing full-precision to re-score against; the codes only reproduce the quantized approximation the beam already ranked by. Requesting `rerankDepth > 0` on an index without originals therefore throws the typed error `QuantizedIndexError.originalsNotRetained` rather than silently falling back to ADC. A silent no-op would hide a ~30%-recall misconfiguration behind normal-looking results — the same silent-degradation failure mode as the demo app's dimension-mismatch `[]` bug (CHA-201), which had to be converted into an actionable error after the fact. Throwing is fail-fast and costs nothing: the legacy non-throwing overload remains for callers who never ask for reranking.

### Memory math: this trades the 32× story for recall
Retention costs the full `4·d` bytes/vector again, on top of the M-byte codes. At 384d / M=48: 1536 B originals + 48 B codes ≈ 1584 B/vector — slightly *more* than plain full-precision HNSW, not 32× less. `memorySavingsRatio` accounts for retained originals (it drops below 1.0), and `originalStorageBytes` reports the retention cost explicitly. The win over plain HNSW is not memory: it is ADC-speed traversal (O(M) table lookups per candidate during the beam, exact O(d) work only on the final `rerankDepth` candidates) with exact final ranking — and the design composes with future on-disk originals (memory-mapped or demand-paged), which would restore the memory story while keeping this exact same search path.

### Persistence: PQHW v2 (ADR-010)
`qhFormatVersion` bumps 1 → 2, with `qhMinSupportedVersion = 1` (readers accept N-1; writers always write 2):

- The previously reserved header bytes at offset 48 become a UInt32 `originalsPresent` flag (0 or 1; any other value is `corruptedData`), leaving 4 reserved bytes at offset 52. Section offsets do not shift (ADR-010 rule 3).
- Iff the flag is 1, an originals section follows the metadata section: `nodeCount × dimension` Float32 (little-endian), row-major, slot-aligned with the PQ codes. Save-time compaction applies to originals exactly as to every other per-slot section.
- v1 files (zero where the flag now lives) load with the documented default `retainOriginals = false`; the originals flag is only read for `version >= 2`.
- Corruption validation parity: the flag is sanity-checked, the originals section is bounds-checked with the same `requireBytes` machinery (truncation inside the section throws `corruptedData`, never reads out of bounds), and a flag of 1 with too few trailing bytes throws.

## Consequences
- Reranked recall on the CHA-91-style clustered fixture recovers to ≥ 0.90 recall@10, asserted in `PQRerankTests` with the observed band documented next to the threshold; pure-ADC behavior and thresholds are unchanged.
- An index built with `retainOriginals: true` no longer has a compression story: memory is original + codes + graph. Callers who want 32× keep the default and accept ADC recall.
- Rerank latency adds O(rerankDepth · d) exact distance computations per query on top of the (slightly wider) beam; `rerankDepth` is the recall/latency dial.
- v2 files — even those saved without originals — are rejected by v1.5.0 and older readers with `unsupportedVersion(2)`. Per ADR-010 this is accepted: apps bundle the library version they were built with.
- Removal still degrades the graph (tombstone-only, no reconnection — ADR-011); reranking re-scores whatever candidates the degraded beam surfaces, it cannot recover vectors the beam never reached. Heavy removal workloads should still rebuild.
- `QuantizedIndexError` is a new public error enum; future quantized-index failure modes should extend it rather than overloading `PersistenceError`/`IndexError`.
