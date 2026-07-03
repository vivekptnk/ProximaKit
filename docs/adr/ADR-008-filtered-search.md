# ADR-008: Post-Filter Strategy for Filtered Search

## Status
Accepted (retrospective + addendum; addendum amended — see Correction, 2026-07)

## Context
`search(query:k:efSearch:filter:)` accepts an optional `(@Sendable (UUID) -> Bool)` predicate. Two candidate strategies: post-filter (search normally, drop non-matching candidates) vs graph-aware filtering (consult the predicate during HNSW traversal so the beam expands toward matching nodes). The filter is an arbitrary closure over UUID, so it cannot be pre-indexed.

## Decision
Post-filter. HNSW runs its normal greedy descent and layer-0 beam search (ef candidates); the predicate is applied while materializing results from the layer-0 candidate set — alongside the tombstone check, before sorting and truncating to k (`HNSWIndex.search`; same shape in `QuantizedHNSWIndex`). `BruteForceIndex` applies the predicate during its full scan. `HybridIndex` passes the filter to both legs before fusion ("Applied on both legs before fusion"); `VectorStore` and `HybridVectorStore` forward it unchanged.

## Rationale
- Simplicity: no change to graph construction; one branch at result-collection time.
- No build-time cost: indexes stay filter-agnostic, so any predicate works against any existing index.
- Rejected candidates don't consume result slots — all ef layer-0 candidates are evaluated — so moderate selectivity still fills k. Callers can raise `efSearch` to compensate.

## Consequences
- Approximate indexes may return fewer than k results under selective filters: only the ef layer-0 candidates are considered, and the beam traversal itself is filter-blind. The documented contract is deliberately "Up to `k` results, sorted by distance (ascending)" (`VectorIndex.swift`, `QuantizedHNSWIndex.swift`, `VectorStore.swift`).
- `BruteForceIndex` is exact under any filter (filter applied across the full scan before top-k selection).
- Under highly selective filters (far fewer than ef matches in the beam), recall degrades and search work is spent evaluating candidates that get rejected.

## Upgrade Path
Graph-aware filtering: consult the predicate during traversal, letting the beam route through non-matching nodes while only matching nodes fill the result set. Acceptance criteria before merge: a selectivity benchmark at 10%, 1%, and 0.1% predicate pass rates, showing recall and result-count improvements over post-filter at equal latency budgets.

## Addendum: Graph-Aware Filtering Implemented for HNSWIndex

### Status
Implemented for `HNSWIndex` only. `QuantizedHNSWIndex`, `ScalarQuantizedHNSWIndex`, and `SparseIndex` keep the post-filter strategy described above; `BruteForceIndex` remains exact under any filter; `HybridIndex` inherits the upgrade on its dense leg because it forwards the filter unchanged. The public API is identical — `search(query:k:efSearch:filter:)` did not change shape.

### Strategy
- **Unfiltered queries (`filter == nil`) are untouched.** They run the original filter-blind `searchLayer` beam and result materialization — the graph-aware path adds zero work to the default case.
- **Filtered queries apply the predicate during the layer-0 beam** (`searchLayer0Filtered`). Rejected nodes — tombstoned slots (identity-based liveness, checked first so the predicate never sees deleted ids) or predicate failures — still join the candidate frontier, so the beam routes *through* them toward matching regions, but they never enter the result heap. The upper-layer greedy descent stays filter-blind: it only routes.
- **Adaptive ef widening.** The effective beam width is recomputed from the acceptance rate observed among evaluated nodes: `effectiveEf = clamp(ceil(k / rate), ef, efCap)` with add-one smoothing (`rate = (accepted + 1) / (evaluated + 1)`) and `efCap = max(ef, min(liveCount, 16 · max(ef, k)))`. When (nearly) every node passes, this stays at `ef` and the beam behaves like the unfiltered one; under selective predicates it widens toward the cap.
- **Termination.** The beam exits early only once the result heap holds `effectiveEf` *accepted* nodes and the nearest frontier candidate is farther than the worst of them. When fewer matching nodes than `effectiveEf` are reachable, the beam therefore explores the entire connected component. This is the deliberate trade: selective filters now spend latency (worst case O(liveCount) distance evaluations, the same trade hnswlib's filtered search makes) instead of returning under-filled, low-recall results. The contract stays "up to `k` results": a predicate matching fewer than `k` live vectors still returns fewer than `k`.

### Measured behavior
All numbers below are asserted thresholds from `FilteredSearchSelectivityTests` (seeded corpus: 2000 vectors, 32d, Euclidean, `m = 16`, `efConstruction = 200`, `efSearch = 50`, `k = 10`, 20 seeded queries; `SeededRandom` + `levelSeed` make data and topology deterministic). No latency figures are claimed — the acceptance comparison is made at the post-filter strategy's own candidate budget (the same `ef = 50` beam), with the graph-aware path's extra exploration bounded by the cap above.

- **~10% pass rate (200 matching):** fills `k` on every query; recall@10 vs. brute-force-filtered ground truth asserted ≥ 0.9.
- **~1% pass rate (20 matching):** fills `k` on every query; recall@10 asserted ≥ 0.9. Control assertion documenting the improvement: emulating the retired post-filter pipeline (apply the predicate to the 50 unfiltered beam candidates) yields fewer than `k` survivors on *every* seeded query — re-running the suite against the pre-addendum implementation shows it returning 0–1 results where the graph-aware path returns 10.
- **~0.1% pass rate (2 matching):** returns exactly the matching set on every query, ordered identically to the brute-force-filtered ranking (asserted as set and order equality).
- **Tombstone interplay:** a removed matching vector disappears from filtered results while the surviving match is still returned (asserted; liveness is checked before the predicate inside the beam).

### Consequences for this ADR
The "Approximate indexes may return fewer than k results under selective filters" consequence above no longer applies to `HNSWIndex`; it still applies to the quantized variants, which retain post-filter until they adopt the same beam.

## Correction (2026-07)

Two claims above overstate what is actually tested or enforced.

- **"at equal latency budgets" (Upgrade Path, above) was never met, and the addendum does not claim it was.** The shipped beam does not hold latency constant against the post-filter control — it deliberately trades latency for recall via adaptive `ef` widening, up to `efCap = max(ef, min(liveCount, 16 · max(ef, k)))` (worst case O(liveCount) distance evaluations under a highly selective filter; see "Termination" above). "Measured behavior"'s own wording is accurate about this ("No latency figures are claimed... at the post-filter strategy's own candidate budget"), but the original pre-merge acceptance criterion this addendum was meant to satisfy still reads "at equal latency budgets" and was relaxed, not met — that relaxation was never recorded as a decision. It is recorded now: the comparison is, and was always going to be, against post-filter's own `ef` budget, not an equal-latency control, because equal-latency graph-aware filtering would require capping `effectiveEf` at `ef` — which defeats the fill-`k` goal this addendum exists for.
- **"re-running the suite against the pre-addendum implementation shows it returning 0–1 results" (Measured behavior, ~1% pass rate) is more specific than what `testOnePercentControlPostFilterUnderfillsK` asserts.** The committed assertion is `XCTAssertLessThan(survivors.count, Self.k)` — under-fill vs. `k = 10`, nothing tighter. "0–1" describes what was observed on the current seeded fixture when this addendum was written; no test pins that range, so a future change to the fixture (corpus, seed, or sampling stride) could shift the survivor count anywhere in `0..<10` without turning CI red.

Both mechanisms remain correctly described above; only these two specific figures are corrected to what the code and committed tests actually guarantee.

## Second Addendum: Graph-Aware Filtering Extended to the Quantized Indexes (2026-07, M3-F31)

### Status
Implemented for `QuantizedHNSWIndex` (ADC scoring path) and `ScalarQuantizedHNSWIndex` (dequantize scoring path), porting the layer-0 filtered beam from the first addendum. The public API is unchanged — `search(query:k:efSearch:filter:)` (and the PQ `rerankDepth:` overload) did not change shape.

### Which index uses which strategy now

| Index | Filtered-search strategy | Why |
| --- | --- | --- |
| `HNSWIndex` | **Graph-aware** (first addendum) | full-precision layer-0 beam |
| `QuantizedHNSWIndex` | **Graph-aware** (this addendum) | ports the beam onto ADC scoring; composes with `rerankDepth` |
| `ScalarQuantizedHNSWIndex` | **Graph-aware** (this addendum) | ports the beam onto dequantize scoring |
| `HybridIndex` | **Graph-aware on its dense leg** | forwards the filter unchanged; inherits whichever dense index it wraps |
| `BruteForceIndex` | **Exact** | filter applied across the full scan before top-k |
| `SparseIndex` | **Post-filter (unchanged)** | see below |

`SparseIndex` stays post-filter deliberately: it is a BM25 inverted index (term → postings lists), not an HNSW graph. There is no layer-0 beam and no `ef`-bounded frontier to route *through*, so the routing mechanism this ADR describes does not apply. Its candidate set is every document whose postings match a query term — an unbounded scan, not an `ef` truncation — so applying the predicate at scoring time does not structurally under-fill `k` the way an `ef`-bounded HNSW beam does. The post-filter contract there is honest and is retained.

### Strategy (both quantized indexes)
Identical to the first addendum's mechanics, re-verified on each scoring path:
- **Unfiltered queries (`filter == nil`) are untouched.** They run the original filter-blind beam (`searchLayerADC` / `searchLayerSQ`) and materialization — the graph-aware path is a separate branch and adds zero work to the default case. Regression-pinned by `testQuantizedNilFilterMatchesAllTruePredicate` (nil path ≡ all-true graph-aware path, id- and distance-for-distance).
- **Filtered queries apply the predicate during the layer-0 beam** (`searchLayer0FilteredADC` / `searchLayer0FilteredSQ`). Liveness (identity-based tombstone check) is evaluated **before** the predicate, so the filter never sees a deleted id — matching the post-filter contract and the HNSW port. Rejected nodes still join the candidate frontier (they route) but never enter the result heap. Upper-layer descent stays filter-blind.
- **Adaptive ef widening** uses the same formula: `effectiveEf = clamp(ceil(target / rate), ef, efCap)`, add-one smoothed `rate = (accepted + 1) / (evaluated + 1)`, `efCap = max(ef, min(liveCount, 16 · max(ef, target)))`.
- **Termination** is the same bounded exploration (early-exit only once the widened beam holds `effectiveEf` *accepted* nodes farther than the nearest frontier candidate; otherwise it explores the reachable component — worst case O(liveCount) scoring evaluations).

### Rerank composition (`QuantizedHNSWIndex` only)
When the index retains originals and reranks (ADR-012), the filtered beam's adaptive **target is `rerankDepth`, not `k`** — it surfaces up to `rerankDepth` *accepted* (live, filter-passing) candidates by ADC distance, and the exact re-score consumes those. This composes with `rerankDepth` **exactly as post-filter did**: only filtered candidates count toward the depth, ranked by ADC, then re-scored on the originals and truncated to `k`. At non-selective filters (≈100% acceptance) `effectiveEf` collapses to `ef` and the reranked result is identical to the retired post-filter-then-rerank pipeline. Pinned by `testQuantizedRerankComposesWithDepthLikePostFilter`, which asserts the index's rerank output equals an independent oracle (the pure-ADC filtered pool, top-`rerankDepth` re-scored exactly, top-`k`) id-for-id with exact L2 distances.

### Measured behavior
All numbers are asserted thresholds in `FilteredSearchSelectivityTests` (seeded corpus: 2000 vectors, 32d, Euclidean, `m = 16`, `efConstruction = 200`, `efSearch = 50`, `k = 10`, 20 seeded queries; PQ `subspaceCount = 8`, `trainingIterations = 20`, training `seed` pinned; SQ metric Euclidean). Data (`SeededRandom`), topology (`levelSeed`), and PQ training are all deterministic — **debug and release produced byte-identical recall**. Recall is measured against `BruteForceIndex` with the same filter (exact ground truth). No latency figures are claimed; as in the first addendum the beam trades latency for fill under selective filters.

Recall floors are the honest measured values asserted **with margin** — they are NOT the full-precision `0.9` target. Pure-ADC PQ sits well below `0.9` (32×-lossy distances reorder near-ties); reranking recovers to the full-precision band; SQ (only ≈4×-lossy) sits just under it.

| Selectivity (matches) | PQ pure-ADC recall@10 | PQ rerank recall@10 | SQ recall@10 | Fills `k`? |
| --- | --- | --- | --- | --- |
| ~10% (200) | 0.745 (floor 0.65) | 0.995 (floor 0.95) | 1.000 (floor 0.95) | yes, all three |
| ~1% (20) | 0.870 (floor 0.78) | 1.000 (floor 0.95) | 1.000 (floor 0.95) | yes, all three |
| ~0.1% (2) | returns exactly the 2-vector matching set | exact set, brute order | exact set | returns min(k, live matches) = 2 |

*(PQ pure-ADC recall is higher at ~1% than ~10% because at ~1% only 20 vectors match, the widened beam exhausts their reachable support, and the k=10 ground truth is a subset of that recovered support; at ~10% the 200 matches leave more room for lossy-ADC reordering below the top-k cut.)*

### Under-fill contract (the upgrade)
The "approximate indexes may return fewer than `k` under selective filters" consequence no longer applies to `QuantizedHNSWIndex` or `ScalarQuantizedHNSWIndex`. `testQuantizedOnePercentControlPostFilterUnderfillsK` pins this on both: emulating the retired post-filter pipeline (the unfiltered `ef = 50` beam, predicate applied afterward) under-fills `k` on **every** seeded query at ~1% selectivity (survivors `< 10`), while the graph-aware beam fills all 10. The contract stays "up to `k` results": a predicate matching fewer than `k` live vectors still returns fewer than `k` (the ~0.1% row returns exactly 2).

### Determinism
Every threshold above is derived from a seeded, reproducible run recorded in the M3-F31 build output (`swift test -c release --filter FilteredSearchSelectivityTests`, cross-checked in debug). No system RNG is used anywhere in the fixture — `SeededRandom` for data/queries, `levelSeed` for topology, PQ `seed` for k-means.
