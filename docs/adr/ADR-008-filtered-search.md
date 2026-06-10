# ADR-008: Post-Filter Strategy for Filtered Search

## Status
Accepted (retrospective)

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
