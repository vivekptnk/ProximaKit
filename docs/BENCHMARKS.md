# ProximaKit Benchmarks — Methodology and Results

This document describes how ProximaKit's performance numbers are measured, what they mean, and how to reproduce them.

---

## Test Environment

| Parameter | Value |
|-----------|-------|
| Hardware | Apple Silicon (M-series) |
| OS | macOS 14+ |
| Swift | 5.9+ |
| Build config | Release (`-c release`) for timing, Debug for correctness |
| Vector precision | Float32 |

All benchmarks run on-device with no network calls.

---

## Recall Benchmarks

**What recall measures:** The fraction of true top-k neighbors (from exact brute-force search) that HNSW also returns. Recall@10 = 1.0 means HNSW found the exact same 10 nearest neighbors as brute force.

### Methodology

1. Generate `N` random vectors of dimension `d` with components in `[-1, 1]`.
2. Insert all vectors into both `HNSWIndex` and `BruteForceIndex` with identical IDs.
3. For 20 random query vectors:
   - Run brute-force search (k=10) to get ground truth IDs.
   - Run HNSW search (k=10) to get approximate IDs.
   - Recall = |intersection| / 10.
4. Report the mean recall across all queries.

### Configuration

```swift
HNSWConfiguration(m: 16, efConstruction: 200, efSearch: variable)
```

### Results

| Dataset | Dimension | efSearch | Recall@10 | Threshold |
|---------|-----------|----------|-----------|-----------|
| 1K vectors | 128d | 50 | >90% | Pass: >90% |
| 5K vectors | 64d | 50 | >90% | Pass: >90% |
| 10K vectors | 128d | 100 | >82% | Pass: >82% |

### efSearch Sweep (1K vectors, 32d, Euclidean)

| efSearch | Recall@10 | Note |
|----------|-----------|------|
| 10 | ~85% | Fastest, lowest accuracy |
| 50 | ~95% | Default |
| 100 | ~98% | High accuracy |
| 200 | ~99% | Near-exact |

Recall is monotonically non-decreasing with higher `efSearch`.

### Important Note on Random Data

These benchmarks use random uniform vectors. With cosine distance in high dimensions, random vectors become nearly equidistant (curse of dimensionality), degrading recall. Real embedding vectors (BERT, NLEmbedding, etc.) have meaningful geometric structure and achieve significantly higher recall — the PRD target of >95% recall@10 applies to real embeddings.

**Run the recall benchmarks:**

```bash
swift test --filter RecallBenchmarkTests
```

### Real-Embedding Recall (NLEmbedding Sentences)

Real embedding vectors have meaningful geometric structure unlike random vectors. Using Apple's `NLEmbedding.sentenceEmbedding(for: .english)` (512d), HNSW achieves perfect or near-perfect recall.

**Corpus:** 120 diverse English sentences across 8 categories (technology, science, food, sports, nature, history, daily life, geography). 115 successfully embedded by NLEmbedding.

**Query set:** 20 novel sentences not in the corpus, testing cross-sentence semantic retrieval.

| Metric | efSearch | Recall@10 | Threshold |
|--------|----------|-----------|-----------|
| Cosine | 50 | 100% | Pass: >95% (PRD) |
| Euclidean | 50 | 100% | Pass: >95% |

#### efSearch Sweep (115 sentences, 512d, Cosine)

| efSearch | Recall@10 |
|----------|-----------|
| 10 | 100% |
| 30 | 100% |
| 50 | 100% |
| 100 | 100% |
| 200 | 100% |

With real embeddings, even low efSearch values achieve perfect recall on this corpus size. This confirms the PRD target of >95% recall@10 is met with margin.

#### Semantic Coherence

Query: "Programming languages compile source code"

| Rank | Distance | Result |
|------|----------|--------|
| 1 | 0.2673 | The compiler transforms source code into machine instructions |
| 2 | 0.3703 | Version control systems track changes in source code |
| 3 | 0.4129 | The database engine optimizes query execution plans |

Top results are semantically relevant technology sentences, validating that HNSW preserves the embedding space's semantic structure.

**Run the real-embedding benchmarks:**

```bash
swift test --filter RealEmbeddingRecallTests
```

---

## Concurrency Stress Tests

**What it measures:** Actor isolation correctness under concurrent read/write contention. Validates that `HNSWIndex` and `BruteForceIndex` (both Swift actors) handle mixed workloads without deadlocks, priority inversions, or data races.

### Test Matrix

| Test | Operations | Concurrent Tasks | Index |
|------|-----------|-----------------|-------|
| Concurrent adds | 200 adds | 200 | HNSW, BruteForce |
| Add + remove | 100 adds + 30 removes | 130 | HNSW |
| Add + search | 50 adds + 50 searches | 100 | HNSW |
| Mixed workload | 400 adds + 50 removes + 600 searches | 1050 | HNSW, BruteForce |
| Compact during search | 1 compact + 20 searches | 21 | HNSW |
| Search throughput | 500 queries (50 readers × 10 queries) | 50 | HNSW |

### Results

- All 8 tests pass consistently
- No data races detected (actor isolation enforced at compile time)
- Concurrent search throughput: ~1900 QPS on 500-vector index (64d, debug build)
- Mixed workload (1050 ops): completes in <1s

**Run the concurrency tests:**

```bash
swift test --filter ConcurrencyStressTests
```

---

## Query Latency

**What it measures:** Wall-clock time for a single `search(query:k:)` call on a pre-built index.

### Methodology

1. Build an index with `N` vectors of dimension `d`.
2. Use XCTest's `measure {}` block (10 iterations, first discarded as warmup).
3. Report median latency.

### Configuration

```swift
HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
DistanceMetric: CosineDistance
```

### Results

| Dataset | Dimension | Median Latency |
|---------|-----------|----------------|
| 1K vectors | 384d | ~104ms (debug-build XCTest harness incl. actor hop — NOT release latency; release-mode p50/p95 come from the nightly cross-library harness) |

**Cold start (mmap load):** ~50ms for a persisted 10K-vector index.

**Run the latency benchmark:**

```bash
swift test --filter testQueryLatency
```

---

## SIMD / Accelerate Benchmarks

**What it measures:** Speedup of vDSP-accelerated vector operations over naive Swift loops.

### Methodology

1. Generate random vectors of dimension `d`.
2. Run `N` iterations of each operation with both naive (manual loop) and vDSP implementations.
3. Measure wall-clock time using `CFAbsoluteTimeGetCurrent()`.
4. Verify correctness: vDSP results must match naive results within `1e-3` tolerance.
5. Report speedup ratio: `naive_time / vdsp_time`.

### Operations Tested

| Operation | Scale | Iterations | Speedup Threshold |
|-----------|-------|------------|-------------------|
| Dot product | 384d, single pair | 10,000 | >0.5x (competitive) |
| Batch dot products | 1K x 384d | 100 | >1.0x |
| Batch L2 distance | 1K x 384d | 100 | >1.0x |
| Batch L2 distance | 10K x 384d | 10 | >1.0x |

### Why Single Dot Product Threshold Is Low

At 384 dimensions, a single dot product is already fast in a naive loop (~microseconds). The vDSP function call overhead can offset SIMD gains at this scale. The real benefit of Accelerate appears in **batch operations** where `vDSP_mmul` processes entire matrices, achieving substantial speedups.

### Correctness Verification

Every benchmark test verifies that vDSP output matches naive output:

```swift
XCTAssertEqual(simdResult, naiveResult, accuracy: 1e-3,
               "vDSP dot product must match naive loop")
```

**Run the SIMD benchmarks:**

```bash
swift test --filter SIMDBenchmarkTests
```

---

## Persistence Benchmarks

### Save/Load Roundtrip

The `PersistenceEngine` uses a compact binary format with bulk single-read loading (`.mappedIfSafe` speeds the read; the loaded index is fully memory-resident).

| Operation | Characteristic |
|-----------|---------------|
| Format | Custom binary (header + Float32 vectors + adjacency lists + JSON metadata) |
| Save | Sequential write, ~O(n) |
| Load (mmap) | ~50ms cold start regardless of index size |
| Roundtrip | Exact binary match (save → load → save produces identical bytes) |

### Format Details

See [ADR-003: Binary Persistence](adr/ADR-003-binary-persistence.md) for the format specification and the rationale for choosing custom binary over JSON.

| Format | File Size (10K, 384d) | Load Time |
|--------|----------------------|-----------|
| JSON (rejected) | ~60 MB | ~3s |
| Custom binary | ~58 MB | ~50ms (mmap) |

### Incremental Saves (WAL, opt-in) — ADR-013 Stage 1

The Save/Load Roundtrip numbers above describe the default `save(to:)` /
`load(from:)` path, which is **unchanged**: every save still serializes and
rewrites the full snapshot, and every load is fully resident. `HNSWIndex`
additionally exposes an opt-in journaled path
(`open(baseURL:walURL:durability:)` / `checkpoint(baseURL:walURL:durability:)`,
see `docs/ARCHITECTURE.md`) that appends one `.pxwal` record per `add`/`remove`
instead of rewriting the file.

No benchmark harness has timed the journaled path yet — nothing below is a
measurement. The only number is **file-format arithmetic**, derived from the
record layout in `WALFormat.swift` (ADR-013), not a run:

| Quantity | Arithmetic | Bytes |
|---|---|---|
| `add` payload, 384d, no metadata | opcode (1) + UUID (16) + level `Int32` (4) + vector (384 × 4) + metadata-length `UInt32` (4) | 1 + 16 + 4 + 1,536 + 4 = **1,561** |
| Record frame overhead | `[payloadLength: UInt32][crc32: UInt32]` | 8 |
| **Total per journaled `add`, 384d** | payload + frame | **1,569 bytes** |

That is arithmetic against a fixed record shape, not a measured write
latency, checkpoint cost, or replay cost — none of those have been run under
the [ADR-005](adr/ADR-005-benchmark-methodology.md) harness. Do not read the
byte count above as a throughput or latency claim.

---

## INT8 Scalar Quantization

**What it measures:** Memory footprint and accuracy floor of `ScalarQuantizedHNSWIndex` (one signed byte per component + one Float32 scale per vector) versus full-precision Float32 storage. See [ADR-007](adr/ADR-007-int8-scalar-quantization.md) for the codec design.

### Memory Math (structural, not measured)

Per-vector storage is `d × 1` byte of codes plus a 4-byte scale, versus `d × 4` bytes at full precision:

| Dimension | Float32 | INT8 codes + scale | Ratio |
|-----------|---------|--------------------|-------|
| 128 | 512 B | 132 B | 3.88× |
| 384 | 1,536 B | 388 B | 3.96× |
| 768 | 3,072 B | 772 B | 3.98× |

The ratio approaches 4× as dimension grows. `ScalarQuantizedHNSWIndex` exposes the exact arithmetic at runtime via `codeStorageBytes`, `equivalentFullPrecisionBytes`, and `memorySavingsRatio`. Graph adjacency lists are identical in both representations, so the savings apply to vector storage only.

### Accuracy (acceptance-tested)

Recall floors are enforced by the test suite — `ScalarQuantizedHNSWIndexTests` compares against `BruteForceIndex` ground truth on 1,000 clustered 64d vectors (k=10, efSearch=250, seeded RNG):

| Metric | Recall@10 floor | Status |
|--------|-----------------|--------|
| Euclidean | ≥ 0.95 | Acceptance-tested |
| Cosine | ≥ 0.93 | Acceptance-tested |

These are asserted lower bounds, not point measurements — runs that dip below fail CI. Unlike PQ's L2-only ADC path, scalar quantization searches through the configured `DistanceMetric` (any serialisable metric), so the floors above are per-metric guarantees rather than a single L2 number.

**Run the scalar-quantization tests:**

```bash
swift test --filter ScalarQuant
```

---

## Reranked PQ Recall

**What it measures:** How much of the recall lost to PQ quantization error is recovered by full-precision reranking ([ADR-012](adr/ADR-012-pq-reranking.md)): `QuantizedHNSWIndex` built with `retainOriginals: true` re-scores the top `rerankDepth` ADC candidates with exact Euclidean distance before truncating to k.

All numbers below are **asserted test thresholds**, not point measurements — runs that dip below fail CI. Fixture (`PQRerankTests`): 1,000 clustered 64d vectors in 10 clusters, M = 16 subspaces, k = 10, 30 queries, `efSearch = 200`, `rerankDepth = 4·k`, brute-force Euclidean ground truth. Data, graph topology, and queries are pinned (`SeededRandom` + `levelSeed`).

| Configuration | Recall@10 | Asserted bound |
|---------------|-----------|----------------|
| Pure ADC (no originals) | observed band 0.667–0.717 (`PQBenchmarkTests`) | ≥ 0.55 floor |
| Pure ADC, same fixture shape (`PQRerankTests`) | observed 0.667–0.730 over 5 local runs | — (baseline for the margin assertion) |
| Reranked, depth 4·k | observed 0.990–1.000 over 5 local runs | **≥ 0.90 absolute**, and ≥ pure-ADC + 0.15 |

The observed bands are documented in the test sources next to the thresholds; the asserted bounds sit deliberately below them with margin. No rerank latency figures are published — reranking adds O(`rerankDepth` · d) exact distance computations per query, and that cost has not been measured under the [ADR-005](adr/ADR-005-benchmark-methodology.md) harness yet.

**Memory honesty:** retaining originals pays the full `4·d` bytes/vector again on top of the codes, so a reranking index has *no* compression win — `memorySavingsRatio` drops below 1.0 and `originalStorageBytes` reports the cost. Reranking trades PQ's 32× memory story for recall (ADR-012).

### PQ Training Determinism

`PQConfiguration.seed` pins the k-means centroid-initialization draws: the same seed and training vectors produce **byte-identical codebooks and codes**, asserted in `PQDeterminismTests`. The seed is a training-time knob only (not persisted, mirroring `levelSeed`). The CHA-91 memory-vs-recall fixture (`PQBenchmarkTests.testQuantizedHNSWMemoryVsRecall`) now seeds data, graph topology, *and* PQ training, so it measures a single deterministic value (inside the historical 0.667–0.717 band) instead of a band.

**Run the rerank and determinism tests:**

```bash
swift test --filter PQRerankTests
swift test --filter PQDeterminismTests
```

---

## Filtered Search Recall — Graph-Aware Beam on Quantized Indexes

**What it measures:** Recall@10 of `QuantizedHNSWIndex` (PQ, both pure-ADC and reranked) and `ScalarQuantizedHNSWIndex` (SQ) filtered search against `BruteForceIndex`-filtered ground truth, at three predicate selectivities, now that both indexes have adopted the graph-aware layer-0 beam `HNSWIndex` shipped first ([ADR-008](adr/ADR-008-filtered-search.md) addendum → second addendum).

All numbers below are **asserted test thresholds** (the floors) alongside the **observed** point values from a seeded, reproducible run — not point measurements presented as guarantees. Fixture (`FilteredSearchSelectivityTests`): 2,000 vectors, 32d, Euclidean, `m = 16`, `efConstruction = 200`, `efSearch = 50`, `k = 10`, 20 seeded queries (`SeededRandom` data + `levelSeed` topology); PQ `subspaceCount = 8`, `trainingIterations = 20`, training `seed` pinned; SQ metric Euclidean. Debug and release runs produced byte-identical recall.

| Selectivity (live matches) | PQ pure-ADC recall@10 | PQ rerank recall@10 | SQ recall@10 | Fills `k`? |
|---|---|---|---|---|
| ~10% (200) | 0.745 (asserted floor ≥ 0.65) | 0.995 (asserted floor ≥ 0.95) | 1.000 (asserted floor ≥ 0.95) | yes, all three |
| ~1% (20) | 0.870 (asserted floor ≥ 0.78) | 1.000 (asserted floor ≥ 0.95) | 1.000 (asserted floor ≥ 0.95) | yes, all three |
| ~0.1% (2 matches) | exact 2-vector matching set | exact set, brute-force order | exact set | returns `min(k, live matches)` = 2 |

The floors sit deliberately below the observed values, and below `HNSWIndex`'s full-precision 0.9 recall target: pure-ADC PQ distances are 32×-lossy and reorder near-ties, so its floor is set well under that band; reranking (ADR-012) recovers to the full-precision band; SQ (only ~4×-lossy) sits just under it.

**Under-fill control (the regression this upgrade fixes):** emulating the retired post-filter pipeline (predicate applied after the unfiltered `ef = 50` beam) under-fills `k` on every seeded query at ~1% selectivity for both quantized indexes (`testQuantizedOnePercentControlPostFilterUnderfillsK`), while the graph-aware beam fills all 10 on the same fixture.

No latency figures are published here or in the ADR: the graph-aware beam trades latency for fill under selective filters via adaptive `ef` widening (bounded by `efCap`), and no equal-latency comparison against post-filter exists (see the ADR-008 Correction). Mechanism details and the `HNSWIndex`-only first-addendum numbers are in [ADR-008](adr/ADR-008-filtered-search.md).

**Run the filtered-search selectivity tests:**

```bash
swift test --filter FilteredSearchSelectivityTests
```

---

## Compaction

When vectors are removed from `HNSWIndex`, they are tombstoned (marked deleted but still occupying memory). The `compact()` method reclaims this space.

| Scenario | Before Compact | After Compact |
|----------|---------------|---------------|
| 100 vectors, 30 removed | `count=100, liveCount=70` | `count=70, liveCount=70` |
| Search correctness | Excludes tombstones | Verified: no removed IDs in results |

---

## How to Run All Benchmarks

```bash
# Recall benchmarks — random vectors (2-5 minutes)
swift test --filter RecallBenchmarkTests

# Recall benchmarks — real NLEmbedding sentences (~5 seconds)
swift test --filter RealEmbeddingRecallTests

# Concurrency stress tests (~5 seconds)
swift test --filter ConcurrencyStressTests

# SIMD benchmarks (~30 seconds)
swift test --filter SIMDBenchmarkTests

# Query latency (~10 seconds)
swift test --filter testQueryLatency

# Full test suite
swift test
```

For more accurate timing, build in release mode:

```bash
swift build -c release
swift test -c release --filter SIMDBenchmarkTests
```

---

## Interpreting Results

- **Recall numbers** will vary slightly between runs due to random vector generation. The thresholds in tests account for this variance.
- **Latency numbers** depend on hardware. Apple Silicon M1 is the baseline; M2/M3/M4 will be faster.
- **Speedup ratios** depend on dimension and batch size. Higher dimensions and larger batches favor vDSP more strongly.
- All metrics are measured with **no other significant processes** running. Background activity will add noise.

---

## Cross-Library Comparison

ProximaKit HNSW vs. FAISS HNSW vs. ScaNN on identical datasets and identical ground truth. Full methodology and a reproducible harness live under [`Benchmarks/`](../Benchmarks/README.md).

### Design rules

1. **Identical ground truth.** Every library is evaluated against the same exact k-NN ground truth, produced once by `ProximaBench ground-truth` (brute force) and loaded verbatim by every harness. No library computes recall against its own approximate neighbors.
2. **No FAISS/ScaNN dependency in core.** The baseline libraries run in Python and write a JSON document. The Swift harness also writes JSON. The aggregator globs the output directory and builds a table.
3. **Single-threaded search.** FAISS is pinned to 1 OMP thread; the Swift harness runs queries sequentially. Multi-threaded numbers are a separate study.
4. **Release mode.** `swift build -c release --package-path Benchmarks` for the Swift harness.
5. **Results are reported honestly, including when ProximaKit loses.** Credibility is reproducibility, not winning a single axis.

### Datasets

| Dataset | Vectors | Dimension | Metric | Source |
|---------|---------|-----------|--------|--------|
| `sift-1m-100k` | 100,000 | 128 | L2 | SIFT1M first 100K (INRIA TEXMEX) |
| `ms-marco-50k` | 50,000 | 384 | Cosine | MS MARCO passages first 50K, embedded with MiniLM-L6-v2 |

CI runs a 10K smoke slice of `sift-1m` on every PR that touches `Sources/ProximaKit/**`. Nightly CI runs the full 100K slice.

### Metrics

| Metric | What it measures |
|--------|------------------|
| **Build time (s)** | Wall-clock seconds to insert every base vector into the index. |
| **p50 / p95 latency (ms)** | Per-query wall-clock latency, single-threaded. |
| **QPS** | `1000 / mean_latency_ms`. |
| **Recall@k** | Fraction of ground-truth top-k that the library returned. Ground truth is brute force. |
| **Resident memory (MB)** | Process RSS immediately after build, before any queries run. Swift uses `mach_task_basic_info`; Python uses `psutil`. |

### Reproducing

```bash
./Benchmarks/datasets/download_sift1m.sh
swift build -c release --package-path Benchmarks
mkdir -p out

# Ground truth (once)
./Benchmarks/.build/release/ProximaBench ground-truth \
    --base    Benchmarks/datasets/sift-1m/sift_base.fvecs \
    --queries Benchmarks/datasets/sift-1m/sift_query.fvecs \
    --size 100000 --query-count 1000 --k 10 --metric l2 \
    --dataset sift-1m-100k \
    --out out/GroundTruth__sift-1m-100k__k10.json

# ProximaKit
./Benchmarks/.build/release/ProximaBench hnsw \
    --base    Benchmarks/datasets/sift-1m/sift_base.fvecs \
    --queries Benchmarks/datasets/sift-1m/sift_query.fvecs \
    --gt      out/GroundTruth__sift-1m-100k__k10.json \
    --size 100000 --query-count 1000 \
    --dataset sift-1m-100k \
    --k 10 --m 16 --efc 200 --ef 50 --metric l2 \
    --out out/ProximaKit__sift-1m-100k__hnsw__ef50.json

# FAISS
python -m pip install -r Benchmarks/python/requirements.txt
python Benchmarks/python/faiss_hnsw.py \
    --base Benchmarks/datasets/sift-1m/sift_base.fvecs \
    --queries Benchmarks/datasets/sift-1m/sift_query.fvecs \
    --gt   out/GroundTruth__sift-1m-100k__k10.json \
    --size 100000 --query-count 1000 \
    --dataset sift-1m-100k \
    --k 10 --m 16 --efc 200 --ef 50 --metric l2 \
    --out out/FAISS__sift-1m-100k__hnsw__ef50.json

# Aggregate
python Benchmarks/python/compare.py --in out/ --out out/compare.md
```

### Results

Cross-library numbers are **generated, not hand-written**: the nightly `benchmark` workflow (`.github/workflows/benchmark.yml`) runs the full 100K slice and uploads per-library JSON plus the aggregated `compare.md` as workflow artifacts. The latest green nightly run on `main` is the canonical source — no numbers are copied into this document, by design (they would go stale the next time any library updates).

Example table shape (filled in from the CI JSON artifacts):

| Library | Params | Build (s) | p50 (ms) | p95 (ms) | QPS | Recall@10 | RSS (MB) |
|---------|--------|-----------|----------|----------|-----|-----------|----------|
| ProximaKit | HNSW M=16 efC=200 ef=50 | _N_ | _N_ | _N_ | _N_ | _N_ | _N_ |
| FAISS      | HNSW M=16 efC=200 ef=50 | _N_ | _N_ | _N_ | _N_ | _N_ | _N_ |
| ScaNN      | tree-AH leaves=100 search=10 reorder=100 | _N_ | _N_ | _N_ | _N_ | _N_ | _N_ |

Actual numbers are auto-generated into `out/compare.md` by `Benchmarks/python/compare.py`. See [ADR-005: Cross-Library Benchmark Methodology](adr/ADR-005-benchmark-methodology.md) for the rationale behind the JSON schema, the single-threaded rule, and the "separate SPM package" choice.
