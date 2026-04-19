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
| 1K vectors | 384d | ~104ms |

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

The `PersistenceEngine` uses a compact binary format with memory-mapped I/O for loading.

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

The first published tables land with the v1.4.0 release. Until then, the CI nightly artifact under the `benchmark` workflow is the canonical source — see the latest green nightly run on `main` for current numbers.

Example table shape (filled in from the CI JSON artifacts):

| Library | Params | Build (s) | p50 (ms) | p95 (ms) | QPS | Recall@10 | RSS (MB) |
|---------|--------|-----------|----------|----------|-----|-----------|----------|
| ProximaKit | HNSW M=16 efC=200 ef=50 | _N_ | _N_ | _N_ | _N_ | _N_ | _N_ |
| FAISS      | HNSW M=16 efC=200 ef=50 | _N_ | _N_ | _N_ | _N_ | _N_ | _N_ |
| ScaNN      | tree-AH leaves=100 search=10 reorder=100 | _N_ | _N_ | _N_ | _N_ | _N_ | _N_ |

Actual numbers are auto-generated into `out/compare.md` by `Benchmarks/python/compare.py`. See [ADR-005: Cross-Library Benchmark Methodology](adr/ADR-005-benchmark-methodology.md) for the rationale behind the JSON schema, the single-threaded rule, and the "separate SPM package" choice.
