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
# Recall benchmarks (2-5 minutes)
swift test --filter RecallBenchmarkTests

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
