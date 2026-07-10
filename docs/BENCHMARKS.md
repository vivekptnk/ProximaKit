# ProximaKit Benchmarks — Methodology and Results

This document describes how ProximaKit's performance numbers are measured, what they mean, and how to reproduce them.

> **Looking for concrete, reproducible numbers?** See the
> [**Benchmark Card**](BENCHMARK-CARD.md): HNSW build / latency / recall /
> memory / disk / cold-open measured at 10K and 100K × 384d in release mode on
> a named machine and commit, with exact seeds and commands for every row.

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

**Cold start (load from disk):** measured 24.3 ms for a persisted 10K × 384d index and 408.4 ms for 100K × 384d (fresh process, median of 3 runs, release mode — [Benchmark Card](BENCHMARK-CARD.md)). The load fully decodes the file into resident memory, so cold-start time scales with file size.

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
| Load | Full decode into resident memory — **O(file size)**, not constant. Measured cold-open: 24.3 ms at 10K × 384d (16.3 MB file), 408.4 ms at 100K × 384d (162.9 MB file) — fresh process, median of 3 ([Benchmark Card](BENCHMARK-CARD.md)) |
| Roundtrip | Exact binary match (save → load → save produces identical bytes) |

### Format Details

See [ADR-003: Binary Persistence](adr/ADR-003-binary-persistence.md) for the format specification and the rationale for choosing custom binary over JSON.

| Format | File Size (10K, 384d) | Load Time |
|--------|----------------------|-----------|
| JSON (rejected) | ~60 MB (estimate from ADR-003; never shipped) | ~3s (estimate) |
| Custom binary | 16.3 MB (measured — [Benchmark Card](BENCHMARK-CARD.md)) | 24.3 ms (measured, median of 3) |

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

**Memory honesty:** retaining originals **resident** pays the full `4·d` bytes/vector again on top of the codes, so a resident reranking index has *no* compression win — `memorySavingsRatio` drops below 1.0 and `originalStorageBytes` reports the cost. Resident reranking trades PQ's 32× memory story for recall (ADR-012). **This is no longer the whole story:** ADR-014 adds an opt-in `.paged` open that serves the same retained originals from a read-only file mapping instead of the resident heap, restoring the 32× story on the vector payload while keeping rerank exact — see "Paged Originals" below for the measured numbers.

### PQ Training Determinism

`PQConfiguration.seed` pins the k-means centroid-initialization draws: the same seed and training vectors produce **byte-identical codebooks and codes**, asserted in `PQDeterminismTests`. The seed is a training-time knob only (not persisted, mirroring `levelSeed`). The CHA-91 memory-vs-recall fixture (`PQBenchmarkTests.testQuantizedHNSWMemoryVsRecall`) now seeds data, graph topology, *and* PQ training, so it measures a single deterministic value (inside the historical 0.667–0.717 band) instead of a band.

**Run the rerank and determinism tests:**

```bash
swift test --filter PQRerankTests
swift test --filter PQDeterminismTests
```

---

## Paged Originals — Memory and Migration

**What it measures:** whether the opt-in `.paged` open of a retaining `QuantizedHNSWIndex` ([ADR-014](adr/ADR-014-paged-originals.md)) actually keeps the retained originals off the resident heap, whether it costs any resident-mode search regression, and what a v2→v3 migration costs. The design and the addenda these numbers are copied from verbatim live in the ADR.

### Resident-mode regression (bail-out gate) — measured

Threading `OriginalsStore` through the single rerank read site must not regress the existing resident path. A/B on the same public-API benchmark (`ResidentRerankBenchTests`: 8K × 128d retained, k=10, rerankDepth=40, 9 reps × 400 queries), pre-change tree (HEAD `5057152`) vs. post-change tree, same machine:

| Tree | median ms/query | p10 | p90 |
|------|-----------------|-----|-----|
| Before (resident `[Vector]?`) | 0.1654 | 0.1478 | 0.1687 |
| After (`OriginalsStore.resident`) | 0.1594 | 0.1536 | 0.1645 |

**Machine:** Apple M4 Max, 36 GB, macOS 26.0.1, Swift 6.2, release. The "after" median is **~3.6% faster** — an improvement, not a regression — though re-measurement puts the true delta closer to 0–2% faster with the sign varying run-to-run (statistical noise at this scale). The ±2% bail-out bound applies to regressions only and was never approached.

### Paged memory acceptance — measured

`PagedOriginalsMemoryTests` (env-gated `PROXIMA_PAGED_BENCH=1`, release-only, CI-excluded — same gate as `PagedVectorMemoryTests`), 100,000 × 384d retaining `PQHW` v3 base, originals payload = 146.5 MB, `phys_footprint` (`task_vm_info`) deltas opening the **same** base `.paged` vs. `.resident`:

| Measurement | Value |
|---|---|
| Originals payload (theoretical) | 146.5 MB |
| Paged open delta | 8.0 MB |
| Paged open + 50 warm reranks | 8.2 MB |
| Resident open delta | 43.1 MB |

**Machine:** Apple M4 Max, 36 GB, macOS 26.0.1, Swift 6.2, release. The paged open costs **8.0 MB for a 146.5 MB payload (18× less)** and stays essentially flat (8.2 MB) after 50 reranked queries — the originals are demonstrably not resident, and warm faults are a bounded working set, not the corpus. The resident open of the same base costs **5.4× more** (43.1 MB).

**Compressor-reality note (why the gate isn't "≥60% of payload recovered"):** on macOS 26, the memory compressor counts freshly-copied anonymous originals pages at their *compressed* size, so `phys_footprint` captures only **~30%** of the theoretical payload on the resident side — a ratio stable across fixture sizes (12.8 MB of 39 MB at 40K × 256d; 43.1 MB of 146.5 MB at 100K × 384d) — an OS accounting reality, not a residency leak. The acceptance test therefore gates on ratios derived from this measured baseline, with margin, rather than an absolute payload-fraction target: paged open `< payload / 8` (originals demonstrably not resident); resident `> 2.5×` paged (resident pays materially more); resident − paged `> payload / 8` (a substantial slice recovered); warm-rerank delta bounded by `payload / 4`.

### Migration cost — arithmetic (unmeasured)

Neither `PersistenceEngine.upgradeToV3(at:)` (`.pxkt`) nor `QuantizedHNSWIndex.upgradeToV3(at:)` (`PQHW`) decodes the graph or materializes a single vector — both are a pure section-copy: read the source once, copy its sections byte-for-byte into a new in-memory image (plus ≤16,383 B of alignment padding before the pageable section, ADR-014 arithmetic), write that image to a temp file, then re-read the whole temp file back and compare every section against the source before the atomic replace (the full-checksum verification gate). That fixed I/O shape — one full read of the source, one full write of the temp file, one full read-back of the temp file for verification — is **≈3× the source file size in I/O**, arithmetic against the upgrader's code shape, not a run: no migration wall-clock latency has been measured under the [ADR-005](adr/ADR-005-benchmark-methodology.md) harness for either family. Do not read the multiplier above as a throughput or latency claim.

**Run the paged-originals benchmarks (both env-gated, release only):**

```bash
PROXIMA_PAGED_BENCH=1 swift test -c release --filter PagedOriginalsMemoryTests
PROXIMA_RESIDENT_BENCH=1 swift test -c release --filter ResidentRerankBenchTests
```

---

## Paged-Access Overhead — Zero-Copy Decision Probe

**What it measures:** whether the copy-on-access step that both ADR-013 Stage 2 (paged HNSW vector reads) and ADR-014 (paged PQHW rerank reads) chose over raw pointer access costs enough of a warm per-query overhead to justify a scoped zero-copy design pass — the ADR text left this an explicit possible future optimization ("Zero-copy scoped access remains a possible future optimization under ADR-005 measurement" / "zero-copy stays on the deferred list"). The new `ProximaBench paged-access-bench` subcommand measures three things against the same fixture: an isolated copy-vs-raw-unsafe-mmap-read microbenchmark, warm resident-vs-paged HNSW search, and warm resident-vs-paged PQ rerank.

### Pre-declared gate

GO only if the **observed** paged-vs-resident delta on a warm per-query median exceeds 5.0% of that median; otherwise NO-GO. The gate keys on this observed delta alone. An earlier draft of the gate additionally weighted an isolation-microbench extrapolation, which could print GO on measurement noise unrelated to in-search cost; that extrapolation is now retained as a diagnostic only (below) and does not feed the decision.

### HNSW search — measured, scoped GO

Fixture: 50,000 × 384d, `m = 16`, `efConstruction = 64`, `efSearch = 50`, `k = 10`, 200 queries, warm pages (pre-faulted before timing). Eight independent warm measurements: the original two timing passes (reps 7, seed 42), plus an adversarial methodology judge's independent re-measurement of six more passes (reps 11 each) judge-replicated across two fixture seeds — three further replicates at seed 42 and three at the independent seed-7 fixture:

| | floor | max | mean |
|---|---|---|---|
| Observed paged-overhead fraction | 8.75% | 17.01% | 11.89% |

**Machine:** Apple M4 Max, 36 GB, macOS 26.0.1, Swift 6.2, release. All eight measurements clear the 5.0% threshold. **Decision: scoped GO** for a zero-copy design pass on the paged HNSW search path only — implementation is deferred to a future mission, gated on re-measurement on target consumer hardware. The observed delta is copy-allocation cost *plus* mmap page-locality effects (cache/TLB/residency behavior), not pure memcpy time, so a zero-copy change recovers at most the allocation/memcpy component: realizable benefit is bounded by, and likely below, the observed 8.75–17.01% warm-best-case delta above.

### PQ rerank — measured, stays copy-on-access

Same eight-measurement protocol over a retained-originals `PQHW` fixture (`rerankDepth = 40`, PQ subspaces = 32): observed paged-overhead fraction ranges **0.00%–3.28%**, well under the 5.0% gate. **Decision: no change** — the rerank path stays copy-on-access; only a future change to `rerankDepth` or candidate selection large enough to cross the threshold would reopen it.

### Non-transferable diagnostic

The isolated copy-vs-raw-read microbenchmark reported roughly 175–219 ns/access at 384d in isolation during this mission's session, with the marginal cost actually observed inside HNSW search only ~35 ns/access on average — distance computation hides most of the access latency once it's in context. These ns-level figures are a session measurement, not reproducible from the committed artifacts in this repository (the backing harness JSON lives only in local, gitignored scratch output) — rerun `paged-access-bench` below to regenerate them. The isolation number is kept only as a diagnostic sanity check; it never feeds the GO/NO-GO decision above.

**Warm-run scope, honestly stated:** these figures are a warm best case for exposing paged-access overhead — pages are pre-faulted before timing, which emphasizes steady-state allocation/memcpy/locality cost. Cold, fault-dominated workloads (paging's actual reason to exist) dilute this overhead toward zero, since major page faults and I/O dominate query time there. Raw per-run JSON lives in the mission's local benchmark harness output, not in this repository.

**Run the paged-access bench:**

```bash
swift build -c release --package-path Benchmarks
Benchmarks/.build/release/ProximaBench paged-access-bench --out out/paged-access.json
```

---

## Embedding Compute-Units — ANE Decision Probe

**What it measures:** whether Core ML's Apple Neural Engine dispatch (`MLComputeUnits.cpuAndNeuralEngine`) beats CPU-only batch embedding throughput by enough to justify exposing a public `computeUnits` knob on `CoreMLEmbeddingProvider` (today it configures none — Core ML picks its own default dispatch). The new `ProximaBench embed-bench` subcommand runs the same seeded batch through `cpuOnly`, `cpuAndGPU`, and `cpuAndNeuralEngine` and reports median batch throughput plus first-load latency for each.

### Pre-declared gate

GO only if `cpuAndNeuralEngine` median batch throughput is ≥ 1.50× `cpuOnly`; otherwise NO-GO — keep Core ML defaults, ship no knob.

### Results — measured, NO-GO

Local MiniLM `.mlpackage`, batch 512, 5 reps (1 warmup), SplitMix64 seed 42 (deterministic inputs, no system RNG):

| computeUnits | median texts/s | spread | first-load total (ms) |
|---|---|---|---|
| cpuOnly | 117.64 | 4.76% | 88.03 |
| cpuAndGPU | 117.93 | 0.39% | 84.23 |
| cpuAndNeuralEngine | 117.48 | 0.46% | 213.88 |

**Machine:** Apple M4 Max, 36 GB, macOS 26.0.1, Swift 6.2, release. Observed gate: `cpuAndNeuralEngine / cpuOnly` = **0.9986×** — dead even, well under the 1.50× threshold. ANE additionally pays roughly 126 ms more first-load latency than CPU (213.88 ms vs. 88.03 ms — the classic Core ML compile tax on the ANE path). **Decision: NO-GO.** No public `computeUnits` knob ships; `CoreMLEmbeddingProvider` keeps Core ML's own dispatch default.

**NaN-model caveat:** the local model used for this probe emits `NaN` through the full `CoreMLEmbeddingProvider` path (`CoreMLEmbeddingProviderTests/testEmbedText` fails asserting a non-zero, and therefore non-finite, embedding), so the table above is a **CoreML runtime-dispatch throughput measurement, not a semantic-correctness validation** — the raw report's checksum field is `null` for every row for exactly this reason. Reopening this decision needs a finite-output sentence-embedding artifact (`.mlpackage` / `.mlmodel` / `.mlmodelc` with `input_ids` / `attention_mask` MLMultiArray inputs) re-measured on consumer hardware, at the same pre-declared 1.50× threshold.

**Run the embedding compute-units bench:**

```bash
swift build -c release --package-path Benchmarks
Benchmarks/.build/release/ProximaBench embed-bench \
    --model Models/MiniLM-L6-v2.mlpackage \
    --batch-size 512 --reps 5 --warmup 1 --seed 42 \
    --out out/embed-bench.json
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
