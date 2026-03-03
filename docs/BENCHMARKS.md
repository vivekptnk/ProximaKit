# ProximaKit Benchmark Strategy

## Why Benchmarks Matter Here

ProximaKit's value proposition is _on-device performance_. If queries take 200ms instead of 20ms, nobody will use it over a cloud API. If recall drops below 90%, the results feel broken. Benchmarks aren't a nice-to-have — they're the product.

## The Three Dimensions

Every benchmark measures along three axes:

```
         Speed (latency/throughput)
            ▲
            │
            │     ★ Sweet spot
            │    (fast + accurate + small)
            │
            ├──────────────────► Quality (recall@K)
           ╱
          ╱
         ╱
        ▼
     Memory (bytes per vector)
```

A change that improves speed but tanks recall is NOT an improvement. The benchmark suite enforces all three simultaneously.

## Benchmark Categories

### 1. Vector Math (Micro-benchmarks)

The inner loop. Called millions of times per search. Nanosecond-level measurement.

| Benchmark | What It Measures | Dimensions | Iterations |
|-----------|-----------------|------------|------------|
| `dot_product` | Single vDSP_dotpr call | 128, 384, 768, 1536 | 100K |
| `cosine_similarity` | Dot + magnitudes + division | 128, 384, 768, 1536 | 100K |
| `l2_distance` | Subtract + sum squares + sqrt | 128, 384, 768, 1536 | 100K |
| `normalize` | Sum squares + divide | 128, 384, 768, 1536 | 100K |
| `batch_dot_1K` | vDSP_mmul: 1 query vs 1K vectors | 384 | 1K |
| `batch_dot_10K` | vDSP_mmul: 1 query vs 10K vectors | 384 | 100 |
| `batch_dot_100K` | vDSP_mmul: 1 query vs 100K vectors | 384 | 10 |

**Why these dimensions:** 128 (small NLP), 384 (MiniLM/NLEmbedding), 768 (BERT), 1536 (OpenAI ada-002). Real-world coverage.

**Target:** Batch dot product for 10K/384d must complete in <5ms. This is the ceiling for the entire search operation.

### 2. Index Build (Throughput)

How fast can we ingest vectors?

| Benchmark | Dataset | Parameters | Metric |
|-----------|---------|------------|--------|
| `brute_build_1K` | 1K random 384d | — | vectors/sec |
| `brute_build_10K` | 10K random 384d | — | vectors/sec |
| `hnsw_build_1K` | 1K random 384d | M=16, efC=200 | vectors/sec, total time |
| `hnsw_build_10K` | 10K random 384d | M=16, efC=200 | vectors/sec, total time |
| `hnsw_build_50K` | 50K random 384d | M=16, efC=200 | vectors/sec, total time |
| `hnsw_build_param_sweep` | 10K random 384d | M={8,16,32}, efC={100,200,400} | grid of build times |

**Targets:**
- HNSW 10K/384d build: < 5 seconds
- HNSW 50K/384d build: < 30 seconds
- Build throughput: > 2,000 vectors/sec at 384d

### 3. Search Latency (The Money Metric)

| Benchmark | Index | Dataset | Parameters | Metric |
|-----------|-------|---------|------------|--------|
| `brute_query_1K` | BruteForce | 1K/384d | k=10 | p50, p95, p99 latency |
| `brute_query_10K` | BruteForce | 10K/384d | k=10 | p50, p95, p99 latency |
| `hnsw_query_1K` | HNSW | 1K/384d | k=10, ef=50 | p50, p95, p99 latency |
| `hnsw_query_10K` | HNSW | 10K/384d | k=10, ef=50 | p50, p95, p99 latency |
| `hnsw_query_50K` | HNSW | 50K/384d | k=10, ef=50 | p50, p95, p99 latency |
| `hnsw_query_ef_sweep` | HNSW | 10K/384d | k=10, ef={10,25,50,100,200} | latency vs recall curve |

**Targets:**
- HNSW query p50 at 10K/384d, ef=50: < 5ms
- HNSW query p95 at 10K/384d, ef=50: < 15ms
- HNSW query p99 at 10K/384d, ef=50: < 50ms
- BruteForce 10K/384d: < 50ms (baseline comparison)

**Why percentiles:** Averages hide tail latency. A p50 of 5ms with p99 of 500ms means 1% of users wait half a second. Not acceptable.

### 4. Recall (Search Quality)

Uses brute-force as ground truth.

| Benchmark | Dataset | Parameters | Metric |
|-----------|---------|------------|--------|
| `recall_uniform_10K` | 10K uniform random 384d | ef={10,25,50,100,200} | recall@1, @5, @10 |
| `recall_clustered_10K` | 10K clustered 384d (8 clusters) | ef={10,25,50,100,200} | recall@1, @5, @10 |
| `recall_high_dim` | 10K uniform 768d | ef=50 | recall@10 |
| `recall_low_dim` | 10K uniform 128d | ef=50 | recall@10 |
| `recall_param_sweep` | 10K/384d | M={8,16,32}, efC={100,200}, ef={25,50,100} | grid of recall values |

**Targets:**
- recall@10, uniform, ef=50: > 95%
- recall@10, clustered, ef=50: > 92%
- recall@1, uniform, ef=50: > 98%

**Why clustered data:** Real embeddings cluster (topics, categories). Naive HNSW implementations get trapped in clusters. Heuristic neighbor selection (ADR-004) specifically addresses this. Clustered benchmarks verify it works.

### 5. Persistence (Cold Start)

| Benchmark | Dataset | Metric |
|-----------|---------|--------|
| `save_10K` | 10K/384d HNSW | save time, file size |
| `save_50K` | 50K/384d HNSW | save time, file size |
| `load_10K` | 10K/384d from disk | load time (cold start) |
| `load_50K` | 50K/384d from disk | load time (cold start) |
| `query_after_load_10K` | 10K/384d, load then query | end-to-end latency |

**Targets:**
- Save 10K index: < 500ms
- Load 10K index (cold start): < 200ms
- Load + first query: < 250ms
- File size 10K/384d: < 20MB (raw float32 = ~15MB)

### 6. Memory (Resource Pressure)

| Benchmark | Dataset | Metric |
|-----------|---------|--------|
| `memory_brute_10K` | 10K/384d BruteForce | peak RSS delta |
| `memory_hnsw_10K` | 10K/384d HNSW (M=16) | peak RSS delta |
| `memory_hnsw_50K` | 50K/384d HNSW (M=16) | peak RSS delta |
| `memory_per_vector` | 1K→50K/384d HNSW | bytes per vector (slope) |

**Targets:**
- HNSW memory per vector at 384d, M=16: < 2KB (raw floats = 1.5KB, overhead < 0.5KB)
- Peak memory during build should not exceed 2x final index size

### 7. Concurrency (Thread Safety Under Load)

| Benchmark | Scenario | Metric |
|-----------|----------|--------|
| `concurrent_reads` | 8 parallel queries on 10K index | throughput (queries/sec), no crashes |
| `concurrent_read_write` | 4 readers + 1 writer on 10K index | throughput, correctness (no data races) |
| `actor_overhead` | Sequential vs actor-isolated queries | overhead percentage |

**Targets:**
- No crashes or data races under ThreadSanitizer
- Actor overhead < 10% vs direct access
- Concurrent query throughput: > 500 queries/sec on 10K/384d

## Data Generation

### Uniform Random
```swift
func generateUniform(count: Int, dimension: Int, seed: UInt64 = 42) -> [Vector]
```
Standard for most benchmarks. Seed fixed for reproducibility.

### Clustered
```swift
func generateClustered(count: Int, dimension: Int, clusters: Int = 8, spread: Float = 0.1) -> [Vector]
```
Generate cluster centers, then points around each center with Gaussian noise. This is how real embeddings behave — text about cooking clusters separately from text about astrophysics.

### Correlated
```swift
func generateCorrelated(count: Int, dimension: Int, rank: Int = 50) -> [Vector]
```
Low-rank structure (first `rank` dimensions carry signal, rest is noise). Simulates the intrinsic dimensionality being lower than the embedding dimension — which is true for all real embedding models.

## Baseline Tracking

Baselines stored in `Benchmarks/baselines.json`:

```json
{
  "device": "Apple M2",
  "swift_version": "5.9",
  "date": "2026-03-01",
  "results": {
    "hnsw_query_10K": {
      "p50_ms": 4.2,
      "p95_ms": 12.1,
      "p99_ms": 38.7
    },
    "recall_uniform_10K_ef50": {
      "recall_at_10": 0.962
    }
  }
}
```

CI compares PR benchmarks against baselines. A regression > 10% on any latency metric or > 2% on recall fails the check.

## Benchmark Report Format

The `/bench` command outputs:

```
ProximaKit Benchmark Report
Device: Apple M2 Pro | Swift 5.9 | macOS 14.2
Date: 2026-03-03

VECTOR MATH (384d, 100K iterations)
  dot_product       0.18µs  ✅
  cosine_similarity 0.31µs  ✅
  l2_distance       0.24µs  ✅
  batch_dot_10K     3.8ms   ✅ (target: <5ms)

INDEX BUILD (384d, M=16, efC=200)
  hnsw_1K           0.4s    ✅
  hnsw_10K          3.2s    ✅ (target: <5s)
  hnsw_50K          18.7s   ✅ (target: <30s)

SEARCH LATENCY (384d, k=10, ef=50)
              p50     p95     p99     target
  hnsw_1K     0.8ms   2.1ms   4.3ms   ✅
  hnsw_10K    4.2ms   12.1ms  38.7ms  ✅ (<50ms p99)
  hnsw_50K    11.3ms  28.4ms  67.2ms  ✅

RECALL (384d, ef=50)
              @1      @5      @10     target
  uniform_10K 0.987   0.974   0.962   ✅ (>95%)
  cluster_10K 0.971   0.948   0.931   ✅ (>92%)

PERSISTENCE (10K/384d)
  save        312ms   ✅
  load        87ms    ✅ (<200ms)
  file_size   16.2MB  ✅ (<20MB)

MEMORY (384d, M=16)
  per_vector  1.72KB  ✅ (<2KB)
  hnsw_10K    17.2MB  ✅
  hnsw_50K    86.0MB  ✅
```
