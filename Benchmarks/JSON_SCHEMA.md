# Cross-library bench JSON contract (v1)

Every harness (Swift `ProximaBench`, Python FAISS, Python ScaNN) writes a
single JSON document per run with this exact shape. The aggregator in
`python/compare.py` groups these documents by `dataset` and `indexParams`
and emits a Markdown table.

Schema is deliberately flat and unambiguous — no nested "results" arrays,
one run per file. Use multiple files for multiple configurations.

```jsonc
{
  "schemaVersion": 1,

  // Identity of the run
  "library": "ProximaKit",          // "ProximaKit" | "FAISS" | "ScaNN"
  "libraryVersion": "1.4.0-dev",    // semver / git describe
  "dataset": "sift-1m-100k",        // see datasets/README.md
  "datasetSize": 100000,            // count of vectors actually indexed
  "dimension": 128,
  "metric": "l2",                   // "l2" | "ip" | "cosine"

  // Index configuration
  "indexParams": {
    "type": "hnsw",                 // free-form, per library
    "M": 16,
    "efConstruction": 200,
    "efSearch": 50
  },

  // Workload
  "k": 10,
  "queryCount": 1000,

  // Results
  "buildTimeSeconds": 12.3,         // wall-clock end-to-end build
  "searchLatencyMeanMs": 1.44,
  "searchLatencyP50Ms": 1.21,
  "searchLatencyP95Ms": 2.56,
  "queriesPerSecond": 826.4,
  "recallAt10": 0.971,              // vs the exact ground truth
  "residentMemoryMb": 138.2,        // RSS after build (peak), steady-state

  // Environment
  "platform": {
    "os": "darwin",                 // uname -s, lowercased
    "kernel": "25.0.0",
    "arch": "arm64",
    "cpuModel": "Apple M1 Pro",
    "swiftVersion": "5.10",         // optional — set by Swift runs
    "pythonVersion": "3.11.6"       // optional — set by Python runs
  },
  "seed": 42,
  "runStartedAt": "2026-04-19T14:50:00Z",
  "runDurationSeconds": 18.7,
  "notes": "smoke slice"            // free-form, e.g. CI context
}
```

## Filename convention

`{library}__{dataset}__{indexParams.type}__ef{efSearch}.json`

Examples:

- `ProximaKit__sift-1m-100k__hnsw__ef50.json`
- `FAISS__sift-1m-100k__hnsw__ef50.json`
- `ScaNN__ms-marco-50k__scann__leaves_to_search100.json`

Ground-truth files (from exact brute-force) use a separate naming:

- `GroundTruth__sift-1m-100k__k10.json`

## Invariants

1. `recallAt10` is measured against the **same** ground truth file for a given
   `(dataset, k)`. No library computes its own recall from its own approximate
   neighbors.
2. `queryCount` is the size of the query set actually timed (not the dataset).
3. All timing measurements are wall clock, single-threaded unless the
   `indexParams` explicitly opt into multi-threaded search.
4. `residentMemoryMb` is the process RSS sampled after index build completes
   and before any queries run, to isolate index memory from query buffers.
