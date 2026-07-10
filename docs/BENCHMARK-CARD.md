# ProximaKit Benchmark Card — HNSW, measured

One page of numbers a stranger can trust and reproduce. Every figure below was
measured on the machine described here, in release mode, in a single session,
from the raw outputs listed at the bottom. Nothing on this card is an estimate
unless it is explicitly labeled a projection.

## Environment (measured on)

| Parameter | Value |
|-----------|-------|
| Machine | MacBook Pro (Mac16,5) |
| Chip | Apple M4 Max — 14 cores (10 performance + 4 efficiency) |
| RAM | 36 GB |
| OS | macOS 26.0.1 (build 25A362) |
| Swift | Apple Swift 6.2 (swiftlang-6.2.0.19.9), target arm64-apple-macosx26.0 |
| Build config | Release (`swift build -c release`) — debug numbers are NOT comparable |
| Commit | `4520ce2fea8df06a3315828fa75b2ce8fe64876d` |
| Date | 2026-07-10 |

## Results — HNSW, 384 dimensions, Euclidean, k=10

Index configuration: `HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50, levelSeed: 1355858142)`.
Data: **synthetic** Gaussian-mixture vectors (exact seeded formula below) — not
real embeddings; see Limitations.

| Metric | 10K vectors | 100K vectors | How measured |
|---|---|---|---|
| Build time (single-threaded inserts) | **16.2 s** | **372.2 s** | `ContinuousClock` around the add loop |
| Query latency p50 | **0.14 ms** | **0.37 ms** | nearest-rank p50; median of 3 reps × 1,000 queries; 200-query warmup excluded |
| Query latency p95 | **0.24 ms** | **0.84 ms** | nearest-rank p95, same protocol |
| Recall@10 vs exact brute force | **1.000** | **0.9999** | 1,000 queries against `BruteForceIndex` ground truth |
| Resident memory after build | **31 MB** | **306 MB** | `phys_footprint` delta (`task_vm_info`) |
| Saved index size on disk | **16.3 MB** (17,082,632 B) | **162.9 MB** (170,819,968 B) | file size of `PersistenceEngine.save` output |
| Cold-open (load) time from disk | **24.3 ms** | **408.4 ms** | fresh process, `PersistenceEngine.loadHNSW`, median of 3 |
| Resident memory after load | **31 MB** | **309 MB** | `phys_footprint` delta in the fresh process |

Per-rep spread (all three reps, so you can judge the noise):
p50 10K = 0.143 / 0.149 / 0.144 ms; p50 100K = 0.370 / 0.371 / 0.393 ms;
p95 10K = 0.258 / 0.244 / 0.234 ms; p95 100K = 0.839 / 0.820 / 0.868 ms.
Load runs (chronological) 10K = 41.1 / 24.3 / 24.2 ms; 100K = 583.0 / 398.0 / 408.4 ms
— the first fresh-process run of each binary is consistently slower; the median is reported.

**Load scaling (the honest headline):** the saved file is 10.0× larger at 100K
and the cold-open is ~17× slower (24.3 ms → 408.4 ms). `loadHNSW` decodes the
entire file into resident memory, so cold-start time is **O(file size)** — it
is *not* constant across index sizes. (docs/BENCHMARKS.md previously claimed
"~50ms cold start regardless of index size"; that was measured only at 10K and
is wrong at scale. This card supersedes it.)

### 1M vectors: not yet published — planned

The measured 100K build is 372.2 s single-threaded. A ×10 linear projection is
**≥ 62 minutes** of build alone — and measured growth is super-linear (10K→100K
was 23× for 10× the vectors), so 62 min is a lower bound. That exceeded this
session's measurement budget (cutoff: 100K build × 10 < 45 min). **The 62-minute
figure is a projection, not a measurement.** No 1M latency/recall/memory numbers
are published because none were measured.

## Workload definition (exact, seeded)

All randomness is deterministic (SplitMix64). Base vectors are drawn from a
Gaussian mixture so that real geometric neighborhoods exist. The rejected
baseline is measured, not asserted: on i.i.d. Uniform[0,1)^384 data (Euclidean,
same rig and seeds, `--data uniform`), recall@10 at ef=50 measures **0.574**
(raw: `uniform-rejection-10k.json`) versus 1.000 on the mixture — on
structureless high-dimensional data, distance concentration turns the exact
top-10 into a near-tie, so recall largely measures tie-breaking rather than
embedding-like search quality (ProximaKit's own `RecallBenchmarkTests` notes
the same degradation on random data; real embeddings cluster). That 0.574 is
specific to this uniform distribution — other unstructured distributions
degrade by different amounts.

- Clusters: `C = max(16, n/100)` → 100 clusters at 10K, 1,000 at 100K
- Center of cluster `c`: Uniform[-1,1)^384, from `SplitMix64(seed: 0xC0FFEE01 * (c+1))`
- Base row `i`: `center[i % C] + N(0, σ)` with σ = **0.55**, noise from
  `SplitMix64(seed: 0xBA5E0001 * (i+1))` via Box–Muller
- Why σ = 0.55: clusters are separable but overlapping — expected intra-cluster
  pair distance `√(2σ²d)` ≈ 15.2 vs expected inter-center distance `√(2d/3)` ≈ 16.0
  (arithmetic from the definitions above, not a measurement), so true neighbors
  are not trivially isolated; the ef-sweep below confirms non-degeneracy
  (recall drops when the beam narrows)
- Query `j`: same formula at global index `100_000_000 + j` (same clusters,
  disjoint noise stream — queries are never verbatim copies of base rows)
- IDs: UUID with the row index little-endian in the first 8 bytes
- HNSW graph topology pinned with `levelSeed: 1355858142` (`0x50D0C0DE`) —
  a re-run reproduces the same graph and the same recall bit-for-bit

Recall sensitivity was verified, not assumed: at efSearch 5/10/20/50 the 10K
recall measured **0.9803 / 0.9803 / 0.9990 / 1.0000** (1,000 queries, same
seeds and protocol as the table; raw: `ef-sweep-10k-ef{5,10,20,50}.json`,
summary `ef-sweep-10k.json`) — the metric responds to the knob, so 1.000 at
ef=50 is a real measurement, not a saturated artifact. ef=5 and ef=10 are
identical by construction: `HNSWIndex.search` clamps the beam width to
`max(efSearch, k)`, so ef=5 runs as ef=10 at k=10.

## Memory measurement method

Reported column: **`phys_footprint` delta** from `task_info(TASK_VM_INFO)` —
probe taken at process start and again after build (or after load), same probe
ProximaKit's own paged-memory acceptance tests use. Cross-checks recorded in
the raw outputs: `mach_task_basic_info.resident_size` (34.7 MB / 298.9 MB after
build) and `/usr/bin/time -l` peak RSS of the whole process (75.7 MB / 575.7 MB
for the measure runs — that includes the brute-force ground-truth index, which
is why the card reports the probe delta, not process peak). The measure process
retains no copy of the input vectors during build, so the after-build delta is
the index alone.

## Cross-check with the tracked harness

The same 10K workload, exported to `.fvecs` and run through the repo's tracked
`Benchmarks/ProximaBench` (`ground-truth` + `hnsw` subcommands, unseeded
topology, seed flag 42): build 15.57 s, p50 0.142 ms, p95 0.240 ms,
recall@10 1.000 — agreeing with the table above within 4% on build (ProximaBench
excludes data generation) and ~1% on latency, with recall identical despite the
unseeded graph. Raw: `crosscheck-hnsw-10k.json`.

## Reproduce it

The measurement rig (`CardRig`) is a standalone SwiftPM executable that links
ProximaKit's public API. It lives under `.harness/scratch/bench-conversion-round/`
(deliberately untracked — nothing ships in the library for benchmarking); its
complete source is in the appendix below, so this card is self-contained.

```bash
# 0. Rig layout: <repo>/.harness/scratch/bench-conversion-round/CardRig/
#    with Package.swift + Sources/CardRig/main.swift from the appendix.
cd .harness/scratch/bench-conversion-round/CardRig && swift build -c release && cd -
BIN=.harness/scratch/bench-conversion-round/CardRig/.build/release/CardRig
OUT=.harness/scratch/bench-conversion-round

# 1. 10K row (~40 s): build/latency/recall/memory/disk in one process
/usr/bin/time -l "$BIN" measure --n 10000 --dim 384 --queries 1000 --k 10 \
  --m 16 --efc 200 --ef 50 --warmup 200 --reps 3 --sigma 0.55 \
  --level-seed 1355858142 --label synthetic-gmm \
  --index-out "$OUT/data/idx-10k.pxkt" --json-out "$OUT/raw/measure-10k.json"

# 2. 10K cold-open (x3, fresh process each; ~1 s total)
for i in 1 2 3; do
  /usr/bin/time -l "$BIN" load --index "$OUT/data/idx-10k.pxkt" --dim 384 \
    --n 10000 --sigma 0.55 --smoke-queries 50 \
    --json-out "$OUT/raw/load-10k-$i.json"
done

# 3. 100K rows: same two commands with --n 100000 and idx-100k.pxkt (~8 min)

# 4. Cross-check via the tracked harness (optional, ~1 min):
"$BIN" gen --n 10000 --dim 384 --queries 1000 --sigma 0.55 \
  --base-out "$OUT/data/gmm-10k-base.fvecs" --query-out "$OUT/data/gmm-10k-query.fvecs"
swift build -c release --package-path Benchmarks
PB=Benchmarks/.build/release/ProximaBench
"$PB" ground-truth --base "$OUT/data/gmm-10k-base.fvecs" \
  --queries "$OUT/data/gmm-10k-query.fvecs" --size 10000 --query-count 1000 \
  --k 10 --metric l2 --dataset gmm-10k --out "$OUT/raw/crosscheck-gt-10k.json"
"$PB" hnsw --base "$OUT/data/gmm-10k-base.fvecs" \
  --queries "$OUT/data/gmm-10k-query.fvecs" --gt "$OUT/raw/crosscheck-gt-10k.json" \
  --size 10000 --query-count 1000 --dataset gmm-10k \
  --k 10 --m 16 --efc 200 --ef 50 --metric l2 --seed 42 \
  --out "$OUT/raw/crosscheck-hnsw-10k.json"

# 5. ef-sweep + uniform-rejection sourcing runs (~4 min total)
for EF in 5 10 20 50; do
  "$BIN" measure --n 10000 --dim 384 --queries 1000 --k 10 --m 16 --efc 200 \
    --ef $EF --warmup 200 --reps 1 --sigma 0.55 --level-seed 1355858142 \
    --index-out "$OUT/data/idx-efsweep-tmp.pxkt" \
    --json-out "$OUT/raw/ef-sweep-10k-ef$EF.json"
done
"$BIN" measure --n 10000 --dim 384 --queries 1000 --k 10 --m 16 --efc 200 \
  --ef 50 --warmup 200 --reps 1 --data uniform --level-seed 1355858142 \
  --index-out "$OUT/data/idx-uniform-tmp.pxkt" \
  --json-out "$OUT/raw/uniform-rejection-10k.json"
```

Note: `Benchmarks/Package.swift` resolves its path dependency by directory
identity, so the cross-check build must run from a checkout whose directory is
named `ProximaKit` (not from a git worktree with a generated name). Deterministic
rows (recall, disk size) reproduce exactly; timing rows on comparable Apple
Silicon should land within normal run-to-run noise (see per-rep spread above).

## Limitations — read before quoting these numbers

- **Synthetic distribution.** The Gaussian-mixture workload has cleaner cluster
  structure than real embedding corpora; real-corpus recall at these settings
  will be high but not necessarily 1.000. No 10K+ real-embedding corpus ships
  offline in this repo, so no real-corpus row is published on this card — the
  tracked `Benchmarks/` harness supports SIFT1M and MS MARCO (network download)
  for that purpose.
- **Single machine, single session.** One M4 Max laptop. The machine had light
  ambient background load during measurement (start load avg ~2.0, end ~2.7 on
  14 cores); the measurement processes are single-threaded and ran serially,
  one at a time.
- **Debug ≠ release.** All numbers are `-c release`. Debug builds are 10–100×
  slower on this code; never compare debug numbers against this card.
- **Cold-open caveat.** "Cold" means a fresh process; the OS page cache was not
  purged, so a first-ever read of the file from a cold disk cache can be slower
  than the medians above (observed first-run values: 41 ms at 10K, 583 ms at 100K).
- **Latency is per-query actor round-trip** (`await index.search`) measured
  in-process — it includes actor hop overhead but no serialization or IPC.

## Raw outputs

Everything above traces to files under `.harness/scratch/bench-conversion-round/raw/`
generated in this session on the machine above: `session-info.txt`,
`measure-10k.json`, `measure-100k.json`, `measure-10k.timelog`,
`measure-100k.timelog`, `load-10k-{1,2,3}.json` + `.timelog`,
`load-100k-{1,2,3}.json` + `.timelog`, `crosscheck-gt-10k.json`,
`crosscheck-hnsw-10k.json`, `ef-sweep-10k-ef{5,10,20,50}.json` +
`ef-sweep-10k.json` (summary), `uniform-rejection-10k.json`, `rig-build.log`,
and the pipeline script `run-all.sh`.

---

## Appendix — full rig source

<details>
<summary><code>CardRig/Package.swift</code></summary>

```swift
// swift-tools-version: 5.9
// CardRig — scratch benchmark driver for the "conversion round" benchmark card.
//
// NOT a product of ProximaKit and NOT under Sources/ or Tests/. Lives under
// .harness/scratch/ only. It reuses ProximaKit's real public APIs (HNSWIndex,
// BruteForceIndex, PersistenceEngine, EuclideanDistance) and mirrors the exact
// measurement methodology of Benchmarks/Sources/ProximaBench (BruteForce ground
// truth, nearest-rank percentiles, ContinuousClock). The one thing it does that
// ProximaBench cannot is pin HNSWConfiguration.levelSeed, so the built graph is
// bit-reproducible and the card survives hostile re-measurement.
//
// `.package(name:path:)` pins the dependency identity to "ProximaKit" because in
// a git worktree the checkout directory is not literally named "ProximaKit", and
// SwiftPM would otherwise derive the identity from the directory name.

import PackageDescription

let package = Package(
    name: "CardRig",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(name: "ProximaKit", path: "../../../.."),
    ],
    targets: [
        .executableTarget(
            name: "CardRig",
            dependencies: [
                .product(name: "ProximaKit", package: "ProximaKit"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),
    ]
)
```

</details>

<details>
<summary><code>CardRig/Sources/CardRig/main.swift</code></summary>

```swift
// CardRig main.swift
//
// Deterministic HNSW benchmark driver for docs/BENCHMARK-CARD.md.
// Subcommands:
//   measure  — build (release), after-build resident memory, disk size,
//              query latency p50/p95, recall@10 vs brute-force ground truth.
//   load     — fresh-process cold-open (load) time + after-load resident memory.
//   gen      — write deterministic .fvecs (base + queries) for the ProximaBench
//              cross-check (Benchmarks/ProximaBench hnsw / ground-truth).
//
// All randomness is seeded. Vectors are SYNTHETIC (seeded Gaussian mixture; see
// the data-model comment below), generated by the exact formula documented on
// the card. HNSW graph topology is pinned with HNSWConfiguration.levelSeed so a
// re-measurement reproduces recall bit-for-bit.

import Foundation
import ProximaKit

#if canImport(Darwin)
import Darwin
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic generators (self-contained; not imported from the library).
// ─────────────────────────────────────────────────────────────────────────────

/// SplitMix64 — identical to the generator used inside ProximaKit's own tests
/// (Tests/…/SeededRandom.swift / PagedOriginalsMemoryTests). Reproduced here so
/// the rig has no @testable dependency.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Query rows are drawn from the SAME distribution as base rows but from a
/// disjoint index range and noise stream, so no query is ever a verbatim copy
/// of a base row.
let queryOffset = 100_000_000

// ── Synthetic data model: seeded Gaussian mixture ────────────────────────────
//
// i.i.d. uniform-random vectors in 384d are an unrepresentative ANN workload:
// distance concentration ("curse of dimensionality") pushes all pairwise
// distances toward each other, so the exact top-10 becomes a near-tie and
// measured recall reflects tie-breaking on structureless data rather than the
// index's behavior on embedding-like data. ProximaKit's own
// RecallBenchmarkTests only claim >0.82 at 128d on random data and note that
// real embeddings — which CLUSTER by semantic similarity — reach >95%.
// (The uniform baseline stays measurable via `--data uniform`; the card cites
// the measured number rather than asserting one.)
//
// To produce a recall number that is both reproducible AND representative of a
// real embedding workload, base vectors are drawn from a Gaussian mixture:
//   C clusters, C = max(16, n/100)              (≈100 vectors per cluster)
//   cluster center c ~ Uniform[-1, 1)^dim       (seeded stream 0xC0FFEE01)
//   base row i        = center[i % C] + N(0, σ) (σ from --sigma, seeded per row)
//   query j           = center[j % C] + N(0, σ) (disjoint noise stream)
// Real geometric neighborhoods exist (a query's true NN are its cluster-mates),
// so recall@10 is meaningful. Everything is seeded, so it is bit-reproducible.

// Set once from --sigma at startup (single-threaded; read by the free
// generator functions). Larger σ ⇒ clusters overlap more ⇒ harder NN.
var noiseSigma: Double = 0.55

func clusterCount(n: Int) -> Int { max(16, n / 100) }

/// Uniform[-1, 1) cluster center, seeded per cluster.
func clusterCenter(_ c: Int, dim: Int) -> [Double] {
    var g = SplitMix64(seed: 0xC0FF_EE01 &* UInt64(c + 1))
    return (0..<dim).map { _ in Double(UInt32(truncatingIfNeeded: g.next()) % 65_536) / 32_768.0 - 1.0 }
}

/// Zero-mean Gaussian noise vector (Box-Muller from SplitMix64), std = σ.
func gaussianNoise(seed: UInt64, dim: Int, sigma: Double) -> [Double] {
    var g = SplitMix64(seed: seed)
    func u01() -> Double { Double(UInt32(truncatingIfNeeded: g.next()) % 1_000_000 + 1) / 1_000_001.0 }
    var out = [Double](repeating: 0, count: dim)
    var i = 0
    while i < dim {
        let u1 = u01(), u2 = u01()
        let r = (-2.0 * Foundation.log(u1)).squareRoot()
        out[i] = r * Foundation.cos(2.0 * Double.pi * u2) * sigma
        if i + 1 < dim { out[i + 1] = r * Foundation.sin(2.0 * Double.pi * u2) * sigma }
        i += 2
    }
    return out
}

/// Deterministic synthetic vector at a global row index, drawn from the mixture.
/// `n` is the BASE scale — it fixes the cluster count so base and query rows
/// share the same clusters.
func syntheticVector(globalIndex g: Int, dim: Int, n: Int) -> [Float] {
    let c = g % clusterCount(n: n)
    let center = clusterCenter(c, dim: dim)
    let noise = gaussianNoise(seed: 0xBA5E_0001 &* UInt64(g + 1), dim: dim, sigma: noiseSigma)
    return (0..<dim).map { Float(center[$0] + noise[$0]) }
}

/// Deterministic i.i.d. Uniform[0,1)^dim vector (`--data uniform`). This is the
/// REJECTED baseline distribution, kept measurable so the card's data-model
/// rationale is sourced by a saved run rather than asserted: at high dimension,
/// i.i.d. uniform distances concentrate and the exact top-10 becomes a
/// near-tie, which is unrepresentative of embedding corpora (real embeddings
/// cluster — see ProximaKit's RecallBenchmarkTests notes).
///   seed      = 0x9E37 * (globalIndex + 1)
///   component = (uint32(next()) % 65536) / 65536.0
func uniformVector(globalIndex g: Int, dim: Int) -> [Float] {
    var gen = SplitMix64(seed: 0x9E37 &* UInt64(g + 1))
    return (0..<dim).map { _ in Float(UInt32(truncatingIfNeeded: gen.next()) % 65_536) / 65_536.0 }
}

/// Dispatch on --data kind ("gmm" default, "uniform" rejected baseline).
func dataVector(globalIndex g: Int, dim: Int, n: Int, kind: String) -> [Float] {
    kind == "uniform" ? uniformVector(globalIndex: g, dim: dim)
                      : syntheticVector(globalIndex: g, dim: dim, n: n)
}

/// Deterministic UUID that encodes the row index in its first 8 bytes, so a
/// search result's UUID round-trips back to a base-set row index with no map.
///
/// `uuid_t` is a 16×UInt8 tuple with alignment 1, so `storeBytes`/`load` of a
/// UInt64 into/from it is a MISALIGNED access (undefined behavior — it silently
/// returned 0 under `-c release`). We copy raw bytes and use `loadUnaligned`.
func uuid(_ i: Int) -> UUID {
    var le = UInt64(i).littleEndian
    var b: uuid_t = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    withUnsafeMutableBytes(of: &b) { dst in
        withUnsafeBytes(of: &le) { src in dst.copyBytes(from: src) }
    }
    return UUID(uuid: b)
}
func rowIndex(of id: UUID) -> Int {
    withUnsafeBytes(of: id.uuid) { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory probes (both the mach probes ProximaKit itself uses).
// ─────────────────────────────────────────────────────────────────────────────

/// phys_footprint via task_vm_info — the probe ProximaKit's paged-memory tests
/// use (PagedOriginalsMemoryTests). This is the compressor-aware footprint.
func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

/// resident_size via mach_task_basic_info — the probe ProximaBench's Platform.swift
/// uses. Reported as a cross-check alongside phys_footprint.
func residentSizeBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? info.resident_size : 0
}

let MB = 1024.0 * 1024.0

// ─────────────────────────────────────────────────────────────────────────────
// Nearest-rank percentiles — copied verbatim from
// Benchmarks/Sources/ProximaBench/Percentiles.swift so latency stats match.
// ─────────────────────────────────────────────────────────────────────────────
func percentile(sorted: [Double], q: Double) -> Double {
    let rank = Int((q * Double(sorted.count)).rounded(.up))
    let idx = max(0, min(sorted.count - 1, rank - 1))
    return sorted[idx]
}
func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    guard !s.isEmpty else { return 0 }
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

// ─────────────────────────────────────────────────────────────────────────────
// Duration helpers.
// ─────────────────────────────────────────────────────────────────────────────
extension Duration {
    var seconds: Double {
        let p = components
        return Double(p.seconds) + Double(p.attoseconds) * 1e-18
    }
    var milliseconds: Double { seconds * 1000.0 }
}

// ─────────────────────────────────────────────────────────────────────────────
// Flags.
// ─────────────────────────────────────────────────────────────────────────────
struct Flags {
    private var map: [String: String] = [:]
    init(_ args: [String]) {
        var i = 0
        while i < args.count {
            let a = args[i]
            if a.hasPrefix("--") {
                if i + 1 < args.count, !args[i + 1].hasPrefix("--") { map[a] = args[i + 1]; i += 2 }
                else { map[a] = "1"; i += 1 }
            } else { i += 1 }
        }
    }
    func str(_ k: String, _ d: String) -> String { map[k] ?? d }
    func int(_ k: String, _ d: Int) -> Int { map[k].flatMap(Int.init) ?? d }
    func req(_ k: String) -> String {
        guard let v = map[k] else { FileHandle.standardError.write(Data("missing \(k)\n".utf8)); exit(2) }
        return v
    }
}

func platformJSON() -> [String: Any] {
    func sysctl(_ n: String) -> String {
        var sz = 0
        guard sysctlbyname(n, nil, &sz, nil, 0) == 0, sz > 0 else { return "unknown" }
        var buf = [CChar](repeating: 0, count: sz)
        guard sysctlbyname(n, &buf, &sz, nil, 0) == 0 else { return "unknown" }
        return String(cString: buf)
    }
    return [
        "cpu": sysctl("machdep.cpu.brand_string"),
        "arch": sysctl("hw.machine"),
    ]
}

func writeJSON(_ obj: [String: Any], to path: String) {
    let data = try! JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
    try! data.write(to: URL(fileURLWithPath: path))
    FileHandle.standardError.write(data)
    FileHandle.standardError.write(Data("\n".utf8))
}

// ─────────────────────────────────────────────────────────────────────────────
// gen — write deterministic .fvecs for the ProximaBench cross-check.
// ─────────────────────────────────────────────────────────────────────────────
func runGen(_ f: Flags) {
    let n = f.int("--n", 10_000)
    let dim = f.int("--dim", 384)
    let q = f.int("--queries", 1_000)
    let baseOut = f.req("--base-out")
    let queryOut = f.req("--query-out")
    noiseSigma = Double(f.str("--sigma", "0.55")) ?? 0.55

    func writeFvecs(path: String, count: Int, indexBase: Int) {
        var data = Data(); data.reserveCapacity(count * (4 + dim * 4))
        var dimLE = Int32(dim).littleEndian
        for k in 0..<count {
            let v = syntheticVector(globalIndex: indexBase + k, dim: dim, n: n)
            withUnsafeBytes(of: &dimLE) { data.append(contentsOf: $0) }
            v.withUnsafeBufferPointer { buf in
                buf.withMemoryRebound(to: UInt8.self) { data.append(contentsOf: $0) }
            }
        }
        try! data.write(to: URL(fileURLWithPath: path))
    }
    writeFvecs(path: baseOut, count: n, indexBase: 0)
    writeFvecs(path: queryOut, count: q, indexBase: queryOffset)
    FileHandle.standardError.write(Data("[gen] base=\(baseOut) (\(n)x\(dim))  queries=\(queryOut) (\(q)x\(dim))\n".utf8))
}

// ─────────────────────────────────────────────────────────────────────────────
// measure — build + after-build memory + disk + latency + recall.
// ─────────────────────────────────────────────────────────────────────────────
func runMeasure(_ f: Flags) async throws {
    let n = f.int("--n", 10_000)
    let dim = f.int("--dim", 384)
    let q = f.int("--queries", 1_000)
    let k = f.int("--k", 10)
    let m = f.int("--m", 16)
    let efc = f.int("--efc", 200)
    let ef = f.int("--ef", 50)
    let warmup = f.int("--warmup", 200)
    let reps = f.int("--reps", 3)
    let levelSeed = UInt64(f.int("--level-seed", 0x50D0_C0DE))
    let indexOut = f.req("--index-out")
    let jsonOut = f.req("--json-out")
    let dataKind = f.str("--data", "gmm")
    let label = f.str("--label", dataKind == "uniform" ? "synthetic-uniform" : "synthetic-gmm")
    noiseSigma = Double(f.str("--sigma", "0.55")) ?? 0.55

    let clock = ContinuousClock()
    let metric = EuclideanDistance()
    let cfg = HNSWConfiguration(m: m, efConstruction: efc, efSearch: ef, levelSeed: levelSeed)

    // ── Baseline footprint (before any large allocation) ──────────────
    let footBaseline = physFootprintBytes()

    // ── Build: generate each vector on the fly and add; retain NO source
    //    array, so the after-build footprint reflects the INDEX only. ───
    let index = HNSWIndex(dimension: dim, metric: metric, config: cfg)
    let buildStart = clock.now
    for i in 0..<n {
        try await index.add(Vector(dataVector(globalIndex: i, dim: dim, n: n, kind: dataKind)), id: uuid(i))
    }
    let buildSeconds = (clock.now - buildStart).seconds

    let footAfterBuild = physFootprintBytes()
    let residentAfterBuild = residentSizeBytes()
    let afterBuildDeltaMB = Double(footAfterBuild &- footBaseline) / MB

    // ── Save to disk + measure file size ──────────────────────────────
    let snapshot = try await index.persistenceSnapshot()
    let indexURL = URL(fileURLWithPath: indexOut)
    try PersistenceEngine.save(snapshot, to: indexURL)
    let diskBytes = (try FileManager.default.attributesOfItem(atPath: indexOut)[.size] as? Int) ?? -1

    // ── Query set ─────────────────────────────────────────────────────
    let queries = (0..<q).map { Vector(dataVector(globalIndex: queryOffset + $0, dim: dim, n: n, kind: dataKind)) }

    // ── Exact ground truth via BruteForceIndex (mirrors GroundTruthBuilder) ──
    let brute = BruteForceIndex(dimension: dim, metric: metric)
    for i in 0..<n {
        try await brute.add(Vector(dataVector(globalIndex: i, dim: dim, n: n, kind: dataKind)), id: uuid(i))
    }
    var gtNeighbors: [Set<Int>] = []
    gtNeighbors.reserveCapacity(q)
    for query in queries {
        let res = await brute.search(query: query, k: k)
        gtNeighbors.append(Set(res.map { rowIndex(of: $0.id) }))
    }

    // ── Warmup (excluded from latency) ────────────────────────────────
    for w in 0..<warmup {
        _ = await index.search(query: queries[w % q], k: k)
    }

    // ── Timed query reps ──────────────────────────────────────────────
    var p50s: [Double] = []
    var p95s: [Double] = []
    var means: [Double] = []
    var recallSum = 0.0
    for rep in 0..<reps {
        var lat: [Double] = []; lat.reserveCapacity(q)
        for (j, query) in queries.enumerated() {
            let t0 = clock.now
            let results = await index.search(query: query, k: k)
            lat.append((clock.now - t0).milliseconds)
            if rep == 0 {
                let hits = results.reduce(0) { $0 + (gtNeighbors[j].contains(rowIndex(of: $1.id)) ? 1 : 0) }
                recallSum += Double(hits) / Double(k)
            }
        }
        lat.sort()
        p50s.append(percentile(sorted: lat, q: 0.50))
        p95s.append(percentile(sorted: lat, q: 0.95))
        means.append(lat.reduce(0, +) / Double(lat.count))
    }
    let recall = recallSum / Double(q)

    let out: [String: Any] = [
        "label": label,
        "scale": n,
        "dimension": dim,
        "metric": "euclidean",
        "k": k,
        "dataModel": dataKind == "uniform"
            ? [
                "kind": "uniform",
                "componentDistribution": "iid Uniform[0,1) per component",
                "seedFormula": "SplitMix64(0x9E37 * (globalIndex+1))",
                "queryOffset": queryOffset,
            ]
            : [
                "kind": "gaussian-mixture",
                "clusterCount": clusterCount(n: n),
                "noiseSigma": noiseSigma,
                "centerStreamSeed": "0xC0FFEE01",
                "baseNoiseSeed": "0xBA5E0001",
                "queryOffset": queryOffset,
            ],
        "hnsw": ["m": m, "efConstruction": efc, "efSearch": ef, "levelSeed": levelSeed],
        "queryCount": q,
        "warmup": warmup,
        "reps": reps,
        "queryOffset": queryOffset,
        "buildTimeSeconds": buildSeconds,
        "afterBuildResidentDeltaMB_physFootprint": afterBuildDeltaMB,
        "afterBuildResidentAbsMB_residentSize": Double(residentAfterBuild) / MB,
        "diskBytes": diskBytes,
        "diskMB": Double(diskBytes) / MB,
        "latencyP50Ms_median": median(p50s),
        "latencyP95Ms_median": median(p95s),
        "latencyP50Ms_perRep": p50s,
        "latencyP95Ms_perRep": p95s,
        "latencyMeanMs_perRep": means,
        "recallAt10": recall,
        "platform": platformJSON(),
        "generatedAt": ISO8601DateFormatter().string(from: Date()),
    ]
    writeJSON(out, to: jsonOut)
}

// ─────────────────────────────────────────────────────────────────────────────
// load — fresh-process cold-open (load) time + after-load resident memory.
// ─────────────────────────────────────────────────────────────────────────────
func runLoad(_ f: Flags) async throws {
    let dim = f.int("--dim", 384)
    let k = f.int("--k", 10)
    let n = f.int("--n", 10_000)
    let smoke = f.int("--smoke-queries", 50)
    let indexIn = f.req("--index")
    let jsonOut = f.req("--json-out")
    noiseSigma = Double(f.str("--sigma", "0.55")) ?? 0.55

    let clock = ContinuousClock()
    let footBaseline = physFootprintBytes()

    let t0 = clock.now
    let index = try PersistenceEngine.loadHNSW(from: URL(fileURLWithPath: indexIn))
    let loadMs = (clock.now - t0).milliseconds

    let footAfterLoad = physFootprintBytes()
    let residentAfterLoad = residentSizeBytes()

    // Smoke: prove the loaded index actually answers queries.
    var nonEmpty = 0
    for j in 0..<smoke {
        let res = await index.search(query: Vector(syntheticVector(globalIndex: queryOffset + j, dim: dim, n: n)), k: k)
        if !res.isEmpty { nonEmpty += 1 }
    }

    let out: [String: Any] = [
        "index": indexIn,
        "loadTimeMs": loadMs,
        "afterLoadResidentDeltaMB_physFootprint": Double(footAfterLoad &- footBaseline) / MB,
        "afterLoadResidentAbsMB_residentSize": Double(residentAfterLoad) / MB,
        "smokeQueries": smoke,
        "smokeNonEmpty": nonEmpty,
        "platform": platformJSON(),
        "generatedAt": ISO8601DateFormatter().string(from: Date()),
    ]
    writeJSON(out, to: jsonOut)
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry.
// ─────────────────────────────────────────────────────────────────────────────
let argv = CommandLine.arguments
guard argv.count >= 2 else {
    FileHandle.standardError.write(Data("usage: CardRig <measure|load|gen> [flags]\n".utf8)); exit(2)
}
let flags = Flags(Array(argv.dropFirst(2)))
switch argv[1] {
case "measure": try await runMeasure(flags)
case "load":    try await runLoad(flags)
case "gen":     runGen(flags)
default:
    FileHandle.standardError.write(Data("unknown subcommand \(argv[1])\n".utf8)); exit(2)
}
```

</details>
