# ProximaKit Architecture

> Read this before modifying any core component. For decision rationale, see ADRs in `docs/adr/`.

## Module Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│            Consumer App  /  ProximaDemo (terminal demo)            │
├────────────────────────────────────────────────────────────────────┤
│  ProximaEmbeddings                                                 │
│    NLEmbeddingProvider · CoreMLEmbeddingProvider (+ WordPiece)     │
│    VisionEmbeddingProvider (image — standalone)                    │
│    EmbeddingProvider (protocol) ── refines ProximaKit.TextEmbedder │
├────────────────────────────────────────────────────────────────────┤
│  ProximaKit (core)                                                 │
│                                                                    │
│  Store/         VectorStore · HybridVectorStore · TextEmbedder     │
│  Index/         HNSWIndex · BruteForceIndex · SparseIndex (BM25)   │
│                 HybridIndex (RRF / weighted-sum fusion)            │
│                 QuantizedHNSWIndex · ScalarQuantizedHNSWIndex      │
│  Quantization/  ProductQuantizer · ScalarQuantizer                 │
│  Distance/      9 metrics + BatchDistance vDSP fast paths          │
│  Persistence/   PersistenceEngine + per-format binary codecs       │
│  Query/         SearchResult        Vector (vDSP value type)       │
│                                                                    │
│  Imports: Foundation, Accelerate ONLY                              │
└────────────────────────────────────────────────────────────────────┘
```

## Source Layout

| Directory | Contents |
|-----------|----------|
| `Sources/ProximaKit/Distance/` | `DistanceMetric` protocol, 9 metric implementations, `DistanceMetricType` (serialization), `BatchDistance` vDSP batch paths |
| `Sources/ProximaKit/Index/` | Six index types, `VectorIndex` / `SparseVectorIndex` protocols, BM25 tokenizers, internal `Heap`, quantized-index codecs |
| `Sources/ProximaKit/Quantization/` | `ProductQuantizer` (+ codec, errors), `ScalarQuantizer` |
| `Sources/ProximaKit/Persistence/` | `PersistenceEngine` (`.pxkt`), `SparseIndexPersistence` (`.pxbm`), `PersistenceError` |
| `Sources/ProximaKit/Store/` | `VectorStore`, `HybridVectorStore`, `TextEmbedder`, `ChunkMetadata`, store errors |
| `Sources/ProximaKit/Query/` | `SearchResult` |
| `Sources/ProximaEmbeddings/` | `EmbeddingProvider` protocol, NL / CoreML / Vision providers, `WordPieceTokenizer` |
| `Sources/ProximaDemo/` | Interactive terminal demo (`swift run ProximaDemo`) |

## Distance Layer

Nine `DistanceMetric` implementations: Cosine, Euclidean, DotProduct, Manhattan,
Hamming, Chebyshev, BrayCurtis, JensenShannon, Mahalanobis.

- Eight are stateless and serializable via `DistanceMetricType` (a `UInt32` enum
  stored in index file headers; `makeMetric()` reconstructs the instance on load).
- `MahalanobisDistance` carries a `dimension × dimension` inverse-covariance
  matrix, so it has no `DistanceMetricType` case — saving an index configured
  with it throws `PersistenceError.unserializableMetric`, same as custom metrics.

`BatchDistance` computes query-vs-N distances against a flat row-major matrix
(ADR-001). Fast paths by metric:

| Metric | Path |
|--------|------|
| Cosine, DotProduct | One `vDSP_mmul` for all dot products, then post-processing |
| Euclidean | `batchL2Distances` (`vDSP_svesq` norms + `vDSP_mmul` dots) |
| Manhattan | `batchL1Distances` (`vDSP_vsub`/`vabs`/`sve` per row) |
| All others | Scalar fallback through `metric.distance(_:_:)` |

Batch and scalar paths are required to agree element-wise — including the
zero-vector convention (cosine distance to a zero vector is `1.0`, neutral, on
both paths).

## Index Layer

<p align="center">
  <img src="assets/hnsw-search.svg" alt="Animated HNSW search: greedy descent across layers, then beam search on layer 0" width="720" />
</p>

| Index | Protocol | Search | Persists as | Notes |
|-------|----------|--------|-------------|-------|
| `BruteForceIndex` | `VectorIndex` | Exact, O(n) batch scan | `.pxkt` | Ground truth / small corpora |
| `HNSWIndex` | `VectorIndex` | Approximate, O(log n) | `.pxkt` | Tombstone deletes + auto-compaction |
| `SparseIndex` | `SparseVectorIndex` | BM25 (Lucene-style IDF) | `.pxbm` | Pluggable `BM25Tokenizer`; same tombstone/compaction model |
| `HybridIndex` | — (actor) | Both legs concurrently, then fusion | per leg | RRF (default, k=60) or weighted-sum fusion |
| `QuantizedHNSWIndex` | — (actor) | HNSW graph over PQ codes, ADC | `PQHW` codec | L2-only by construction (ADR-011) |
| `ScalarQuantizedHNSWIndex` | — (actor) | HNSW graph over INT8 codes, dequantize-per-candidate | `SQHW` codec | Metric-general except Hamming (ADR-007) |

All dense indexes support **filtered search** (ADR-008): an optional
`@Sendable (UUID) -> Bool` predicate, with default overloads supplied by
protocol extension. Every HNSW-graph index now shares one strategy:
`HNSWIndex` (ADR-008 first addendum), `QuantizedHNSWIndex`, and
`ScalarQuantizedHNSWIndex` (ADR-008 second addendum) all apply the predicate
**during** the layer-0 beam with the same adaptive `ef` widening formula
(`effectiveEf = clamp(ceil(target / rate), ef, efCap)`) — rejected nodes
(tombstoned, checked first, or filter-failing) still route the beam toward
matching regions but never occupy result slots, so selective filters still
fill `k` (acceptance gated by `FilteredSearchSelectivityTests` on all three
indexes). On `QuantizedHNSWIndex` with `retainOriginals` reranking
(ADR-012), the beam's adaptive target is `rerankDepth`, not `k`, composing
exactly as post-filter used to. `SparseIndex` remains **post-filter** — it is
a BM25 inverted index with an unbounded postings scan, not an `ef`-bounded
beam to route through, so the graph-aware mechanism doesn't structurally
apply (rationale in the ADR-008 second addendum); a selective filter can
still return fewer than `k` there. `BruteForceIndex` is exact under any
filter (full scan). `HybridIndex.search` applies the filter to both legs
before fusion, inheriting each leg's strategy — graph-aware on its dense leg
regardless of which of the three HNSW-family indexes it wraps.

`Heap` is an internal support type (beam search priority queues), not public API.

### Hybrid retrieval

`HybridIndex` fans writes out to a dense leg (`any VectorIndex`) and a sparse
leg (`any SparseVectorIndex`) under one shared UUID, queries both with
`async let`, and fuses the two rankings with a `HybridFusionStrategy`:

- `.rrf(k: 60)` — Reciprocal Rank Fusion (Cormack et al., SIGIR 2009). Scale-free; the robust default.
- `.weightedSum(alpha:)` — min-max-normalized weighted sum; `alpha` weights the dense leg, validated to `[0, 1]`.

Each leg contributes a candidate pool of `candidatePoolK` (default
`max(k * 5, 50)`) before fusion. See `docs/HYBRID.md` for the full design.

### The build-then-quantize pattern

Both quantized indexes share one construction pipeline:

1. Insert all vectors into a **full-precision** `HNSWIndex` — graph edges are
   chosen using exact distances, so graph quality is not degraded by
   quantization error.
2. Encode every vector (train a `ProductQuantizer` first for PQ; the scalar
   quantizer is stateless and needs no training).
3. `build(...)` extracts the finished graph + codes; the Float32 vectors are
   discarded by default. Search runs on compressed codes.

The two differ in how search computes distances:

- **PQ / ADC** (`QuantizedHNSWIndex`): a per-query distance table over the
  codebooks; asymmetric distance = full-precision query vs. compressed codes.
  ~48-byte codes for 384d (M=48) ≈ **32× memory reduction**, L2 only.
  **Opt-in reranking (ADR-012):** `build(retainOriginals: true)` keeps the
  Float32 vectors alongside the codes instead of discarding them. The layer-0
  beam still runs on ADC/compressed distances, but `search(rerankDepth:)`
  re-scores the top `rerankDepth` live, filter-passing candidates with exact
  Euclidean distance against those retained originals before the final sort
  — reranked recall@10 is asserted ≥ 0.90 in `PQRerankTests`. **Resident**
  retention forfeits the compression story above (originals cost the full
  `4·d` bytes/vector again); an opt-in **paged** open restores it while
  keeping rerank exact (ADR-014 — see "Paged originals for reranking" under
  Quantization Layer, below). The default (`retainOriginals: false`) behaves
  exactly as described in step 3.
- **INT8** (`ScalarQuantizedHNSWIndex`): each candidate is dequantized on the
  fly and fed to the configured metric, so cosine, euclidean, dot product,
  Manhattan, Chebyshev, and Bray-Curtis all work. 384d: 1536 B → 388 B ≈
  **3.96× memory reduction**. Hamming is excluded — lossy reconstruction
  destroys bit-equality semantics.

## Quantization Layer

- `ProductQuantizer` (ADR-011): k-means-trained codebooks over `M` subspaces,
  `PQConfiguration` for shape/training knobs, ADC via per-query distance
  tables. Has its own codec (`PQTT`) so trained codebooks can be reused.
- `ScalarQuantizer` (ADR-007): stateless symmetric INT8 — one signed byte per
  component plus a per-vector Float32 scale (`scale = maxAbs / 127`). No
  training phase, no codebook to persist; handles zero vectors and subnormal
  scales explicitly.

### Paged originals for reranking (ADR-014, opt-in)

A retaining `QuantizedHNSWIndex` (`retainOriginals: true`) holds its
full-precision originals behind an internal `OriginalsStore` — `.resident`
(an array, the historical and still-default behavior) or `.paged` (a
read-only file mapping). `.paged` serves the **single** rerank read site
inside `search(rerankDepth:)` copy-on-access from a `MappedVectorRegion` over
a `PQHW` v3 base's originals section — the same mapping machinery ADR-013
Stage 2 built for `.pxkt` vector sections, generalized to a resolved layout
rather than duplicated. The ADC beam itself never touches originals, so the
fault ceiling is bounded by `rerankDepth` and lands entirely *after* the beam
completes, never on the traversal critical path. Copy-on-access — the same
Stage-2 discipline — makes paged rerank results bit-identical to resident by
construction: the single read site is `originalsStore.vector(at:)`, a value
copied out of the mapping inside one synchronous, actor-isolated call.

Opt in with `QuantizedHNSWIndex.load(from:mode: .paged)` on a base written
`save(to:layout: .pagedV3)` (or upgraded in place from an existing v2 base via
`upgradeToV3(at:)` — see the persistence format table below); `.resident`
stays the default and is byte-identical to `load(from:)` before this ADR.
Accounting stays honest about which mode is active: a paged index reports
`originalStorageBytes == 0` (its originals are on flash, not resident) and
exposes the on-flash size separately via `mappedOriginalStorageBytes`;
`originalsArePaged` distinguishes the two `retainsOriginals` modes, and
`memorySavingsRatio` rises back to the pure-PQ ratio once paged, since
`originalStorageBytes` no longer counts against it. Measured (100K × 384d,
146.5 MB originals payload): paged open resident at 8.0 MB versus 43.1 MB for
the same base opened resident — see `docs/BENCHMARKS.md` for the full table
and the memory-accounting caveat it documents.

## Persistence Layer

One engine plus per-format codecs, all little-endian binary with
magic + version headers (ADR-003, ADR-010):

| Format | Magic | Current / min version | Codec | Used by |
|--------|-------|-----------------------|-------|---------|
| `.pxkt` | `PXKT` | 2 / 1 (v3 max-readable¹) | `PersistenceEngine` | `HNSWIndex`, `BruteForceIndex` |
| `.pxwal` (sidecar) | `PXWL` | 1 / 1 | `WALFormat` / `WALJournal` | `HNSWIndex`, opt-in journaling only |
| `.pxbm` | `PXBM` | 2 / 1 | `SparseIndexPersistence` | `SparseIndex` |
| quantized HNSW | `PQHW` | 2 / 1 (v3 max-readable²) | `QuantizedHNSWIndexPersistence` | `QuantizedHNSWIndex` |
| scalar-quantized HNSW | `SQHW` | 1 | `ScalarQuantizedHNSWIndexPersistence` | `ScalarQuantizedHNSWIndex` |
| PQ codebooks | `PQTT` | 1 | `ProductQuantizerPersistence` | `ProductQuantizer` |

¹ `.pxkt` v3 (ADR-013) appends a fixed trailer — a per-section offset/length
table plus a `snapshotGeneration: UInt64` — after the same v2 body. It is
written **only** by the streaming-persistence `checkpoint(...)` API below;
the legacy `save(to:)` keeps writing v2 byte-for-byte, unchanged. The
sequential loader stops after the metadata section, so v3 files load through
the identical resident path as v2 — only the WAL layer reads the trailer.
`minSupportedVersion` stays 1: v1/v2 files load exactly as before.
`checkpoint(...)` also zero-pads the vector section to a 16 KiB boundary (the
Apple-Silicon page size) so it can be mapped read-only for paged access
(Stage 2, below); the section table records the padded offset, and padded
and unpadded v3 bases both still decode identically through the resident
path.

² `PQHW` v3 (ADR-014) mirrors the `.pxkt` v3 shape: the 56-byte header and the
v2 body (codebooks → codes → UUIDs → node levels → adjacency → metadata) stay
byte-for-byte identical, and a 128-byte trailer is appended — a seven-section
`UInt64` offset/length table plus a reserved `snapshotGeneration`. When
originals are retained, the originals section is additionally zero-padded to
start on a 16 KiB boundary so it can be mapped read-only. **Writers stay
conservative by default:** `save(to:)` always writes v2, unchanged and
byte-identical to prior releases; `save(to:layout: .pagedV3)` is the new
opt-in surface, and it falls back to v2 when there is nothing to page (no
retained originals). `qhMinSupportedVersion` stays 1, so v1/v2/v3 all load
`.resident`; `load(from:mode: .paged)` additionally requires a padded v3 base
that retains originals (see "Paged originals for reranking" above).
`PersistenceEngine.upgradeToV3(at:)` (`.pxkt`) and
`QuantizedHNSWIndex.upgradeToV3(at:)` (`PQHW`) rewrite an existing v1/v2 (or
unpadded-v3) base to a padded v3 base **in place**, without decoding the graph
or materializing a single vector — a pure section-copy (bit-identical
payloads, full-checksum-verified, temp-file + atomic replace) so an on-device
app can enable paging on a base it already has without a full rebuild. See
`docs/BENCHMARKS.md` for the migration-cost arithmetic.

Shared invariants:

- **Format versioning** (ADR-010): every header carries a version; loaders
  accept `minSupportedVersion...formatVersion` and throw
  `PersistenceError.unsupportedVersion` otherwise. v1 `.pxkt` files (which
  predate threshold serialization) still load with documented defaults.
- **Corruption hardening**: bounds-checked little-endian readers throw
  `PersistenceError.corruptedData` instead of trapping — section truncation,
  out-of-range graph indices, implausible header fields, and invalid metric
  types are all validated before any state is constructed.
- **Atomic writes**: every save goes through `Data.write(options: .atomic)`,
  so a crash mid-save never leaves a torn file.
- **Loading**: files are read with `.mappedIfSafe` to make the decode pass
  cheap, but loaded indexes are *fully resident* — floats are copied into
  Swift arrays during decode. See the 2026-06 correction in ADR-003.

### Streaming persistence: WAL incremental saves (ADR-013 Stage 1, opt-in)

The default path above (`save(to:)` / `load(from:)`) is untouched: every call
still serializes and atomically rewrites the entire snapshot, exactly as
before. `HNSWIndex` additionally exposes an **opt-in** journaled path so
routine mutations no longer cost a full rewrite:

```swift
// Establish (or re-open) a journaled index bound to a base + sidecar file.
let index = try await HNSWIndex.open(
    baseURL: base, walURL: wal, durability: .everyBatch
)
try await index.add(vector, id: id)              // appends one WAL record
if await index.needsCheckpoint() {
    try await index.checkpoint(baseURL: base, walURL: wal)
}
```

- **Journal binding and cross-check.** Each `.pxwal` header binds a
  `parentGeneration: UInt64`, `dimension`, and metric raw value to the base
  it was written against. `open` validates all three against the *loaded*
  index: a generation mismatch throws `PersistenceError.walGenerationMismatch`
  (stale or checkpointed-past WAL), a dimension mismatch throws
  `.walDimensionMismatch`, and a metric mismatch throws `.walMetricMismatch`
  — a crafted or mispaired sidecar is rejected before a single record
  replays, rather than feeding mismatched-length vectors past `add`'s
  dimension guard.
- **Snapshot generation.** `.pxkt` v3's trailer carries the
  `snapshotGeneration` a WAL is bound to (see the persistence-format table
  above); `checkpoint` bumps it by one on every base rewrite.
- **Deterministic replay.** An `add` record journals the vector's
  **assigned HNSW level**, not just its UUID and components — insertion
  normally draws the level from an RNG, so replaying a bare `add` would
  produce a *valid but different* graph on every recovery. With the level
  recorded, replay reproduces the exact producing state — asserted
  structurally (adjacency, levels, entry point, tombstones, vectors,
  metadata), not merely by search validity, in `WALRecoveryTests`.
- **Checkpoint policy.** `checkpoint(baseURL:walURL:durability:)` compacts
  first if any tombstones are pending, writes a fresh v3 base with the
  generation bumped, `F_FULLFSYNC`s it, and resets the WAL to a new empty
  journal bound to that generation.
  `needsCheckpoint(policy:)` reports when to call it, per
  `WALCheckpointPolicy(walBytesFractionOfBase: 0.10, maxOps: 10_000)` —
  both bounds configurable; either being exceeded triggers a checkpoint.
- **fsync levels — Darwin honesty.** `WALDurability` offers `.everyRecord`,
  `.everyBatch` (default, one `fsync` per mutation's record), and `.manual`
  (no `fsync` on append; a power loss before the next checkpoint can lose
  the unsynced tail). On Darwin, `fsync(2)` only pushes writes to the
  **drive cache**, not the physical media — these levels do **not**
  guarantee media durability. Only `fcntl(_:F_FULLFSYNC)`, which
  `checkpoint` always calls on the base file, forces a media write. Choose
  the durability level knowing which guarantee it does and doesn't give.
- **Recovery is prefix-tolerant, not silently lossy.** Records are
  CRC-framed; the decoder stops at the first torn or bit-damaged record and
  returns the longest intact prefix — a torn tail from a crash mid-append is
  *expected* and recovers cleanly, never throwing. Only a damaged WAL
  *header* or a stale generation throws. This is proven, not just designed:
  an in-process sweep truncates at every byte boundary of the final record
  and every record boundary of the whole WAL and asserts exact-prefix
  recovery (`WALTruncationSweepTests`), and an out-of-process rig spawns a
  writer process and `SIGKILL`s it at randomized points, reopening in the
  parent and asserting the same semantics — a 5-iteration smoke runs in
  every CI run, and a ≥100-iteration heavy class runs opt-in behind
  `PROXIMA_RUN_KILL_RIG` (`WALKillRecoveryTests`).
- **Opt-in benchmark gates, one place.** A handful of env vars, each set to
  `1`, unlock heavyweight test classes that stay out of default CI:
  `PROXIMA_RUN_KILL_RIG` is the SIGKILL rig's ≥100-iteration heavy class
  above; `PROXIMA_PAGED_BENCH` unlocks both `PagedVectorMemoryTests` (backs
  the ADR-013 Stage 2 HNSW vector-region memory claims) and
  `PagedOriginalsMemoryTests` (backs the ADR-014 `PQHW` originals memory
  claims below), each a release-build test measuring `phys_footprint` (via
  `task_vm_info`) deltas between `.paged` and `.resident` open modes on the
  same base file; `PROXIMA_RESIDENT_BENCH` unlocks `ResidentRerankBenchTests`,
  the ADR-014 resident-mode no-regression A/B; and `PROXIMA_GPU_BENCH`
  unlocks `MetalBuildIntegrationDecisionTests`, an opt-in Metal build-scale
  (N=100,000) GPU-vs-vDSP correctness/parity check behind the ADR-009
  insert-loop integration NO-GO decision, skipped whenever no Metal device is
  present or the variable isn't set. None of these run by default; each earns
  its opt-in cost only when its specific question is being asked.
- **Scope, honestly stated.** Stage 1 shipped **index-level only** journaling;
  wiring it into `VectorStore` / `HybridVectorStore` was a deferred follow-up
  at the time — `HybridVectorStore` freezes the `VectorStore` v1 contract
  (CHA-107), the sparse leg had no WAL codec, and none of the Stage 1
  acceptance criteria required it. **That follow-up has since shipped** — see
  "Store-level journaling" below. Auto-compaction is suppressed
  while a journal is attached (compaction changes `count` in a way an
  append-only WAL can't replay) and deferred to the next `checkpoint`.
  Checkpoint itself has one narrow crash window: the base rename is the
  commit point, so a crash after it but before the WAL reset leaves a
  complete new base beside a stale WAL — the next `open` surfaces that as a
  typed `walGenerationMismatch`, not silent data loss (the new base already
  holds every committed record). Stage 2 of ADR-013 (paged, on-demand vector
  loading) has shipped — see the paged vector region below.
- **Paged vector region (ADR-013 Stage 2, opt-in).** `HNSWOpenMode.paged`
  serves the vector section from a read-only `mmap` (`MappedVectorRegion`)
  over the 16 KiB-padded v3 base instead of decoding it resident. Access is
  copy-on-access, not zero-copy: each read copies the requested vector out of
  the mapping into a value-typed `Vector` inside one synchronous,
  actor-isolated call, so no raw mapping pointer is ever held across an
  `await` — the deliberate trade that keeps paged results bit-identical to
  resident and makes the design trivially sound against actor re-entrancy.
  Checkpointing a paged index **remaps**: it writes the fresh padded base,
  `F_FULLFSYNC`s it, then opens a new mapping over the new inode and swaps
  the provider back to paged — compact, write, and swap all happen inside the
  same single synchronous, actor-isolated critical section, so a concurrent
  search can never observe torn state. Honestly stated: the mapping is opened
  read-only and ProximaKit never truncates its own live files, but
  truncating a mapped base from outside the library is out of contract and
  raises an uncatchable SIGBUS — the same risk class the `.mappedIfSafe`
  decode pass already carries at load time, now extended to the paged
  index's whole lifetime.
- **Unified residency vocabulary + store-level residency (ADR-015 Stages
  A/B, opt-in).** `IndexResidency` (`.resident` / `.paged`) is now the
  canonical enum shared by both HNSW-graph families; `HNSWOpenMode` (above)
  and `PQHWOpenMode` (Quantization Layer, below) are zero-breakage
  typealiases for it — both spellings compile and mean the same thing.
  `HNSWIndex.load(from:mode:)` mirrors `QuantizedHNSWIndex.load(from:mode:)`:
  `.resident` is byte-identical to `load(from:)`, and `.paged` loads through
  the same `MappedVectorRegion` path described above, throwing a typed
  `PersistenceError` (never a trap) when the base isn't a padded v3 file.
  At the store layer, `VectorStore.open(...)` and `HybridVectorStore.open(...)`
  (see "Store-level journaling" below) accept `dense: IndexResidency =
  .resident`, plumbed straight to the dense leg's `HNSWIndex.open(...,
  mode:)` — a journaled store with a paged dense leg is a one-parameter
  opt-in, not a second code path.

### Store-level journaling (opt-in, closes ADR-013 deviation 5)

`VectorStore` and `HybridVectorStore` gain async `open(...)` factories —
`VectorStore.open(name:embedder:storageDirectory:metric:config:durability:checkpointAutomatically:dense:)`
and
`HybridVectorStore.open(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:durability:checkpointAutomatically:dense:)`
— that establish WAL journaling at the store level: a fresh directory
creates an empty index and checkpoints it immediately (generation-1 base +
fresh WAL); an existing base, including a plain v2 base written by the
historical `save()`, opens and replays its WAL. Under journaling, `save()`
becomes an O(1) durability flush of the dense leg's WAL (`syncJournal()`) —
the sidecars are not rewritten, because **they are derived, not persisted
authorities**. `checkpoint()` is the periodic O(corpus) fold: compact → write
a fresh v3 base (generation bumped, `F_FULLFSYNC`) → reset the WAL → rewrite
the `docmap.json` / `index.pxbm` / `hybrid.json` caches for the non-journaled
reload path; `needsCheckpoint(policy:)` reports when to call it.

**The multi-file crash-consistency problem is solved by derivation, not
ordering.** A store is an index plus sidecars; naively journaling only the
dense index would let a crash leave a replayed index *ahead of* its
sidecars — an orphaned vector with no doc-map entry, or (worse, for hybrid)
a document searchable in the dense leg but absent from the WAL-less sparse
leg. Instead, the dense index and its WAL are the **single source of
truth**, and a journaled `open` always *rebuilds* the document map — and,
for `HybridVectorStore`, the entire BM25 sparse leg — from the recovered
index's live entries via the new `HNSWIndex.liveEntries()` hook, rather than
trusting whatever `docmap.json` / `index.pxbm` / `hybrid.json` happens to be
on disk. Because both projections are computed *from* the recovered index,
a doc-map entry exists iff a live dense vector exists, and a sparse entry
exists iff a live dense vector exists — orphans and phantoms are
structurally impossible, and any on-disk sidecar (stale, absent, or even
hand-corrupted) is simply **ignored** on a journaled open, never reconciled.

The historical (non-journaled) initializers, `save()`, `query()`,
`addChunks`, `removeDocument`, and `loadDocumentMap()` are unchanged and
byte-identical for every store not built through `open` —
`loadDocumentMap()` is a no-op on a journaled store, since its map is already
the authoritative projection. The CHA-107 contract (`HybridVectorStore` never
mutates `VectorStore`) is untouched; PXWL v1 and the v3 `.pxkt` format are
untouched too — this is store-layer wiring over the already-shipped
index-level WAL, not a new format.

**Automatic checkpointing (opt-in, ADR-015 Stage A).** Both `open(...)`
factories additionally accept `checkpointAutomatically: WALCheckpointPolicy?
= nil`. When set, the store checks the policy after every serialized
mutation batch (`addChunks`, `removeDocument`) and folds the WAL inside that
same actor-serialized mutation chain, so a concurrent batch can never land
between the apply and the fold. The honest failure contract: if the fold
itself throws, the error surfaces from the mutation call that triggered it,
but that triggering mutation has already been applied and made durable — do
not retry it, since `addChunks` assigns fresh UUIDs and there is no
idempotency key to make a retry safe. There is no failure latch: the store
stays consistent, and the next mutation, `save()`, or `checkpoint()` simply
re-attempts the fold. The default `nil` path is untouched and byte-identical
to before this option existed.

## Store Layer

Document-level API over the indexes, designed for RAG pipelines (ADR-006):

- `VectorStore` — wraps an `HNSWIndex` + any `TextEmbedder`. Add text chunks
  (auto-embedded), query by text, remove by document ID. Persists to a named
  directory: `index.pxkt` + `docmap.json` (document → UUID map).
- `HybridVectorStore` — same API shape over a `HybridIndex`
  (`index.pxkt` + `index.pxbm` + `hybrid.json`). Deliberately a sibling, not a
  subclass: the `VectorStore` contract stays frozen and hybrid is opt-in.
- `TextEmbedder` — minimal embedding protocol defined in ProximaKit so stores
  have no dependency on ProximaEmbeddings; `EmbeddingProvider` refines it, so
  every provider plugs in directly.

Compound operations (`addChunks`, `removeDocument`, `save`) are **serialized
through an internal operation chain**, and `save()` snapshots a monotonic
mutation-generation counter — the store is only marked clean if no mutation
landed while the write was in flight. This closes the actor-reentrancy
lost-update window without blocking reads.

Both stores additionally support an **opt-in journaled** construction path —
`VectorStore.open(...)` / `HybridVectorStore.open(...)` — that streams
mutations to the dense leg's WAL instead of paying a full rewrite on every
`save()`; see "Store-level journaling" under Persistence Layer, above, for
the derivation design that keeps the sidecars consistent across a crash. The
historical initializers are untouched by this — journaling is opt-in.

## Data Flow

```
Index   : text ─▶ TextEmbedder.embed() ─▶ Vector ─▶ index.add(vector, id:)
Query   : text ─▶ embed() ─▶ Vector ─▶ index.search(query:k:) ─▶ [SearchResult]
Hybrid  : add/search fan out to dense + sparse legs ─▶ fuse (RRF | weighted)
Quantize: vectors ─▶ full-precision HNSW build ─▶ encode ─▶ drop Float32s (or retain, ADR-012)
Persist : actor snapshot ─▶ codec ─▶ atomic write ─▶ load() ─▶ new actor
```

## Module Rules

| Module | Can Import | Cannot Import |
|--------|-----------|---------------|
| ProximaKit | Foundation, Accelerate | ProximaEmbeddings, UIKit, SwiftUI |
| ProximaEmbeddings | ProximaKit, CoreML, NaturalLanguage, Vision | UIKit, SwiftUI |
| ProximaDemo | Everything | — |

## Concurrency Model (ADR-002)

- Every index and store is an `actor`. Reads queue, writes serialize.
- Immutable configuration is readable without data races; `HNSWIndex` and
  the quantized indexes mark `dimension`/`configuration` (and, on
  `ScalarQuantizedHNSWIndex`, `metricType`) as `nonisolated` for
  `await`-free access.
- `Vector`, `SearchResult`, snapshots, configurations, and both quantizers are
  `Sendable` value types; search filters are `@Sendable`.
- Actor *reentrancy* is handled explicitly where it matters: stores serialize
  compound operations and gate `save()` on a mutation-generation counter
  (see Store Layer above).
- Batch embedding uses `TaskGroup`; `HybridIndex` queries its legs with
  `async let` so neither leg waits on the other's queue.

## Reproducibility

`HNSWConfiguration.levelSeed: UInt64?` seeds layer assignment (SplitMix64).
Set it for bit-reproducible graph construction in tests and benchmarks; leave
`nil` for production. The seed is a build-time knob and is not persisted.

## Performance-Critical Paths

1. Distance computation → vDSP batch ops, single `vDSP_mmul` for dot products (ADR-001)
2. HNSW neighbor selection → heuristic algorithm (ADR-004)
3. PQ search → per-query ADC distance tables, no decode in the inner loop (ADR-011)
4. Beam search → internal min/max `Heap` on layer 0
5. Cold start → `.mappedIfSafe` read + bounds-checked single-pass decode (ADR-003)

## Extension Points

| Add a... | Location | Conform to |
|----------|----------|-----------|
| Distance metric | `Sources/ProximaKit/Distance/` | `DistanceMetric` (add a `DistanceMetricType` case if stateless, so it persists) |
| Dense index | `Sources/ProximaKit/Index/` | `VectorIndex` (actor) |
| Sparse index | `Sources/ProximaKit/Index/` | `SparseVectorIndex` (actor) |
| BM25 tokenizer | `Sources/ProximaKit/Index/` | `BM25Tokenizer` |
| Embedding provider | `Sources/ProximaEmbeddings/` | `EmbeddingProvider` |
| Store embedder | anywhere (no ProximaEmbeddings needed) | `TextEmbedder` |
