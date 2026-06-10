# ADR-013: Streaming Persistence — Incremental Saves and On-Demand Paging

## Status
**Proposed — design only, NOT implemented.** Unlike every Accepted ADR in this
directory, no shipping code corresponds to this document. It is a worked
design for a future change, written so a future implementer does not have to
re-derive the analysis. All byte figures below are arithmetic from code facts
and the per-vector costs documented in ADR-007/ADR-011 — none are
measurements, because the design has not been built.

## Context

### What the code does today (code facts)

- **Every save rewrites the entire file.** `PersistenceEngine.save(_:to:)`
  serializes the full snapshot into one `Data` value (header, UUIDs, vectors,
  node levels, adjacency, metadata) and writes it with
  `data.write(to:options:.atomic)` — a temp file plus rename. The HNSW path
  does not even `reserveCapacity` (only the BruteForce path does), so the
  `Data` grows by geometric reallocation while ~the whole index is appended
  byte-by-section. Adding **one** vector and saving costs a full re-serialize
  and a full rewrite.
- **Saving may first rebuild the whole graph.** `HNSWIndex.persistenceSnapshot()`
  calls `compact()` whenever `liveCount < count`. `compact()` clears all
  storage and re-inserts every live node — O(n log n) graph construction —
  before a single byte is written.
- **Every load materializes everything.** `loadHNSW(from:)` reads the file
  with `.mappedIfSafe`, then copies every section into Swift arrays: one
  fresh `[Float]` heap allocation per vector via `readFloats` (~1,000,000
  small allocations at 1M scale), adjacency decoded into `[[[Int]]]`
  (8-byte `Int` in memory vs 4-byte `Int32` on disk), and
  `HNSWIndex.init(restoring:)` additionally rebuilds the derived reverse
  adjacency `inEdges: [[Set<Int>]]` plus the `uuidToNode` dictionary. The
  mapping is released when decoding finishes; the live index is fully
  resident (ADR-003 Correction, 2026-06).
- **The store layer compounds it.** `VectorStore.performSave()` writes
  `index.pxkt` + `docmap.json`; `HybridVectorStore.performSave()` writes
  `index.pxkt` + `index.pxbm` + `hybrid.json`. Each is dirty-flag gated, but
  *any* single mutation re-dirties the store, so the next `save()` is a full
  rewrite of every leg.

### Problem quantification: 1M × 384d, m = 16 (defaults)

Using the documented per-vector costs (Float32 vector = 1,536 B, ADR-007;
graph overhead ≈ 200 B/node at m = 16, ADR-011), the `.pxkt` sections are:

| Section | Arithmetic | Bytes |
|---|---|---|
| Header | fixed | 64 |
| UUIDs | 1,000,000 × 16 | 16,000,000 |
| Vectors | 1,000,000 × 384 × 4 | **1,536,000,000** |
| Node levels | 1,000,000 × 4 | 4,000,000 |
| Adjacency | ~200 B/node (layer-0 worst case alone: 4 + 32×4 = 132 B/node = 132,000,000) | ≈ 200,000,000 |
| Metadata | ≥ 4 B length word/node + payloads | ≥ 4,000,000 |
| **Total** | | **≈ 1.76 GB** |

Consequences of the current design at this scale:

- **Save:** one inserted vector → ≈ 1.76 GB serialized into a transient
  `Data` (a second copy of the index in RAM during save, on top of the
  resident index) → ≈ 1.76 GB written to flash. Write amplification is
  ~1,000,000× the logical change (≈ 1.6 KB of new payload). On-device flash
  wear and energy cost scale with save frequency.
- **Load:** ≈ 1.76 GB streamed through the decoder; ~1,000,000 small
  `[Float]` allocations; adjacency inflated 2× (Int32 → Int in memory) and
  then duplicated again into `inEdges` sets.
  Resident memory after load is the 1.536 GB vector payload **plus** graph
  arrays, reverse-edge sets, dictionaries, and per-allocation headers — a
  multiple of the vector payload's companions, comfortably north of 2 GB.
  That does not fit a foreground-app memory budget on most iPhones; the OS
  will jetsam the app long before the index finishes loading.
- **Cold start is linear in corpus size.** The README's measured ~50 ms load
  is for a 10K-vector index; the decode is O(file size), so 1M vectors is
  ~100× the bytes through the same loader.

ADR-003's correction already concedes the mapping is decode-transient only.
This ADR is the follow-through: make the *live* index file-backed where that
is sound, and make saves proportional to the change, not the corpus.

## Options

### Option A — Write-ahead log (WAL) over a base snapshot

A sidecar journal (`index.pxwal`) of mutation records appended after each
`add`/`remove`, replayed over the last full snapshot on load; periodic
checkpoint = today's atomic full save + WAL truncation.

- **Record framing:** `[length: UInt32][crc32: UInt32][payload]`, little-endian
  per house convention. `add` payload: UUID (16 B) + **assigned level** (4 B)
  + vector (d × 4 B) + metadata length/bytes. `remove` payload: UUID.
  ≈ 1.6 KB per add at 384d — versus 1.76 GB for today's save.
- **Why the level must be journaled:** `assignLevel()` draws from an RNG
  (`SplitMix64` only when `levelSeed` is set, system RNG otherwise, and
  `levelSeed` is deliberately not persisted). Replaying a bare `add` would
  re-draw levels and produce a *valid but different* graph each recovery.
  With the level in the record, insertion is deterministic end-to-end
  (`searchLayer` + heuristic selection are deterministic given levels), so
  recovery can be tested for **exact** state equality, not just validity.
- **Replay cost:** O(WAL ops × insertion cost). Bounded by the checkpoint
  policy (checkpoint when WAL bytes > ~10% of base or > N ops); worst-case
  recovery is one checkpoint interval of re-insertion.
- **Tradeoffs:** Fixes save cost only — load and resident memory are
  untouched. The crash-safety story changes shape: today `.atomic` rename
  means a save either fully happens or fully doesn't; a WAL is appended in
  place, so a crash leaves a torn tail that the loader must detect (CRC) and
  truncate, with documented prefix semantics. The base snapshot keeps the
  rename trick. Actor fit is clean: appends happen synchronously inside the
  index actor's mutation methods, so WAL order *is* actor serialization
  order; the stores' existing generation-counter discipline
  (`mutationGeneration`/`savedGeneration`) carries over to checkpointing.

### Option B — Segment-based persistence (LSM-flavored)

Immutable segments (each a small self-contained index file, written once with
`.atomic`) plus a tombstone set; new vectors accumulate in a resident
memtable that is flushed as a new segment; background merge compacts
segments and drops tombstones. This is the Lucene/LSM shape.

- **Pros:** Every file is immutable → per-file crash safety stays exactly
  today's rename model; no torn-write handling at all. Flushes are
  proportional to the memtable, not the corpus. Merge is incremental and
  schedulable (composes with the roadmap's "background HNSW compaction").
- **Cons (why it fights this codebase):** Search must fan out across all
  segments and k-way-merge results — k searches over k small HNSW graphs lose
  the single-graph O(log n) advantage; recall/latency now depend on segment
  count and merge policy, which invalidates the published single-graph
  benchmark story (ADR-005) until re-measured. Deletes become cross-segment
  tombstone lookups on every result. It is also the largest implementation:
  a segment manager, merge scheduler, and per-segment readers — effectively a
  storage engine. The actor model handles it (one coordinator actor, segment
  actors or value snapshots), but the surface area is the cost.

### Option C — True mmap paging of the vector region

Keep the graph resident; serve vectors directly from a read-only file
mapping, faulted in by the OS on demand.

- **Why the vector region is the one pageable section:** it is the only
  fixed-stride section. Vector `i` lives at
  `vectorSectionStart + i × dimension × 4` — O(1) addressing with no decode.
  It is also 87% of the file (1.536 GB of 1.76 GB), so paging it captures
  almost the whole win. Today the section starts at `64 + 16n` (16-byte
  aligned — already sufficient for `Float` binding), but **not page-aligned**;
  the v-next format should pad the section start to 16 KiB (Apple Silicon
  page size) so it can be mapped independently and each fault pulls a clean
  ~10.6-vector page.
- **What changes in `readFloats`:** for the paged region, nothing is "read"
  at load. The copying `readFloats` remains for non-paged sections; the
  vector section instead becomes a *view*: replace the actor's
  `vectors: [Vector]` with a vector-provider abstraction whose paged
  implementation returns scoped `UnsafeBufferPointer<Float>` access into the
  mapping (the metric layer already works through
  `withUnsafeBufferPointer` — see `Vector.components`), and whose resident
  implementation wraps today's arrays. Newly added (post-snapshot) vectors
  live in a resident tail array; node id < snapshotCount → mapped, else →
  tail.
- **What CANNOT page — the graph adjacency — and why:**
  1. *Variable-length encoding.* Adjacency rows are count-prefixed
     (`UInt32` count + neighbors), so there is no O(1) node → offset map;
     random access would require an offset index, i.e., a different format.
  2. *It is mutated in place.* Every `add` rewrites neighbor lists
     (pruning/heuristic reconnection); a read-only mapping cannot serve it,
     and a writable mapping forfeits crash atomicity entirely (no rename
     barrier — partially flushed pages after a crash are silent corruption,
     the exact failure class `.atomic` exists to prevent).
  3. *It is the traversal hot path.* Beam search dereferences a fresh node's
     neighbor row on every hop with random access across the whole region;
     a page fault per hop puts disk latency on the search critical path.
     Faulting *vector* pages costs one fault per ~10 candidates and only on
     cold pages; faulting *graph* pages stalls navigation itself.
  4. *Half the graph state is derived and memory-only anyway:* `inEdges`
     (reverse adjacency `Set`s) and `uuidToNode` are rebuilt at restore and
     have no on-disk representation to map.
  A fixed-stride padded adjacency (mMax0 slots per node) would make the
  graph *readable* by offset, but points 2–3 still hold; out of scope here.
- **mmap lifetime, file handles, Sendable (the Swift-specific tradeoffs):**
  - Ownership: a `final class MappedVectorRegion: @unchecked Sendable`
    owning the file descriptor and mapping, unmapping/closing in `deinit`,
    stored as a `let` on the index actor. The `@unchecked` is justified the
    same way the codebase justifies actor isolation generally (ADR-002):
    the region is confined to the actor; raw pointers are exposed only
    through scoped, actor-isolated calls and **must never escape across an
    `await`** (actor re-entrancy means a suspension can interleave a
    checkpoint/remap). `UnsafeBufferPointer` is not `Sendable`; the public
    surface stays value-typed, so `StrictConcurrency` continues to hold the
    line at compile time.
  - Crash safety vs `.atomic`: rename-replace stays safe *for the writer and
    for already-open readers* — the mapping pins the old inode, so a paged
    index keeps serving pre-checkpoint bytes until it reopens. The new
    exposure is external truncation of the mapped file: touching a faulted
    page past EOF raises SIGBUS, which Swift cannot catch. Mitigations:
    open read-only, `fstat` the size once at open, never truncate our own
    live files (checkpoints always rename a new inode), and document
    external mutation as out of contract. Note this risk class already
    exists today during the `.mappedIfSafe` decode pass; paging extends the
    window from load-time to index lifetime — that must be stated in the
    public docs, not hidden.
- **Tradeoffs:** Cold-start and resident memory are transformed; save cost is
  untouched. First-search-after-open is slower (faults), warm searches are
  byte-identical math on the same Float32 values, so recall is unchanged by
  construction.

### Option D — Hybrid: paged vectors + resident graph + WAL for mutations

Option C for reads + Option A for writes, composed: open = map the
page-aligned vector section, decode only graph/UUIDs/levels/metadata
(≈ 220 MB of the 1.76 GB file), replay the WAL into the resident tail and
graph. Mutations append WAL records; checkpoint rewrites base + truncates WAL.

This is not a fourth mechanism — it is the observation that A and C fix
disjoint halves of the problem (writes vs reads) and share no moving parts,
while B replaces both halves with a storage engine. The actor model makes the
composition tractable: one actor owns the mapping, the tail, the graph, and
the WAL handle, and its serialization order is the WAL order.

## Recommendation (v1)

**Option D, staged — Stage 1: WAL (Option A). Stage 2: paged vector region
(Option C). Reject B for v1** (largest cost, only option that degrades the
single-graph search story). Stage 1 ships value alone (saves go from O(corpus)
to O(change)) and is pure-Foundation, low-risk, reusing today's snapshot code
as its checkpoint. Stage 2 rides the format bump Stage 1 already forces.

### Format-version plan (per ADR-010)

- `.pxkt` v2 consumed the last reserved header bytes (ADR-010, Consequences),
  so this is the bump that "needs a new section": **`.pxkt` v3** keeps the
  64-byte legacy header and appends a section table (per-section offset +
  length), a **snapshot generation** (UInt64), and pads the vector section to
  a 16 KiB boundary. Monotonic bump, writers write 3, `minSupportedVersion`
  stays 1 (v1/v2 load exactly as today — resident — since they lack the
  table; the N-1 window of ADR-010 rule 2 is exceeded, not just met).
- **`PXWL` v1** (new sidecar, own magic + version): header carries the parent
  snapshot's generation; a WAL whose parent generation does not match the
  base header is discarded as stale (typed error or silent-discard —
  decide and document; recommendation: typed `PersistenceError` case, since
  silent discard hides data loss).
- ADR-010 rule 5 applies to both: corruption tests per format change (torn
  tail, CRC mismatch, stale generation, truncated section table, non-aligned
  vector section), plus the N-1 backward-compat test patching v3 → v2.
- `SQHW`/`PQHW` adopt the same section-table shape at their first bump;
  ADR-012's originals section (explicitly designed to compose with
  "memory-mapped or demand-paged" on-disk originals) is the first
  beneficiary — paged originals restore PQ's 32× resident-memory story while
  keeping exact reranking.

### Acceptance criteria

1. **Recovery-after-kill.**
   - *In-process truncation sweep (CI, deterministic):* build a fixture
     (seeded via `levelSeed` + journaled levels so replay is byte-exact),
     checkpoint, append K ops; for every truncation point of the WAL
     (every byte boundary across at least the final record, every record
     boundary across all of it), load and assert: no crash, typed errors
     only, recovered state == base + the longest valid record prefix.
   - *Out-of-process kill (macOS CI leg):* spawn a writer via `Process`,
     SIGKILL it at randomized delays while it ingests, reopen in the parent,
     assert the same prefix semantics. Repeated ≥ 100 iterations; any
     `corruptedData` on a file the library itself wrote is a failure.
   - Recovery must never trap — same standard the existing
     `PersistenceCorruptionTests` enforce for the base format.
2. **fsync policy (explicit, configurable, documented).** Default: one
   `fsync` per WAL batch (batch = the records appended by one mutation call),
   `F_FULLFSYNC` at checkpoint commit points. Darwin's `fsync(2)` reaches the
   drive cache, not the media — only `fcntl(F_FULLFSYNC)` forces media write;
   the docs must say which guarantee each policy level gives instead of
   implying durability the platform does not provide. Offer
   `.everyRecord` / `.everyBatch` (default) / `.manual` as the
   durability/throughput dial.
3. **Paged-mode memory.** A benchmark-class test (excluded from the PR job,
   like `RecallBenchmarkTests`) opens a large fixture in paged mode and
   asserts resident-delta (via `task_info`/`phys_footprint`) stays under the
   non-vector bound — i.e., the 4·d·n vector payload is demonstrably not
   resident. The threshold is set from a measured baseline when implemented,
   not invented now.
4. **Search parity.** Paged-mode search results are byte-identical to
   resident-mode on the same file (same Float32 bytes, same math); recall
   gates in CI run unchanged against paged mode. Cold/warm latency is
   measured and published via the existing benchmark harness before any
   performance claim is made (ADR-005 discipline — no hand-written numbers).
5. **Additive API only.** Existing `load(from:)`/`save(to:)` keep full
   resident/atomic behavior. New surface: an opt-in open mode (e.g.,
   `HNSWIndex.open(at:mode:)` with `.resident`/`.paged`) and a store-level
   journaling option. No existing call site changes behavior.

### Honest cost estimate

- Stage 1 (WAL: framing, replay, generation binding, checkpoint policy,
  store integration, recovery test rigs): **≈ 2 engineering weeks** — the
  kill-test harness is a deliverable of its own, not an afterthought.
- Stage 2 (v3 section table + alignment, `MappedVectorRegion`,
  vector-provider abstraction threaded through `searchLayer`/`add`/
  `compact`, paged-mode memory/latency benchmarks): **≈ 2–3 weeks**; the
  provider abstraction touches the hottest loop in the library and must be
  proven free of regression in resident mode first.
- Hardening + docs (corruption matrix, SIGBUS contract documentation,
  BENCHMARKS/README updates): **≈ 1–2 weeks**.
- Total: **~5–7 engineering weeks**, with Stage 1 shippable alone at ~2.
  Primary schedule risks: the provider abstraction's effect on resident-mode
  search latency, and actor-re-entrancy bugs around checkpoint/remap — both
  are why the stages are ordered this way.

## What this unlocks (Tier 3)

- **Corpora larger than RAM on iPhone.** Today 1M × 384d demands a
  multi-gigabyte resident footprint that a foreground iOS app does not get.
  Paged vectors cut residency to graph + dictionaries + the working set of
  vector pages the OS chooses to keep — the corpus on flash can exceed
  physical RAM, with the OS, not the app, managing the cache.
- **Instant cold start.** Open stops being O(file): it becomes map + decode
  of the non-vector ~12% of the file. The first query pays faults instead of
  the app paying a full-file decode at launch — the difference between a
  loading spinner and a usable search box.
- **Saves proportional to change.** Ingest-as-you-go (Lumen-style continuous
  indexing) stops costing a full-corpus rewrite per flush — ~1.6 KB of WAL
  per added chunk instead of gigabytes, which is also the difference between
  "save on every mutation" being an anti-pattern and a default.
- **PQ with a memory story *and* exact ranking.** ADR-012 paid 4·d bytes per
  vector to retain originals for reranking and explicitly deferred the fix to
  on-disk originals; a paged originals section is that fix.

## Consequences

- Two new durability concepts (prefix-valid WAL, paged read-only mappings)
  join the existing rename-atomic model; the docs must present all three
  honestly, including the SIGBUS contract and what each fsync level does and
  does not guarantee.
- The corruption-test surface grows by the full matrix in the format plan
  above; per ADR-010 rule 5 that is the price of every bump and is budgeted
  in the estimate.
- Until implemented, this ADR makes no performance or recall claims — every
  number above is either file-format arithmetic or an explicitly labeled
  acceptance gate to be measured under ADR-005 methodology.
