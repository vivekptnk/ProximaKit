# ADR-013: Streaming Persistence — Incremental Saves and On-Demand Paging

## Status
**Accepted (Stage 1: WAL / Option A + Stage 2: paged vectors / Option C).**
Stage 1 (the `.pxwal` sidecar, `.pxkt` v3 generation binding, deterministic
replay, checkpoint policy, fsync dial, and the recovery test rigs) shipped
first. Stage 2 (16 KiB vector-section padding in v3, `MappedVectorRegion`, the
resident/paged vector-provider abstraction threaded through the search/insert
hot path, the additive `.resident`/`.paged` open mode, and the checkpoint-remap
story) now ships too, proven free of resident-mode regression before landing.
See the **Stage 1 implementation notes** and the **Stage 2 implementation
notes** addenda at the end of this document for what was built, the format
bytes and measured numbers as implemented, and the deviations from the original
design (all documented, none silent). Acceptance criteria 3–4 are now met with
measured evidence; the byte figures in the Context section remain arithmetic
projections at 1M scale (the acceptance numbers below are real measurements at
100K scale, labeled with machine + build mode).

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

## Stage 1 implementation notes (addendum, M3-F29)

This addendum records what Stage 1 actually shipped and where it deviates from
the design above. Correction/addendum style per the ADR-003 precedent: the
body of this ADR is the original design; this section is authoritative for the
built code.

### What shipped

- **`PXWL` v1 sidecar** (`WALFormat.swift`, `WALJournal.swift`). Fixed 32-byte
  header: magic `"PXWL"`, version, `parentGeneration: UInt64`, dimension,
  metric raw, and a **header CRC32 over the first 24 bytes** (added beyond the
  design so a damaged header is a typed error, not a mis-parse). Records are
  framed `[payloadLength: UInt32][crc32: UInt32][payload]`, little-endian.
  `add` payload = opcode + UUID + **assigned level (Int32)** + vector
  (d × Float32) + metadata length/bytes; `remove` = opcode + UUID. CRC is
  IEEE-802.3 (`0xEDB88320`), a small pure-Foundation implementation to keep the
  zero-dependency contract.
- **Torn-tail / prefix semantics.** The decoder stops at the first record whose
  frame runs past EOF or whose CRC fails, returning the longest intact prefix
  and the dropped-byte count. A torn tail is **not** an error — recovery
  returns the recovered index. Only a damaged header or a stale
  `parentGeneration` throws. Stale generation is the typed
  `PersistenceError.walGenerationMismatch(expected:found:)` (the ADR's
  recommended choice over silent discard).
- **`.pxkt` v3** (`PersistenceEngine.saveHNSW(_:generation:to:)`,
  `readGeneration(from:)`). The 64-byte legacy header and the v2 body are kept
  byte-for-byte; a fixed 96-byte **trailer** is appended after the metadata
  section: `sectionCount`, a per-section `(offset,length)` table for the five
  sections, `snapshotGeneration: UInt64`, and a trailer magic `"PXK3"`. The
  sequential loader stops after the metadata section, so **v3 loads through the
  unchanged resident path exactly like v2**; only the WAL layer reads the
  trailer. `minSupportedVersion` stays 1; `maxReadableVersion` is 3. v1/v2 load
  exactly as today (covered by the existing corruption/version tests, still
  green).
- **Deterministic replay with exact equality.** Because the assigned level is
  journaled, replay re-inserts through the same deterministic
  `searchLayer`/heuristic path and reproduces the **byte-exact** producing
  state — asserted structurally (adjacency, levels, entry point, tombstones,
  vectors, metadata), not merely by search validity
  (`WALRecoveryTests.testReplayReproducesExactState`).
- **Checkpoint** (`HNSWIndex.checkpoint(baseURL:walURL:durability:)`) reuses the
  atomic full save as the base rewrite: compact → write v3 base with generation
  bumped → `F_FULLFSYNC` → reset the WAL to a fresh empty journal on the new
  generation. Policy helper `needsCheckpoint(policy:)` with
  `WALCheckpointPolicy(walBytesFractionOfBase: 0.10, maxOps: 10_000)` — both
  configurable, defaults per the ADR.
- **fsync dial** `WALDurability { .everyRecord, .everyBatch (default), .manual }`
  with `F_FULLFSYNC` at checkpoint commit. Doc comments state the Darwin
  guarantee precisely (`fsync` reaches the drive cache; only `F_FULLFSYNC`
  forces media) — no overpromising.
- **Additive, opt-in API.** `save(to:)`/`load(from:)` are untouched and still
  write/read **v2** (byte-identical). Journaling is a new surface:
  `HNSWIndex.open(baseURL:walURL:durability:)`,
  `checkpoint(...)`, `syncJournal()`, `needsCheckpoint(...)`,
  `closeJournal()`. Existing non-journaled call sites take an unchanged path
  (the `journal == nil` branch).
- **Recovery test rigs.** In-process truncation sweep at every record boundary
  and every byte of the final record, asserting exact-prefix equality
  (`WALTruncationSweepTests`); out-of-process `Process` + SIGKILL rig with a
  5-iteration smoke that runs in CI and a ≥100-iteration heavy class gated
  behind `PROXIMA_RUN_KILL_RIG` / `--skip WALKillRecoveryTests`
  (`WALKillRecoveryTests`, spawn target `WALKillWriter`). Every new
  regression/acceptance test was shown to fail under a code perturbation and
  pass on restore.

### Deviations from the design (documented)

1. **Legacy `save(to:)` keeps writing v2, not v3.** The design says "writers
   write 3." Acceptance criterion 5 and the harness ground rules require the
   existing `save(to:)`/`load(from:)` to stay byte-identical, so only the new
   streaming-persistence checkpoint writer stamps v3. v3 is otherwise a strict
   superset (v2 body + trailer), so this costs nothing and preserves the frozen
   contract. Consequence: a base only carries a generation once it has been
   through a `checkpoint`; a plain-saved v2 base reads generation 0.
2. **16 KiB vector-section padding deferred to Stage 2.** The design pads the
   vector section start to a page boundary as part of the v3 bump. That padding
   has **zero function in Stage 1** (which never memory-maps the vector region)
   and would bloat every v3 file plus require threading a padded offset through
   the sequential reader — pure Stage-2 (Option C) plumbing. The v3 trailer
   ships a per-section offset/length table precisely so Stage 2 can add the
   padding as a localized change without another version bump.
3. **Auto-compaction is suppressed while a journal is attached.** Compaction
   physically drops tombstone slots, changing `count` in a way an append-only
   WAL cannot reproduce on replay. So a *journaled* index defers compaction to
   the next `checkpoint` (which rewrites the base and truncates the WAL);
   non-journaled indexes keep today's exact auto-compaction behavior. This is an
   additive branch on the new opt-in mode, so existing call sites are unchanged.
4. **Checkpoint crash window.** True cross-file atomicity of "rewrite base +
   truncate WAL" is not possible with a single rename barrier. The base rename
   is the commit point; a crash after it but before the WAL reset leaves a
   complete new base beside a stale (generation − 1) WAL, which the next `open`
   surfaces as a typed `walGenerationMismatch` (no silent loss, no corruption —
   the new base already holds every committed record; the operator deletes the
   stale WAL to proceed). The kill rig exercises kills during *ingest* (the
   torn-tail path), not this checkpoint window.
5. **Store-level journaling deferred.** The index-level WAL API is designed to
   compose with the stores' `mutationGeneration`/`savedGeneration` discipline
   (WAL append order = actor serialization order; `save()` → `checkpoint()`),
   but wiring it into `VectorStore`/`HybridVectorStore` was not shipped in
   Stage 1: `HybridVectorStore` explicitly freezes the `VectorStore` v1 contract
   (CHA-107), the sparse leg has no WAL codec yet, and none of the in-scope
   acceptance criteria (1, 2, 5) require it. It is a thin, additive wrapper over
   the shipped index API and is left as a scoped follow-up.

### Acceptance criteria status

- **1 (recovery-after-kill): met.** In-process truncation sweep + out-of-process
  SIGKILL rig, prefix semantics asserted; no `corruptedData` on a
  library-written WAL; recovery never traps.
- **2 (fsync policy): met.** Configurable dial, documented Darwin guarantees,
  `F_FULLFSYNC` at checkpoint.
- **3, 4 (paged memory / search parity): shipped in Stage 2** — see the Stage 2
  implementation notes below.
- **5 (additive API only): met at the index level; store wiring deferred
  (deviation 5).** `save`/`load` unchanged; journaling is opt-in.

## Stage 2 implementation notes (addendum, paged vectors / Option C)

This addendum records what Stage 2 actually shipped. Same convention as the
Stage 1 addendum: the body of this ADR is the original design; this section is
authoritative for the built code. Stage 2 rides the v3 format Stage 1 already
forced and adds paging as a localized change — no new version bump.

### Measurement provenance

All numbers below were measured on: **Apple M4 Max, macOS 26.0.1 (Darwin
25.0.0), 36 GB, page size 16384, Swift 6.2, `-c release`.** Search-latency
numbers come from the `ProximaBench search-provider-bench` harness (public API
only, seeded fixture); memory numbers from `PagedVectorMemoryTests` (release,
`PROXIMA_PAGED_BENCH=1`). Per ADR-005 discipline these are the only performance
claims Stage 2 makes; both are reproducible from a seed.

### What shipped

- **16 KiB vector-section padding in v3 (`PersistenceEngine`).** The v3
  checkpoint writer (`saveHNSW(_:generation:to:)`) now zero-pads so the vector
  section START lands on a 16384-byte (Apple-Silicon page) boundary; the v3
  section table records the padded offset. Only the vector section is padded, so
  the sections after it stay contiguous with it. The resident loader
  (`loadHNSW`) reads the vector-section offset from the table and jumps to it, so
  it decodes **padded and unpadded v3 identically** — Stage-1-written unpadded v3
  files keep loading (verified by `PagedVectorCorruptionTests.testPaddedAndUnpadded
  V3LoadIdentically` and `.testUnpaddedV3PagedRejectedButResidentLoads`, which
  round-trip an unpadded v3 produced via the internal
  `saveHNSW(…, padVectorSection: false)`). The trailer parse
  (`readV3Trailer`) now bounds-checks every section entry, so a section-table
  offset past EOF or a non-aligned claimed offset is a typed `PersistenceError`,
  never a trap (`PagedVectorCorruptionTests.testVectorOffsetPastEOFRejected`,
  `.testNonAlignedVectorOffsetRejected`).
- **`MappedVectorRegion` (`Persistence/MappedVectorRegion.swift`).** A
  `final class … @unchecked Sendable` that opens the base read-only, `fstat`s its
  size once, and `mmap`s just the (page-aligned) vector section `PROT_READ`,
  `MAP_PRIVATE`; `munmap` + `close` in `deinit`; it never truncates. The
  `@unchecked Sendable` is justified exactly as ADR-002 justifies actor-isolated
  state and as Stage 1 justifies `WALJournal`: the region is confined to the
  `HNSWIndex` actor and its raw pointer is dereferenced only in synchronous,
  actor-isolated calls, never escaping an `await`. The **SIGBUS contract** is
  documented verbatim in the file's header doc and on the public `open` mode:
  read-only + `fstat`-once + never-truncate-our-own-files closes every window the
  library controls; the residual exposure is *external* truncation of a mapped
  file (out of contract), the same risk class the `.mappedIfSafe` decode pass
  already carries, extended from load-time to index lifetime.
- **Vector-provider abstraction (`VectorProvider`, same file).** A value type
  storing actor-isolated state that unifies vector access: resident mode holds
  everything in a `[Vector]` tail (`snapshotBoundary == 0`), so `vector(at:)`
  reduces to a single array subscript; paged mode serves node ids
  `< region.count` from the mapping and post-snapshot ids from the resident tail.
  It is threaded through `searchLayer`, `searchLayer0Filtered`, `insertNode`,
  `selectNeighborsHeuristic`, `pruneConnections`, `primitiveRemove`, `compact`,
  and the snapshot builders. **Copy-on-access, not zero-copy pointers:** the
  paged accessor copies each requested vector into a value-typed `Vector` inside
  one synchronous scope. This is a deliberate simplification over the ADR's
  sketch of scoped `UnsafeBufferPointer` access — it makes paged results
  bit-identical to resident (same Float32 bytes, same math) and makes the
  actor-reentrancy/remap story *trivially* sound (no raw mapping pointer ever
  exists across a suspension, because none is ever handed out), at the cost of a
  transient per-access copy on the paged path only. The resident path is
  untouched by this. (Zero-copy scoped access remains a possible future
  optimization under ADR-005 measurement.)
- **Additive `.resident` / `.paged` open mode (`HNSWOpenMode`).**
  `open(baseURL:walURL:durability:mode:)` gains a defaulted `mode` parameter;
  `.resident` is the default and byte-identical to before, so every existing call
  site is unchanged. `.paged` requires a padded v3 base (any `checkpoint` writes
  one); a non-v3 or unpadded base throws a typed error. Post-open adds and WAL
  replay compose: they append to the resident tail (verified by
  `PagedVectorParityTests.testPagedParityAfterWALReplay` — 300 mapped + 120
  replayed).
- **Checkpoint-remap choice (documented).** A checkpoint necessarily renumbers
  nodes (compaction) and writes an all-resident base, so the pre-checkpoint
  mapping cannot serve the new numbering. The chosen story: the checkpoint
  **re-maps** — after the padded v3 base is written and `F_FULLFSYNC`-ed, a paged
  index opens a fresh `MappedVectorRegion` over the *new* inode and swaps the
  provider back to paged (`snapshotBoundary = count`, tail cleared). The entire
  `checkpoint` is a single synchronous, actor-isolated critical section — no
  `await` between compact, write, and swap — so a concurrent search can never
  observe a torn intermediate state (the ADR's "no window where search sees torn
  state" requirement). This is the actor-local realization of the ADR's "keep
  serving the old inode until reopen" note: the old inode is pinned by the old
  mapping until it is dropped by the swap. Peak residency during the checkpoint
  equals a resident checkpoint's; steady-state paged residency is restored
  immediately (`PagedVectorParityTests.testPagedCheckpointRemapKeepsParity`).

### Acceptance criteria 3 & 4 — measured

- **3 (paged-mode memory): met.** `PagedVectorMemoryTests` (release,
  `PROXIMA_PAGED_BENCH=1`) builds a 100,000 × 384d fixture (vector payload =
  146.5 MB), then measures `phys_footprint` (`task_vm_info`) deltas on the SAME
  base: **paged open = 18.1 MB, resident open = 112.3 MB** → **94.1 MB (64% of
  the payload) recovered** by not residenting the vectors, and **+50 warm
  searches faulted only 0.2 MB** (bounded working set, not the corpus). The test
  asserts `residentDelta − pagedDelta ≥ 60%` of the payload (threshold derived
  from this measured baseline with margin). It is benchmark-class and
  env-gated, excluded from the PR job like `RecallBenchmarkTests`. This 64%
  isn't a downward revision of the original design's 87% projection (1.536 GB
  of 1.76 GB, above) — that figure was against total file bytes, while this
  one isolates the vector-section payload alone (146.5 MB), a different and
  much smaller denominator.
- **4 (search parity): met.** `PagedVectorParityTests` proves `.paged` results
  are **byte-identical** to `.resident` on the same file — same ids AND
  bit-equal Float32 distances (raw bit patterns compared) — across seeded
  queries, including graph-aware **filtered** search and **post-WAL-replay**
  state, plus a checkpoint-remap parity case and a recall check against brute
  force. Parity holds by construction: the paged accessor copies the identical
  Float32 bytes the resident loader reads, and the beam traversal is
  deterministic given identical distances. Every new test was red-green proven
  (e.g. a +0.001 perturbation of the paged accessor breaks the distance-bit
  assertion; disabling the padded-offset jump breaks the padded-v3 load).

### Resident-mode regression proof (the gate the ADR ordered first)

The provider abstraction touches the hottest loop in the library, so it was
proven free of resident-mode regression **before** landing. Harness:
`ProximaBench search-provider-bench`, seeded 50,000 × 128d fixture, k=10,
efSearch=50, 2,000 queries/rep, 9 reps + 2 warmup, `-c release`, identical seed
(so the graph is bit-identical before/after; only per-query search compute
differs). Per-query medians (ns):

| | run 1 | run 2 |
|---|---|---|
| before (resident `[Vector]`) | 264,889 | 256,919 |
| after (provider abstraction) | 257,029 | 258,207 |

The before/after distributions fully overlap and the medians interleave; the
most adversarial comparison (after-worst 258,207 vs before-best 256,919) is
**+0.5%**, far under the ADR's 2% bail-out threshold — no measurable regression
(if anything a wash in the favorable direction). The resident accessor compiles
to `tail[node − 0]`, a couple of integer ops against a vDSP distance, consistent
with the measurement. Search-result checksums were identical across all runs.

### Out of scope (unchanged from the design)

Save cost is untouched by Stage 2 (that is Stage 1's win). The graph adjacency
stays resident and is not paged, for the four reasons the Options/Option C
section gives (variable-length encoding, in-place mutation, traversal hot path,
derived-only reverse edges). `SQHW`/`PQHW` paged originals (ADR-012's deferred
fix) remain a follow-up: they adopt the same section-table + padding shape but
are a separate change.
