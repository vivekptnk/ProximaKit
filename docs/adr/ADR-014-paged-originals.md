# ADR-014: Paged Originals for Quantized Reranking

## Status

**Proposed — design only, NOT implemented.** Nothing in this document is built.
Every byte, percentage, and count below is either file-format arithmetic or an
acceptance gate to be measured under ADR-005 methodology **when** implemented;
this ADR makes no performance or recall claim of any kind. It is the design
[ADR-012](ADR-012-pq-reranking.md) deferred — its "Memory math" section closes
by noting the design "composes with future on-disk originals (memory-mapped or
demand-paged), which would restore the memory story while keeping this exact
same search path" — and the design [ADR-013](ADR-013-streaming-persistence.md)
Stage 2 enabled: the v3 section-table trailer, 16 KiB vector-section padding,
`MappedVectorRegion`, and copy-on-access parity all shipped for HNSW vectors,
and this ADR applies that same shape to the quantized index's **originals**
section.

## Context

[ADR-012](ADR-012-pq-reranking.md) shipped exact reranking via
`retainOriginals: true`, but the retained originals are fully **resident**. A
retaining index therefore has no compression story: memory ≈ original + codes +
graph — ADR-012's "Memory math" states it plainly, "1536 B originals + 48 B
codes ≈ 1584 B/vector — slightly *more* than plain full-precision HNSW, not 32×
less," and its Consequences repeat it: "An index built with
`retainOriginals: true` no longer has a compression story." ADR-013 Stage 2
shipped the enabling plumbing for the HNSW vector section. This ADR designs the
application of that plumbing to `PQHW`'s originals section, restoring the 32×
resident story on the vector payload while keeping reranking exact by
construction.

### What the code does today (code facts)

- **The retained originals are resident on the actor.**
  `QuantizedHNSWIndex` stores `var originals: [Vector]?`
  (`QuantizedHNSWIndex.swift:96`), documented slot-aligned with `codes` so that
  `originals[i]` is the full-precision vector PQ-encoded into `codes[i]`
  (`:88–94`). Retention is an opt-in build parameter,
  `retainOriginals: Bool = false` (`:219`), wired as
  `originals: retainOriginals ? snapshot.vectors : nil` (`:277`) from the same
  compacted snapshot the codes are encoded from, so codes and originals share
  one node order. Accounting derives from it: `retainsOriginals` (`:125`),
  `originalStorageBytes` (`:129–131`), and
  `memorySavingsRatio = equivalentFullPrecisionBytes / (codeStorageBytes + originalStorageBytes)`
  (`:139–143`), which drops below 1.0 once originals are retained.
- **Exactly one site in the search path dereferences an original.**
  `searchImpl(query:k:efSearch:rerankDepth:filter:)` spans
  `QuantizedHNSWIndex.swift:380–486`. Reranking engages only when
  `rerankDepth > 0 && originals != nil` (`:391`); the beam is widened to
  `ef = max(efSearch ?? hnswConfig.efSearch, k, reranking ? rerankDepth : 0)`
  (`:395`). The rerank loop (`:449–464`) walks ADC-ordered candidates,
  `guard taken < rerankDepth else { break }` (`:451`); the liveness gate
  `uuidToNode[uuid] == node` (`:456`) and the filter gate (`:457`) **both**
  precede any originals access; `taken += 1` (`:458`); and the single original
  read in the whole search path is `metric.distance(query, originals[node])`
  (`:461`, with `metric = EuclideanDistance()` at `:448`). One query therefore
  touches **at most `rerankDepth`** distinct originals, and dead or filtered
  slots are skipped before any originals access. The ADC beam itself
  (`searchLayerADC` and the filtered variant) runs entirely on resident codes
  plus the query's distance table and never touches originals.
- **The default search auto-reranks; rerank-without-retention throws.** The
  non-throwing `search(query:k:efSearch:filter:)` computes
  `autoDepth = originals != nil ? 4 * max(k, 0) : 0` (`:316`). The throwing
  overload throws `QuantizedIndexError.originalsNotRetained` when
  `rerankDepth > 0` and `originals == nil` (`:369–370`) rather than silently
  falling back to ADC — a deliberate fail-fast so a ~30%-recall
  misconfiguration cannot hide behind normal-looking results (ADR-012).
- **The mutation surface is remove-only; new vectors mean a full rebuild.**
  `public func remove(id: UUID) -> Bool` (`:500`) tombstones: it disconnects
  edges with a full reverse sweep (`:504–510`) and calls
  `uuidToNode.removeValue` (`:512`), with **no** neighbor reconnection (the
  diversity heuristic's vectors were discarded at build time — ADR-011). There
  is no `add`/`insert`: the only construction path is `static func build(...)`
  (`:212`), which internally builds a transient full-precision `HNSWIndex` via
  `add` (`:235`), trains PQ, and snapshots. A new vector is a full rebuild.
- **`PQHW` is a cursor-walk format with no offset table.** `qhMagic = 0x50514857`
  ("PQHW"), `qhFormatVersion = 2`, `qhMinSupportedVersion = 1`,
  `qhHeaderSize = 56` (`QuantizedHNSWIndexPersistence.swift:39–42`); the read
  guard is `version >= qhMinSupportedVersion, version <= qhFormatVersion`
  (`:240`) — there is no separate max-readable constant. The 56-byte header
  (write `:127–142`, read `:233–267`) is: magic@0, version@4, dimension@8,
  nodeCount@12, subspaceCount@16, m@20, efConstruction@24, efSearch@28,
  maxLevel@32 (Int32), entryPoint@36 (Int32, −1 if nil), layerCount@40,
  trainingIterations@44, an `originalsPresent` flag (UInt32 0/1) @48 (write
  `:140`, read `:260–263`, validated ≤ 1 at `:270–273`), and 4 reserved bytes
  @52 (`:142`) — exactly the ADR-011/012 layout. Sections follow in a single
  sequential pass, tightly packed, with **no section table, no trailer, no
  padding, and no stored offset of any kind**: header → PQ codebooks
  (M × 256 × ds Float32, `:147–149`/`:313–329`) → PQ codes (nodeCount × M UInt8,
  `:151–154`/`:334–346`) → UUIDs (16 B each, `:156–160`/`:348–365`) → node
  levels (Int32, `:162–165`/`:367–377`) → per-layer adjacency (UInt32
  count-prefixed rows, `:167–175`/`:379–399`) → JSON metadata (UInt32
  length-prefixed, `:177–180`/`:401–416`) → the **originals section iff the flag
  is 1** (nodeCount × dimension Float32, row-major, slot-aligned with codes;
  write `:182–190`, read `:418–445`). The originals section is located purely by
  **cursor walk** — its start is wherever the cursor lands after
  `offset += metadataSize` (`:416`). Save-time compaction `compactedForSave()`
  (`:55–113`) keeps originals slot-aligned in the same renumbering loop as codes
  (`:77`, `:84–86`), and the write-side asserts "compacted originals must stay
  slot-aligned with codes" (`:185–186`). The read bounds-checks the section with
  `requireBytes(originalsBytes, "originals section")` (`:429`) behind
  overflow-checked size math (`:423–428`). The buffer is written `.atomic`
  (`:192`) and read via `.mappedIfSafe` (`:197`), but every original is copied
  out into a fresh `[Float]`/`Vector` (`:433–443`): the mapping is
  decode-transient only and the constructed index is fully resident.
- **The Stage-2 plumbing this ADR reuses already exists.**
  `MappedVectorRegion` is a `final class … @unchecked Sendable`
  (`MappedVectorRegion.swift:43`, internal, not public) with
  `static let requiredAlignment = 16_384` (`:49`); its `init(baseURL:)` (`:67`)
  resolves layout via `PersistenceEngine.pagedVectorLayout(of:)` (`:68`),
  enforces 16 KiB alignment of the section offset (`:69–73`), opens `O_RDONLY`
  (`:77`), `fstat`s once (`:83–87`), bounds-checks section end ≤ file size
  (`:89–94`), and maps `PROT_READ, MAP_PRIVATE` (`:106`); its copy-on-access
  accessor `vector(at:)` (`:123–129`) hardcodes the stride
  `node * dimension * 4` and returns a value-typed `Vector` copied inside the
  synchronous scope so the raw pointer never escapes; `deinit` unmaps and closes
  (`:131–136`). The class body is **already stride-correct for a PQHW originals
  section** — also dimension × Float32 — so the only HNSW coupling is layout
  resolution: `pagedVectorLayout` guards `header.indexType == indexTypeHNSW`
  (`PersistenceEngine.swift:616`), hardcodes `v3VectorSectionIndex = 1`
  (`:50`, used `:628`), and validates `length == count × dimension × 4`
  (`:631–636`). `VectorProvider` (same file, `:154`) unifies resident and paged
  behind a resident tail whose sole purpose is HNSW's post-snapshot `add()` —
  a purpose `PQHW` has no analogue for.
- **The v3 machinery `PQHW` v3 would mirror.** `PersistenceEngine` carries
  `formatVersion = 2`, `maxReadableVersion = 3`, `v3FormatVersion = 3`,
  `headerSize = 64` (`:33–38`); trailer constants `v3SectionCount = 5`,
  `v3TrailerMagic = 0x5058_4B33` ("PXK3"), `v3TrailerSize = 4 + 5·16 + 8 + 4 = 96`,
  `v3VectorSectionIndex = 1` (`:46–53`); `vectorSectionAlignment = 16_384`
  (`:54–56`). The padded writer's public entry `saveHNSW(_:generation:to:)`
  (`:506–508`) delegates to the internal `padVectorSection:` overload
  (`:519–521`), captures section boundaries (`:544–550`), pads before the vector
  section (`:555–561`), and appends the trailer — sectionCount, per-section
  **UInt64 offset + UInt64 length**, generation, magic — then writes `.atomic`
  (`:586–593`, `:595`). `padToAlignment` (`:796–801`) and `readV3Trailer`
  (`:665–698`, overflow-checked bounds `:679–693`) round it out; alignment is
  checked at map time in `MappedVectorRegion`, not in the trailer parse.
- **The migration gap is a producer gap.** The public padded writer has exactly
  one production call site — `HNSWIndex.checkpoint(...)` at
  `HNSWIndex.swift:892` (the checkpoint method begins `:869`), reachable only
  through the journaled `open(baseURL:walURL:…)` path (`:730–742`, contract doc
  `:719–729`). Legacy `save(to:)` still writes v2 via `PersistenceEngine.save`
  (`HNSWIndex.swift:690–694`) — ADR-013 Stage-1 deviation 1, the byte-identity
  ground rule. A `.paged` open requires a padded v3 base; `loadHNSWPaged`
  (`PersistenceEngine.swift:354`) rejects non-v3 (`:361–363`). **No
  migrate/rewrite/upgrade/convert utility exists anywhere** — not in `Sources/`,
  not in the `ProximaBench` CLI, whose five subcommands are `hnsw`,
  `ground-truth`, `distance-kernel`, `insert-shape`, and `search-provider-bench`
  (`Benchmarks/Sources/ProximaBench/ProximaBenchCLI.swift:21–31`).
- **`SQHW` is scoped out — honestly, because it has no originals to page.**
  `sqMagic = 0x53514857` ("SQHW"), `sqFormatVersion = 1`, `sqHeaderSize = 64`
  (`ScalarQuantizedHNSWIndexPersistence.swift:34–36`). The scalar-quantized
  index **discards** the originals at build time: its header doc says "Discard
  the original vectors — search uses dequantized candidates only"
  (`ScalarQuantizedHNSWIndex.swift:20–22`) and "The full vectors are discarded
  after building" (`:157–158`); search re-scores every candidate against
  `reconstruct(_:)` (dequantize Int8 → Float32, `:378`) with no rerank stage.
  There are zero grep hits for `retainOriginals`/`originals`/`rerank` in the SQ
  index or its persistence, and no originals section in its format (sections:
  header, scales, codes, UUIDs, levels, graph, metadata). Its remove-only
  mutation surface matches `PQHW` (`remove(id:)` at `:342`, no `add`).
  **Consequence for this ADR:** the deferred-list item titled "SQHW/PQHW paged
  originals" (below) overstates — there are no SQHW originals to page. This ADR
  scopes paged originals to `PQHW` only. `SQHW`'s connection is limited to (a)
  adopting the same section-table trailer shape at its own first bump, per
  ADR-013's format-plan sentence, and (b) a *hypothetical* future
  `retainOriginals` for SQ rerank — which would be a **new feature** (a rerank
  stage SQ has never had), not this ADR.

### Problem quantification: 1M × 384d, M = 48 subspaces, m = 16

Using the documented per-vector costs (Float32 vector = 1,536 B, ADR-007; graph
overhead ≈ 200 B/node at m = 16 and code cost = M bytes/vector, ADR-011; PQ
codebook = M·K·ds·4 = D·1024 bytes, ADR-011), a `PQHW` v2 file **with**
originals lays out (all figures arithmetic):

| Section | Arithmetic | Bytes |
|---|---|---|
| Header | fixed | 56 |
| PQ codebooks | 384 × 1,024 (= M·K·ds·4) | 393,216 |
| PQ codes | 1,000,000 × 48 | 48,000,000 |
| UUIDs | 1,000,000 × 16 | 16,000,000 |
| Node levels | 1,000,000 × 4 | 4,000,000 |
| Adjacency | ≈ 200 B/node | ≈ 200,000,000 |
| Metadata | ≥ 4 B length word/node | ≥ 4,000,000 |
| Originals | 1,000,000 × 384 × 4 | **1,536,000,000** |
| **Total** | | **≈ 1.81 GB** |

The originals section alone is **≈ 85%** of the file (1,536,000,000 of
≈ 1,808,393,272, arithmetic). Resident memory today for a retaining index is the
originals (1.536 GB) plus codes (48 MB) plus codebooks (≈ 0.4 MB) plus graph
(≈ 200 MB) plus UUIDs and dictionaries — **1,536 + 48 = 1,584 B/vector**
resident, "slightly *more* than plain full-precision HNSW, not 32× less"
(ADR-012), ≈ **1.79–1.8 GB** total (arithmetic), which no foreground iPhone app
is granted. The paged design target keeps codes + graph + codebooks +
dictionaries resident (≈ **250–270 MB** arithmetic: 48 MB codes + ≈ 200 MB graph
+ 0.4 MB codebooks + 16 MB UUIDs + dictionaries) and maps the originals, so the
per-vector resident cost returns to **48 B** codes — the 32× story on the vector
payload — with 1,536 B/vector living on flash.

**Per-query fault ceiling (arithmetic).** One 384d original is 1,536 B; one
16 KiB page holds 16,384 / 1,536 = **10.67 vectors/page**. Because the rerank
loop touches ≤ `rerankDepth` distinct originals (`:451`/`:461`), the worst case
is each landing on a distinct cold page ⇒ ≤ `rerankDepth` faults. At the default
depth 4·k with k = 10, that is ≤ **40 faults ≈ 640 KiB** paged in per fully cold
query; were those 40 slots contiguous it would be
⌈40 × 1,536 / 16,384⌉ = **4 pages** (64 KiB), but candidate slots are
effectively arbitrary, so the design targets the 40-page ceiling.

**The structural point that makes this cheaper than HNSW Stage 2.** In ADR-013
Stage 2, every candidate distance computed during the beam reads a vector
through the mapping, so cold faults land inside the traversal loop — a cost
ADR-013 accepted at roughly one fault per ~10.6 candidates on cold pages (the
graph itself was never mapped). Here the ADC beam runs **entirely** on resident
codes and distance tables; `:461` is the **only** originals read in the search
path, and it runs **after** the beam completes. Paging originals
therefore never puts a fault on the traversal critical path; the fault budget is
strictly the post-beam rerank, bounded by `rerankDepth`. This is a strictly
better starting position than the one Stage 2 shipped from.

## Options

### Option A — Status quo (resident originals)

The rejected baseline. Retention costs 4·d B/vector resident; a retaining
1M × 384d index needs ≈ 1.8 GB resident (arithmetic, above), which no foreground
iPhone app gets. Reranking works; the memory story does not exist. ADR-012's
Consequences already concede this ("no longer has a compression story"). Keeping
it is the null design.

### Option B — Demand reads without mmap (`pread` per rerank candidate)

Open the base read-only; the rerank loop fetches each candidate's 1,536 B row
via `pread` at `originalsOffset + node·d·4`.

- **Pros:** no mapping lifetime to manage, no SIGBUS exposure class, and the OS
  page cache still amortizes repeated reads.
- **Cons:** a syscall per candidate on the query path (≤ `rerankDepth`
  syscalls/query); buffer management would be new code replacing the
  already-shipped, already-tested `MappedVectorRegion` machinery; it loses the
  Stage-2 uniformity (two different ways to serve an on-disk Float32 vector
  section); and it still needs the same v3 format work — `pread` does not need
  the 16 KiB padding, but it **does** need the section table, because the
  originals offset must be discoverable without a full cursor walk of the file.

Viable but strictly duplicative — rejected in favor of reusing what Stage 2
already proved.

### Option C — mmap the originals section, copy-on-access (recommended)

The ADR-013 Option-C shape applied to `PQHW`: pad the originals section start to
16 KiB in a new `PQHW` v3, map it read-only through the existing
`MappedVectorRegion`, and serve rerank reads copy-on-access. Everything under
**Recommendation** details it.

### Option D — Rerank against decoded codes (no originals at all)

Rejected on principle. The codes only reproduce the quantized approximation the
beam already ranked by — re-scoring against them recovers nothing. ADR-012 said
exactly this when it made rerank-without-retention a typed error rather than a
silent fallback ("the codes only reproduce the quantized approximation the beam
already ranked by").

## Recommendation

**Option C, staged.** Reject A (no memory story), B (duplicative of shipped
Stage-2 machinery), and D (recovers no recall). The design below is settled;
every measurement it names is deferred to ADR-005 at implementation time.

### PQHW v3 format

- Bump `qhFormatVersion` 2 → 3; `qhMinSupportedVersion` stays 1, so v1/v2/v3 all
  load. The N-1 window of ADR-010 rule 2 is exceeded, not merely met — matching
  the `.pxkt` v3 precedent, where `minSupportedVersion` also stayed 1.
- Keep the 56-byte header **byte-for-byte** (the flag @48 stays authoritative
  for originals presence). Append a fixed **trailer** after the last section,
  mirroring `PXK3`: `sectionCount: UInt32 = 7`, then 7 × (offset: UInt64,
  length: UInt64) for [codebooks, codes, uuids, nodeLevels, adjacency, metadata,
  originals], then `snapshotGeneration: UInt64`, then trailer magic
  `"PQH3" = 0x5051_4833`. Trailer size = 4 + 7·16 + 8 + 4 = **128 bytes**
  (arithmetic). The originals entry is (0, 0) when the flag is 0;
  flag/entry consistency is part of the corruption matrix.
- **`snapshotGeneration`** would be written 0 by every v3 writer today; it is
  reserved so a future remove-journal can bind to a base without another bump.
  Note the mutation-surface simplification: `PQHW` is remove-only (`:500`, no
  `add`), so a hypothetical PQ WAL is a 16-byte-UUID-per-record journal with
  none of HNSW's level-journaling/replay complexity. It is deliberately **out of
  scope** — paging does not force it (today's save-rewrites-everything model
  already persists removals) — but the 8 bytes keep the door open. This is a
  simpler WAL story than HNSW's, by construction.
- **Padding.** Zero-pad so the **originals** section start lands on a 16 KiB
  boundary — only that section is padded, and only when the flag is 1.
  Everything before the originals section stays cursor-walkable byte-identically
  to v2, because the padding sits between metadata and originals: a v3 resident
  loader walks the sections exactly as v2 and jumps via the trailer only for the
  originals offset. Retain an internal unpadded-v3 knob for corruption tests
  (the Stage-2 `padVectorSection: false` precedent). Padding cost is ≤ 16,383 B
  per file (arithmetic; negligible against 1.81 GB).
- **Writer policy.** Legacy `save(to:)` on the quantized index would **keep
  writing v2 byte-identically** — the ADR-013 Stage-1 deviation-1 precedent and
  its reasoning: the additive-API ground rule beats ADR-010's "writers write the
  current version" until a deliberate major. v3 would be written by a new
  additive surface — sketch `save(to:layout:)` with a `.v2` default and a
  `.pagedV3` option, exact spelling left to the implementer. The ADR-010 tension
  is real; it is resolved here by the same precedent Stage 1 set.
- **Corruption matrix (ADR-010 rule 5).** The implementation must ship tests
  for: (1) file too small for the trailer; (2) bad trailer magic; (3)
  `sectionCount ≠ 7`; (4) any section (offset, length) overflow or past
  `bodyEnd`, with overflow-checked adds (the `PXK3` precedent, `:679–693`); (5)
  originals offset not 16 KiB-aligned → paged open rejected with a typed error,
  resident load unaffected (mirror `testUnpaddedV3PagedRejectedButResidentLoads`);
  (6) flag/entry mismatch both directions (flag 1 with a zero-length entry; flag
  0 with a nonzero entry); (7) originals length ≠ count × dimension × 4
  (overflow-checked); (8) N-1/N-2: a v2 file loads resident byte-identically, a
  v1 file loads with the documented default `retainOriginals = false`; (9)
  padded and unpadded v3 load identically through the resident path; (10) a
  `.paged` open of a v2 base → typed error whose message names the upgrade path,
  and a `.paged` open of a flag-0 v3 base → typed error (nothing to page). The
  last mirrors ADR-012's `originalsNotRetained` rationale: a silent resident
  fallback would hide a memory-budget misconfiguration.

### Paged read path

- **Reuse `MappedVectorRegion` as-is; generalize only layout resolution.** The
  class body is already stride-correct (dimension × Float32, `:126`) and carries
  the proven mmap / `fstat`-once / `deinit` / SIGBUS machinery; the single HNSW
  coupling is `pagedVectorLayout` (indexType guard
  `PersistenceEngine.swift:616`, section index `:50`/`:628`). The design gives
  the region an internal init taking a resolved section layout and adds a
  PQHW-side resolver (sketch `pagedOriginalsLayout(of:)`) that reads the 56-byte
  header plus the `PQH3` trailer and validates
  `length == count × dimension × 4`. Rejected alternatives: a second
  mapped-region class (duplicates the SIGBUS contract in two places); a fully
  stride-parametric generic section mapper (YAGNI — Float32 vector sections are
  the only paged sections the library has).
- **Do not reuse `VectorProvider`.** Its resident tail and `snapshotBoundary`
  exist solely for HNSW's post-snapshot `add()`; `PQHW` cannot add, so every id
  is `< region.count` forever and the tail would be vacuously empty machinery a
  future reader could not distinguish from load-bearing state. Instead, replace
  `originals: [Vector]?` (`:96`) with a minimal two-case store — sketch
  `enum OriginalsStore { case resident([Vector]); case paged(MappedVectorRegion) }`,
  optional — served copy-on-access through the region's existing `vector(at:)`.
- **Exactly one search-path call site changes.** `:461`
  `originals[node]` → the store accessor. The liveness gate (`:456`) and the
  filter gate (`:457`) already precede the read, so tombstoned or filtered slots
  never fault — ADR-012's invariant "originals for dead slots are simply never
  read" carries over to paging unchanged. Copy-on-access (the Stage-2
  precedent) buys bit-identical parity by construction and a trivially sound
  reentrancy story (no raw pointer crosses an `await`); zero-copy stays on the
  deferred list.
- **No remap machinery at all** — the second simplification over HNSW Stage 2.
  No `add` ⇒ no resident tail; no `checkpoint` ⇒ no compact-renumber-swap.
  `remove(id:)` mutates only resident state (tombstones, graph, `uuidToNode`);
  the mapped originals are never written post-build. A `save(to:)` of a paged
  index would compact into a **new** file via `.atomic` rename while the live
  mapping pins the old inode (Stage-2's inode-pinning story) — the live index's
  numbering is untouched because `compactedForSave()` builds the save image
  without mutating live state (`:55–113`). One honest cost: saving a paged
  retaining index must read every live original back through the mapping to
  write the compacted section — a sequential fault sweep of the originals
  section per save (arithmetic: the full 1.536 GB at 1M × 384d). An implementer
  option, not a commitment, is a raw section byte-copy for the tombstone-free
  case.
- **API.** An additive open mode mirroring Stage 2 — sketch
  `QuantizedHNSWIndex.open(at:mode:)` with `.resident` (default, the
  byte-identical decode path) and `.paged` (graph/codes/metadata decoded
  resident exactly as today; the originals section mapped, not copied). Existing
  `load(from:)`/`save(to:)`/`search` signatures stay unchanged (an acceptance
  criterion: additive only).
- **Accounting.** `originalStorageBytes` (`:129–131`) and `memorySavingsRatio`
  (`:139–143`) currently assume resident originals; under paging, "storage" and
  "resident" diverge. This is left **open** for the implementer, constrained so
  that whatever ships must never report a paged index as though its originals
  were resident.

### The memory story restored

The arithmetic above is the whole argument: today ≈ 1.79–1.8 GB resident for a
retaining index versus ≈ 250–270 MB paged (codes 48 MB + graph ≈ 200 MB +
codebooks 0.4 MB + UUIDs 16 MB + dictionaries), with the originals payload —
85% of the file — demonstrably not resident, and per-vector residency dropping
1,584 B → 48 B. Every figure is arithmetic. The acceptance test would measure
`phys_footprint` the way Stage-2's `PagedVectorMemoryTests` does and set its
threshold from a measured baseline at implementation time, not from a number
invented here.

### Migration: the v2 → v3 upgrade path for both format families

This is likely the most consequential section, and the M3 dogfooding trace names
it directly. The harness feature list records, verbatim: title
"v2->v3 base upgrade path (enable paging on an existing save()d index without
full rebuild)", reason "M3 demo dogfooding: only checkpoint() (add()-built)
produces padded v3; apps with large v2 bases pay full compact+rewrite. Design
question for ADR-014/mission 4." The producer facts confirm the shape of the
gap: HNSW's only padded-v3 producer is `checkpoint()` (sole call site
`HNSWIndex.swift:892`), reachable only through the journaled path; quantized
indexes have no `checkpoint` and no `add` at all, so once `PQHW` v3 exists its
existing v2 bases have **no** upgrade path but full rebuild — which for PQ
requires the original corpus the app may no longer hold. No migrate/rewrite
utility exists anywhere (verified across `Sources/` and the five `ProximaBench`
subcommands).

- **HNSW workaround that exists today** (documented, not endorsed):
  `open(v2 base, fresh WAL)` → `checkpoint()` → padded v3. Costs: it forces
  journal attachment on a non-journaling app, compacts when tombstones exist
  (O(n log n) re-insertion), and pays a full resident load plus a full rewrite.
- **Option M-A — offline section-copy rewrite (recommended, both families).** A
  format upgrader that never decodes the index. It cursor-walks the v2 section
  boundaries (each boundary is computable from header fields plus the
  count-prefixed adjacency walk and the length-prefixed metadata — for `.pxkt`
  v2 from n, d, layerCount; for `PQHW` v2 from n, d, M and the same walks), then
  bulk byte-copies sections into a new file, inserting the alignment padding
  before the pageable section, stamps v3, appends the trailer, and
  `.atomic`-renames. No graph decode, no `[Vector]` materialization, no
  re-insertion, no RNG — the output's section payloads are **bit-identical** to
  the input's. Peak memory is O(1)-streaming (or one `.mappedIfSafe` read plus a
  buffered write); a crash mid-upgrade leaves the source untouched (temp +
  rename). It **must** live in-library (sketch
  `PersistenceEngine.upgradeToV3(at:)` and a `PQHW` sibling) because the primary
  consumer is an iPhone app upgrading its own on-device base — a Mac-side CLI
  cannot run there — and a thin `ProximaBench migrate` subcommand should wrap it
  for operators and CI fixtures. Two focused per-family rewriters share the
  pad/trailer helpers; a unified generic rewriter is rejected — the two formats
  share shape, not layout, so the generic version would be parameter soup for
  two call sites.
- **Option M-B — save-time format option (fresh writes).** The
  `save(to:layout: .pagedV3)` surface from the format section. It covers new
  builds and the rebuild cadence ADR-011 already prescribes for heavy-removal
  workloads, but is useless alone for large existing bases: load + re-save is a
  full decode + re-encode at full resident peak — hostile exactly where paging
  matters, on-device.
- **Option M-C — rebuild-only (status quo).** Rejected: it defeats the ADR's
  purpose, and PQ rebuild needs the original corpus.
- **Recommendation: M-A + M-B together.** M-A is the migration story; M-B is the
  steady state. M-A for `.pxkt` is valuable **independently** of everything else
  here — it lets existing v2 HNSW bases adopt the already-shipped Stage-2 paging
  with no journaling and no compaction, closing the dogfooding gap on day one —
  which is why it anchors Stage 1 below.
- **Migration fidelity gates** (folded into acceptance): section payloads
  bit-identical between input and output (checksum-asserted); output parses
  (trailer bounds, alignment) and opens `.paged`; a resident load of the output
  equals a resident load of the input (structural equality / search parity); v1
  `PQHW` and flag-0 inputs upgrade uniformly (legal v3, (0,0) originals entry, no
  padding needed — they gain nothing until rebuilt with retention); an
  interrupted upgrade leaves the source intact.

### metadataOffset widening rides this bump

- **`PQHW` v3:** nothing to widen. `PQHW` stores no offsets, and the new trailer
  is UInt64 from day one. The format-level ceilings that remain in `PQHW` are the
  UInt32 metadata-JSON length prefix (`:179`/`:402`) and the UInt32 header counts
  (nodeCount caps slots at ≈ 4.29 billion) — noted, out of scope.
- **`.pxkt`:** the header field `FileHeader.metadataOffset: UInt32`
  (`PersistenceEngine.swift:715`, read at header offset 52, `:771`) cannot be
  widened in place — `autoCompactionThreshold` occupies @56 and there are no
  reserved bytes left (ADR-010's "v2 consumed the last reserved bytes"). It needs
  **no new `.pxkt` bump**, because v3's trailer already carries the metadata
  section offset as UInt64 (`:588–591`): (a) `loadHNSWPaged` (`:482`) — the only
  v3-era consumer — switches to the trailer's UInt64 entry; (b) the v3 writer
  replaces the trapping `UInt32(data.count)` (`:580`) with "≤ UInt32.max ⇒ write
  it (bytes unchanged for every file writable today); else write the sentinel
  `0xFFFF_FFFF` and rely on the trailer," documented where the constant lives
  (ADR-010 rule 3's documentation discipline). The BruteForce consumer
  (`:190–191`) and the legacy v2 writer keep the UInt32 ceiling — pre-existing,
  unreachable without the writer trapping first (the converting `UInt32(_:)`
  traps past UInt32.max, so a v3 base whose metadata starts beyond 4 GiB crashes
  the writer **today**, a trap not a mis-write), noted not fixed. The sibling
  `.pxbm` sparse `metadataOffset` UInt32 at header offset 36
  (`SparseIndexPersistence.swift:147`) is out of scope.
- Provenance, verbatim from the harness feature list: title
  "Header metadataOffset UInt32 -> UInt64 (4 GiB metadata-section ceiling)",
  reason "M3 judges: pre-existing, unreachable at tested scales; widen in the
  next format bump."

### Consumer fit — TinyBrain

Framed as design pressure, not requirements — no specs are invented here. The
consumer is building "TinyBrain," an agentic on-device assistant using
ProximaKit as its memory: continuous ingest, constant recall, rerank quality
that matters, iPhone memory budgets. ADR-006 already names it a reuse consumer
("Other Chakravyuha products (TinyBrain) can reuse the same pattern"), and its
CHA-107 addendum established the "drop-in shape-compatible sibling" /
"one-line construction-site swap" precedent. The honest connections:

- **Continuous ingest** lands in the WAL-journaled HNSW leg (ADR-013 Stage 1),
  because the quantized families are build-once; the natural shape is an
  ingest-hot HNSW plus periodic distillation into a retaining `PQHW` for the
  long tail. M-B makes every distillation output paged by default; M-A upgrades
  whatever bases already sit on device.
- **Constant recall with rerank quality:** rerank stays exact by construction
  (copy-on-access parity), the recall dial `rerankDepth` is untouched, and the
  cost moves to a bounded ≤ `rerankDepth` cold faults per query — off the beam's
  critical path.
- **iPhone budgets:** the resident arithmetic above is the difference between a
  jetsam-bait ≈ 1.8 GB and ≈ 260 MB for the same exact-reranking index.
- **One honest caveat:** `build()` itself still materializes a transient
  full-precision `HNSWIndex` (`:235`), so build-time peak memory is untouched by
  this ADR and remains the on-device distillation constraint — design pressure
  for a future streaming build, explicitly not in scope.

### Staged recommendation and honest cost

The stages are shippable alone, in ADR-013's register.

- **Stage 1 — formats + migration (≈ 2 engineering weeks).** `PQHW` v3 (trailer,
  padding, corruption matrix, N-1/N-2 tests), the `.pxkt` metadataOffset
  trailer-sourcing + sentinel, both section-copy upgraders + `ProximaBench
  migrate`, and the fidelity rigs (checksum equality, parity-after-upgrade).
  Ships value alone: existing v2 `.pxkt` bases adopt the already-shipped
  Stage-2 paging with no journal and no rebuild.
- **Stage 2 — paged originals read path (≈ 1.5–2 weeks).** The
  layout-resolution generalization + `MappedVectorRegion` reuse, the two-case
  originals store threaded through the single rerank read site + the save path +
  accounting, `open(at:mode:)`, parity tests (byte-identical ids + bit-equal
  distances, including filtered and post-remove), the `phys_footprint` memory
  acceptance, and a resident-retaining no-regression measurement before landing
  (ADR-005; the change is one call site, but the Stage-2 discipline of proving
  the resident path unregressed comes first regardless).
- **Hardening + docs (≈ 1 week).** The SIGBUS contract extended to the `PQHW`
  mapping in public docs, upgrade-utility docs, and BENCHMARKS / README /
  ROADMAP / DocC updates.
- **Total ≈ 4–5 engineering weeks, Stage 1 shippable alone at ~2.** This is
  cheaper than ADR-013's 5–7 because the hot path is untouched (one post-beam
  call site versus the eight sites Stage 2 threaded its provider through,
  including the hottest traversal loops), with no WAL, no tail, and no
  remap. Primary schedule risks: rewriter boundary-walk fidelity on adversarial
  v2 files (mitigated by checksum gates + the corruption corpus), and the
  temptation to grow the generic mapper (mitigated by the scope fence in this
  ADR).

### Acceptance criteria

1. **Paged-retaining memory.** A benchmark-class, env-gated test (the Stage-2
   `PagedVectorMemoryTests` precedent) opens the **same** `PQHW` v3 retaining
   base `.resident` versus `.paged` and asserts the resident-vs-paged
   `phys_footprint` delta shows the originals payload not resident, with the
   threshold derived from a measured baseline when implemented — not invented
   now — plus a warm-rerank sweep asserting the paged-in delta after N reranked
   queries stays bounded by the working set, not the corpus.
2. **Rerank parity.** `.paged` results are byte-identical to `.resident` on the
   same file — same ids **and** bit-equal Float32 distances — across seeded
   queries × {rerank on/off, filtered, post-remove}; every parity test is
   red-green proven (Stage-2 discipline).
3. **Migration fidelity.** Both upgraders produce section-payload-bit-identical
   outputs (checksummed), the outputs open `.paged` and parity-match the
   resident-loaded source, interrupted upgrades leave sources intact, and
   v1 / flag-0 inputs upgrade to legal v3.
4. **Corruption matrix.** Every item enumerated in the format section ships as a
   test; recovery never traps; typed errors only (ADR-010 rules 4/5).
5. **Additive API only.** Existing `save` / `load` / `search` / `build` stay
   byte-identical in behavior and on-disk output; `.paged` open,
   `save(to:layout:)`, and the upgraders are new opt-in surface.
6. **No performance claim until measured** under ADR-005 via the existing
   harness; this document contains arithmetic only.

## What this unlocks

Today `VectorStore` and `HybridVectorStore` hardcode `HNSWIndex`
(`VectorStore.swift:35`; `HybridVectorStore.swift:48–55`) and nothing in the
store layer touches quantized indexes (zero grep hits). Paged originals is the
piece that makes a quantized-backed store **sibling** (the CHA-107 "drop-in
shape-compatible sibling" precedent) viable on-device: a store leg with codes +
graph resident, originals on flash, and exact rerank — `index.pqhw` beside
`index.pxkt` in the store directory, a one-line construction-site swap for
consumers whose corpus outgrew the full-precision leg's budget. This ADR designs
the index-level capability only; the store sibling is the unlock, not a
commitment.

## Consequences

- A second format family gains the trailer/padding shape; the corruption-test
  surface grows by the enumerated matrix — the ADR-010 rule-5 price, budgeted in
  the estimate.
- The SIGBUS / read-only-mapping contract now spans two formats and must be
  documented once and referenced twice, not duplicated.
- The deferred-list item's "SQHW" half is resolved by **scoping**, not by code:
  `SQHW` has no originals to page; a future SQ rerank feature would re-open it.
- Until implemented, every number here is arithmetic or an acceptance gate to be
  measured under ADR-005 methodology.

## Open questions

Deliberately left to the implementer:

1. Whether the quantized `save(to:)` default ever flips to v3 (ADR-010 rule 2
   versus the byte-identity ground rule; the ADR-013 deviation-1 precedent says
   keep v2 until a deliberate major).
2. The accounting API shape for mapped originals
   (`memorySavingsRatio` / `originalStorageBytes` semantics) — constrained to
   never report paged as resident.
3. Upgrade verification depth at scale (full checksum versus sampled) for
   multi-GB bases on-device.
4. Whether save-of-a-paged-index should special-case the tombstone-free path as
   a raw section byte-copy instead of a mapped read-back sweep.
5. `snapshotGeneration` semantics if a PQ remove-journal is ever built (reserved,
   stamped 0 until then).
