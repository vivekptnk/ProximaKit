# ADR-010: Serialization Format Evolution Policy

## Status
Accepted

## Context
ADR-003 mandated a magic + version field in the persistence header (`PXKT`, validated on load via `PersistenceError.unsupportedVersion`). v1.4 forced the first real evolution: `autoCompactionThreshold` existed in configuration but was not serialized, so it silently reset to the default on load. The fix bumped `.pxkt` to v2 (threshold in the previously reserved header bytes at offset 56) and `.pxbm` to v2 (offset 40), with all-zero bits encoding `nil` and v1 files loading with the documented default `0.7`.

## Decision
Codify the rules that change demonstrated, for all on-disk formats (`PXKT`, `PXBM`, `PQTT`, `PQHW`):

1. **Monotonic version bumps.** Any layout change increments the format version. Versions are never reused or forked.
2. **Readers support N and N-1 at minimum.** Loaders accept `minSupportedVersion...formatVersion` and throw `PersistenceError.unsupportedVersion` outside that range. Writers always write the current version.
3. **New fields get documented defaults when absent.** A field missing from an older file loads with an explicitly documented default (v1 → compaction threshold `0.7`), never an implicit zero value. Prefer previously reserved header bytes so section offsets don't shift.
4. **Typed errors, never crashes.** Bad magic, over-version, and corrupt files throw typed `PersistenceError` cases (`invalidMagic`, `unsupportedVersion`, `corruptedData`). Header values are sanity-checked before they reach type preconditions that would trap the process (e.g. `m >= 2`, `mMax0 == 2*m`, threshold in `(0, 1)`).
5. **Corruption tests required per format change.** Every bump ships corruption tests plus an N-1 backward-compatibility test (`PersistenceCorruptionTests` patches a v2 file down to v1, zeroes the new bytes, and asserts the documented default is restored).

## Consequences
- No forward compatibility: an older library fails on a newer file with `unsupportedVersion(N)` rather than misreading it. Acceptable — apps bundle the library version they were built with.
- `.pxkt` v2 consumed the last reserved bytes in the 64-byte header; the next field needs a new section (and a bump).
- `PQTT` and `PQHW` are still v1 and currently require an exact version match; the N-1 read window starts applying at their first bump.
- Rule 3 means every format change is also a documentation change: the default for absent fields must be written down where the version constant is defined.
