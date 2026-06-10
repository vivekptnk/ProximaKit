# ADR-003: Custom Binary Persistence Format

## Status
Accepted (amended — see Correction, 2026-06)

## Context
10K+ vector indices need fast save/load. Options: JSON/Codable, Protocol Buffers, custom binary.

## Decision
Custom binary format, read via `.mappedIfSafe` and decoded into memory-resident Swift arrays (see Correction below).

## Rationale
- JSON for 10K/384d: ~60MB, ~3s load. Unacceptable.
- Custom binary: ~58MB, ~50ms load via mmap. OS pages on demand.
- Protocol Buffers: adds dependency. Violates zero-dep rule.

## Format
Header (64B) + contiguous Float32 vectors (mmap) + graph adjacency lists + JSON metadata.

## Consequences
- Must version the format (magic + version in header)
- Graph requires full deserialization (variable-length adjacency lists)
- Must document format for cross-language readers

## Correction (2026-06)

The original rationale overstated the memory-mapping benefit. As implemented,
`PersistenceEngine` reads files with `Data(contentsOf:options:.mappedIfSafe)`,
which accelerates the **decode pass** (the OS pages file bytes in lazily while
the loader walks them). It does not keep the live index file-backed: the
decoder copies every Float32 into Swift arrays (`Vector` storage), so a loaded
index is **fully resident in memory** and the mapping is released once decoding
finishes. "OS pages on demand" applies only to the transient `Data` during
load — not to search-time memory.

The load-time numbers above still hold; the claim that did not is the implied
resident-memory saving. For actual memory reduction, use
`QuantizedHNSWIndex` (product quantization, ADR-011) or
`ScalarQuantizedHNSWIndex` (INT8, ADR-007).
