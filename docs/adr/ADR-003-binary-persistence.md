# ADR-003: Custom Binary Persistence Format

## Status
Accepted

## Context
10K+ vector indices need fast save/load. Options: JSON/Codable, Protocol Buffers, custom binary.

## Decision
Custom binary with memory-mapped vector data.

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
