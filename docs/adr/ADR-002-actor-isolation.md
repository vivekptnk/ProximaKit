# ADR-002: Actor Isolation for Index Types

## Status
Accepted

## Context
Indices are read-heavy, write-light. Need thread safety without destroying search performance.

## Decision
Make indices conform to `VectorIndex` which requires `Actor` conformance.

## Rationale
- Compile-time data race safety via Sendable checking
- Actor re-entrancy allows multiple searches to queue
- Writes naturally serialize (HNSW insertion needs exclusive access)
- Alternative (readers-writer lock) loses compile-time safety

## Consequences
- All callers use `await` for index operations
- Vector and SearchResult must be Sendable (they are — value types)
- If profiling shows overhead: `nonisolated(unsafe)` for read-only data
