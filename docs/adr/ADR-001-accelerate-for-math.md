# ADR-001: Use Accelerate/vDSP for All Vector Math

## Status
Accepted

## Context
Need to compute vector distances millions of times per search. Options: manual loops, Swift SIMD types, Accelerate/vDSP, Metal.

## Decision
Use Accelerate/vDSP for all vector math.

## Rationale
- vDSP auto-vectorizes across NEON SIMD lanes on Apple Silicon
- No dimension constraints (unlike fixed-width SIMD4/SIMD8)
- Batch matrix multiply (vDSP_mmul) computes query-vs-N-vectors in one call
- Ships with every Apple OS — zero dependency
- Metal has high dispatch overhead, overkill for < 100K vectors

## Consequences
- All vector code uses `withUnsafeBufferPointer` for vDSP interop
- Contributors must understand C-style pointer APIs
- Cannot port to non-Apple platforms (acceptable for Apple-native library)
- Patterns guide at `.claude/skills/accelerate-patterns.md`

## Benchmarks
At 384d, 10K vectors: manual loop ~120ms, vDSP sequential ~25ms, vDSP_mmul batch ~8ms.
