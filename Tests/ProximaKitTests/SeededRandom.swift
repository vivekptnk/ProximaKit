// SeededRandom.swift
// ProximaKitTests
//
// Shared deterministic RNG for tests whose assertions depend on the
// generated data (recall thresholds, quantization quality). The global
// `Float.random(in:)` draws from SystemRandomNumberGenerator, so a
// threshold like "recall > 0.82" measures a different dataset on every
// run — the repo has already shipped one flake fix (CHA-105) caused by
// exactly this. Tests that assert on data-dependent quantities should
// pass a `SeededRandom` instance via the `using:` overloads instead.
//
// SparseIndexTests keeps its own private xorshift RNG predating this
// helper; new tests should use this one.

/// SplitMix64 — tiny, fast, and passes BigCrush; the standard choice for
/// seeding/deterministic test data (Steele, Lea & Flood, OOPSLA 2014).
/// Not cryptographic; test-data generation only.
struct SeededRandom: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
