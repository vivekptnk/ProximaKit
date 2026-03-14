// ProximaKit — Pure-Swift vector similarity search.
//
// This is the core module. It provides:
// - Vector type with Accelerate-powered math
// - Distance metrics (cosine, L2, dot product)
// - Index structures (BruteForce, HNSW)
// - Persistence (memory-mapped binary files)
//
// Imports: Foundation + Accelerate ONLY. No third-party dependencies.

/// The ProximaKit namespace. Version constant for runtime checks.
public enum ProximaKit {
    /// The current library version.
    public static let version = "1.0.0"
}
