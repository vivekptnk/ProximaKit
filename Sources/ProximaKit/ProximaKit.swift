// ProximaKit — Pure-Swift vector similarity search.
//
// This is the core module. It provides:
// - Vector type with Accelerate-powered math
// - Distance metrics (cosine, L2, dot product, Manhattan, Hamming,
//   Chebyshev, Bray-Curtis, Mahalanobis)
// - Index structures (BruteForce, HNSW, Sparse/BM25, Hybrid,
//   QuantizedHNSW, ScalarQuantizedHNSW)
// - Persistence (versioned binary files)
//
// Imports: Foundation + Accelerate ONLY. No third-party dependencies.

/// The ProximaKit namespace. Version constant for runtime checks.
public enum ProximaKit {
    /// The current library version.
    public static let version = "1.6.1"
}
