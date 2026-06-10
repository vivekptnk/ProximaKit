// DistanceMetricType.swift
// ProximaKit
//
// Type discriminator for serializing distance metrics.
// The stateless built-in metrics need only their type persisted; stateful
// metrics (Mahalanobis) are unserializable, like custom metrics.

/// Identifies which distance metric an index uses, for persistence.
///
/// ProximaKit's serializable built-in metrics are stateless — they have no
/// constructor parameters. This enum maps each one to a `UInt32` value for
/// binary serialization. Raw values are append-only and never reused
/// (per ADR-010); older readers reject unknown values with a typed
/// `PersistenceError` rather than misreading the file.
///
/// ``MahalanobisDistance`` is deliberately absent: it carries a
/// `dimension × dimension` matrix payload, so indices configured with it
/// throw `PersistenceError.unserializableMetric` on save, exactly like
/// custom user-defined metrics.
///
/// ```swift
/// let type = DistanceMetricType(metric: CosineDistance())  // .cosine
/// let metric = type?.makeMetric()                           // CosineDistance()
/// ```
public enum DistanceMetricType: UInt32, Sendable, CaseIterable {
    case cosine = 0
    case euclidean = 1
    case dotProduct = 2
    case manhattan = 3
    case hamming = 4
    case chebyshev = 5
    case brayCurtis = 6

    /// Creates the corresponding `DistanceMetric` instance.
    public func makeMetric() -> any DistanceMetric {
        switch self {
        case .cosine: return CosineDistance()
        case .euclidean: return EuclideanDistance()
        case .dotProduct: return DotProductDistance()
        case .manhattan: return ManhattanDistance()
        case .hamming: return HammingDistance()
        case .chebyshev: return ChebyshevDistance()
        case .brayCurtis: return BrayCurtisDistance()
        }
    }

    /// Identifies which type a given metric is.
    /// Returns `nil` for unknown custom metrics (including ``MahalanobisDistance``).
    public init?(metric: any DistanceMetric) {
        switch metric {
        case is CosineDistance: self = .cosine
        case is EuclideanDistance: self = .euclidean
        case is DotProductDistance: self = .dotProduct
        case is ManhattanDistance: self = .manhattan
        case is HammingDistance: self = .hamming
        case is ChebyshevDistance: self = .chebyshev
        case is BrayCurtisDistance: self = .brayCurtis
        default: return nil
        }
    }
}
