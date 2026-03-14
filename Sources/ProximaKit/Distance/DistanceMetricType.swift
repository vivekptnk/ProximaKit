// DistanceMetricType.swift
// ProximaKit
//
// Type discriminator for serializing distance metrics.
// All built-in metrics are stateless structs — only the type needs to be persisted.

/// Identifies which distance metric an index uses, for persistence.
///
/// ProximaKit's three built-in metrics are stateless — they have no constructor
/// parameters. This enum maps each one to a `UInt32` value for binary serialization.
///
/// ```swift
/// let type = DistanceMetricType(metric: CosineDistance())  // .cosine
/// let metric = type?.makeMetric()                           // CosineDistance()
/// ```
public enum DistanceMetricType: UInt32, Sendable, CaseIterable {
    case cosine = 0
    case euclidean = 1
    case dotProduct = 2

    /// Creates the corresponding `DistanceMetric` instance.
    public func makeMetric() -> any DistanceMetric {
        switch self {
        case .cosine: return CosineDistance()
        case .euclidean: return EuclideanDistance()
        case .dotProduct: return DotProductDistance()
        }
    }

    /// Identifies which type a given metric is.
    /// Returns `nil` for unknown custom metrics.
    public init?(metric: any DistanceMetric) {
        switch metric {
        case is CosineDistance: self = .cosine
        case is EuclideanDistance: self = .euclidean
        case is DotProductDistance: self = .dotProduct
        default: return nil
        }
    }
}
