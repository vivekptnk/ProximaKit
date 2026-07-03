// DistanceMetric.swift
// ProximaKit
//
// Protocol for pluggable distance functions.
// Indices are generic over their distance metric, so you can swap
// cosine for L2 without changing any search code.

import Accelerate

/// A protocol for measuring the distance (or dissimilarity) between two vectors.
///
/// Distance metrics are used by index types to rank search results.
/// Lower distance = more similar. All conforming types must be `Sendable`
/// so they can be stored inside actor-isolated indices.
///
/// ProximaKit ships nine metrics:
/// - ``CosineDistance``: 1 - cosine_similarity. Best for text embeddings.
/// - ``EuclideanDistance``: L2 (straight-line) distance.
/// - ``DotProductDistance``: Negative dot product (for pre-normalized vectors).
/// - ``ManhattanDistance``: L1 (taxicab) distance. Good for sparse data.
/// - ``HammingDistance``: Count of differing positions. For binary/quantized vectors.
/// - ``ChebyshevDistance``: L∞ (maximum component difference). For grid-like spaces.
/// - ``BrayCurtisDistance``: Normalized L1 dissimilarity. For non-negative count vectors.
/// - ``JensenShannonDistance``: Distribution distance for non-negative histograms.
/// - ``MahalanobisDistance``: Covariance-aware distance. Not serializable.
///
/// ```swift
/// let metric = CosineDistance()
/// let d = metric.distance(vectorA, vectorB)  // 0.0 = identical, 2.0 = opposite
/// ```
public protocol DistanceMetric: Sendable {
    /// Computes the distance between two vectors of equal dimension.
    ///
    /// - Parameters:
    ///   - a: The first vector.
    ///   - b: The second vector.
    /// - Returns: A distance value where lower = more similar.
    func distance(_ a: Vector, _ b: Vector) -> Float
}

// ── Cosine Distance ──────────────────────────────────────────────────

/// Cosine distance: `1 - cosine_similarity(a, b)`.
///
/// Range: [0, 2] where 0 = identical direction, 1 = perpendicular, 2 = opposite.
///
/// This is the default metric for text embeddings (BERT, sentence-transformers,
/// NLEmbedding) because it measures direction regardless of magnitude.
/// We convert similarity → distance so that "lower = more similar" is consistent
/// across all metrics.
public struct CosineDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        // cosine_similarity returns [-1, 1].
        // Subtracting from 1 flips it to [0, 2] where 0 = identical.
        1.0 - a.cosineSimilarity(b)
    }
}

// ── Euclidean (L2) Distance ──────────────────────────────────────────

/// Euclidean (L2) distance: the straight-line distance between two points.
///
/// Range: [0, ∞) where 0 = identical vectors.
///
/// Good for when absolute position matters (image features, geographic
/// coordinates). Less common for text embeddings because it's sensitive
/// to vector magnitude.
public struct EuclideanDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        a.l2Distance(b)
    }
}

// ── Dot Product Distance ─────────────────────────────────────────────

/// Negative dot product distance: `-dot(a, b)`.
///
/// For **pre-normalized** (unit) vectors, this is equivalent to cosine distance
/// but faster — it skips the magnitude computation.
///
/// Range: [-1, 1] for unit vectors, where -1 = identical, 1 = opposite.
/// (Negated so that lower = more similar, consistent with other metrics.)
///
/// **Important:** Only use this with normalized vectors. For unnormalized vectors,
/// use ``CosineDistance`` instead.
public struct DotProductDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        // Negate because higher dot product = more similar,
        // but our convention is lower distance = more similar.
        -a.dot(b)
    }
}

// ── Manhattan (L1) Distance ────────────────────────────────────────────

/// Manhattan (L1) distance: the sum of absolute differences between components.
///
/// Range: [0, ∞) where 0 = identical vectors.
///
/// Also known as "taxicab" or "city block" distance. Good for sparse data,
/// grid-based problems, or when outliers should have proportional (not squared)
/// influence on distance. Uses vDSP for SIMD-accelerated computation.
public struct ManhattanDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        precondition(a.dimension == b.dimension,
                     "Dimension mismatch: \(a.dimension) vs \(b.dimension)")

        let count = vDSP_Length(a.dimension)

        // Step 1: Compute difference vector (b - a)
        var difference = [Float](repeating: 0, count: a.dimension)
        a.components.withUnsafeBufferPointer { aPtr in
            b.components.withUnsafeBufferPointer { bPtr in
                vDSP_vsub(aPtr.baseAddress!, 1,
                          bPtr.baseAddress!, 1,
                          &difference, 1, count)
            }
        }

        // Step 2: Absolute values in-place
        var absDiff = [Float](repeating: 0, count: a.dimension)
        vDSP_vabs(difference, 1, &absDiff, 1, count)

        // Step 3: Sum the absolute differences
        var sum: Float = 0
        vDSP_sve(absDiff, 1, &sum, count)

        return sum
    }
}

// ── Hamming Distance ───────────────────────────────────────────────────

/// Hamming distance: the count of positions where two vectors differ.
///
/// Range: [0, dimension] where 0 = identical vectors.
///
/// For float vectors, two components are considered equal if they have
/// the same bit pattern (exact equality). This metric is most useful for
/// binary or quantized vectors where elements take on discrete values
/// (e.g., 0.0/1.0 binary codes).
public struct HammingDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        precondition(a.dimension == b.dimension,
                     "Dimension mismatch: \(a.dimension) vs \(b.dimension)")

        var count: Float = 0
        a.components.withUnsafeBufferPointer { aPtr in
            b.components.withUnsafeBufferPointer { bPtr in
                for i in 0..<a.dimension {
                    if aPtr[i] != bPtr[i] {
                        count += 1
                    }
                }
            }
        }
        return count
    }
}

// ── Chebyshev (L∞) Distance ────────────────────────────────────────────

/// Chebyshev (L∞) distance: the maximum absolute difference between components.
///
/// Range: [0, ∞) where 0 = identical vectors.
///
/// Also known as "chessboard" distance — the number of moves a king needs
/// on a chessboard. Useful for grid/game-AI pathfinding over embedded state
/// spaces, and whenever the single worst-case component deviation matters
/// more than the aggregate. A genuine metric (symmetric, satisfies the
/// triangle inequality). Uses vDSP for SIMD-accelerated computation.
public struct ChebyshevDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        precondition(a.dimension == b.dimension,
                     "Dimension mismatch: \(a.dimension) vs \(b.dimension)")

        let count = vDSP_Length(a.dimension)

        // Step 1: Compute difference vector (b - a)
        var difference = [Float](repeating: 0, count: a.dimension)
        a.components.withUnsafeBufferPointer { aPtr in
            b.components.withUnsafeBufferPointer { bPtr in
                vDSP_vsub(aPtr.baseAddress!, 1,
                          bPtr.baseAddress!, 1,
                          &difference, 1, count)
            }
        }

        // Step 2: Maximum magnitude = max |a_i - b_i|
        var maxAbs: Float = 0
        vDSP_maxmgv(difference, 1, &maxAbs, count)

        return maxAbs
    }
}

// ── Bray-Curtis Distance ───────────────────────────────────────────────

/// Bray-Curtis dissimilarity: `Σ|a_i - b_i| / Σ|a_i + b_i|`.
///
/// Range: [0, 1] for non-negative vectors, where 0 = identical and
/// 1 = no overlap (disjoint supports).
///
/// Standard in ecology and other compositional analyses for comparing
/// **non-negative count vectors** (species abundances, term frequencies,
/// histograms). When both vectors are all-zero the ratio is 0/0; ProximaKit
/// defines that case as distance **0** (two empty samples are identical).
/// A zero denominator with a nonzero numerator (only reachable with
/// mixed-sign inputs, e.g. `b = -a`) returns distance **1**.
///
/// **Important:** Only use this with non-negative inputs. Negative components
/// make the measure ill-behaved: the denominator can shrink toward zero while
/// the numerator does not, so values escape [0, 1] and lose their
/// dissimilarity interpretation.
///
/// **Not a true metric:** even on its intended non-negative domain,
/// Bray-Curtis violates the triangle inequality (it is a semimetric — see
/// `NewMetricsTests.testBrayCurtisViolatesTriangleInequality` for a concrete
/// counterexample). It is still a valid ranking dissimilarity for search.
public struct BrayCurtisDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        precondition(a.dimension == b.dimension,
                     "Dimension mismatch: \(a.dimension) vs \(b.dimension)")

        let count = vDSP_Length(a.dimension)
        var difference = [Float](repeating: 0, count: a.dimension)
        var elementSum = [Float](repeating: 0, count: a.dimension)

        a.components.withUnsafeBufferPointer { aPtr in
            b.components.withUnsafeBufferPointer { bPtr in
                // difference = b - a, elementSum = a + b
                vDSP_vsub(aPtr.baseAddress!, 1,
                          bPtr.baseAddress!, 1,
                          &difference, 1, count)
                vDSP_vadd(aPtr.baseAddress!, 1,
                          bPtr.baseAddress!, 1,
                          &elementSum, 1, count)
            }
        }

        // Numerator: Σ|a_i - b_i|
        var absDiff = [Float](repeating: 0, count: a.dimension)
        vDSP_vabs(difference, 1, &absDiff, 1, count)
        var numerator: Float = 0
        vDSP_sve(absDiff, 1, &numerator, count)

        // Denominator: Σ|a_i + b_i|
        var absSum = [Float](repeating: 0, count: a.dimension)
        vDSP_vabs(elementSum, 1, &absSum, 1, count)
        var denominator: Float = 0
        vDSP_sve(absSum, 1, &denominator, count)

        // Zero denominator:
        // - 0/0 (both vectors all-zero): identical empty samples → 0.
        // - x/0 with x > 0 (only reachable with mixed signs, e.g. b = -a):
        //   undefined ratio on inputs outside the documented non-negative
        //   domain → 1 (maximal dissimilarity), never a silent perfect match.
        guard denominator > 0 else { return numerator > 0 ? 1 : 0 }
        return numerator / denominator
    }
}

// ── Jensen-Shannon Distance ───────────────────────────────────────────

/// Jensen-Shannon distance: `sqrt(JSD(a, b))` for non-negative distributions.
///
/// Range: [0, 1] where 0 = identical distributions and 1 = disjoint
/// support. Inputs are treated as unnormalized finite, non-negative
/// distributions and L1-normalized internally before comparison. The
/// divergence uses base-2 logarithms, so `sqrt(JSD)` is bounded by 1.
///
/// This is the metric form of Jensen-Shannon: raw Jensen-Shannon divergence
/// is useful as a dissimilarity, but the square root is the true metric
/// (symmetric and satisfies the triangle inequality).
///
/// Zero components are valid: terms with probability 0 contribute 0 by the
/// standard `0 * log(0) = 0` convention. When both vectors are all-zero,
/// ProximaKit defines the distance as **0** (two empty distributions are
/// identical). When exactly one vector is all-zero, the distance is **1**,
/// the maximal value on the normalized distribution domain.
///
/// **Important:** ProximaKit does not clamp negatives or take absolute values
/// because either choice silently changes the input distribution and can hide
/// data-quality bugs.
///
/// Out-of-domain input:
/// - negative or non-finite components: outside the documented finite,
///   non-negative distribution domain → 1 (maximal dissimilarity), never a
///   silent perfect match or process trap.
/// - non-finite intermediate result: numerical overflow/NaN during
///   normalization or divergence → 1, keeping distances bounded and sortable.
///   In the subnormal total-mass regime, this applies even to identical inputs
///   (d(a,a)=1, sacrificing the identity property there).
public struct JensenShannonDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        precondition(a.dimension == b.dimension,
                     "Dimension mismatch: \(a.dimension) vs \(b.dimension)")

        var sumA: Float = 0
        var sumB: Float = 0
        for i in 0..<a.dimension {
            let ai = a.components[i]
            let bi = b.components[i]
            guard ai.isFinite && ai >= 0 && bi.isFinite && bi >= 0 else {
                return 1
            }
            sumA += ai
            sumB += bi
        }

        switch (sumA > 0, sumB > 0) {
        case (false, false):
            return 0
        case (false, true), (true, false):
            return 1
        case (true, true):
            break
        }

        let invA = 1 / sumA
        let invB = 1 / sumB
        var divergence: Float = 0

        for i in 0..<a.dimension {
            let p = a.components[i] * invA
            let q = b.components[i] * invB
            let midpoint = 0.5 * (p + q)

            if p > 0 {
                divergence += 0.5 * p * log2(p / midpoint)
            }
            if q > 0 {
                divergence += 0.5 * q * log2(q / midpoint)
            }
        }

        let distance = sqrt(max(divergence, 0))
        return distance.isFinite ? distance : 1
    }
}

// ── LAPACK Inversion Shim ──────────────────────────────────────────────

/// Inverts a square general matrix in place via LU factorization.
///
/// Exists so ``MahalanobisDistance/init(covariance:)`` can call CLAPACK
/// (`sgetrf` + `sgetri`) without tripping `-warnings-as-errors`: Apple
/// deprecated the CLAPACK interface in favor of `ACCELERATE_NEW_LAPACK`,
/// which would require a package-wide compile flag. The calls live in a
/// deprecated protocol witness — deprecated contexts may use deprecated API
/// without diagnostics, and dispatching through the (non-deprecated)
/// protocol requirement keeps call sites warning-free.
private protocol MatrixInverting {
    /// Inverts a row-major `dimension × dimension` matrix in place.
    /// Returns the first nonzero LAPACK `info` code, or 0 on success.
    static func invertInPlace(_ matrix: inout [Float], dimension: Int) -> Int32
}

private enum CLAPACKInverter: MatrixInverting {
    @available(*, deprecated, message: "Contains the CLAPACK calls; use via MatrixInverting")
    static func invertInPlace(_ matrix: inout [Float], dimension: Int) -> Int32 {
        // LAPACK is column-major, but that is harmless for inversion:
        // interpreting a row-major buffer column-major hands LAPACK Aᵀ,
        // and inv(Aᵀ) = inv(A)ᵀ — reading the result back row-major yields
        // inv(A). (For a symmetric covariance matrix Aᵀ = A anyway.)
        var rows = __CLPK_integer(dimension)
        var n = __CLPK_integer(dimension)
        var lda = __CLPK_integer(dimension)
        var ipiv = [__CLPK_integer](repeating: 0, count: dimension)
        var info: __CLPK_integer = 0

        // Step 1: LU factorization in place.
        sgetrf_(&rows, &n, &matrix, &lda, &ipiv, &info)
        guard info == 0 else { return info }

        // Step 2: Workspace query, then invert from the LU factors.
        var lwork: __CLPK_integer = -1
        var workQuery: Float = 0
        sgetri_(&n, &matrix, &lda, &ipiv, &workQuery, &lwork, &info)
        guard info == 0 else { return info }

        lwork = __CLPK_integer(workQuery)
        var work = [Float](repeating: 0, count: max(Int(lwork), 1))
        sgetri_(&n, &matrix, &lda, &ipiv, &work, &lwork, &info)
        return info
    }
}

/// Inverts via the shim's protocol witness, so call sites never reference
/// the deprecated implementation directly.
private func invertMatrixInPlace(_ matrix: inout [Float], dimension: Int) -> Int32 {
    func dispatch<I: MatrixInverting>(_: I.Type, _ matrix: inout [Float], dimension: Int) -> Int32 {
        I.invertInPlace(&matrix, dimension: dimension)
    }
    return dispatch(CLAPACKInverter.self, &matrix, dimension: dimension)
}

// ── Mahalanobis Distance ───────────────────────────────────────────────

/// Mahalanobis distance: `sqrt((a-b)ᵀ · S⁻¹ · (a-b))` for a covariance
/// matrix `S`.
///
/// Range: [0, ∞) where 0 = identical vectors.
///
/// Covariate-aware similarity: components are weighted by the inverse of
/// their (co)variance, so dimensions with different scales or correlated
/// dimensions are compared fairly. With `S = I` this reduces to Euclidean
/// distance.
///
/// The covariance matrix must be **symmetric positive-definite** — that is
/// what makes the quadratic form non-negative and the result a genuine
/// metric (symmetric, satisfies the triangle inequality). Supplying a
/// matrix that is not positive-definite produces meaningless (clamped)
/// distances.
///
/// **Not serializable:** unlike the stateless built-in metrics, this metric
/// carries a `dimension × dimension` matrix payload, so it has no
/// ``DistanceMetricType`` case. Saving an index configured with it throws
/// `PersistenceError.unserializableMetric`, exactly like custom user-defined
/// metrics. Rebuild the metric from your covariance data after loading.
///
/// ```swift
/// let metric = MahalanobisDistance(covariance: covarianceMatrix)
/// let d = metric.distance(vectorA, vectorB)
/// ```
public struct MahalanobisDistance: DistanceMetric, Sendable {
    /// The dimension this metric's matrix was built for. Vectors passed to
    /// ``distance(_:_:)`` must match it.
    public let dimension: Int

    /// Row-major flattened `dimension × dimension` inverse covariance matrix.
    private let inverseCovariance: [Float]

    /// Creates the metric from a precomputed inverse covariance matrix `S⁻¹`.
    ///
    /// Use this when you already have the inverse (e.g. computed offline or
    /// shared across processes). The matrix must be square and should be the
    /// inverse of a symmetric positive-definite covariance matrix.
    ///
    /// - Parameter inverseCovariance: A square row-major matrix
    ///   (`[row][column]`).
    public init(inverseCovariance: [[Float]]) {
        let d = inverseCovariance.count
        precondition(d > 0, "Inverse covariance matrix must be non-empty")
        for row in inverseCovariance {
            precondition(row.count == d,
                         "Inverse covariance matrix must be square: row has \(row.count) columns, expected \(d)")
        }
        self.dimension = d
        self.inverseCovariance = inverseCovariance.flatMap { $0 }
    }

    /// Creates the metric from a covariance matrix `S`, inverting it via
    /// LAPACK (`sgetrf` LU factorization + `sgetri` inversion).
    ///
    /// - Parameter covariance: A square, symmetric **positive-definite**
    ///   row-major covariance matrix. Traps with a precondition failure if
    ///   the matrix is singular (LU factorization hits an exact zero pivot).
    public init(covariance: [[Float]]) {
        let d = covariance.count
        precondition(d > 0, "Covariance matrix must be non-empty")
        for row in covariance {
            precondition(row.count == d,
                         "Covariance matrix must be square: row has \(row.count) columns, expected \(d)")
        }

        var matrix = covariance.flatMap { $0 }
        let info = invertMatrixInPlace(&matrix, dimension: d)
        precondition(info == 0,
                     "Covariance matrix is singular — cannot invert (LAPACK info = \(info))")

        self.dimension = d
        self.inverseCovariance = matrix
    }

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        precondition(a.dimension == b.dimension,
                     "Dimension mismatch: \(a.dimension) vs \(b.dimension)")
        precondition(a.dimension == dimension,
                     "Vector dimension \(a.dimension) does not match inverse covariance dimension \(dimension)")

        let count = vDSP_Length(dimension)

        // Step 1: Difference vector d = b - a
        // (the sign cancels in the quadratic form dᵀ·S⁻¹·d)
        var difference = [Float](repeating: 0, count: dimension)
        a.components.withUnsafeBufferPointer { aPtr in
            b.components.withUnsafeBufferPointer { bPtr in
                vDSP_vsub(aPtr.baseAddress!, 1,
                          bPtr.baseAddress!, 1,
                          &difference, 1, count)
            }
        }

        // Step 2: y = S⁻¹ · d via matrix–vector multiply (D×D · D×1)
        var transformed = [Float](repeating: 0, count: dimension)
        inverseCovariance.withUnsafeBufferPointer { mPtr in
            vDSP_mmul(mPtr.baseAddress!, 1,
                      difference, 1,
                      &transformed, 1,
                      count, 1, count)
        }

        // Step 3: Quadratic form dᵀ·y; clamp tiny negative rounding before sqrt.
        var quadraticForm: Float = 0
        vDSP_dotpr(difference, 1, transformed, 1, &quadraticForm, count)

        return sqrt(max(quadraticForm, 0))
    }
}
