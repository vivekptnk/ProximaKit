// FVecLoader.swift
// Reader for the SIFT1M .fvecs / .ivecs binary format.
//
// Format (little-endian, Corpus of TEXMEX): for each vector,
//   int32 dim
//   float32 * dim   (for .fvecs)
//   int32 * dim     (for .ivecs)
// repeated back-to-back. We mmap the file to avoid loading the whole thing
// into heap when taking subsets.

import Foundation

enum FVecLoader {
    struct FloatMatrix {
        let dimension: Int
        let count: Int
        /// Row-major contiguous floats, length = count * dimension.
        let data: [Float]
    }

    struct IntMatrix {
        let dimension: Int
        let count: Int
        /// Row-major contiguous ints, length = count * dimension.
        let data: [Int32]
    }

    enum LoadError: Error, CustomStringConvertible {
        case fileNotFound(String)
        case truncated(String)
        case inconsistentDimension(expected: Int, got: Int, at: Int)

        var description: String {
            switch self {
            case .fileNotFound(let p): return "fvec file not found: \(p)"
            case .truncated(let p): return "fvec file truncated: \(p)"
            case .inconsistentDimension(let exp, let got, let at):
                return "fvec dim mismatch at vector \(at): expected \(exp), got \(got)"
            }
        }
    }

    /// Load at most `limit` vectors from a .fvecs file.
    static func loadFvecs(path: String, limit: Int? = nil) throws -> FloatMatrix {
        let data = try readMapped(path)
        return try data.withUnsafeBytes { raw in
            try parseFvecs(bytes: raw, limit: limit)
        }
    }

    /// Load at most `limit` vectors from a .ivecs file.
    static func loadIvecs(path: String, limit: Int? = nil) throws -> IntMatrix {
        let data = try readMapped(path)
        return try data.withUnsafeBytes { raw in
            try parseIvecs(bytes: raw, limit: limit)
        }
    }

    // MARK: - Private

    private static func readMapped(_ path: String) throws -> Data {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            throw LoadError.fileNotFound(path)
        }
        return try Data(contentsOf: url, options: [.mappedIfSafe])
    }

    private static func parseFvecs(bytes: UnsafeRawBufferPointer, limit: Int?) throws -> FloatMatrix {
        var offset = 0
        var firstDim = -1
        var out: [Float] = []
        var count = 0

        while offset + 4 <= bytes.count {
            let dim = Int(bytes.load(fromByteOffset: offset, as: Int32.self))
            if firstDim < 0 { firstDim = dim }
            if dim != firstDim {
                throw LoadError.inconsistentDimension(expected: firstDim, got: dim, at: count)
            }
            offset += 4
            let bytesForVec = dim * MemoryLayout<Float>.size
            guard offset + bytesForVec <= bytes.count else {
                throw LoadError.truncated("fvecs body")
            }
            out.reserveCapacity(out.count + dim)
            for i in 0..<dim {
                let f = bytes.load(fromByteOffset: offset + i * 4, as: Float.self)
                out.append(f)
            }
            offset += bytesForVec
            count += 1
            if let limit, count >= limit { break }
        }

        return FloatMatrix(dimension: max(firstDim, 0), count: count, data: out)
    }

    private static func parseIvecs(bytes: UnsafeRawBufferPointer, limit: Int?) throws -> IntMatrix {
        var offset = 0
        var firstDim = -1
        var out: [Int32] = []
        var count = 0

        while offset + 4 <= bytes.count {
            let dim = Int(bytes.load(fromByteOffset: offset, as: Int32.self))
            if firstDim < 0 { firstDim = dim }
            if dim != firstDim {
                throw LoadError.inconsistentDimension(expected: firstDim, got: dim, at: count)
            }
            offset += 4
            let bytesForVec = dim * MemoryLayout<Int32>.size
            guard offset + bytesForVec <= bytes.count else {
                throw LoadError.truncated("ivecs body")
            }
            out.reserveCapacity(out.count + dim)
            for i in 0..<dim {
                let v = bytes.load(fromByteOffset: offset + i * 4, as: Int32.self)
                out.append(v)
            }
            offset += bytesForVec
            count += 1
            if let limit, count >= limit { break }
        }

        return IntMatrix(dimension: max(firstDim, 0), count: count, data: out)
    }
}
