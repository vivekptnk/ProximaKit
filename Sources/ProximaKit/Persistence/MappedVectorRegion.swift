// MappedVectorRegion.swift
// ProximaKit
//
// ADR-013 Stage 2 (Option C): the read-only file mapping that serves the
// vector section of a `.pxkt` v3 base without making it resident, plus the
// `VectorProvider` that unifies resident and paged vector access behind one
// hot-path accessor.
//
// Concurrency (ADR-002 discipline): a `MappedVectorRegion` is created once and
// then confined to the `HNSWIndex` actor that owns it (stored inside the
// actor-isolated `VectorProvider`). Its raw mapping pointer is dereferenced
// only inside synchronous, actor-isolated calls and is NEVER handed out or held
// across an `await` — every access copies the requested vector into a
// value-typed `Vector` within a single synchronous scope, so a concurrent
// checkpoint/remap (also actor-isolated and synchronous) can never observe a
// pointer mid-flight. `UnsafeBufferPointer` is not `Sendable`; because the
// public surface stays value-typed (`Vector`), `StrictConcurrency` holds the
// line at compile time. That confinement — the same argument the codebase makes
// for actor-isolated mutable state and for `WALJournal` — is what justifies the
// `@unchecked Sendable` here.
//
// SIGBUS contract (documented, not hidden — see ADR-013 "mmap lifetime"): the
// mapping is opened read-only, its size `fstat`-ed once, and the library never
// truncates its own live files (a checkpoint always renames a fresh inode, so
// an open mapping keeps serving the pinned pre-checkpoint inode). The residual
// exposure is EXTERNAL truncation of the mapped file by another process:
// touching a faulted page past a shrunken EOF raises SIGBUS, which Swift cannot
// catch. This is the same risk class the `.mappedIfSafe` decode pass already
// carries at load time; paging only extends the window from load-time to index
// lifetime. External mutation of a file ProximaKit has open is out of contract.

import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// A read-only memory mapping over the fixed-stride vector section of a `.pxkt`
/// v3 base. Vector `i` lives at `vectorSectionStart + i × dimension × 4`, so
/// access is O(1) with no decode — the OS faults pages in on demand.
final class MappedVectorRegion: @unchecked Sendable {

    /// The byte alignment the v3 writer pads the vector section to, and which a
    /// paged open requires — the Apple-Silicon page size. Enforced so each
    /// fault pulls a clean vector page rather than one straddling another
    /// section.
    static let requiredAlignment = 16_384

    private let fileDescriptor: Int32
    /// Base of the mapped vector section, or `nil` for an empty base (nothing
    /// to map; `vector(at:)` is never called when `count == 0`).
    private let mapping: UnsafeRawPointer?
    private let mappingLength: Int

    let dimension: Int
    /// Number of vectors served from the mapping (= the base snapshot's node
    /// count). The resident tail for post-snapshot adds lives in the owning
    /// `VectorProvider`, not here.
    let count: Int

    /// Opens `baseURL` read-only and maps its vector section. Throws a typed
    /// `PersistenceError` (never traps) for a non-v3 base, a mis-sized or
    /// out-of-bounds section table, a vector section that is not page-aligned
    /// (an unpadded base — re-checkpoint to enable paging), or an mmap failure.
    init(baseURL: URL) throws {
        let layout = try PersistenceEngine.pagedVectorLayout(of: baseURL)
        guard layout.vectorOffset % Self.requiredAlignment == 0 else {
            throw PersistenceError.corruptedData(
                "vector section offset \(layout.vectorOffset) is not "
                + "\(Self.requiredAlignment)-byte aligned; re-checkpoint the base to enable paging")
        }
        self.dimension = layout.dimension
        self.count = layout.count

        let fd = baseURL.path.withCString { open($0, O_RDONLY) }
        guard fd >= 0 else {
            throw PersistenceError.corruptedData(
                "Could not open base for paging at \(baseURL.path) (errno \(errno))")
        }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd)
            throw PersistenceError.corruptedData("fstat failed for \(baseURL.path) (errno \(errno))")
        }
        let fileSize = Int(st.st_size)
        guard layout.vectorOffset + layout.vectorLength <= fileSize else {
            close(fd)
            throw PersistenceError.corruptedData(
                "vector section (offset \(layout.vectorOffset), length \(layout.vectorLength)) "
                + "exceeds file size \(fileSize)")
        }

        self.fileDescriptor = fd

        guard layout.vectorLength > 0 else {
            // Empty base: no pages to map. `mmap` with length 0 is EINVAL, and
            // an index with no vectors never calls `vector(at:)`.
            self.mapping = nil
            self.mappingLength = 0
            return
        }

        let raw = mmap(nil, layout.vectorLength, PROT_READ, MAP_PRIVATE, fd, off_t(layout.vectorOffset))
        guard let raw = raw, raw != MAP_FAILED else {
            close(fd)
            throw PersistenceError.corruptedData("mmap failed for vector section (errno \(errno))")
        }
        self.mapping = UnsafeRawPointer(raw)
        self.mappingLength = layout.vectorLength
    }

    /// Copies vector `node` out of the mapping into a value-typed `Vector`.
    ///
    /// The copy is deliberate: it keeps the raw mapping pointer inside this
    /// synchronous scope (never escaping across an `await`) and makes paged
    /// results bit-identical to resident ones (same Float32 bytes, same math).
    /// The vector payload stays file-backed — only the touched pages fault in,
    /// and only the transient copy is heap-resident, freed after the caller's
    /// distance evaluation.
    @inline(__always)
    func vector(at node: Int) -> Vector {
        let floatPtr = mapping!
            .advanced(by: node * dimension * 4)
            .assumingMemoryBound(to: Float.self)
        return Vector(UnsafeBufferPointer(start: floatPtr, count: dimension))
    }

    deinit {
        if let mapping = mapping, mappingLength > 0 {
            munmap(UnsafeMutableRawPointer(mutating: mapping), mappingLength)
        }
        close(fileDescriptor)
    }
}

/// Serves per-node vectors to the HNSW hot path, hiding whether they live in a
/// resident array (the default — byte-identical to the historical `[Vector]`)
/// or a read-only file mapping with a resident tail for post-snapshot adds
/// (ADR-013 Stage 2 `.paged`).
///
/// Layout: node ids `[0, snapshotBoundary)` are served from `region`; ids
/// `[snapshotBoundary, count)` from the resident `tail`. In resident mode
/// `snapshotBoundary == 0` and `tail` holds everything, so `vector(at:)`
/// reduces to a single array subscript (the extra index arithmetic against a
/// stored zero is a couple of integer ops, measured free against a vDSP
/// distance — see ADR-013 Stage 2 notes).
///
/// This is a value type storing actor-isolated state; it never crosses a
/// concurrency boundary itself, and its only reference member
/// (`MappedVectorRegion`) is `Sendable`.
struct VectorProvider {

    /// Read-only mapping of the base snapshot's vectors, or `nil` in resident
    /// mode.
    private let region: MappedVectorRegion?

    /// Node ids `< snapshotBoundary` are mapped; ids `>=` are in `tail`. Always
    /// 0 in resident mode.
    private let snapshotBoundary: Int

    /// Post-snapshot (resident) vectors. In resident mode holds ALL vectors; in
    /// paged mode only those added after the mapped base (WAL replay + live
    /// adds append here).
    private var tail: [Vector]

    let dimension: Int

    // MARK: - Construction

    /// A resident provider seeded with `vectors` (or empty). `region == nil`.
    static func resident(_ vectors: [Vector] = [], dimension: Int) -> VectorProvider {
        VectorProvider(region: nil, snapshotBoundary: 0, tail: vectors, dimension: dimension)
    }

    /// A paged provider serving `region.count` vectors from the mapping, with an
    /// empty resident tail ready for post-snapshot adds.
    static func paged(region: MappedVectorRegion, dimension: Int) -> VectorProvider {
        VectorProvider(region: region, snapshotBoundary: region.count, tail: [], dimension: dimension)
    }

    // MARK: - Hot path

    /// Total node slots (mapped + tail) — the historical `vectors.count`.
    @inline(__always)
    var count: Int { snapshotBoundary + tail.count }

    /// Whether vectors are served from a file mapping.
    var isPaged: Bool { region != nil }

    /// The vector for `node`. Resident: a single `tail` subscript. Paged: a
    /// tail subscript for post-snapshot ids, else a scoped copy out of the
    /// mapping. Never returns or retains a raw mapping pointer.
    @inline(__always)
    func vector(at node: Int) -> Vector {
        let t = node - snapshotBoundary
        if t >= 0 { return tail[t] }
        // node ∈ [0, snapshotBoundary) ⇒ region is non-nil by construction.
        return region!.vector(at: node)
    }

    /// Appends a live/added vector to the resident tail.
    @inline(__always)
    mutating func append(_ vector: Vector) { tail.append(vector) }

    // MARK: - Bulk

    /// Materializes ALL vectors into a resident array — used by `compact` and
    /// snapshotting. Resident: returns `tail` (a cheap COW share). Paged: faults
    /// every mapped vector and copies it out, then appends the tail.
    func materialized() -> [Vector] {
        guard let region = region else { return tail }
        var out = [Vector]()
        out.reserveCapacity(count)
        for i in 0..<snapshotBoundary { out.append(region.vector(at: i)) }
        out.append(contentsOf: tail)
        return out
    }
}
