// WALJournal.swift
// ProximaKit
//
// File-backed writer for the PXWL v1 sidecar (ADR-013, Stage 1). Owns an
// append-mode file descriptor and applies the durability policy. The pure
// framing/decoding lives in `WALFormat.swift`.
//
// Concurrency: a `WALJournal` is confined to the `HNSWIndex` actor that owns
// it — every `append*` call happens synchronously inside an actor-isolated
// mutation method, so the writer is never touched concurrently. That
// confinement is what justifies `@unchecked Sendable` here, the same argument
// the codebase makes for actor-isolated mutable state generally (ADR-002).

import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Durability / throughput dial for WAL appends (ADR-013, criterion 2).
///
/// Darwin honesty (documented, not overpromised): a plain `fsync(2)` on Darwin
/// pushes writes to the **drive cache**, not the physical media — only
/// `fcntl(_:F_FULLFSYNC)` forces a media write. Checkpoint commits always use
/// `F_FULLFSYNC`; these per-record/per-batch levels use `fsync`.
public enum WALDurability: Sendable {
    /// `fsync` after every individual record. Highest durability, lowest
    /// throughput. On Darwin this reaches the drive cache after each record;
    /// it does NOT force the media (only a checkpoint's `F_FULLFSYNC` does).
    case everyRecord

    /// `fsync` once per append call (the batch of records from one mutation).
    /// Default. On Darwin, reaches the drive cache once per mutation call.
    /// "Batch" is the records appended by one mutation call, not a
    /// caller-controlled group: every index mutation appends exactly one record
    /// today, so `.everyBatch` and `.everyRecord` currently coincide. A future
    /// batched-append surface (many records per mutation) must revisit whether
    /// this per-write `fsync` is still the intended `.everyBatch` behavior.
    case everyBatch

    /// No `fsync` on append. Records reach the OS page cache only; a power loss
    /// before the next checkpoint can lose the tail. Fastest; use when the base
    /// snapshot cadence is the real durability boundary.
    case manual
}

/// Policy governing when an accumulated WAL should be folded back into a fresh
/// base snapshot (ADR-013, criterion 4). A checkpoint fires when either bound
/// is exceeded.
public struct WALCheckpointPolicy: Sendable {
    /// Checkpoint once the WAL grows past this fraction of the base snapshot's
    /// size. Default 0.10 (10%).
    public var walBytesFractionOfBase: Double
    /// Checkpoint once this many records have accumulated. Default 10_000.
    public var maxOps: Int

    public init(walBytesFractionOfBase: Double = 0.10, maxOps: Int = 10_000) {
        self.walBytesFractionOfBase = walBytesFractionOfBase
        self.maxOps = maxOps
    }
}

/// Appends PXWL records to a `.pxwal` file and flushes per the durability dial.
final class WALJournal: @unchecked Sendable {

    let url: URL
    let parentGeneration: UInt64
    let dimension: Int
    private let durability: WALDurability
    private let handle: FileHandle

    /// Total bytes written to the file (header + records). Drives the
    /// checkpoint policy without an extra `stat`.
    private(set) var byteCount: Int

    /// Number of records appended since the last checkpoint. Drives the
    /// op-count checkpoint policy.
    private(set) var recordCount: Int

    /// Deferred error from a non-throwing append (`appendRemove`, called from
    /// the actor's non-throwing `remove`). Surfaced by the next throwing
    /// journaled operation so a write failure is never swallowed.
    private(set) var pendingError: Error?

    /// Opens a fresh WAL, writing (and `F_FULLFSYNC`-committing) its header so
    /// the parent-generation binding is durable before any record is appended.
    init(
        creatingAt url: URL,
        parentGeneration: UInt64,
        dimension: Int,
        metricRaw: UInt32,
        durability: WALDurability
    ) throws {
        self.url = url
        self.parentGeneration = parentGeneration
        self.dimension = dimension
        self.durability = durability

        let header = WALFormat.encodeHeader(
            parentGeneration: parentGeneration, dimension: dimension, metricRaw: metricRaw
        )
        // Truncate-create so a checkpoint's WAL reset starts from a clean file.
        try header.write(to: url, options: .atomic)
        guard let handle = try? FileHandle(forWritingTo: url) else {
            throw PersistenceError.corruptedData("Could not open WAL for writing at \(url.path)")
        }
        self.handle = handle
        try handle.seekToEnd()
        self.byteCount = header.count
        self.recordCount = 0
        try Self.fullSync(handle)   // header must be durable before records
    }

    /// Reopens an existing WAL in append mode (records already present are
    /// preserved — the replay that just read them keeps its prefix). The
    /// header's parent generation is trusted to have been validated by the
    /// decoder before this call.
    ///
    /// Both counters are seeded from what the replay carried in so they keep
    /// their "since the last checkpoint" meaning across a reopen: the WAL is
    /// truncate-created fresh at each checkpoint, so every record in it (and
    /// every byte past the header) was written since that checkpoint.
    /// `existingRecordCount` is the number of records the decoder replayed from
    /// the valid prefix — it must match `existingByteCount`'s prefix exactly,
    /// so the byte- and op-count checkpoint rules agree from the first append.
    init(
        appendingTo url: URL,
        parentGeneration: UInt64,
        dimension: Int,
        existingByteCount: Int,
        existingRecordCount: Int,
        durability: WALDurability
    ) throws {
        self.url = url
        self.parentGeneration = parentGeneration
        self.dimension = dimension
        self.durability = durability
        guard let handle = try? FileHandle(forWritingTo: url) else {
            throw PersistenceError.corruptedData("Could not open WAL for append at \(url.path)")
        }
        self.handle = handle
        try handle.seekToEnd()
        self.byteCount = existingByteCount
        self.recordCount = existingRecordCount
    }

    // MARK: - Append

    /// Appends an `add` record (throwing path — called from `HNSWIndex.add`,
    /// which already throws). Surfaces any deferred error first.
    func appendAdd(id: UUID, level: Int, vector: [Float], metadata: Data?) throws {
        try surfacePending()
        try write(WALFormat.encodeRecord(.add(id: id, level: level, vector: vector, metadata: metadata)))
    }

    /// Appends a `remove` record. `HNSWIndex.remove` is non-throwing, so a write
    /// failure here is captured into `pendingError` and re-raised by the next
    /// throwing journaled op rather than crashing or being lost.
    func appendRemove(id: UUID) {
        guard pendingError == nil else { return }
        do {
            try write(WALFormat.encodeRecord(.remove(id: id)))
        } catch {
            pendingError = error
        }
    }

    /// Forces the OS to flush appended records. `fsync` on Darwin reaches the
    /// drive cache; see ``WALDurability``. Throws any deferred append error.
    func sync() throws {
        try surfacePending()
        try Self.dataSync(handle)
    }

    /// Full-media flush (`F_FULLFSYNC` on Darwin). Used at checkpoint commit.
    func fullSync() throws {
        try surfacePending()
        try Self.fullSync(handle)
    }

    func close() {
        try? handle.close()
    }

    // MARK: - Internals

    private func write(_ frame: Data) throws {
        try handle.write(contentsOf: frame)
        byteCount += frame.count
        recordCount += 1
        switch durability {
        case .everyRecord, .everyBatch:
            // Index-level appends are single-record, so per-record and
            // per-batch coincide here; both fsync now. `.manual` skips.
            try Self.dataSync(handle)
        case .manual:
            break
        }
    }

    private func surfacePending() throws {
        if let error = pendingError {
            pendingError = nil
            throw error
        }
    }

    /// `fsync` — Darwin: reaches the drive cache, not the media.
    private static func dataSync(_ handle: FileHandle) throws {
        let fd = handle.fileDescriptor
        guard fsync(fd) == 0 else {
            throw PersistenceError.corruptedData("fsync failed (errno \(errno))")
        }
    }

    /// `F_FULLFSYNC` on Darwin forces a media write; elsewhere falls back to
    /// `fsync` (the platform's strongest available flush).
    private static func fullSync(_ handle: FileHandle) throws {
        let fd = handle.fileDescriptor
        #if canImport(Darwin)
        if fcntl(fd, F_FULLFSYNC) == 0 { return }
        // Fall through to fsync if F_FULLFSYNC is unsupported (e.g. some FS).
        #endif
        guard fsync(fd) == 0 else {
            throw PersistenceError.corruptedData("fullSync failed (errno \(errno))")
        }
    }
}
