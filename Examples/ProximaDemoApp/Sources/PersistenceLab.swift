// PersistenceLab.swift
// ProximaDemoApp
//
// v1.6.0 Persistence panel controller. Dogfoods the streaming-persistence
// surface end to end on a dedicated, reproducible synthetic corpus (kept
// separate from the live search index so experimenting never disturbs search):
//
//   • HNSWIndex.open(baseURL:walURL:durability:mode:)  — journaled open
//   • .journalByteCount / .journalRecordCount / .needsCheckpoint()  — readouts
//   • checkpoint(baseURL:walURL:)                       — fold WAL → v3 base
//   • mode: .resident vs .paged  + task_vm_info footprint — live memory delta
//
// The honest bit: a fresh corpus is written with a plain `save()` (a v2 base),
// so a `.paged` open FAILS with the library's real typed error until you
// checkpoint. The panel surfaces that error and the one-tap recovery.

import Foundation
import ProximaKit

@Observable
@MainActor
final class PersistenceLab {

    // ── Fixed corpus shape (memory demo needs a real vector section, not a
    //    high-quality graph, so m/efConstruction are tuned for fast builds) ──
    private let dimension = 384
    private let graphM = 8
    private let graphEfConstruction = 16
    let corpusSizeOptions = [3_000, 6_000, 12_000]
    /// Warm searches `measureMemory()` runs against the paged index to fault
    /// mapped pages in. Non-private so the panel caption can quote the same
    /// number instead of hardcoding it separately.
    let warmSearchCount = 20

    enum Phase: Equatable { case empty, building, ready }
    enum OpenMode: String { case resident = "Resident", paged = "Paged" }

    // Inputs
    var corpusSize = 6_000

    // Lifecycle / status
    var phase: Phase = .empty
    var buildProgress = 0
    var buildSeconds: Double = 0
    var payloadMB: Double = 0
    var statusNote = ""
    var errorMessage: String?

    // Base / WAL state
    var baseExists = false
    var basePagedReady = false          // true once a checkpoint has written a v3 base
    var isOpen = false
    var openMode: OpenMode = .resident
    var generation: UInt64 = 0
    var walBytes = 0
    var walOps = 0
    var needsCheckpoint = false

    // The typed error from a `.paged` open on an unpadded base — the honesty moment.
    var pagedBlockedMessage: String?

    // Live memory measurement
    var isMeasuringMemory = false
    var residentMemoryMB: Double?
    var pagedMemoryMB: Double?
    /// Extra resident footprint faulted in by the warm searches on the paged,
    /// mmap'd index (f1b − f1). nil until a measurement completes.
    var warmSearchDeltaMB: Double?

    private var index: HNSWIndex?
    private var addedOps = 0

    // MARK: - Paths

    private static var dir: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("ProximaDemoApp/PersistenceLab")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    private static var baseURL: URL { dir.appendingPathComponent("lab.pxkt") }
    private static var walURL: URL { dir.appendingPathComponent("lab.pxwal") }

    // MARK: - Build

    /// Builds a fresh seeded corpus and writes it with a plain `save()` — a v2
    /// base that is deliberately NOT paged-ready, then opens it journaled
    /// (resident) so the WAL readouts come alive.
    func build() async {
        phase = .building
        buildProgress = 0
        errorMessage = nil
        pagedBlockedMessage = nil
        residentMemoryMB = nil
        pagedMemoryMB = nil
        warmSearchDeltaMB = nil
        addedOps = 0
        await closeIndex()
        try? FileManager.default.removeItem(at: Self.baseURL)
        try? FileManager.default.removeItem(at: Self.walURL)

        let started = DispatchTime.now().uptimeNanoseconds
        do {
            let fresh = HNSWIndex(
                dimension: dimension,
                metric: EuclideanDistance(),
                config: HNSWConfiguration(
                    m: graphM, efConstruction: graphEfConstruction, efSearch: 50,
                    autoCompactionThreshold: nil, levelSeed: 7))
            for i in 0..<corpusSize {
                try await fresh.add(SyntheticCorpus.vector(i, dimension: dimension),
                                    id: SyntheticCorpus.id(i))
                if i % 200 == 0 { buildProgress = i }
            }
            buildProgress = corpusSize
            // Plain save → v2 base (unpadded): paging will refuse it until a checkpoint.
            try await fresh.save(to: Self.baseURL)
            payloadMB = Double(corpusSize * dimension * 4) / 1_048_576
            baseExists = true
            basePagedReady = false
            buildSeconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
            statusNote = "Built \(corpusSize) vectors in \(String(format: "%.1f", buildSeconds)) s, saved a v2 base."
            // Attach a WAL by opening journaled (resident works on a v2 base).
            try await openJournaled(mode: .resident, announce: false)
            phase = .ready
        } catch {
            errorMessage = error.localizedDescription
            phase = baseExists ? .ready : .empty
        }
    }

    // MARK: - Open (journaled)

    /// Opens the base journaled in the requested mode, attaching/replaying the
    /// WAL. A `.paged` open on an unpadded base throws a typed
    /// `PersistenceError`; we surface it and fall back to a resident open so the
    /// panel stays live.
    func openJournaled(mode: OpenMode, announce: Bool = true) async throws {
        do {
            // Open the NEW index before releasing the current one, so a failed
            // paged open leaves the existing (resident) index — and its live WAL
            // readouts — untouched. No disruptive reopen, no op-count reset.
            let opened = try await HNSWIndex.open(
                baseURL: Self.baseURL, walURL: Self.walURL,
                mode: mode == .paged ? .paged : .resident)
            await closeIndex()
            index = opened
            openMode = mode
            isOpen = true
            if mode == .paged { pagedBlockedMessage = nil }   // paging now works
            await refreshReadouts()
            if announce { statusNote = "Opened journaled in \(mode.rawValue) mode." }
        } catch let error as PersistenceError where mode == .paged {
            // The honesty moment: surface the library's own guidance and keep
            // the current index open so the panel stays live.
            pagedBlockedMessage = error.errorDescription ?? "\(error)"
            statusNote = "Paged open refused — checkpoint to write a page-aligned base."
        }
    }

    func switchMode(to mode: OpenMode) {
        Task {
            do { try await openJournaled(mode: mode) }
            catch { errorMessage = error.localizedDescription }
        }
    }

    // MARK: - Grow the WAL

    /// Appends `count` journaled adds so the WAL grows and `needsCheckpoint`
    /// eventually flips — the streaming-persistence value proposition (saves are
    /// O(change), not O(corpus)).
    func addOps(_ count: Int) async {
        guard let index else { return }
        do {
            for _ in 0..<count {
                let i = corpusSize + addedOps
                try await index.add(SyntheticCorpus.vector(i, dimension: dimension),
                                    id: SyntheticCorpus.id(i))
                addedOps += 1
            }
            try await index.syncJournal()
            await refreshReadouts()
            statusNote = "Appended \(count) ops to the WAL (\(walOps) since last checkpoint)."
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Checkpoint

    /// Folds the WAL back into a fresh, page-aligned v3 base (generation bumped,
    /// WAL reset). This is what makes `.paged` open possible.
    func checkpoint() async {
        guard let index else { return }
        do {
            try await index.checkpoint(baseURL: Self.baseURL, walURL: Self.walURL)
            basePagedReady = true
            pagedBlockedMessage = nil
            await refreshReadouts()
            statusNote = "Checkpoint complete — v3 base at generation \(generation), WAL reset."
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Memory measurement (resident vs paged)

    /// Measures the live `phys_footprint` cost of opening the SAME base resident
    /// vs paged, using the exact methodology of `PagedVectorMemoryTests`: from a
    /// clean baseline, open paged (+warm searches), then open resident on top
    /// and attribute the additional footprint to the vector payload the resident
    /// open copies and the paged open leaves on disk.
    func measureMemory() async {
        guard basePagedReady else {
            statusNote = "Checkpoint first — paged open needs a v3 base."
            return
        }
        isMeasuringMemory = true
        errorMessage = nil
        let priorMode = openMode
        await closeIndex()   // clean baseline: release the panel's open index

        do {
            let f0 = MemoryProbe.physFootprint()
            let paged = try await HNSWIndex.open(baseURL: Self.baseURL, walURL: Self.walURL, mode: .paged)
            _ = await paged.count
            let f1 = MemoryProbe.physFootprint()
            for q in 0..<warmSearchCount {
                _ = await paged.search(query: SyntheticCorpus.vector(2_000_000 + q, dimension: dimension), k: 10)
            }
            let f1b = MemoryProbe.physFootprint()

            let resident = try await HNSWIndex.open(baseURL: Self.baseURL, walURL: Self.walURL, mode: .resident)
            _ = await resident.count
            let f2 = MemoryProbe.physFootprint()

            await paged.closeJournal()
            await resident.closeJournal()

            pagedMemoryMB = Double(f1 &- f0) / 1_048_576
            warmSearchDeltaMB = Double(f1b &- f1) / 1_048_576
            residentMemoryMB = Double(f2 &- f1b) / 1_048_576
            statusNote = "Measured live: resident open costs "
                + String(format: "%.1f MB vs %.1f MB paged.", residentMemoryMB ?? 0, pagedMemoryMB ?? 0)
        } catch {
            errorMessage = error.localizedDescription
        }

        // Reopen the panel's index in whatever mode it was in.
        try? await openJournaled(mode: priorMode, announce: false)
        isMeasuringMemory = false
    }

    // MARK: - Reset

    func reset() async {
        await closeIndex()
        try? FileManager.default.removeItem(at: Self.baseURL)
        try? FileManager.default.removeItem(at: Self.walURL)
        phase = .empty
        baseExists = false
        basePagedReady = false
        isOpen = false
        generation = 0
        walBytes = 0
        walOps = 0
        needsCheckpoint = false
        pagedBlockedMessage = nil
        residentMemoryMB = nil
        pagedMemoryMB = nil
        warmSearchDeltaMB = nil
        addedOps = 0
        statusNote = "Reset — lab files deleted."
    }

    // MARK: - Helpers

    private func refreshReadouts() async {
        guard let index else { return }
        generation = await index.currentGeneration
        walBytes = await index.journalByteCount
        walOps = await index.journalRecordCount
        needsCheckpoint = await index.needsCheckpoint()
    }

    private func closeIndex() async {
        if let index { await index.closeJournal() }
        index = nil
        isOpen = false
    }
}
