// PersistencePanel.swift
// ProximaDemoApp
//
// v1.6.0 Persistence panel UI: journaled open, live WAL readouts, checkpoint,
// the unpadded-base error path, and a live resident-vs-paged memory delta.

import SwiftUI
import ProximaKit

struct PersistencePanel: View {
    @Bindable var lab: PersistenceLab
    #if !os(macOS)
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    #endif

    // Stable scroll anchor so a completed measurement can scroll the memory card
    // into view (it sits below the fold on iPhone compact width).
    private let memoryCardID = "memoryCard"

    // Compact widths (iPhone, narrow splits) can't fit the three bordered action
    // buttons' full titles side by side; regular widths (iPad split, macOS) can.
    private var isCompact: Bool {
        #if os(macOS)
        false
        #else
        horizontalSizeClass == .compact
        #endif
    }

    var body: some View {
        ScrollView {
            ScrollViewReader { proxy in
                VStack(alignment: .leading, spacing: 20) {
                    header

                    switch lab.phase {
                    case .empty:
                        buildCard
                    case .building:
                        buildingCard
                    case .ready:
                        readoutsCard
                        pagedBlockedBanner
                        modeAndActionsCard
                        memoryCard.id(memoryCardID)
                        statusFooter
                    }

                    if let error = lab.errorMessage {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption).foregroundStyle(.red)
                    }
                }
                .padding(20)
                .frame(maxWidth: 640, alignment: .leading)
                .frame(maxWidth: .infinity)
                // When a measurement lands, scroll the memory card into view so its
                // result is visible without manual scrolling (it's below the fold on
                // compact width; also fixes the -demoFlow memory screenshot). Not
                // gated on any launch flag — a real "Measure memory" tap scrolls too.
                .onChange(of: lab.residentMemoryMB) { _, newValue in
                    guard newValue != nil else { return }
                    withAnimation { proxy.scrollTo(memoryCardID, anchor: .top) }
                }
            }
        }
        .navigationTitle("Persistence")
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("WAL + Paged Index")
                .font(.title2.weight(.semibold))
            Text("Open the index journaled, watch the write-ahead log grow, checkpoint it into a page-aligned base, then compare resident vs paged memory — measured live.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    // MARK: - Build

    private var buildCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Build a lab corpus")
                .font(.headline)
            Text("A reproducible synthetic corpus (\(lab.corpusSize) × 384d) written with a plain save() — a v2 base that is deliberately not paged-ready yet.")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Corpus size", selection: $lab.corpusSize) {
                ForEach(lab.corpusSizeOptions, id: \.self) { Text("\($0) vectors").tag($0) }
            }
            .pickerStyle(.segmented)
            Button {
                Task { await lab.build() }
            } label: {
                Label("Build & Save", systemImage: "hammer.fill").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
        }
        .cardBackground()
    }

    private var buildingCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            ProgressView(value: Double(lab.buildProgress), total: Double(lab.corpusSize)) {
                Text("Indexing \(lab.buildProgress) / \(lab.corpusSize)…").font(.callout)
            }
            Text("Building a real HNSW graph so the base has a genuine vector section to map.")
                .font(.caption).foregroundStyle(.tertiary)
        }
        .cardBackground()
    }

    // MARK: - Readouts

    private var readoutsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("WAL state").font(.headline)
            LabeledContent("Generation", value: "\(lab.generation)")
            LabeledContent("WAL size on disk", value: byteString(lab.walBytes))
            LabeledContent("Ops since checkpoint", value: "\(lab.walOps)")
            HStack {
                Text("Needs checkpoint").foregroundStyle(.secondary)
                Spacer()
                badge(lab.needsCheckpoint ? "Yes" : "No",
                      color: lab.needsCheckpoint ? .orange : .green,
                      systemImage: lab.needsCheckpoint ? "arrow.triangle.2.circlepath" : "checkmark")
            }
            HStack {
                Text("Base format").foregroundStyle(.secondary)
                Spacer()
                badge(lab.basePagedReady ? "v3 · paged-ready" : "v2 · not paged-ready",
                      color: lab.basePagedReady ? .green : .secondary,
                      systemImage: lab.basePagedReady ? "externaldrive.badge.checkmark" : "externaldrive")
            }
        }
        .font(.callout)
        .cardBackground()
    }

    // MARK: - Paged-blocked banner (the honesty moment)

    @ViewBuilder
    private var pagedBlockedBanner: some View {
        if let message = lab.pagedBlockedMessage {
            VStack(alignment: .leading, spacing: 10) {
                Label("Paged open blocked", systemImage: "lock.doc")
                    .font(.headline).foregroundStyle(.orange)
                Text(message)
                    .font(.callout).foregroundStyle(.primary)
                Text("The library refused to page an unpadded base rather than trap. Checkpoint writes a page-aligned v3 base and unblocks paging.")
                    .font(.caption).foregroundStyle(.secondary)
                Button {
                    Task { await lab.checkpoint() }
                } label: {
                    Label("Checkpoint to enable paging", systemImage: "arrow.down.doc.fill")
                }
                .buttonStyle(.borderedProminent)
            }
            .cardBackground(tint: .orange)
        }
    }

    // MARK: - Mode + actions

    private var modeAndActionsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Open mode").font(.headline)
            Picker("Mode", selection: Binding(
                get: { lab.openMode },
                set: { lab.switchMode(to: $0) })) {
                Text("Resident").tag(PersistenceLab.OpenMode.resident)
                Text("Paged").tag(PersistenceLab.OpenMode.paged)
            }
            .pickerStyle(.segmented)
            Text(lab.openMode == .paged
                 ? "Vectors served from a read-only file mapping, faulted in on demand."
                 : "Whole base decoded into resident memory — the classic path.")
                .font(.caption).foregroundStyle(.secondary)

            Divider()

            actionButtons
                .buttonStyle(.bordered)
                .font(.callout)
        }
        .cardBackground()
    }

    // The three row actions. On compact width their full titles can't fit side by
    // side — the middle "Checkpoint" label wraps mid-word — so we collapse to
    // icon-only glyphs there. Label keeps its title as the accessibility label
    // under .iconOnly, so VoiceOver is unaffected; regular width keeps icon+text.
    @ViewBuilder
    private var actionButtons: some View {
        let buttons = HStack(spacing: 12) {
            Button {
                Task { await lab.addOps(25) }
            } label: {
                Label("Add 25 ops", systemImage: "plus.rectangle.on.rectangle")
            }
            Button {
                Task { await lab.checkpoint() }
            } label: {
                Label("Checkpoint", systemImage: "arrow.down.doc")
            }
            .disabled(lab.walOps == 0 && lab.basePagedReady)
            Spacer()
            Button(role: .destructive) {
                Task { await lab.reset() }
            } label: {
                Label("Reset", systemImage: "trash")
            }
        }
        if isCompact {
            buttons.labelStyle(.iconOnly)
        } else {
            buttons.labelStyle(.titleAndIcon)
        }
    }

    // MARK: - Memory

    private var memoryCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Memory: resident vs paged").font(.headline)

            if let resident = lab.residentMemoryMB, let paged = lab.pagedMemoryMB {
                Text("Index memory: \(String(format: "%.1f", resident)) MB (resident) vs \(String(format: "%.1f", paged)) MB (paged)")
                    .font(.system(.title3, design: .rounded).weight(.semibold))
                    .foregroundStyle(.blue)
                Text("Vector payload on disk: \(String(format: "%.1f", lab.payloadMB)) MB · phys_footprint delta measured live via task_vm_info.")
                    .font(.caption).foregroundStyle(.secondary)
                if let delta = lab.warmSearchDeltaMB {
                    Text("After \(lab.warmSearchCount) warm searches: \(String(format: "%+.1f", delta)) MB")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            } else {
                Text("Open the same base both ways and compare the live process footprint. Paged keeps the vector payload on disk.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if lab.isMeasuringMemory {
                ProgressView("Measuring footprint…").font(.callout)
            } else {
                Button {
                    Task { await lab.measureMemory() }
                } label: {
                    Label("Measure memory", systemImage: "gauge.with.dots.needle.67percent")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(!lab.basePagedReady)
            }
            if !lab.basePagedReady {
                Text("Checkpoint first — paged open needs a page-aligned v3 base.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .cardBackground()
    }

    private var statusFooter: some View {
        Group {
            if !lab.statusNote.isEmpty {
                Label(lab.statusNote, systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Small helpers

    private func badge(_ text: String, color: Color, systemImage: String) -> some View {
        Label(text, systemImage: systemImage)
            .font(.caption.weight(.medium))
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }

    private func byteString(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}

private extension View {
    func cardBackground(tint: Color = .clear) -> some View {
        padding(16)
            .background(tint == .clear ? AnyShapeStyle(.quaternary.opacity(0.4)) : AnyShapeStyle(tint.opacity(0.12)),
                        in: RoundedRectangle(cornerRadius: 12))
    }
}
