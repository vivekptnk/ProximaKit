import SwiftUI
import UniformTypeIdentifiers

#if os(macOS)
import AppKit
#endif

struct ResultsExportView: View {
    @Bindable var engine: SearchEngine
    @Binding var queryText: String
    @State private var csvURL: URL?
    @State private var jsonURL: URL?
    @State private var exportStatus: String?
    @State private var exportError: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                queryCard
                actionsCard
                previewCard

                if let exportStatus {
                    Label(exportStatus, systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
                if let exportError {
                    Label(exportError, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            .padding(20)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("Export")
        .task {
            refreshExportFiles()
        }
        .onChange(of: engine.results.count) {
            refreshExportFiles()
        }
        .onChange(of: engine.currentQuery) {
            refreshExportFiles()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Results Export")
                .font(.title2.weight(.semibold))
            Text("Export the current semantic search results with query, UUIDs, scores, document titles, categories, and text.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private var queryCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Search results")
                .font(.headline)
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search by meaning...", text: $queryText)
                    .textFieldStyle(.plain)
                    .onSubmit {
                        runSearch()
                    }
                Button {
                    runSearch()
                } label: {
                    Image(systemName: "arrow.right.circle.fill")
                }
                .buttonStyle(.plain)
                .disabled(queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(10)
            .background(.background, in: RoundedRectangle(cornerRadius: 8))
            .overlay {
                RoundedRectangle(cornerRadius: 8)
                    .stroke(.quaternary, lineWidth: 1)
            }

            LabeledContent("Query", value: engine.currentQuery.isEmpty ? "None" : engine.currentQuery)
            LabeledContent("Result count", value: "\(engine.results.count)")
        }
        .font(.callout)
        .exportCardBackground()
    }

    private var actionsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Formats")
                .font(.headline)
            HStack(spacing: 12) {
                #if os(macOS)
                Button {
                    save(format: .csv)
                } label: {
                    Label("Save CSV", systemImage: "square.and.arrow.down")
                }
                Button {
                    save(format: .json)
                } label: {
                    Label("Save JSON", systemImage: "curlybraces.square")
                }
                #else
                if let csvURL {
                    ShareLink(item: csvURL) {
                        Label("Share CSV", systemImage: "square.and.arrow.up")
                    }
                }
                if let jsonURL {
                    ShareLink(item: jsonURL) {
                        Label("Share JSON", systemImage: "curlybraces.square")
                    }
                }
                #endif
            }
            .buttonStyle(.borderedProminent)
            .disabled(engine.results.isEmpty)

            Text(engine.results.isEmpty
                 ? "Run a search to enable export."
                 : "Prepared \(engine.results.count) rows for each format.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .exportCardBackground()
    }

    private var previewCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Preview")
                    .font(.headline)
                Spacer()
                Text(engine.currentQuery)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            if engine.results.isEmpty {
                Text("No results available.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(Array(engine.results.prefix(6))) { result in
                    HStack(alignment: .top, spacing: 10) {
                        Text(String(format: "%.3f", Double(result.distance)))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.blue)
                            .frame(width: 54, alignment: .leading)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(result.documentTitle)
                                .font(.callout.weight(.medium))
                                .lineLimit(1)
                            Text(result.text)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                    }
                    Divider()
                }
            }
        }
        .exportCardBackground()
    }

    private func runSearch() {
        Task {
            await engine.search(queryText)
            refreshExportFiles()
        }
    }

    private func refreshExportFiles() {
        guard !engine.results.isEmpty else {
            csvURL = nil
            jsonURL = nil
            return
        }
        do {
            csvURL = try engine.writeExportFile(format: .csv)
            jsonURL = try engine.writeExportFile(format: .json)
            exportError = nil
        } catch {
            exportError = error.localizedDescription
        }
    }

    #if os(macOS)
    private func save(format: ResultsExportFormat) {
        do {
            let data = try engine.exportData(format: format)
            let panel = NSSavePanel()
            panel.canCreateDirectories = true
            panel.nameFieldStringValue = engine.defaultExportFilename(format: format)
            panel.allowedContentTypes = [format == .csv ? .commaSeparatedText : .json]
            panel.begin { response in
                guard response == .OK, let url = panel.url else { return }
                do {
                    try data.write(to: url, options: .atomic)
                    DispatchQueue.main.async {
                        exportStatus = "Saved \(url.lastPathComponent)"
                        exportError = nil
                    }
                } catch {
                    DispatchQueue.main.async {
                        exportError = error.localizedDescription
                    }
                }
            }
        } catch {
            exportError = error.localizedDescription
        }
    }
    #endif
}

private extension View {
    func exportCardBackground() -> some View {
        padding(14)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }
}
