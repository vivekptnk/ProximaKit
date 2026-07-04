import SwiftUI

struct CorpusImportView: View {
    @Bindable var engine: SearchEngine
    let openImporter: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                actionCard
                progressCard
                importedFilesCard
                failuresCard
            }
            .padding(20)
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("Import")
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Custom Corpus")
                .font(.title2.weight(.semibold))
            Text("Import `.txt` and `.md` files, chunk them locally, embed them with the active provider, and append them to the demo index.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private var actionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Documents")
                        .font(.headline)
                    Text(engine.embeddingSource.isEmpty ? "Embedding provider pending" : engine.embeddingSource)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    openImporter()
                } label: {
                    Label("Choose", systemImage: "doc.badge.plus")
                }
                .buttonStyle(.borderedProminent)
                .disabled(engine.isIndexing || engine.isImporting)
            }

            Divider()

            LabeledContent("Indexed vectors", value: "\(engine.indexedCount)")
            LabeledContent("Imported files", value: "\(engine.importedFiles.count)")
            LabeledContent("Imported chunks", value: "\(engine.importedChunkCount)")
        }
        .font(.callout)
        .importCardBackground()
    }

    @ViewBuilder
    private var progressCard: some View {
        if engine.isImporting || engine.importTotal > 0 || engine.importErrorMessage != nil {
            VStack(alignment: .leading, spacing: 10) {
                Text("Import run")
                    .font(.headline)
                if engine.importTotal > 0 {
                    ProgressView(value: Double(engine.importProgress), total: Double(engine.importTotal)) {
                        Text("\(engine.importProgress) / \(engine.importTotal) chunks")
                            .font(.callout)
                    }
                } else if engine.isImporting {
                    ProgressView(engine.importStatus.isEmpty ? "Reading documents..." : engine.importStatus)
                }

                if !engine.importStatus.isEmpty {
                    Text(engine.importStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let error = engine.importErrorMessage {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            .importCardBackground()
        }
    }

    private var importedFilesCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Imported documents")
                    .font(.headline)
                Spacer()
                Text("\(engine.importedFiles.count)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            if engine.importedFiles.isEmpty {
                Text("No custom documents indexed in this session.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(Array(engine.importedFiles.suffix(8).reversed())) { file in
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Image(systemName: "doc.text")
                            .foregroundStyle(.blue)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(file.title)
                                .font(.callout.weight(.medium))
                                .lineLimit(1)
                            Text(file.sourcePath)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        Spacer()
                        Text("\(file.chunkCount)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Divider()
                }
            }
        }
        .importCardBackground()
    }

    @ViewBuilder
    private var failuresCard: some View {
        if !engine.importFailures.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text("Skipped files")
                    .font(.headline)
                ForEach(engine.importFailures) { failure in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(failure.sourcePath)
                            .font(.callout.weight(.medium))
                        Text(failure.message)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Divider()
                }
            }
            .importCardBackground()
        }
    }
}

private extension View {
    func importCardBackground() -> some View {
        padding(14)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }
}
