import SwiftUI
import UniformTypeIdentifiers
import ProximaKit

struct MainView: View {
    @Bindable var engine: SearchEngine
    @State private var searchText = ""
    @State private var newNote = ""
    @State private var showImagePicker = false

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 220, ideal: 260)
        } detail: {
            VStack(spacing: 0) {
                // Search bar
                HStack(spacing: 10) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("Search by meaning...", text: $searchText)
                        .textFieldStyle(.plain)
                        .font(.title3)
                    if !searchText.isEmpty {
                        Button { searchText = ""; engine.results = [] } label: {
                            Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                        }.buttonStyle(.plain)
                    }
                }
                .padding(12)

                Divider()

                // Content
                if engine.isIndexing {
                    Spacer()
                    ProgressView("Indexing \(engine.indexedCount) / \(SampleData.sentences.count)...")
                    Spacer()
                } else if engine.results.isEmpty && searchText.isEmpty {
                    welcomeView
                } else if engine.results.isEmpty {
                    Spacer()
                    Text("No results for \"\(searchText)\"").foregroundStyle(.secondary)
                    Spacer()
                } else {
                    resultsList
                }

                Divider()
                statusBar
            }
        }
        .onChange(of: searchText) {
            Task { await engine.search(searchText) }
        }
        .fileImporter(
            isPresented: $showImagePicker,
            allowedContentTypes: [.image],
            allowsMultipleSelection: true
        ) { result in
            guard let urls = try? result.get() else { return }
            Task { await engine.addImages(urls) }
        }
    }

    // MARK: - Welcome

    private var welcomeView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "magnifyingglass")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Semantic Search").font(.title2)
            Text("Search \(engine.indexedCount) items by meaning.\nAdd your own notes in the sidebar.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)
            Spacer()
        }
    }

    // MARK: - Results

    private var resultsList: some View {
        List(engine.results) { result in
            HStack(alignment: .top, spacing: 12) {
                Text(String(format: "%.2f", result.distance))
                    .font(.system(.caption, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(distanceColor(result.distance).opacity(0.15))
                    .foregroundStyle(distanceColor(result.distance))
                    .clipShape(RoundedRectangle(cornerRadius: 4))

                VStack(alignment: .leading, spacing: 4) {
                    Text(result.text)
                    Text(result.category)
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(categoryColor(result.category).opacity(0.1))
                        .foregroundStyle(categoryColor(result.category))
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                }
            }
            .padding(.vertical, 2)
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List {
            Section("Search Settings") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("efSearch: \(Int(engine.efSearch))")
                        .font(.caption).foregroundStyle(.secondary)
                    Slider(value: $engine.efSearch, in: 10...200, step: 10)
                        .onChange(of: engine.efSearch) {
                            // Re-run current search with new efSearch
                            if !searchText.isEmpty {
                                Task { await engine.search(searchText) }
                            }
                        }
                    Text("Higher = better results, slower").font(.caption2).foregroundStyle(.tertiary)
                }
                .padding(.vertical, 4)

                Button("Rebuild Index") {
                    Task { await engine.rebuildIndex() }
                }

                if engine.loadedFromDisk {
                    Text("Loaded from disk (instant)")
                        .font(.caption2)
                        .foregroundStyle(.green)
                }
            }

            Section("Your Notes") {
                HStack {
                    TextField("Add a note...", text: $newNote)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            Task { await engine.addNote(newNote); newNote = "" }
                        }
                    Button {
                        Task { await engine.addNote(newNote); newNote = "" }
                    } label: {
                        Image(systemName: "plus.circle.fill")
                    }
                    .disabled(newNote.trimmingCharacters(in: .whitespaces).isEmpty)
                }

                if engine.userNotes.isEmpty {
                    Text("Notes you add appear in search results")
                        .font(.caption).foregroundStyle(.tertiary)
                } else {
                    ForEach(engine.userNotes, id: \.self) { note in
                        Text(note).font(.caption).lineLimit(2)
                    }
                }
            }

            Section("Images") {
                Button("Add Images...") { showImagePicker = true }
                if engine.imageCount > 0 {
                    Text("\(engine.imageCount) images indexed")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Add photos to search visually")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            }

            Section("Index Stats") {
                LabeledContent("Vectors", value: "\(engine.indexedCount)")
                LabeledContent("efSearch", value: "\(Int(engine.efSearch))")
                if engine.lastQueryTimeMs > 0 {
                    LabeledContent("Last query", value: String(format: "%.1f ms", engine.lastQueryTimeMs))
                }
                LabeledContent("Notes", value: "\(engine.userNotes.count)")
                LabeledContent("Images", value: "\(engine.imageCount)")
                if !engine.embeddingSource.isEmpty {
                    Text(engine.embeddingSource)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .listStyle(.sidebar)
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack {
            Label("\(engine.indexedCount) vectors", systemImage: "square.stack.3d.up")
            Spacer()
            if engine.lastQueryTimeMs > 0 {
                Label(String(format: "%.1f ms", engine.lastQueryTimeMs), systemImage: "clock")
            }
            Label("ef=\(Int(engine.efSearch))", systemImage: "slider.horizontal.3")
        }
        .font(.caption).foregroundStyle(.secondary)
        .padding(.horizontal, 12).padding(.vertical, 6)
    }

    // MARK: - Colors

    // NLEmbedding distances are typically 0.5-0.8 range.
    // Adjusted thresholds for Apple's built-in model quality.
    private func distanceColor(_ d: Float) -> Color {
        d < 0.55 ? .green : d < 0.68 ? .orange : .red
    }

    private func categoryColor(_ c: String) -> Color {
        switch c {
        case "Animals": .brown; case "Food": .orange; case "Technology": .blue
        case "Nature": .green; case "Sports": .red; case "Science": .purple
        case "Travel": .cyan; case "Music": .pink; case "Arts": .mint
        case "Your Notes": .indigo; default: .gray
        }
    }
}
