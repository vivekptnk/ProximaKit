// ContentView.swift
// ProximaDemo
//
// Main UI: search bar, results list, settings panel, notes input.

import SwiftUI

struct ContentView: View {
    @State private var engine = SearchEngine()
    @State private var searchText = ""
    @State private var showSettings = false
    @State private var newNote = ""

    var body: some View {
        NavigationSplitView {
            // Sidebar: settings + notes
            sidebar
                .navigationSplitViewColumnWidth(min: 220, ideal: 260)
        } detail: {
            // Main: search + results
            VStack(spacing: 0) {
                if engine.isIndexing {
                    indexingView
                } else if engine.results.isEmpty && !searchText.isEmpty {
                    emptyResultsView
                } else if engine.results.isEmpty {
                    welcomeView
                } else {
                    resultsList
                }

                Divider()
                statusBar
            }
            .navigationTitle("ProximaKit Demo")
            .searchable(text: $searchText, prompt: "Search by meaning...")
            .onChange(of: searchText) {
                Task { await engine.search(searchText) }
            }
            .task {
                await engine.buildIndex()
            }
        }
        .frame(minWidth: 700, minHeight: 500)
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List {
            // Settings section
            Section("Search Settings") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("efSearch: \(Int(engine.efSearch))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Slider(value: $engine.efSearch, in: 10...200, step: 10)
                    Text("Higher = better results, slower search")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.vertical, 4)

                Button("Rebuild Index") {
                    Task { await engine.buildIndex() }
                }
            }

            // Add notes section
            Section("Your Notes") {
                HStack {
                    TextField("Add a note...", text: $newNote)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            Task {
                                await engine.addNote(newNote)
                                newNote = ""
                            }
                        }
                    Button {
                        Task {
                            await engine.addNote(newNote)
                            newNote = ""
                        }
                    } label: {
                        Image(systemName: "plus.circle.fill")
                    }
                    .disabled(newNote.trimmingCharacters(in: .whitespaces).isEmpty)
                }

                if engine.userNotes.isEmpty {
                    Text("Add notes to include them in search results")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                } else {
                    ForEach(engine.userNotes, id: \.self) { note in
                        Text(note)
                            .font(.caption)
                            .lineLimit(2)
                    }
                }
            }

            // Stats section
            Section("Index Stats") {
                LabeledContent("Vectors", value: "\(engine.indexedCount)")
                LabeledContent("efSearch", value: "\(Int(engine.efSearch))")
                if engine.lastQueryTimeMs > 0 {
                    LabeledContent("Last query", value: String(format: "%.1f ms", engine.lastQueryTimeMs))
                }
                LabeledContent("Notes", value: "\(engine.userNotes.count)")
            }
        }
        .listStyle(.sidebar)
    }

    // MARK: - Main Content

    private var welcomeView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "magnifyingglass")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Semantic Search")
                .font(.title2)
            Text("Type anything to search \(engine.indexedCount) items by meaning.\nAdd your own notes in the sidebar.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)
            Spacer()
        }
        .padding()
    }

    private var indexingView: some View {
        VStack(spacing: 16) {
            Spacer()
            ProgressView()
                .scaleEffect(1.5)
            Text("Indexing \(engine.indexedCount) / \(sampleSentences.count + engine.userNotes.count)...")
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding()
    }

    private var emptyResultsView: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "questionmark.circle")
                .font(.system(size: 36))
                .foregroundStyle(.secondary)
            Text("No results for \"\(searchText)\"")
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding()
    }

    private var resultsList: some View {
        List(engine.results) { result in
            HStack(alignment: .top, spacing: 12) {
                // Distance badge
                Text(String(format: "%.2f", result.distance))
                    .font(.system(.caption, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(distanceColor(result.distance).opacity(0.15))
                    .foregroundStyle(distanceColor(result.distance))
                    .clipShape(RoundedRectangle(cornerRadius: 4))

                VStack(alignment: .leading, spacing: 4) {
                    Text(result.text)
                        .font(.body)
                    HStack(spacing: 4) {
                        Text(result.category)
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(categoryColor(result.category).opacity(0.1))
                            .foregroundStyle(categoryColor(result.category))
                            .clipShape(RoundedRectangle(cornerRadius: 3))
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    private var statusBar: some View {
        HStack {
            Label("\(engine.indexedCount) vectors", systemImage: "square.stack.3d.up")

            Spacer()

            if engine.lastQueryTimeMs > 0 {
                Label(String(format: "%.1f ms", engine.lastQueryTimeMs), systemImage: "clock")
            }

            Label("ef=\(Int(engine.efSearch))", systemImage: "slider.horizontal.3")

            if let error = engine.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
    }

    // MARK: - Helpers

    private func distanceColor(_ distance: Float) -> Color {
        if distance < 0.3 { return .green }
        if distance < 0.6 { return .orange }
        return .red
    }

    private func categoryColor(_ category: String) -> Color {
        switch category {
        case "Animals": return .brown
        case "Food": return .orange
        case "Technology": return .blue
        case "Nature": return .green
        case "Sports": return .red
        case "Science": return .purple
        case "Travel": return .cyan
        case "Music": return .pink
        case "Your Notes": return .indigo
        default: return .gray
        }
    }
}
