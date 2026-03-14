// ContentView.swift
// ProximaDemo
//
// Main UI: search bar, results list, index status.

import SwiftUI

struct ContentView: View {
    @State private var engine = SearchEngine()
    @State private var searchText = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Results
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

                // Status bar
                statusBar
            }
            .navigationTitle("ProximaKit Demo")
            .searchable(text: $searchText, prompt: "Search for anything...")
            .onChange(of: searchText) {
                Task {
                    await engine.search(searchText)
                }
            }
            .task {
                await engine.buildIndex()
            }
        }
        .frame(minWidth: 500, minHeight: 400)
    }

    // MARK: - Subviews

    private var welcomeView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "magnifyingglass")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Semantic Search")
                .font(.title2)
            Text("Type anything to search \(engine.indexedCount) sentences by meaning, not keywords.")
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
            Text("Indexing \(engine.indexedCount) / \(sampleSentences.count) sentences...")
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
                    Text(result.category)
                        .font(.caption)
                        .foregroundStyle(.secondary)
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
}
