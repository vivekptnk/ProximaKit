// SearchEngine.swift
// ProximaDemo
//
// The brain of the demo: manages the HNSW index, embeddings, and search.

import Foundation
import ProximaKit
import ProximaEmbeddings

/// Metadata stored alongside each vector in the index.
struct SentenceMetadata: Codable, Sendable {
    let text: String
    let category: String
}

/// A search result with the matched sentence text.
struct DemoResult: Identifiable, Sendable {
    let id: UUID
    let text: String
    let distance: Float
    let category: String
}

/// Manages the search index and embedding pipeline.
@Observable
@MainActor
final class SearchEngine {
    var isIndexing = false
    var indexedCount = 0
    var lastQueryTimeMs: Double = 0
    var results: [DemoResult] = []
    var errorMessage: String?

    private var index: HNSWIndex?
    private var provider: NLEmbeddingProvider?

    /// Builds the HNSW index from sample sentences.
    func buildIndex() async {
        isIndexing = true
        indexedCount = 0
        errorMessage = nil
        results = []

        do {
            let nlProvider = try NLEmbeddingProvider()
            self.provider = nlProvider

            let config = HNSWConfiguration(m: 16, efConstruction: 100, efSearch: 50)
            let hnsw = HNSWIndex(dimension: nlProvider.dimension, metric: CosineDistance(), config: config)

            // Categorize sentences by their position in the array
            let categories = [
                "Animals", "Animals", "Animals", "Animals", "Animals", "Animals", "Animals",
                "Food", "Food", "Food", "Food", "Food", "Food", "Food",
                "Technology", "Technology", "Technology", "Technology", "Technology", "Technology", "Technology",
                "Nature", "Nature", "Nature", "Nature", "Nature", "Nature", "Nature",
                "Sports", "Sports", "Sports", "Sports", "Sports", "Sports", "Sports",
                "Science", "Science", "Science", "Science", "Science", "Science", "Science",
                "Travel", "Travel", "Travel", "Travel", "Travel",
                "Music", "Music", "Music", "Music", "Music",
            ]

            for (i, sentence) in sampleSentences.enumerated() {
                let vector = try await nlProvider.embed(sentence)
                let category = i < categories.count ? categories[i] : "Other"
                let meta = SentenceMetadata(text: sentence, category: category)
                let encoded = try JSONEncoder().encode(meta)
                try await hnsw.add(vector, id: UUID(), metadata: encoded)
                indexedCount = i + 1
            }

            self.index = hnsw
        } catch {
            errorMessage = "Indexing failed: \(error.localizedDescription)"
        }

        isIndexing = false
    }

    /// Searches the index for sentences similar to the query.
    func search(_ query: String) async {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            results = []
            return
        }

        guard let index = index, let provider = provider else {
            errorMessage = "Index not built yet"
            return
        }

        do {
            let start = DispatchTime.now()
            let queryVector = try await provider.embed(query)
            let searchResults = await index.search(query: queryVector, k: 10)
            let end = DispatchTime.now()

            let nanos = end.uptimeNanoseconds - start.uptimeNanoseconds
            lastQueryTimeMs = Double(nanos) / 1_000_000

            results = searchResults.compactMap { result in
                guard let meta = result.decodeMetadata(as: SentenceMetadata.self) else {
                    return nil
                }
                return DemoResult(
                    id: result.id,
                    text: meta.text,
                    distance: result.distance,
                    category: meta.category
                )
            }

            errorMessage = nil
        } catch {
            errorMessage = "Search failed: \(error.localizedDescription)"
        }
    }
}
