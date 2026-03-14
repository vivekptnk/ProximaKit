import Foundation
import ProximaKit
import ProximaEmbeddings

struct Sentence: Codable, Sendable {
    let text: String
    let category: String
}

struct DemoResult: Identifiable {
    let id: UUID
    let text: String
    let category: String
    let distance: Float
}

@Observable
@MainActor
final class SearchEngine {
    var isIndexing = false
    var indexedCount = 0
    var lastQueryTimeMs: Double = 0
    var results: [DemoResult] = []
    var errorMessage: String?
    var efSearch: Double = 50
    var userNotes: [String] = []

    private var index: HNSWIndex?
    private var provider: NLEmbeddingProvider?

    func buildIndex() async {
        isIndexing = true
        indexedCount = 0
        results = []
        errorMessage = nil

        do {
            let nlProvider = try NLEmbeddingProvider()
            self.provider = nlProvider

            let config = HNSWConfiguration(m: 16, efConstruction: 100, efSearch: Int(efSearch))
            let hnsw = HNSWIndex(dimension: nlProvider.dimension, metric: CosineDistance(), config: config)

            for (i, sentence) in SampleData.sentences.enumerated() {
                let vector = try await nlProvider.embed(sentence.text)
                let meta = try JSONEncoder().encode(sentence)
                try await hnsw.add(vector, id: UUID(), metadata: meta)
                indexedCount = i + 1
            }

            for note in userNotes {
                let vector = try await nlProvider.embed(note)
                let meta = try JSONEncoder().encode(Sentence(text: note, category: "Your Notes"))
                try await hnsw.add(vector, id: UUID(), metadata: meta)
                indexedCount += 1
            }

            self.index = hnsw
        } catch {
            errorMessage = error.localizedDescription
        }

        isIndexing = false
    }

    func addNote(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        userNotes.append(trimmed)

        if let index, let provider {
            do {
                let vector = try await provider.embed(trimmed)
                let meta = try JSONEncoder().encode(Sentence(text: trimmed, category: "Your Notes"))
                try await index.add(vector, id: UUID(), metadata: meta)
                indexedCount += 1
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func search(_ query: String) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { results = []; return }
        guard let index, let provider else { return }

        do {
            let start = DispatchTime.now()
            let qv = try await provider.embed(trimmed)
            let sr = await index.search(query: qv, k: 10, efSearch: Int(efSearch))
            lastQueryTimeMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000

            results = sr.compactMap { r in
                guard let s = r.decodeMetadata(as: Sentence.self) else { return nil }
                return DemoResult(id: r.id, text: s.text, category: s.category, distance: r.distance)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
