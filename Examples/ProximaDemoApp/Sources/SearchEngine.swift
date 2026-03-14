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

/// Protocol-erased embedding wrapper so we can use either NL or CoreML.
private protocol AnyEmbedder: Sendable {
    var dimension: Int { get }
    func embed(_ text: String) async throws -> Vector
}

extension NLEmbeddingProvider: AnyEmbedder {}
extension CoreMLEmbeddingProvider: AnyEmbedder {}

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
    var embeddingSource: String = ""

    private var index: HNSWIndex?
    private var embedder: (any AnyEmbedder)?

    /// Finds the Models/ directory relative to the app bundle or working directory.
    private func findModelFiles() -> (modelURL: URL, vocabURL: URL)? {
        // Try multiple locations
        let candidates = [
            // Next to the executable
            Bundle.main.bundleURL.deletingLastPathComponent(),
            // Working directory
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
            // ProximaKit repo root (for development)
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Models"),
            // Up from Examples/
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Models"),
        ]

        for base in candidates {
            let modelURL = base.appendingPathComponent("MiniLM-L6-v2.mlmodel")
            let vocabURL = base.appendingPathComponent("vocab.txt")
            // Also check if Models/ is a subdirectory
            let altModelURL = base.appendingPathComponent("Models/MiniLM-L6-v2.mlmodel")
            let altVocabURL = base.appendingPathComponent("Models/vocab.txt")

            if FileManager.default.fileExists(atPath: modelURL.path)
                && FileManager.default.fileExists(atPath: vocabURL.path) {
                return (modelURL, vocabURL)
            }
            if FileManager.default.fileExists(atPath: altModelURL.path)
                && FileManager.default.fileExists(atPath: altVocabURL.path) {
                return (altModelURL, altVocabURL)
            }
        }
        return nil
    }

    func buildIndex() async {
        isIndexing = true
        indexedCount = 0
        results = []
        errorMessage = nil

        do {
            // Try CoreML with WordPiece tokenizer first (better quality)
            let emb: any AnyEmbedder
            if let files = findModelFiles() {
                let coreml = try CoreMLEmbeddingProvider(modelAt: files.modelURL, vocabURL: files.vocabURL)
                emb = coreml
                embeddingSource = "CoreML (MiniLM-L6-v2, \(coreml.dimension)d)"
            } else {
                let nl = try NLEmbeddingProvider()
                emb = nl
                embeddingSource = "NLEmbedding (\(nl.dimension)d) — add Models/ for better quality"
            }
            self.embedder = emb

            let config = HNSWConfiguration(m: 16, efConstruction: 100, efSearch: Int(efSearch))
            let hnsw = HNSWIndex(dimension: emb.dimension, metric: CosineDistance(), config: config)

            for (i, sentence) in SampleData.sentences.enumerated() {
                let vector = try await emb.embed(sentence.text)
                let meta = try JSONEncoder().encode(sentence)
                try await hnsw.add(vector, id: UUID(), metadata: meta)
                indexedCount = i + 1
            }

            for note in userNotes {
                let vector = try await emb.embed(note)
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

        if let index, let embedder {
            do {
                let vector = try await embedder.embed(trimmed)
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
        guard let index, let embedder else { return }

        do {
            let start = DispatchTime.now()
            let qv = try await embedder.embed(trimmed)
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
