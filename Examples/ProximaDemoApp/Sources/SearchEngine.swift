import AppKit
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
    var loadedFromDisk = false
    var imageCount = 0

    private var index: HNSWIndex?
    private var embedder: (any AnyEmbedder)?

    // MARK: - File Paths

    private static var indexURL: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("ProximaDemoApp")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("demo-index.proximakit")
    }

    private static var notesURL: URL {
        indexURL.deletingLastPathComponent().appendingPathComponent("user-notes.json")
    }

    /// Finds the Models/ directory relative to the app or working directory.
    private func findModelFiles() -> (modelURL: URL, vocabURL: URL)? {
        let candidates = [
            Bundle.main.bundleURL.deletingLastPathComponent(),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Models"),
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

    // MARK: - Setup Embedder

    private func setupEmbedder() throws -> any AnyEmbedder {
        if let files = findModelFiles() {
            let coreml = try CoreMLEmbeddingProvider(modelAt: files.modelURL, vocabURL: files.vocabURL)
            embeddingSource = "CoreML (MiniLM-L6-v2, \(coreml.dimension)d)"
            return coreml
        } else {
            let nl = try NLEmbeddingProvider()
            embeddingSource = "NLEmbedding (\(nl.dimension)d) — add Models/ for better quality"
            return nl
        }
    }

    // MARK: - Build or Load Index

    func buildIndex() async {
        isIndexing = true
        indexedCount = 0
        results = []
        errorMessage = nil
        loadedFromDisk = false

        do {
            let emb = try setupEmbedder()
            self.embedder = emb

            // Try loading from disk first
            if FileManager.default.fileExists(atPath: Self.indexURL.path) {
                let loaded = try HNSWIndex.load(from: Self.indexURL)
                let count = await loaded.count
                if count > 0 {
                    self.index = loaded
                    self.indexedCount = count
                    self.loadedFromDisk = true
                    // Load saved notes
                    loadNotes()
                    isIndexing = false
                    return
                }
            }

            // Build fresh index
            let config = HNSWConfiguration(m: 16, efConstruction: 100, efSearch: Int(efSearch))
            let hnsw = HNSWIndex(dimension: emb.dimension, metric: CosineDistance(), config: config)

            for (i, sentence) in SampleData.sentences.enumerated() {
                let vector = try await emb.embed(sentence.text)
                let meta = try JSONEncoder().encode(sentence)
                try await hnsw.add(vector, id: UUID(), metadata: meta)
                indexedCount = i + 1
            }

            self.index = hnsw

            // Save to disk for next launch
            try await hnsw.save(to: Self.indexURL)
        } catch {
            errorMessage = error.localizedDescription
        }

        isIndexing = false
    }

    /// Forces a full rebuild, ignoring saved index.
    func rebuildIndex() async {
        // Delete saved files
        try? FileManager.default.removeItem(at: Self.indexURL)
        try? FileManager.default.removeItem(at: Self.notesURL)
        userNotes = []
        await buildIndex()
    }

    // MARK: - Notes

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

                // Re-save index with new note
                try await index.save(to: Self.indexURL)
                saveNotes()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func saveNotes() {
        if let data = try? JSONEncoder().encode(userNotes) {
            try? data.write(to: Self.notesURL)
        }
    }

    private func loadNotes() {
        if let data = try? Data(contentsOf: Self.notesURL),
           let notes = try? JSONDecoder().decode([String].self, from: data) {
            userNotes = notes
        }
    }

    // MARK: - Images

    func addImages(_ urls: [URL]) async {
        let vision = VisionEmbeddingProvider()

        // Image embeddings go in a separate index (different dimension than text)
        // For now, we skip images if no text index exists yet
        guard let index else { return }

        for url in urls {
            guard url.startAccessingSecurityScopedResource() else { continue }
            defer { url.stopAccessingSecurityScopedResource() }

            do {
                guard let nsImage = NSImage(contentsOf: url),
                      let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                    continue
                }

                let vector = try await vision.embed(cgImage)

                // Vision vectors have different dimension than text vectors
                // Skip if dimensions don't match the current index
                let dim = await index.count > 0 ? index.dimension : 0
                guard vector.dimension == dim || dim == 0 else {
                    // Different dimension — can't mix in same index
                    errorMessage = "Image vectors (\(vector.dimension)d) don't match text index (\(dim)d)"
                    continue
                }

                let filename = url.lastPathComponent
                let meta = try JSONEncoder().encode(Sentence(text: "Image: \(filename)", category: "Images"))
                try await index.add(vector, id: UUID(), metadata: meta)
                imageCount += 1
                indexedCount += 1
            } catch {
                errorMessage = "Failed to embed image: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - Search

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
