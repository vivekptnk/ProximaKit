import Foundation
import ProximaKit
import ProximaEmbeddings

func buildNoteIndex(with embedder: NLEmbeddingProvider) async throws -> HNSWIndex {
    let index = HNSWIndex(dimension: embedder.dimension, metric: CosineDistance())

    let notes = [
        "The cat napped on the warm windowsill",
        "Golden retrievers love a game of fetch",
        "Fresh basil lifts a simple pasta sauce",
        "The hiking trail ends at a hidden waterfall",
    ]

    for note in notes {
        let vector = try await embedder.embed(note)
        let metadata = try JSONEncoder().encode(["text": note])
        try await index.add(vector, id: UUID(), metadata: metadata)
    }

    return index
}

func searchNotes(
    in index: HNSWIndex,
    for query: String,
    using embedder: NLEmbeddingProvider
) async throws -> [String] {
    let queryVector = try await embedder.embed(query)
    let results = await index.search(query: queryVector, k: 2)

    return results.compactMap { result in
        result.decodeMetadata(as: [String: String].self)?["text"]
    }
}

func runNoteSearch() async throws {
    let embedder = try NLEmbeddingProvider(language: .english)
    let index = try await buildNoteIndex(with: embedder)

    let matches = try await searchNotes(in: index, for: "a dog chasing a ball", using: embedder)
    for match in matches {
        print(match)
    }
}
