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
