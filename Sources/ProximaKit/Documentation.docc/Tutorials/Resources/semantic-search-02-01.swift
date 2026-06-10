import Foundation
import ProximaKit
import ProximaEmbeddings

func buildNoteIndex(with embedder: NLEmbeddingProvider) async throws -> HNSWIndex {
    let index = HNSWIndex(dimension: embedder.dimension, metric: CosineDistance())

    return index
}
