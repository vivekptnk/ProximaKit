import Foundation
import ProximaKit

func saveIndex(_ index: HNSWIndex, to url: URL) async throws {
    try await index.save(to: url)
}

func loadIndex(from url: URL) throws -> HNSWIndex {
    try HNSWIndex.load(from: url)
}
