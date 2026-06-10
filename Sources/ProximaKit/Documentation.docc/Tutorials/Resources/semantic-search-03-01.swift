import Foundation
import ProximaKit

func saveIndex(_ index: HNSWIndex, to url: URL) async throws {
    try await index.save(to: url)
}
