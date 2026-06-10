import Foundation
import ProximaKit

func saveIndex(_ index: HNSWIndex, to url: URL) async throws {
    try await index.save(to: url)
}

func loadIndex(from url: URL) throws -> HNSWIndex {
    try HNSWIndex.load(from: url)
}

func roundTrip(_ index: HNSWIndex) async throws {
    let url = URL.documentsDirectory.appending(path: "notes.proxima")

    try await saveIndex(index, to: url)
    let reloaded = try loadIndex(from: url)

    let savedCount = await index.count
    let loadedCount = await reloaded.count
    print("Saved \(savedCount) vectors, reloaded \(loadedCount).")
}
