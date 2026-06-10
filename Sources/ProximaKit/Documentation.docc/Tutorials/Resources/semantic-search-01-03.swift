import Foundation
import ProximaKit

func buildFruitIndex() async throws -> HNSWIndex {
    let index = HNSWIndex(dimension: 3, metric: CosineDistance())

    let items: [(name: String, vector: Vector)] = [
        ("apple", Vector([0.9, 0.1, 0.0])),
        ("banana", Vector([0.8, 0.3, 0.1])),
        ("bicycle", Vector([0.1, 0.2, 0.9])),
    ]

    for item in items {
        let metadata = try JSONEncoder().encode(["name": item.name])
        try await index.add(item.vector, id: UUID(), metadata: metadata)
    }

    return index
}
