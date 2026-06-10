import Foundation
import ProximaKit

func buildFruitIndex() async throws -> HNSWIndex {
    let index = HNSWIndex(dimension: 3, metric: CosineDistance())

    return index
}
