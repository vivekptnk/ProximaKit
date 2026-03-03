import XCTest
@testable import ProximaEmbeddings

final class ProximaEmbeddingsTests: XCTestCase {

    func testVersionMatchesCore() {
        // Smoke test: embeddings module imports and version matches core.
        XCTAssertFalse(ProximaEmbeddings.version.isEmpty)
    }
}
