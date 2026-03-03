import XCTest
@testable import ProximaKit

final class ProximaKitTests: XCTestCase {

    func testVersionExists() {
        // Smoke test: the module imports and the version string is set.
        XCTAssertFalse(ProximaKit.version.isEmpty)
    }
}
