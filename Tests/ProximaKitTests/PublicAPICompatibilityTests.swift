import XCTest
import ProximaKit

final class PublicAPICompatibilityTests: XCTestCase {

    @available(*, deprecated)
    func testLegacyAndCanonicalIndexNamesRemainTypeCompatible() {
        let canonicalLayout: IndexSaveLayout = .resident
        let legacyLayout: PQHWSaveLayout = canonicalLayout
        let canonicalFromLegacy: IndexSaveLayout = legacyLayout
        let pagedLegacy: PQHWSaveLayout = .pagedV3

        let canonicalResidency: IndexResidency = .resident
        let legacyHNSW: HNSWOpenMode = canonicalResidency
        let legacyPQHW: PQHWOpenMode = .paged
        let residencyFromLegacy: IndexResidency = legacyPQHW
        let legacyFromLegacy: PQHWOpenMode = legacyHNSW

        func describe(_ layout: IndexSaveLayout) -> String {
            switch layout {
            case .resident:
                return "resident"
            case .pagedV3:
                return "pagedV3"
            }
        }

        func describe(_ residency: IndexResidency) -> String {
            switch residency {
            case .resident:
                return "resident"
            case .paged:
                return "paged"
            }
        }

        XCTAssertEqual(describe(canonicalFromLegacy), "resident")
        XCTAssertEqual(describe(pagedLegacy), "pagedV3")
        XCTAssertEqual(describe(legacyHNSW), "resident")
        XCTAssertEqual(describe(residencyFromLegacy), "paged")
        XCTAssertEqual(describe(legacyFromLegacy), "resident")
    }
}
