// VersionConsistencyTests.swift
// Regression tests for the audit finding where `ProximaKit.version` shipped
// as "1.0.0" through the v1.1.0 and v1.4.0 tags because nothing validated it.
//
// These tests mirror .github/ci-scripts/check_version.sh (used by ci.yml and release.yml)
// so the drift is also caught by a plain `swift test` run, without CI.

import XCTest
@testable import ProximaKit

final class VersionConsistencyTests: XCTestCase {

    // ── Helpers ───────────────────────────────────────────────────────

    /// Repo root, derived from this file's location (Tests/ProximaKitTests/).
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // ProximaKitTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // repo root
    }

    /// The newest released `## [x.y.z]` section heading in CHANGELOG.md,
    /// skipping the `## [Unreleased]` placeholder.
    private func newestChangelogVersion() throws -> String {
        let changelog = try String(
            contentsOf: repoRoot.appendingPathComponent("CHANGELOG.md"),
            encoding: .utf8
        )
        let pattern = #"^## \[(\d+\.\d+\.\d+)\]"#
        let regex = try NSRegularExpression(pattern: pattern, options: [.anchorsMatchLines])
        let range = NSRange(changelog.startIndex..., in: changelog)
        guard let match = regex.firstMatch(in: changelog, range: range),
              let versionRange = Range(match.range(at: 1), in: changelog) else {
            XCTFail("CHANGELOG.md has no released '## [x.y.z]' section")
            return ""
        }
        return String(changelog[versionRange])
    }

    // ── Tests ─────────────────────────────────────────────────────────

    /// `ProximaKit.version` must be a semantic version, not a placeholder.
    func testVersionConstantIsSemanticVersion() {
        let parts = ProximaKit.version.split(separator: ".")
        XCTAssertEqual(parts.count, 3, "version '\(ProximaKit.version)' is not x.y.z")
        for part in parts {
            XCTAssertNotNil(Int(part), "version component '\(part)' is not numeric")
        }
    }

    /// The runtime constant must match the newest released CHANGELOG section.
    /// This is the in-repo guard against the constant drifting behind tags
    /// again (it reported "1.0.0" for two releases).
    func testVersionConstantMatchesChangelog() throws {
        let changelogVersion = try newestChangelogVersion()
        XCTAssertEqual(
            ProximaKit.version, changelogVersion,
            "ProximaKit.version (\(ProximaKit.version)) != newest CHANGELOG entry "
            + "(\(changelogVersion)) — update Sources/ProximaKit/ProximaKit.swift"
        )
    }
}
