// swift-tools-version: 5.9
// Standalone benchmark harness. NOT a product of ProximaKit — it exists only
// to generate cross-library comparison numbers (ProximaKit vs FAISS vs ScaNN).
//
// The core ProximaKit package (../Package.swift) is intentionally unaware of
// this target so that `swift build` / `swift test` on the library stay
// dependency-free.

import PackageDescription

let package = Package(
    name: "ProximaBench",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        .package(path: ".."),
    ],
    targets: [
        .executableTarget(
            name: "ProximaBench",
            dependencies: [
                .product(name: "ProximaKit", package: "ProximaKit"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),
    ]
)
