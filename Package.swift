// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.
// 5.9 gives us macros, parameter packs, and the latest concurrency features.

import PackageDescription

let package = Package(
    name: "ProximaKit",

    // Minimum OS versions — Accelerate's modern vDSP API requires these.
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1),
    ],

    // What consumers can import. Two libraries:
    // - ProximaKit: core vector math + indices (zero dependencies beyond Apple frameworks)
    // - ProximaEmbeddings: turns content into vectors using CoreML/NaturalLanguage/Vision
    products: [
        .library(name: "ProximaKit", targets: ["ProximaKit"]),
        .library(name: "ProximaEmbeddings", targets: ["ProximaEmbeddings"]),
        // Runnable example: on-device RAG over your notes (docs/RAG-TUTORIAL.md).
        .executable(name: "OnDeviceRAG", targets: ["OnDeviceRAG"]),
    ],

    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.0.0"),
    ],

    targets: [
        // ── Core Library ──────────────────────────────────────────────
        // Imports: Foundation, Accelerate ONLY.
        .target(
            name: "ProximaKit",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),

        // ── Embeddings Library ────────────────────────────────────────
        // Depends on ProximaKit (for the Vector type).
        // Imports: CoreML, NaturalLanguage, Vision.
        .target(
            name: "ProximaEmbeddings",
            dependencies: ["ProximaKit"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),

        // ── Demo App ──────────────────────────────────────────────────
        .executableTarget(
            name: "ProximaDemo",
            dependencies: ["ProximaKit", "ProximaEmbeddings"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),

        // ── On-Device RAG Example ─────────────────────────────────────
        // Lives under Examples/ so the code sits next to its tutorial
        // (docs/RAG-TUTORIAL.md). Run with: swift run OnDeviceRAG
        .executableTarget(
            name: "OnDeviceRAG",
            dependencies: ["ProximaKit", "ProximaEmbeddings"],
            path: "Examples/OnDeviceRAG",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),

        // ── Tests ─────────────────────────────────────────────────────
        .testTarget(
            name: "ProximaKitTests",
            dependencies: ["ProximaKit"]
        ),
        .testTarget(
            name: "ProximaEmbeddingsTests",
            dependencies: ["ProximaEmbeddings"]
        ),
    ]
)
