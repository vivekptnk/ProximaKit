// ProximaDemoApp.swift
// ProximaDemo
//
// Interactive terminal demo for ProximaKit semantic search.
// Run with: swift run ProximaDemo

import Foundation
import ProximaKit
import ProximaEmbeddings

@main
struct ProximaDemoApp {
    static func main() async throws {
        print("""

        ╔══════════════════════════════════════════════╗
        ║          ProximaKit Semantic Search           ║
        ║     Search by meaning, not keywords.          ║
        ╚══════════════════════════════════════════════╝

        """)

        // Step 1: Set up embedding provider
        print("Loading NLEmbedding model...")
        let embedder = try NLEmbeddingProvider(language: .english)
        print("  Dimension: \(embedder.dimension)")

        // Step 2: Create index
        let config = HNSWConfiguration(m: 16, efConstruction: 100, efSearch: 50)
        let index = HNSWIndex(dimension: embedder.dimension, metric: CosineDistance(), config: config)

        // Step 3: Index sample data
        print("Indexing \(sampleSentences.count) sentences...\n")

        let categories = [
            "Animals", "Animals", "Animals", "Animals", "Animals", "Animals", "Animals",
            "Food", "Food", "Food", "Food", "Food", "Food", "Food",
            "Technology", "Technology", "Technology", "Technology", "Technology", "Technology", "Technology",
            "Nature", "Nature", "Nature", "Nature", "Nature", "Nature", "Nature",
            "Sports", "Sports", "Sports", "Sports", "Sports", "Sports", "Sports",
            "Science", "Science", "Science", "Science", "Science", "Science", "Science",
            "Travel", "Travel", "Travel", "Travel", "Travel",
            "Music", "Music", "Music", "Music", "Music",
        ]

        let buildStart = DispatchTime.now()
        for (i, sentence) in sampleSentences.enumerated() {
            let vector = try await embedder.embed(sentence)
            let category = i < categories.count ? categories[i] : "Other"
            let meta = try JSONEncoder().encode(["text": sentence, "category": category])
            try await index.add(vector, id: UUID(), metadata: meta)
        }
        let buildTime = Double(DispatchTime.now().uptimeNanoseconds - buildStart.uptimeNanoseconds) / 1_000_000

        let count = await index.count
        print("  Indexed \(count) sentences in \(String(format: "%.0f", buildTime))ms")
        print("""

        ─────────────────────────────────────────────
        Type a search query and press Enter.
        Type "quit" to exit.
        ─────────────────────────────────────────────

        """)

        // Step 4: Interactive search loop
        while true {
            print("🔍 Search: ", terminator: "")
            guard let input = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !input.isEmpty else {
                continue
            }

            if input.lowercased() == "quit" || input.lowercased() == "exit" {
                print("\nBye!")
                break
            }

            let searchStart = DispatchTime.now()
            let queryVector = try await embedder.embed(input)
            let results = await index.search(query: queryVector, k: 5)
            let searchTime = Double(DispatchTime.now().uptimeNanoseconds - searchStart.uptimeNanoseconds) / 1_000_000

            print()
            if results.isEmpty {
                print("  No results found.")
            } else {
                for (i, result) in results.enumerated() {
                    if let data = result.metadata,
                       let info = try? JSONDecoder().decode([String: String].self, from: data) {
                        let text = info["text"] ?? "?"
                        let category = info["category"] ?? "?"
                        let bar = distanceBar(result.distance)
                        print("  \(i + 1). \(bar) \(String(format: "%.3f", result.distance))  \(text)")
                        print("     \(category)")
                    }
                }
            }
            print("  ⏱  \(String(format: "%.1f", searchTime))ms\n")
        }
    }

    static func distanceBar(_ distance: Float) -> String {
        if distance < 0.3 { return "🟢" }
        if distance < 0.6 { return "🟡" }
        return "🔴"
    }
}
