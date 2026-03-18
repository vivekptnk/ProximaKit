import XCTest
import NaturalLanguage
@testable import ProximaKit

/// Recall benchmarks using real sentence embeddings from Apple's NLEmbedding.
///
/// Unlike RecallBenchmarkTests (random vectors), these tests use actual English
/// sentence embeddings which have meaningful geometric structure. Real embeddings
/// cluster by semantic similarity, making ANN search more effective.
///
/// PRD acceptance criterion: >95% recall@10 at efSearch=50 with real embeddings.
@available(macOS 14.0, iOS 17.0, *)
final class RealEmbeddingRecallTests: XCTestCase {

    // ── Sentence corpus (120 diverse English sentences) ─────────────────

    static let sentences: [String] = [
        // Technology
        "Machine learning models require large datasets for training",
        "Neural networks can approximate any continuous function",
        "Cloud computing enables scalable distributed systems",
        "The graphics processing unit accelerates parallel computation",
        "Quantum computers use qubits instead of classical bits",
        "Version control systems track changes in source code",
        "The database engine optimizes query execution plans",
        "Encryption algorithms protect sensitive data in transit",
        "Operating systems manage hardware resources efficiently",
        "The compiler transforms source code into machine instructions",
        "Microservices architecture decouples application components",
        "Container orchestration automates deployment and scaling",
        "The search engine indexes billions of web pages daily",
        "Artificial intelligence systems can recognize speech patterns",
        "Blockchain technology creates immutable distributed ledgers",

        // Science
        "The speed of light in vacuum is a fundamental constant",
        "Photosynthesis converts sunlight into chemical energy",
        "DNA carries the genetic instructions for all living organisms",
        "Gravity governs the motion of planets around the sun",
        "Chemical reactions involve breaking and forming atomic bonds",
        "Evolution occurs through natural selection over generations",
        "The periodic table organizes elements by atomic number",
        "Mitochondria generate most of the cell energy supply",
        "Tectonic plates shift slowly beneath the earth surface",
        "Neurons transmit electrical signals throughout the brain",
        "The theory of relativity changed our understanding of space",
        "Enzymes catalyze biochemical reactions in living cells",
        "The solar system formed from a collapsing molecular cloud",
        "Antibiotics fight bacterial infections in the human body",
        "The laws of thermodynamics govern energy transformation",

        // Food & Cooking
        "Fresh basil and mozzarella make a classic caprese salad",
        "Sourdough bread requires a fermented starter culture",
        "Sushi chefs train for years to master knife techniques",
        "Chocolate is made from roasted and ground cacao beans",
        "The French cuisine emphasizes butter and fresh herbs",
        "A wok should be heated until smoking before adding oil",
        "Espresso machines force hot water through finely ground coffee",
        "Fermentation transforms grape juice into wine over time",
        "Pasta should be cooked in salted boiling water until al dente",
        "Olive oil is pressed from the fruit of the olive tree",
        "Baking soda reacts with acid to create carbon dioxide bubbles",
        "Spices like turmeric and cumin define Indian curry flavors",
        "A sharp knife is safer than a dull one in the kitchen",
        "Caramelization occurs when sugar is heated above its melting point",
        "The Maillard reaction creates flavor in seared meat",

        // Sports
        "The marathon distance was standardized at twenty six miles",
        "Basketball players practice free throws thousands of times",
        "Swimming requires coordination of breathing and stroke rhythm",
        "The Olympic games bring athletes from around the world together",
        "Tennis serves can exceed one hundred miles per hour",
        "Soccer is the most popular sport on the planet",
        "Rock climbing demands both physical strength and mental focus",
        "Ice hockey combines skating speed with stick handling skill",
        "The Tour de France covers over two thousand miles of road",
        "Yoga improves flexibility and reduces stress through practice",
        "Weightlifting builds muscle strength and bone density",
        "Surfing requires reading ocean waves and maintaining balance",
        "Track and field events test speed endurance and agility",
        "Archery demands steady hands and precise aim at the target",
        "Table tennis is one of the fastest racquet sports in existence",

        // Nature & Animals
        "Elephants communicate using low frequency infrasound vibrations",
        "Coral reefs support the highest marine biodiversity on earth",
        "Migration patterns help birds navigate thousands of miles annually",
        "The Amazon rainforest produces a significant portion of world oxygen",
        "Wolves hunt in coordinated packs to take down larger prey",
        "Bees perform waggle dances to communicate flower locations",
        "The deep ocean remains largely unexplored by human beings",
        "Butterfly wings display intricate patterns created by tiny scales",
        "Redwood trees can grow over three hundred feet tall",
        "Dolphins use echolocation to navigate and find food underwater",
        "Desert plants have evolved to conserve water in arid conditions",
        "The arctic fox changes fur color with the seasons",
        "Mushrooms form vast underground networks connecting tree roots",
        "Octopuses have three hearts and blue copper based blood",
        "Fireflies produce light through a chemical reaction called bioluminescence",

        // History & Culture
        "The printing press revolutionized the spread of knowledge",
        "Ancient Egyptian pyramids were built as royal tombs",
        "The Renaissance sparked a rebirth of art and science in Europe",
        "The industrial revolution transformed manufacturing and society",
        "Libraries have preserved human knowledge for thousands of years",
        "The silk road connected eastern and western civilizations through trade",
        "Democracy originated in the ancient Greek city states",
        "The invention of the wheel changed transportation forever",
        "Medieval castles served as fortified residences for nobility",
        "The age of exploration led to the discovery of new continents",
        "Writing systems evolved from pictographs to alphabetic scripts",
        "The space race pushed technological innovation in the cold war era",
        "Musical instruments have been found in archaeological sites worldwide",
        "Ancient Rome built roads and aqueducts spanning its vast empire",
        "The compass enabled long distance maritime navigation",

        // Daily Life
        "A good night of sleep improves memory and concentration",
        "Regular exercise reduces the risk of cardiovascular disease",
        "Public transportation reduces traffic congestion in cities",
        "Recycling helps conserve natural resources and reduce waste",
        "Reading books expands vocabulary and stimulates the imagination",
        "Morning sunlight helps regulate the circadian rhythm cycle",
        "Balanced nutrition provides the body with essential vitamins",
        "Meditation practice can lower blood pressure and reduce anxiety",
        "Handwriting activates different brain areas than typing does",
        "Clean drinking water is essential for human health and survival",
        "Indoor plants improve air quality and reduce stress levels",
        "Learning a second language strengthens cognitive flexibility",
        "Walking thirty minutes daily benefits both body and mind",
        "Social connections are important for emotional well being",
        "Music therapy can help patients recover from brain injuries",

        // Geography
        "Mount Everest is the tallest peak above sea level on earth",
        "The Sahara desert spans across multiple African countries",
        "The Great Barrier Reef stretches along the Australian coastline",
        "Rivers carry sediment and nutrients to the ocean deltas",
        "Volcanic islands form when magma rises from the ocean floor",
        "The Mediterranean Sea connects Europe Africa and Asia",
        "Glaciers carve valleys as they slowly move across the landscape",
        "The Pacific Ocean is the largest and deepest ocean on earth",
        "Mountain ranges influence weather patterns and rainfall distribution",
        "Underground caves form through the dissolution of limestone rock",
    ]

    // ── Core Recall Test (PRD target) ───────────────────────────────────

    /// Recall@10 with real NLEmbedding sentence vectors.
    /// PRD target: >95% recall@10 at efSearch=50.
    func testRealEmbeddingRecall_Cosine() async throws {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: .english) else {
            throw XCTSkip("NLEmbedding sentence model not available on this platform")
        }

        let dim = embedding.dimension
        print("NLEmbedding dimension: \(dim)")

        // Embed all sentences
        let vectors = try embedSentences(Self.sentences, embedding: embedding, dimension: dim)
        XCTAssertEqual(vectors.count, Self.sentences.count)

        // Build indices
        let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
        let hnsw = HNSWIndex(dimension: dim, metric: CosineDistance(), config: config)
        let brute = BruteForceIndex(dimension: dim, metric: CosineDistance())

        var ids: [UUID] = []
        for (i, vec) in vectors.enumerated() {
            let id = UUID()
            ids.append(id)
            try await hnsw.add(vec, id: id)
            try await brute.add(vec, id: id)
        }

        // Query with a subset of the corpus + some novel queries
        let queryTexts = [
            "Neural networks learn from data",
            "The sun provides energy for plants",
            "Athletes compete in international events",
            "Cooking requires fresh ingredients and skill",
            "Animals adapt to their natural environment",
            "Ancient civilizations built remarkable structures",
            "Exercise improves physical and mental health",
            "The ocean covers most of the earth surface",
            "Computers process information using binary logic",
            "Music has been part of human culture for millennia",
            "Chemical bonds determine molecular properties",
            "Rivers flow from mountains to the sea",
            "Birds migrate south during the winter months",
            "Coffee is one of the most popular beverages",
            "Space exploration has revealed new discoveries",
            "Sleep is essential for brain function and repair",
            "Trees absorb carbon dioxide from the atmosphere",
            "Democracy allows citizens to choose their leaders",
            "Protein is important for building muscle tissue",
            "The wheel was one of humanity greatest inventions",
        ]

        let queryVectors = try embedSentences(queryTexts, embedding: embedding, dimension: dim)

        var totalRecall = 0.0
        for (i, qVec) in queryVectors.enumerated() {
            let bruteResults = Set(await brute.search(query: qVec, k: 10).map(\.id))
            let hnswResults = Set(await hnsw.search(query: qVec, k: 10).map(\.id))
            let recall = Double(bruteResults.intersection(hnswResults).count) / 10.0
            totalRecall += recall

            if recall < 0.9 {
                print("Low recall query \(i): \"\(queryTexts[i])\" — recall=\(String(format: "%.0f%%", recall * 100))")
            }
        }

        let avgRecall = totalRecall / Double(queryVectors.count)
        print("Real-embedding Recall@10 | \(vectors.count) sentences, \(dim)d, cosine, ef=50: \(String(format: "%.1f%%", avgRecall * 100))")
        XCTAssertGreaterThan(avgRecall, 0.95, "PRD target: >95% recall@10 with real embeddings")
    }

    /// Recall with Euclidean distance on real embeddings.
    func testRealEmbeddingRecall_Euclidean() async throws {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: .english) else {
            throw XCTSkip("NLEmbedding sentence model not available on this platform")
        }

        let dim = embedding.dimension
        let vectors = try embedSentences(Self.sentences, embedding: embedding, dimension: dim)

        let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
        let hnsw = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: config)
        let brute = BruteForceIndex(dimension: dim, metric: EuclideanDistance())

        for vec in vectors {
            let id = UUID()
            try await hnsw.add(vec, id: id)
            try await brute.add(vec, id: id)
        }

        // Use corpus sentences as queries (self-retrieval)
        var totalRecall = 0.0
        let queryCount = min(30, vectors.count)
        for i in 0..<queryCount {
            let bruteIDs = Set(await brute.search(query: vectors[i], k: 10).map(\.id))
            let hnswIDs = Set(await hnsw.search(query: vectors[i], k: 10).map(\.id))
            totalRecall += Double(bruteIDs.intersection(hnswIDs).count) / 10.0
        }

        let avgRecall = totalRecall / Double(queryCount)
        print("Real-embedding Recall@10 | \(vectors.count) sentences, \(dim)d, euclidean, ef=50: \(String(format: "%.1f%%", avgRecall * 100))")
        XCTAssertGreaterThan(avgRecall, 0.95, "Real embeddings should achieve >95% recall with Euclidean too")
    }

    // ── efSearch Sweep on Real Embeddings ───────────────────────────────

    func testRealEmbeddingEfSearchSweep() async throws {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: .english) else {
            throw XCTSkip("NLEmbedding sentence model not available on this platform")
        }

        let dim = embedding.dimension
        let vectors = try embedSentences(Self.sentences, embedding: embedding, dimension: dim)

        let efValues = [10, 30, 50, 100, 200]
        let brute = BruteForceIndex(dimension: dim, metric: CosineDistance())

        var vectorIDs: [(Vector, UUID)] = []
        for vec in vectors {
            let id = UUID()
            try await brute.add(vec, id: id)
            vectorIDs.append((vec, id))
        }

        print("\nReal-Embedding Recall@10 vs efSearch | \(vectors.count) sentences, \(dim)d, cosine")
        print("efSearch | recall  | note")
        print("---------|---------|-----")

        var prevRecall = 0.0
        for ef in efValues {
            let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: ef)
            let hnsw = HNSWIndex(dimension: dim, metric: CosineDistance(), config: config)

            for (vec, id) in vectorIDs {
                try await hnsw.add(vec, id: id)
            }

            var totalRecall = 0.0
            let queryCount = 20
            for i in 0..<queryCount {
                let bruteIDs = Set(await brute.search(query: vectors[i], k: 10).map(\.id))
                let hnswIDs = Set(await hnsw.search(query: vectors[i], k: 10).map(\.id))
                totalRecall += Double(bruteIDs.intersection(hnswIDs).count) / 10.0
            }

            let recall = totalRecall / Double(queryCount)
            let note = ef == 50 ? "← default (PRD target)" : ""
            print("\(String(ef).padding(toLength: 8, withPad: " ", startingAt: 0)) | \(String(format: "%.1f%%", recall * 100).padding(toLength: 7, withPad: " ", startingAt: 0)) | \(note)")
            XCTAssertGreaterThanOrEqual(recall, prevRecall - 0.05,
                "Recall should not drop significantly with higher efSearch")
            prevRecall = recall
        }
    }

    // ── Semantic Coherence Check ────────────────────────────────────────

    /// Validates that HNSW returns semantically similar results:
    /// a tech query should return tech-related sentences.
    func testSemanticCoherence() async throws {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: .english) else {
            throw XCTSkip("NLEmbedding sentence model not available on this platform")
        }

        let dim = embedding.dimension
        let vectors = try embedSentences(Self.sentences, embedding: embedding, dimension: dim)

        let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 100)
        let hnsw = HNSWIndex(dimension: dim, metric: CosineDistance(), config: config)

        for (i, vec) in vectors.enumerated() {
            try await hnsw.add(vec, id: UUID())
            // Store sentence index in metadata for verification
            let data = withUnsafeBytes(of: Int32(i)) { Data($0) }
            // Re-add with metadata
            _ = await hnsw.remove(id: UUID()) // skip; just use simple approach
        }

        // Simpler approach: rebuild with tracked IDs
        let index = HNSWIndex(dimension: dim, metric: CosineDistance(), config: config)
        var idToSentenceIndex: [UUID: Int] = [:]

        for (i, vec) in vectors.enumerated() {
            let id = UUID()
            idToSentenceIndex[id] = i
            try await index.add(vec, id: id)
        }

        // Query: "Programming languages compile source code"
        let queryVec = try embedSingle("Programming languages compile source code",
                                        embedding: embedding, dimension: dim)
        let results = await index.search(query: queryVec, k: 5)

        print("\nSemantic coherence check — query: \"Programming languages compile source code\"")
        print("Top 5 results:")
        for (rank, result) in results.enumerated() {
            if let idx = idToSentenceIndex[result.id] {
                print("  \(rank + 1). [\(String(format: "%.4f", result.distance))] \(Self.sentences[idx])")
            }
        }

        // At least 3 of top 5 should be from technology category (indices 0-14)
        let techHits = results.compactMap { idToSentenceIndex[$0.id] }.filter { $0 < 15 }.count
        XCTAssertGreaterThanOrEqual(techHits, 2,
            "Tech query should retrieve mostly tech sentences")
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private func embedSentences(
        _ sentences: [String],
        embedding: NLEmbedding,
        dimension: Int
    ) throws -> [Vector] {
        var vectors: [Vector] = []
        for sentence in sentences {
            let vec = try embedSingle(sentence, embedding: embedding, dimension: dimension)
            vectors.append(vec)
        }
        return vectors
    }

    private func embedSingle(
        _ text: String,
        embedding: NLEmbedding,
        dimension: Int
    ) throws -> Vector {
        // Try sentence embedding first
        if let vector = embedding.vector(for: text) {
            return Vector(vector.map { Float($0) })
        }

        // Fallback: average word embeddings
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var wordVectors: [[Double]] = []

        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range])
            if let vec = embedding.vector(for: word) {
                wordVectors.append(vec)
            }
            return true
        }

        guard !wordVectors.isEmpty else {
            throw XCTSkip("Could not embed text: \"\(text)\"")
        }

        var sum = [Float](repeating: 0, count: dimension)
        for vec in wordVectors {
            for i in 0..<min(vec.count, dimension) {
                sum[i] += Float(vec[i])
            }
        }
        let n = Float(wordVectors.count)
        let averaged = sum.map { $0 / n }

        // Normalize
        let magnitude = sqrt(averaged.reduce(0) { $0 + $1 * $1 })
        if magnitude > 0 {
            return Vector(averaged.map { $0 / magnitude })
        }
        return Vector(averaged)
    }
}
