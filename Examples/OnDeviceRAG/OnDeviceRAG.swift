// OnDeviceRAG.swift
// OnDeviceRAG
//
// Private retrieval-augmented answers over your notes — entirely on-device.
// Tutorial: docs/RAG-TUTORIAL.md
//
// Interactive:  swift run OnDeviceRAG
// Scripted:     swift run OnDeviceRAG -question "How long should I steep cold brew?"
// Force the deterministic stand-in model:  swift run OnDeviceRAG -llm template

import Foundation
import ProximaKit
import ProximaEmbeddings

@main
struct OnDeviceRAGApp {
    static func main() async throws {
        print("""

        ╔══════════════════════════════════════════════╗
        ║         OnDeviceRAG  ·  ProximaKit            ║
        ║   Private answers over your notes. No cloud.  ║
        ╚══════════════════════════════════════════════╝

        """)

        // 1. EMBED — turn text into vectors that capture meaning.
        print("Loading NLEmbedding model...")
        let embedder = try NLEmbeddingProvider(language: .english)
        print("  Dimension: \(embedder.dimension)")

        // 2. INDEX — every note goes into an HNSW graph for fast similarity search.
        let index = HNSWIndex(dimension: embedder.dimension, metric: CosineDistance())
        print("Indexing \(sampleNotes.count) notes...")
        for note in sampleNotes {
            let vector = try await embedder.embed(note)
            let metadata = try JSONEncoder().encode(["text": note])
            try await index.add(vector, id: UUID(), metadata: metadata)
        }
        let count = await index.count
        print("  Indexed \(count) notes.\n")

        // Pick the best language model this machine offers (see LanguageModel.swift).
        let llm = makeLanguageModel(preference: launchValue(for: "-llm"))
        print("Answering with: \(llm.name)\n")

        // Scripted mode: -question "..." (repeatable) answers each and exits.
        let scripted = questionArguments()
        if !scripted.isEmpty {
            for question in scripted {
                print("❓ Question: \(question)")
                try await answer(question, embedder: embedder, index: index, llm: llm)
            }
            return
        }

        // Interactive mode.
        print("Ask a question about your notes. Type \"quit\" to exit.\n")
        while true {
            print("❓ Question: ", terminator: "")
            // nil means EOF (piped input exhausted or terminal closed) —
            // exit cleanly rather than spinning on the prompt forever.
            guard let line = readLine() else {
                print("\nBye!")
                break
            }
            let question = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if question.isEmpty { continue }
            if question.lowercased() == "quit" || question.lowercased() == "exit" {
                print("\nBye!")
                break
            }
            try await answer(question, embedder: embedder, index: index, llm: llm)
        }
    }

    /// The RAG core: RETRIEVE the most relevant notes, AUGMENT the question
    /// with them, and ANSWER with citations back to the retrieved passages.
    static func answer(
        _ question: String,
        embedder: NLEmbeddingProvider,
        index: HNSWIndex,
        llm: any LanguageModel
    ) async throws {
        // 3. RETRIEVE — embed the question, find the k nearest notes.
        let searchStart = DispatchTime.now()
        let queryVector = try await embedder.embed(question)
        let results = await index.search(query: queryVector, k: 3)
        let searchMs = Double(DispatchTime.now().uptimeNanoseconds - searchStart.uptimeNanoseconds) / 1_000_000

        var passages: [String] = []
        print("\n📚 Retrieved notes (lower distance = more relevant, \(String(format: "%.1f", searchMs)) ms):")
        for (i, result) in results.enumerated() {
            guard let data = result.metadata,
                  let info = try? JSONDecoder().decode([String: String].self, from: data),
                  let text = info["text"] else { continue }
            passages.append(text)
            print("  [\(i + 1)] \(String(format: "%.3f", result.distance))  \(text)")
        }

        // 4 + 5. AUGMENT & ANSWER — the model sees ONLY the retrieved notes.
        let reply = try await llm.reply(question: question, context: passages)
        print("\n💬 Answer: \(reply)\n")
    }

    /// Returns the value following a `-flag` launch argument, if present.
    static func launchValue(for flag: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    /// Collects every `-question "..."` pair from the launch arguments.
    static func questionArguments() -> [String] {
        var questions: [String] = []
        let args = CommandLine.arguments
        var i = 1
        while i < args.count {
            if args[i] == "-question", i + 1 < args.count {
                questions.append(args[i + 1])
                i += 2
            } else {
                i += 1
            }
        }
        return questions
    }
}
