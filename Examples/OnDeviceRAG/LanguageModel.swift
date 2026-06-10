// LanguageModel.swift
// OnDeviceRAG
//
// The answer-generation seam. RAG retrieval (ProximaKit) is model-agnostic:
// anything that can turn "question + retrieved passages" into prose plugs in
// here — Apple's FoundationModels, an MLX model, a llama.cpp wrapper.

import Foundation

/// Anything that can turn a question plus retrieved context into an answer.
///
/// `context` is the retrieved note passages **in retrieval order** — cite
/// them by their 1-based position, like `[1]` or `[2]`, so answers point
/// back at the passages the user just saw printed.
protocol LanguageModel: Sendable {
    /// Human-readable backend description, printed at startup so transcripts
    /// are honest about which model produced the answers.
    var name: String { get }

    /// Produces an answer to `question` grounded in `context`.
    func reply(question: String, context: [String]) async throws -> String
}

// ── TemplateLLM: deterministic, dependency-free stand-in ──────────────

/// A deterministic stand-in for a real language model.
///
/// **This is NOT a generative model.** It is purely extractive: it picks the
/// retrieved passages that share the most content words with the question and
/// stitches them into a templated answer with citation markers. It exists so
/// this example runs on every supported machine with zero downloads and zero
/// dependencies. Swap in a real model via the ``LanguageModel`` protocol —
/// see ``FoundationModelsLLM`` below for the Apple-provided option.
struct TemplateLLM: LanguageModel {
    let name = "TemplateLLM (extractive stand-in — selects from retrieved notes, does not generate text)"

    func reply(question: String, context: [String]) async throws -> String {
        guard !context.isEmpty else {
            return "I couldn't find anything relevant in your notes."
        }

        // Score each passage by how many of the question's content words it shares.
        let questionWords = Self.contentWords(in: question)
        let scored = context.indices.map { i in
            (index: i, score: questionWords.intersection(Self.contentWords(in: context[i])).count)
        }

        // Best overlap first; retrieval order breaks ties (fully deterministic).
        let ranked = scored.sorted {
            $0.score != $1.score ? $0.score > $1.score : $0.index < $1.index
        }
        let picks = ranked.prefix(2).filter { $0.score > 0 }

        // No word overlap at all? Fall back to the closest retrieved note.
        guard let best = picks.first else {
            return "Your closest note says: \"\(context[0])\" [1]"
        }

        var answer = "From your notes: \(context[best.index]) [\(best.index + 1)]"
        if picks.count > 1 {
            let second = picks[1]
            answer += " Related: \(context[second.index]) [\(second.index + 1)]"
        }
        return answer
    }

    /// Words that carry no topical signal for overlap scoring.
    private static let stopwords: Set<String> = [
        "the", "and", "for", "are", "was", "you", "your", "how", "what",
        "when", "where", "why", "who", "which", "should", "could", "would",
        "can", "does", "did", "with", "that", "this", "from", "have", "has",
        "had", "not", "but", "any", "all", "out", "about", "into", "over",
        "long", "much", "many", "need", "get", "use",
    ]

    private static func contentWords(in text: String) -> Set<String> {
        Set(
            text.lowercased()
                .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
                .map(String.init)
                .filter { $0.count >= 3 && !stopwords.contains($0) }
        )
    }
}

// ── FoundationModelsLLM: Apple's on-device LLM, when the OS has one ───

#if canImport(FoundationModels)
import FoundationModels

/// Apple's on-device foundation model (macOS 26+ / iOS 26+ / visionOS 26+
/// with Apple Intelligence enabled). Fully private: prompts never leave the
/// device, matching ProximaKit's on-device retrieval.
@available(macOS 26.0, iOS 26.0, visionOS 26.0, *)
struct FoundationModelsLLM: LanguageModel {
    let name = "FoundationModels (Apple's on-device LLM — generative)"

    /// True only when the OS ships the model AND it is ready to use
    /// (Apple Intelligence enabled, model assets downloaded).
    static var isAvailable: Bool {
        if case .available = SystemLanguageModel.default.availability { return true }
        return false
    }

    func reply(question: String, context: [String]) async throws -> String {
        let notes = context.enumerated()
            .map { "[\($0.offset + 1)] \($0.element)" }
            .joined(separator: "\n")
        let session = LanguageModelSession(instructions: """
            You answer questions using ONLY the user's numbered notes. \
            Cite every fact with its note number in brackets, like [1] or [2]. \
            If the notes do not answer the question, say so plainly.
            """)
        let response = try await session.respond(to: "Notes:\n\(notes)\n\nQuestion: \(question)")
        return response.content
    }
}
#endif

/// Picks the language model. Pass `preference: "template"` (the `-llm template`
/// launch arg) to force the deterministic stand-in — handy for scripted runs.
/// Otherwise auto-selects the best backend this machine offers, falling back
/// to ``TemplateLLM`` so the example always runs.
func makeLanguageModel(preference: String? = nil) -> any LanguageModel {
    if preference == "template" { return TemplateLLM() }
    #if canImport(FoundationModels)
    if #available(macOS 26.0, iOS 26.0, visionOS 26.0, *), FoundationModelsLLM.isAvailable {
        return FoundationModelsLLM()
    }
    #endif
    return TemplateLLM()
}
