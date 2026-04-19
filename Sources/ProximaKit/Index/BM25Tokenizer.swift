// BM25Tokenizer.swift
// ProximaKit
//
// Tokenization protocol + default implementation for ``SparseIndex``.
// The core library stays Foundation-only, so the default tokenizer avoids
// NaturalLanguage / NSLinguisticTagger — it relies on Foundation's built-in
// Unicode word segmentation (`String.enumerateSubstrings(options: .byWords)`).
//
// Consumers who want language-aware tokenization (e.g., Lumen with NLTokenizer)
// can implement their own ``BM25Tokenizer`` without pulling NaturalLanguage
// into the ProximaKit core.

import Foundation

/// A type that splits a string into the terms indexed and queried by ``SparseIndex``.
///
/// Conformers must be deterministic: the same input string must always yield
/// the same token sequence, otherwise persisted BM25 statistics and live queries
/// will disagree.
///
/// Conformers must also be `Sendable` so an index actor can hold one safely.
public protocol BM25Tokenizer: Sendable {
    /// Tokenizes a string into an ordered list of terms.
    ///
    /// The returned array preserves order and duplicates — ``SparseIndex``
    /// uses the raw sequence to compute term frequency.
    func tokenize(_ text: String) -> [String]
}

/// Default BM25 tokenizer: Unicode-aware word segmentation with lowercasing.
///
/// Uses Foundation's `String.enumerateSubstrings(options: .byWords)` for
/// language-agnostic word boundary detection, then:
///
/// 1. Lowercases each word (`.lowercased()` — Unicode-correct).
/// 2. Filters out empty strings.
///
/// This is deliberately minimal. It does **not** stem, remove stopwords, or
/// apply any language-specific rules — that's the job of a richer conformer
/// (e.g., one backed by `NLTokenizer`).
///
/// ```swift
/// let tokenizer = DefaultBM25Tokenizer()
/// tokenizer.tokenize("Hello, world! こんにちは")
/// // ["hello", "world", "こんにちは"]
/// ```
public struct DefaultBM25Tokenizer: BM25Tokenizer {
    public init() {}

    public func tokenize(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }

        var tokens: [String] = []
        let range = text.startIndex..<text.endIndex
        text.enumerateSubstrings(
            in: range,
            options: [.byWords, .localized]
        ) { substring, _, _, _ in
            guard let word = substring, !word.isEmpty else { return }
            tokens.append(word.lowercased())
        }
        return tokens
    }
}
