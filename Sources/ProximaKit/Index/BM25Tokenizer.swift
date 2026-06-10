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

/// Default BM25 tokenizer: locale-independent Unicode word segmentation with
/// lowercasing.
///
/// Uses Foundation's `String.enumerateSubstrings(options: .byWords)` for
/// language-agnostic word boundary detection (Unicode UAX #29 segmentation,
/// with no locale-specific tailoring), then:
///
/// 1. Drops whitespace-only substrings — the segmenter emits whitespace runs
///    as "words" when the input mixes scripts (e.g., Latin + CJK), and those
///    are segmentation artifacts, not terms.
/// 2. Lowercases each word (`.lowercased()` — the locale-independent canonical
///    Unicode mapping, so `"I"` always becomes `"i"`, never Turkish `"ı"`).
/// 3. Filters out empty strings.
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
///
/// > Note: Earlier versions passed `.localized` to `enumerateSubstrings`,
/// > which made word boundaries depend on the device's current locale and
/// > violated the ``BM25Tokenizer`` determinism contract. Removing it only
/// > affects tokens produced for documents added (or queries run) **after**
/// > the change — postings already persisted in a `.pxbm` file keep whatever
/// > tokens they were built with. For mixed-language corpora indexed under a
/// > non-default locale on an older version, re-indexing is recommended so
/// > stored postings and query-time tokens agree.
public struct DefaultBM25Tokenizer: BM25Tokenizer {
    public init() {}

    public func tokenize(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }

        var tokens: [String] = []
        let range = text.startIndex..<text.endIndex
        text.enumerateSubstrings(
            in: range,
            options: .byWords
        ) { substring, _, _, _ in
            guard let word = substring, !word.isEmpty else { return }
            // `.byWords` can yield whitespace runs between scripts in
            // mixed-script text; never index those as terms.
            guard !word.allSatisfy(\.isWhitespace) else { return }
            tokens.append(word.lowercased())
        }
        return tokens
    }
}
