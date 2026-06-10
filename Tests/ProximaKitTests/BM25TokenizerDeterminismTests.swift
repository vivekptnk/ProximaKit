// BM25TokenizerDeterminismTests.swift
// ProximaKit
//
// Determinism regression tests for DefaultBM25Tokenizer (audit cluster: bm25):
// - Tokenization must be locale-independent (no `.localized` segmentation),
//   otherwise persisted BM25 postings and live query tokens can disagree
//   after a device locale change — exactly what the BM25Tokenizer protocol
//   doc forbids.
// - Mixed-script input (Latin + CJK) must not emit whitespace-run tokens.
//   Foundation's `.byWords` enumeration yields whitespace runs between
//   scripts; before the fix those leaked into the index as terms like " ".
//
// Note on locale coverage: `Locale.current` cannot be swapped inside a test
// process (Foundation caches it, and LANG/LC_ALL are ignored on macOS), so
// these tests cannot literally re-run tokenization under a second locale.
// Instead they (a) compare against an inline locale-independent reference
// segmentation, and (b) pin golden outputs for mixed-language strings, so any
// reintroduction of locale-tailored segmentation shows up as a diff on
// machines whose locale tailors word breaking.

import XCTest
@testable import ProximaKit

final class BM25TokenizerDeterminismTests: XCTestCase {

    private let tokenizer = DefaultBM25Tokenizer()

    /// Mixed-language corpus used across the determinism checks.
    private let mixedCorpus: [String] = [
        "Hello, world! こんにちは",
        "Vector search ベクトル検索 is fast",
        "naïve café meets 東京 and Москва",
        "스크립트 boundaries between scripts",
        "tabs\tand\nnewlines around こんにちは too",
        "aujourd'hui l'été",
        "1,234.56 and 1.234,56",
    ]

    // MARK: - Locale independence

    /// Reference segmentation: plain `.byWords` (UAX #29, no locale
    /// tailoring), lowercased, whitespace runs dropped. The production
    /// tokenizer must match this exactly; passing `.localized` (the pre-fix
    /// behavior) diverges from it on any machine whose locale tailors word
    /// boundaries.
    func testMatchesLocaleIndependentReferenceSegmentation() {
        for text in mixedCorpus {
            var reference: [String] = []
            text.enumerateSubstrings(
                in: text.startIndex..<text.endIndex,
                options: .byWords
            ) { substring, _, _, _ in
                guard let word = substring, !word.isEmpty,
                      !word.allSatisfy(\.isWhitespace) else { return }
                reference.append(word.lowercased())
            }
            XCTAssertEqual(
                tokenizer.tokenize(text), reference,
                "Tokenizer must use locale-independent .byWords segmentation for: \(text)"
            )
        }
    }

    /// Same input → same tokens, every time, within a process. Catches any
    /// hidden dependence on ambient mutable state.
    func testRepeatedTokenizationIsStable() {
        for text in mixedCorpus {
            let first = tokenizer.tokenize(text)
            for _ in 0..<50 {
                XCTAssertEqual(tokenizer.tokenize(text), first)
            }
        }
    }

    /// Lowercasing must use the canonical (locale-independent) Unicode
    /// mapping — "I" → "i", never the Turkish dotless "ı".
    func testLowercasingIsLocaleIndependent() {
        XCTAssertEqual(tokenizer.tokenize("TITLE II"), ["title", "ii"])
        XCTAssertFalse(tokenizer.tokenize("INDEX").contains("ındex"))
    }

    // MARK: - Mixed-script whitespace artifacts (fails without the fix)

    /// `.byWords` emits whitespace runs as "words" when Latin and CJK text
    /// are mixed. Those must never become BM25 terms — before the fix this
    /// produced " ", "\t", and "\n" tokens with huge document frequency.
    func testNoWhitespaceTokensInMixedScriptText() {
        for text in mixedCorpus {
            let tokens = tokenizer.tokenize(text)
            for token in tokens {
                XCTAssertFalse(
                    token.allSatisfy(\.isWhitespace),
                    "Whitespace-only token \(token.debugDescription) leaked from: \(text)"
                )
                XCTAssertFalse(token.isEmpty)
            }
        }
    }

    /// Golden output for the doc-comment example: the mixed Latin + CJK
    /// string must tokenize without the inter-script space artifact.
    func testMixedLanguageGoldenTokens() {
        XCTAssertEqual(
            tokenizer.tokenize("Hello, world! こんにちは"),
            ["hello", "world", "こんにちは"]
        )
        XCTAssertEqual(
            tokenizer.tokenize("tabs\tand\nnewlines spaced"),
            ["tabs", "and", "newlines", "spaced"]
        )
    }

    // MARK: - Contract edges

    func testWhitespaceOnlyAndPunctuationOnlyInputs() {
        XCTAssertEqual(tokenizer.tokenize("   \t\n  "), [])
        XCTAssertEqual(tokenizer.tokenize("!?., — …"), [])
    }

    /// Duplicates and order must survive (SparseIndex derives term frequency
    /// from the raw sequence) — including around mixed-script boundaries.
    func testOrderAndDuplicatesPreservedAcrossScripts() {
        XCTAssertEqual(
            tokenizer.tokenize("cat 東京 cat 東京 cat"),
            ["cat", "東京", "cat", "東京", "cat"]
        )
    }
}
