// WordPieceTokenizer.swift
// ProximaEmbeddings
//
// BERT WordPiece tokenizer for CoreML sentence-transformer models.
// Loads a vocab.txt file and tokenizes text into integer token IDs.

import Foundation

/// A WordPiece tokenizer compatible with BERT-based models.
///
/// This tokenizer splits text into subword tokens using the WordPiece algorithm:
/// 1. Lowercase and strip accents
/// 2. Split on whitespace and punctuation
/// 3. For each word, greedily match the longest vocab prefix
/// 4. Unmatched subwords get the `##` prefix and are tried again
/// 5. Unknown tokens map to `[UNK]`
///
/// ```swift
/// let tokenizer = try WordPieceTokenizer(vocabURL: vocabFileURL)
/// let ids = tokenizer.tokenize("Hello world", maxLength: 128)
/// // [101, 7592, 2088, 102, 0, 0, ...] (padded to maxLength)
/// ```
public struct WordPieceTokenizer: Sendable {
    private let vocab: [String: Int32]
    private let unkID: Int32
    private let clsID: Int32
    private let sepID: Int32
    private let padID: Int32

    /// Loads a WordPiece vocabulary from a vocab.txt file.
    ///
    /// Each line in the file is one token. The line number (0-indexed) is the token ID.
    ///
    /// - Parameter url: Path to the vocab.txt file.
    /// - Throws: If the file can't be read.
    public init(vocabURL url: URL) throws {
        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)

        var vocab: [String: Int32] = [:]
        vocab.reserveCapacity(lines.count)
        for (i, line) in lines.enumerated() {
            let token = line.trimmingCharacters(in: .whitespaces)
            if !token.isEmpty {
                vocab[token] = Int32(i)
            }
        }

        self.vocab = vocab
        self.padID = vocab["[PAD]"] ?? 0
        self.unkID = vocab["[UNK]"] ?? 100
        self.clsID = vocab["[CLS]"] ?? 101
        self.sepID = vocab["[SEP]"] ?? 102
    }

    /// Tokenizes text into token IDs with padding.
    ///
    /// Returns `(inputIDs, attentionMask)` both of length `maxLength`.
    /// - `inputIDs`: `[CLS] token1 token2 ... [SEP] [PAD] [PAD] ...`
    /// - `attentionMask`: `[1, 1, 1, ..., 1, 0, 0, ...]` (1 for real tokens, 0 for padding)
    public func tokenize(_ text: String, maxLength: Int) -> (inputIDs: [Int32], attentionMask: [Int32]) {
        let tokens = wordPieceTokenize(text)

        // Truncate to fit [CLS] ... [SEP] within maxLength
        let maxTokens = maxLength - 2
        let truncated = Array(tokens.prefix(maxTokens))

        // Build input IDs: [CLS] + tokens + [SEP] + padding
        var inputIDs: [Int32] = [clsID]
        inputIDs.append(contentsOf: truncated)
        inputIDs.append(sepID)

        let realLength = inputIDs.count
        while inputIDs.count < maxLength {
            inputIDs.append(padID)
        }

        // Attention mask: 1 for real tokens, 0 for padding
        var mask = [Int32](repeating: 0, count: maxLength)
        for i in 0..<realLength {
            mask[i] = 1
        }

        return (inputIDs, mask)
    }

    // MARK: - WordPiece Algorithm

    private func wordPieceTokenize(_ text: String) -> [Int32] {
        let words = basicTokenize(text)
        var tokens: [Int32] = []

        for word in words {
            let subTokens = wordPieceSplit(word)
            tokens.append(contentsOf: subTokens)
        }

        return tokens
    }

    /// Basic pre-tokenization: lowercase, split on whitespace and punctuation.
    private func basicTokenize(_ text: String) -> [String] {
        let lowered = text.lowercased()
        var words: [String] = []
        var current = ""

        for char in lowered {
            if char.isWhitespace {
                if !current.isEmpty { words.append(current); current = "" }
            } else if char.isPunctuation || char.isSymbol {
                if !current.isEmpty { words.append(current); current = "" }
                words.append(String(char))
            } else {
                current.append(char)
            }
        }
        if !current.isEmpty { words.append(current) }

        return words
    }

    /// WordPiece: greedily split a word into the longest matching vocab subwords.
    private func wordPieceSplit(_ word: String) -> [Int32] {
        // If the whole word is in vocab, return it directly
        if let id = vocab[word] {
            return [id]
        }

        var tokens: [Int32] = []
        var start = word.startIndex
        let end = word.endIndex

        while start < end {
            var found = false
            var subEnd = end

            // Try longest match first, shrink until we find one
            while subEnd > start {
                var substr = String(word[start..<subEnd])
                if start > word.startIndex {
                    substr = "##" + substr  // WordPiece continuation prefix
                }

                if let id = vocab[substr] {
                    tokens.append(id)
                    start = subEnd
                    found = true
                    break
                }

                subEnd = word.index(before: subEnd)
            }

            if !found {
                // Character not in vocab at all — use [UNK]
                tokens.append(unkID)
                start = word.index(after: start)
            }
        }

        return tokens
    }
}
