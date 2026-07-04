import CoreGraphics
import Foundation
import ImageIO
import ProximaKit
import ProximaEmbeddings

struct Sentence: Codable, Sendable {
    let text: String
    let category: String
    let documentTitle: String?
    let documentID: String?
    let chunkIndex: Int?
    let sourcePath: String?

    init(
        text: String,
        category: String,
        documentTitle: String? = nil,
        documentID: String? = nil,
        chunkIndex: Int? = nil,
        sourcePath: String? = nil
    ) {
        self.text = text
        self.category = category
        self.documentTitle = documentTitle
        self.documentID = documentID
        self.chunkIndex = chunkIndex
        self.sourcePath = sourcePath
    }
}

struct DemoResult: Identifiable {
    let id: UUID
    let text: String
    let category: String
    let distance: Float
    let documentTitle: String
    let documentID: String?
    let chunkIndex: Int?
}

struct ImportedCorpusFile: Identifiable, Sendable {
    let id: String
    let title: String
    let sourcePath: String
    let chunkCount: Int
}

struct CorpusImportFailure: Identifiable, Sendable {
    let id = UUID()
    let sourcePath: String
    let message: String
}

enum ResultsExportFormat: String, CaseIterable, Identifiable {
    case csv
    case json

    var id: String { rawValue }
    var title: String {
        switch self {
        case .csv: "CSV"
        case .json: "JSON"
        }
    }

    var fileExtension: String { rawValue }
}

struct IndexInspectorNode: Identifiable, Hashable, Sendable {
    let id: UUID
    let internalIndex: Int
    let title: String
    let subtitle: String
    let text: String
    let category: String
    let level: Int
    let degree: Int
}

struct IndexInspectorEdge: Identifiable, Hashable, Sendable {
    let source: UUID
    let target: UUID

    var id: String {
        source.uuidString < target.uuidString
            ? "\(source.uuidString)-\(target.uuidString)"
            : "\(target.uuidString)-\(source.uuidString)"
    }
}

struct IndexLayerBucket: Identifiable, Hashable, Sendable {
    let level: Int
    let count: Int
    var id: Int { level }
}

struct IndexInspectorGraph: Sendable {
    let nodes: [IndexInspectorNode]
    let edges: [IndexInspectorEdge]
    let totalNodeCount: Int
    let sampledNodeCount: Int
    let layerBuckets: [IndexLayerBucket]
    let averageLayer0Degree: Double
    let maxLayer: Int

    static let empty = IndexInspectorGraph(
        nodes: [],
        edges: [],
        totalNodeCount: 0,
        sampledNodeCount: 0,
        layerBuckets: [],
        averageLayer0Degree: 0,
        maxLayer: 0)
}

/// Protocol-erased embedding wrapper so we can use either NL or CoreML.
private protocol AnyEmbedder: Sendable {
    var dimension: Int { get }
    func embed(_ text: String) async throws -> Vector
}

extension NLEmbeddingProvider: AnyEmbedder {}
extension CoreMLEmbeddingProvider: AnyEmbedder {}

@Observable
@MainActor
final class SearchEngine {
    var isIndexing = false
    var indexedCount = 0
    var lastQueryTimeMs: Double = 0
    var results: [DemoResult] = []
    var currentQuery = ""
    var errorMessage: String?
    var efSearch: Double = 50
    var userNotes: [String] = []
    var embeddingSource: String = ""
    var loadedFromDisk = false
    var imageCount = 0
    var isImporting = false
    var importProgress = 0
    var importTotal = 0
    var importStatus = ""
    var importErrorMessage: String?
    var importFailures: [CorpusImportFailure] = []
    var importedFiles: [ImportedCorpusFile] = []
    var importedChunkCount = 0

    private var index: HNSWIndex?
    private var embedder: (any AnyEmbedder)?
    private let importTargetCharacters = 900
    private let importOverlapCharacters = 160

    // MARK: - File Paths

    private static var indexURL: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("ProximaDemoApp")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("demo-index.proximakit")
    }

    private static var notesURL: URL {
        indexURL.deletingLastPathComponent().appendingPathComponent("user-notes.json")
    }

    /// Finds the Models/ directory relative to the app or working directory.
    private func findModelFiles() -> (modelURL: URL, vocabURL: URL)? {
        let candidates = [
            Bundle.main.bundleURL.deletingLastPathComponent(),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Models"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Models"),
        ]

        for base in candidates {
            let modelURL = base.appendingPathComponent("MiniLM-L6-v2.mlmodel")
            let vocabURL = base.appendingPathComponent("vocab.txt")
            let altModelURL = base.appendingPathComponent("Models/MiniLM-L6-v2.mlmodel")
            let altVocabURL = base.appendingPathComponent("Models/vocab.txt")

            if FileManager.default.fileExists(atPath: modelURL.path)
                && FileManager.default.fileExists(atPath: vocabURL.path) {
                return (modelURL, vocabURL)
            }
            if FileManager.default.fileExists(atPath: altModelURL.path)
                && FileManager.default.fileExists(atPath: altVocabURL.path) {
                return (altModelURL, altVocabURL)
            }
        }
        return nil
    }

    // MARK: - Setup Embedder

    private func setupEmbedder() throws -> any AnyEmbedder {
        if let files = findModelFiles() {
            let coreml = try CoreMLEmbeddingProvider(modelAt: files.modelURL, vocabURL: files.vocabURL)
            embeddingSource = "CoreML (MiniLM-L6-v2, \(coreml.dimension)d)"
            return coreml
        } else {
            let nl = try NLEmbeddingProvider()
            embeddingSource = "NLEmbedding (\(nl.dimension)d) — add Models/ for better quality"
            return nl
        }
    }

    // MARK: - Build or Load Index

    func buildIndex() async {
        isIndexing = true
        indexedCount = 0
        results = []
        errorMessage = nil
        loadedFromDisk = false

        do {
            let emb = try setupEmbedder()
            self.embedder = emb

            // Try loading from disk first — but only if its dimension matches
            // the CURRENT embedder. NLEmbedding picks sentence (512d) or
            // word-averaging (300d) mode per provider instance depending on
            // which language assets the OS has, and that can change between
            // launches while assets download. A stale-dimension index would
            // make every search silently return [] (dimension-mismatch guard).
            if FileManager.default.fileExists(atPath: Self.indexURL.path) {
                let loaded = try HNSWIndex.load(from: Self.indexURL)
                let count = await loaded.count
                if count > 0 && loaded.dimension == emb.dimension {
                    self.index = loaded
                    self.indexedCount = count
                    self.loadedFromDisk = true
                    // Load saved notes
                    loadNotes()
                    isIndexing = false
                    return
                }
            }

            // Build fresh index
            let config = HNSWConfiguration(m: 16, efConstruction: 100, efSearch: Int(efSearch))
            let hnsw = HNSWIndex(dimension: emb.dimension, metric: CosineDistance(), config: config)

            for (i, sentence) in SampleData.sentences.enumerated() {
                let vector = try await emb.embed(sentence.text)
                let enriched = Sentence(
                    text: sentence.text,
                    category: sentence.category,
                    documentTitle: "Sample Corpus",
                    documentID: "sample-\(sentence.category.lowercased())",
                    chunkIndex: i,
                    sourcePath: "SampleData.swift")
                let meta = try JSONEncoder().encode(enriched)
                try await hnsw.add(vector, id: UUID(), metadata: meta)
                indexedCount = i + 1
            }

            self.index = hnsw

            // Save to disk for next launch
            try await hnsw.save(to: Self.indexURL)
        } catch {
            errorMessage = error.localizedDescription
        }

        isIndexing = false
    }

    /// Forces a full rebuild, ignoring saved index.
    func rebuildIndex() async {
        // Delete saved files
        try? FileManager.default.removeItem(at: Self.indexURL)
        try? FileManager.default.removeItem(at: Self.notesURL)
        userNotes = []
        await buildIndex()
    }

    // MARK: - Notes

    func addNote(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        userNotes.append(trimmed)

        if let index, let embedder {
            do {
                let vector = try await embedder.embed(trimmed)
                let meta = try JSONEncoder().encode(Sentence(
                    text: trimmed,
                    category: "Your Notes",
                    documentTitle: "Your Notes",
                    documentID: "notes",
                    chunkIndex: userNotes.count - 1,
                    sourcePath: "User Notes"))
                try await index.add(vector, id: UUID(), metadata: meta)
                indexedCount += 1

                // Re-save index with new note
                try await index.save(to: Self.indexURL)
                saveNotes()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func saveNotes() {
        if let data = try? JSONEncoder().encode(userNotes) {
            try? data.write(to: Self.notesURL)
        }
    }

    private func loadNotes() {
        if let data = try? Data(contentsOf: Self.notesURL),
           let notes = try? JSONDecoder().decode([String].self, from: data) {
            userNotes = notes
        }
    }

    // MARK: - Images

    func addImages(_ urls: [URL]) async {
        let vision = VisionEmbeddingProvider()

        // Image embeddings go in a separate index (different dimension than text)
        // For now, we skip images if no text index exists yet
        guard let index else { return }

        for url in urls {
            guard url.startAccessingSecurityScopedResource() else { continue }
            defer { url.stopAccessingSecurityScopedResource() }

            do {
                // ImageIO instead of NSImage/UIImage: identical code path on
                // macOS, iOS, iPadOS, and visionOS.
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                    continue
                }

                let vector = try await vision.embed(cgImage)

                // Vision vectors have different dimension than text vectors
                // Skip if dimensions don't match the current index
                let dim = await index.count > 0 ? index.dimension : 0
                guard vector.dimension == dim || dim == 0 else {
                    // Different dimension — can't mix in same index
                    errorMessage = "Image vectors (\(vector.dimension)d) don't match text index (\(dim)d)"
                    continue
                }

                let filename = url.lastPathComponent
                let meta = try JSONEncoder().encode(Sentence(
                    text: "Image: \(filename)",
                    category: "Images",
                    documentTitle: filename,
                    documentID: "image-\(filename)",
                    sourcePath: filename))
                try await index.add(vector, id: UUID(), metadata: meta)
                imageCount += 1
                indexedCount += 1
            } catch {
                errorMessage = "Failed to embed image: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - Search

    func search(_ query: String) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            currentQuery = ""
            results = []
            return
        }
        guard let index, let embedder else { return }

        do {
            currentQuery = trimmed
            let start = DispatchTime.now()
            let qv = try await embedder.embed(trimmed)
            // Surface a dimension mismatch instead of the index's documented
            // silent-[] behavior — it means the index needs a rebuild.
            guard qv.dimension == index.dimension else {
                errorMessage = "Query dimension (\(qv.dimension)d) doesn't match the index (\(index.dimension)d) — tap Rebuild Index."
                results = []
                return
            }
            let sr = await index.search(query: qv, k: 10, efSearch: Int(efSearch))
            lastQueryTimeMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000

            results = sr.compactMap { r in
                guard let s = r.decodeMetadata(as: Sentence.self) else { return nil }
                return DemoResult(
                    id: r.id,
                    text: s.text,
                    category: s.category,
                    distance: r.distance,
                    documentTitle: displayTitle(for: s),
                    documentID: s.documentID,
                    chunkIndex: s.chunkIndex)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Index Inspector

    func makeInspectorGraph(sampleLimit: Int = 150) async throws -> IndexInspectorGraph {
        guard let index else { return .empty }
        let snapshot = try await index.persistenceSnapshot()
        let total = snapshot.nodeToUUID.count
        guard total > 0 else { return .empty }

        let layer0 = snapshot.layers.first ?? []
        let degrees = (0..<total).map { node in
            node < layer0.count ? layer0[node].count : 0
        }
        let sample = deterministicSampleIndices(total: total, limit: sampleLimit)
        let included = Set(sample)

        let nodes = sample.map { node -> IndexInspectorNode in
            let id = snapshot.nodeToUUID[node]
            let metadata = node < snapshot.metadata.count ? snapshot.metadata[node] : nil
            let sentence = decodeSentence(metadata)
            let title = sentence.map(displayTitle(for:)) ?? "Vector \(node + 1)"
            let category = sentence?.category ?? "Unknown"
            let text = sentence?.text ?? "No metadata text stored for this vector."
            let chunk = sentence?.chunkIndex.map { " · chunk \($0 + 1)" } ?? ""
            let subtitle = "\(category)\(chunk) · \(id.uuidString.prefix(8))"
            return IndexInspectorNode(
                id: id,
                internalIndex: node,
                title: title,
                subtitle: subtitle,
                text: text,
                category: category,
                level: snapshot.nodeLevels[safe: node] ?? 0,
                degree: degrees[node])
        }

        var seenEdges = Set<String>()
        var edges: [IndexInspectorEdge] = []
        for from in sample where from < layer0.count {
            for to in layer0[from] where included.contains(to) && from != to {
                let a = min(from, to)
                let b = max(from, to)
                let key = "\(a)-\(b)"
                guard seenEdges.insert(key).inserted else { continue }
                edges.append(IndexInspectorEdge(
                    source: snapshot.nodeToUUID[a],
                    target: snapshot.nodeToUUID[b]))
            }
        }

        let maxLayer = snapshot.nodeLevels.max() ?? 0
        let layerBuckets = (0...maxLayer).map { level in
            IndexLayerBucket(level: level, count: snapshot.nodeLevels.filter { $0 == level }.count)
        }
        let averageDegree = degrees.isEmpty
            ? 0
            : Double(degrees.reduce(0, +)) / Double(degrees.count)

        return IndexInspectorGraph(
            nodes: nodes,
            edges: edges,
            totalNodeCount: total,
            sampledNodeCount: nodes.count,
            layerBuckets: layerBuckets,
            averageLayer0Degree: averageDegree,
            maxLayer: maxLayer)
    }

    private func deterministicSampleIndices(total: Int, limit: Int) -> [Int] {
        guard limit > 0 else { return [] }
        guard total > limit else { return Array(0..<total) }

        var rng = DemoRNG(seed: 0x1A5E_D00D)
        var indices = Array(0..<total)
        for i in stride(from: total - 1, through: 1, by: -1) {
            let j = Int(rng.next() % UInt64(i + 1))
            indices.swapAt(i, j)
        }
        return Array(indices.prefix(limit)).sorted()
    }

    // MARK: - Custom Corpus Import

    func importCorpus(from urls: [URL]) async {
        guard !isImporting else { return }
        guard let index, let embedder else {
            importErrorMessage = "Search index is still building. Try again once indexing finishes."
            return
        }

        isImporting = true
        importProgress = 0
        importTotal = 0
        importStatus = "Reading documents..."
        importErrorMessage = nil
        importFailures = []
        defer { isImporting = false }

        var documents: [PendingImportDocument] = []
        var failures: [CorpusImportFailure] = []
        for url in urls {
            let loaded = loadImportDocuments(from: url)
            documents.append(contentsOf: loaded.documents)
            failures.append(contentsOf: loaded.failures)
        }

        documents.sort { lhs, rhs in
            lhs.sourcePath.localizedStandardCompare(rhs.sourcePath) == .orderedAscending
        }

        guard !documents.isEmpty else {
            importFailures = failures
            importErrorMessage = failures.isEmpty
                ? "No .txt or .md files were found."
                : "No readable .txt or .md files were found."
            importStatus = "Import did not add any chunks."
            return
        }

        let chunked = documents.map { doc in
            ChunkedImportDocument(
                document: doc,
                chunks: chunkText(doc.text, targetCharacters: importTargetCharacters, overlapCharacters: importOverlapCharacters))
        }
        importTotal = chunked.reduce(0) { $0 + $1.chunks.count }
        guard importTotal > 0 else {
            importFailures = failures
            importErrorMessage = "The selected documents did not contain indexable text."
            importStatus = "Import did not add any chunks."
            return
        }

        let startingImportedFiles = importedFiles.count
        do {
            for entry in chunked {
                guard !entry.chunks.isEmpty else { continue }
                let documentID = stableDocumentID(for: entry.document)
                var addedForDocument = 0
                for (chunkIndex, chunk) in entry.chunks.enumerated() {
                    importStatus = "Embedding \(entry.document.title) chunk \(chunkIndex + 1) of \(entry.chunks.count)"
                    let vector = try await embedder.embed(chunk)
                    guard vector.dimension == index.dimension else {
                        throw IndexError.dimensionMismatch(expected: index.dimension, got: vector.dimension)
                    }
                    let metadata = try JSONEncoder().encode(Sentence(
                        text: chunk,
                        category: "Imported",
                        documentTitle: entry.document.title,
                        documentID: documentID,
                        chunkIndex: chunkIndex,
                        sourcePath: entry.document.sourcePath))
                    try await index.add(vector, id: UUID(), metadata: metadata)
                    importProgress += 1
                    importedChunkCount += 1
                    addedForDocument += 1
                }
                importedFiles.append(ImportedCorpusFile(
                    id: documentID,
                    title: entry.document.title,
                    sourcePath: entry.document.sourcePath,
                    chunkCount: addedForDocument))
            }

            try await index.save(to: Self.indexURL)
            indexedCount = await index.liveCount
            importFailures = failures
            let importedNow = importedFiles.count - startingImportedFiles
            importStatus = "Imported \(importProgress) chunks from \(importedNow) files."
            if !currentQuery.isEmpty {
                await search(currentQuery)
            }
        } catch {
            importErrorMessage = error.localizedDescription
            importFailures = failures
            importStatus = "Import stopped after \(importProgress) of \(importTotal) chunks."
        }
    }

    func importDemoCorpus() async {
        do {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("ProximaDemoImport", isDirectory: true)
            try? FileManager.default.removeItem(at: dir)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

            let fieldNotes = """
            # Field Notes

            Solar charging stations kept the coastal sensor array online through a cloudy week.

            The maintenance crew replaced two corroded housings and tagged the repaired units for follow-up inspection.
            """
            let researchMemo = """
            # Research Memo

            A small transformer model summarized the inspection notes before the team indexed them for semantic retrieval.

            The best search queries mixed operational details with the equipment names the field crew actually used.
            """
            try fieldNotes.write(
                to: dir.appendingPathComponent("field-notes.md"),
                atomically: true,
                encoding: .utf8)
            try researchMemo.write(
                to: dir.appendingPathComponent("research-memo.txt"),
                atomically: true,
                encoding: .utf8)
            await importCorpus(from: [dir])
        } catch {
            importErrorMessage = error.localizedDescription
        }
    }

    private struct PendingImportDocument {
        let title: String
        let sourcePath: String
        let text: String
    }

    private struct ChunkedImportDocument {
        let document: PendingImportDocument
        let chunks: [String]
    }

    private func loadImportDocuments(from url: URL) -> (documents: [PendingImportDocument], failures: [CorpusImportFailure]) {
        let didStartScope = url.startAccessingSecurityScopedResource()
        defer {
            if didStartScope {
                url.stopAccessingSecurityScopedResource()
            }
        }

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return ([], [CorpusImportFailure(sourcePath: url.lastPathComponent, message: "File does not exist.")])
        }

        if isDirectory.boolValue {
            return loadImportDocuments(inDirectory: url)
        }

        guard isSupportedTextDocument(url) else {
            return ([], [CorpusImportFailure(sourcePath: url.lastPathComponent, message: "Only .txt and .md files are supported.")])
        }

        do {
            return ([try readImportDocument(url)], [])
        } catch {
            return ([], [CorpusImportFailure(sourcePath: url.lastPathComponent, message: error.localizedDescription)])
        }
    }

    private func loadImportDocuments(inDirectory url: URL) -> (documents: [PendingImportDocument], failures: [CorpusImportFailure]) {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return ([], [CorpusImportFailure(sourcePath: url.lastPathComponent, message: "Folder could not be opened.")])
        }

        var documents: [PendingImportDocument] = []
        var failures: [CorpusImportFailure] = []
        for case let fileURL as URL in enumerator {
            guard isSupportedTextDocument(fileURL) else { continue }
            do {
                documents.append(try readImportDocument(fileURL))
            } catch {
                failures.append(CorpusImportFailure(
                    sourcePath: fileURL.lastPathComponent,
                    message: error.localizedDescription))
            }
        }
        return (documents, failures)
    }

    private func isSupportedTextDocument(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return ext == "txt" || ext == "md"
    }

    private func readImportDocument(_ url: URL) throws -> PendingImportDocument {
        let data = try Data(contentsOf: url)
        let text = String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .unicode)
            ?? String(data: data, encoding: .ascii)
        guard let text else {
            throw ImportReadError(message: "Document is not valid UTF-8, UTF-16, or ASCII text.")
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ImportReadError(message: "Document is empty.")
        }
        return PendingImportDocument(
            title: url.deletingPathExtension().lastPathComponent,
            sourcePath: url.lastPathComponent,
            text: trimmed)
    }

    private func chunkText(_ text: String, targetCharacters: Int, overlapCharacters: Int) -> [String] {
        let paragraphs = text
            .components(separatedBy: CharacterSet.newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !paragraphs.isEmpty else { return [] }

        var chunks: [String] = []
        var current = ""
        for paragraph in paragraphs {
            if paragraph.count > targetCharacters {
                if !current.isEmpty {
                    chunks.append(current)
                    current = ""
                }
                chunks.append(contentsOf: splitLongText(paragraph, targetCharacters: targetCharacters, overlapCharacters: overlapCharacters))
                continue
            }

            let candidate = current.isEmpty ? paragraph : "\(current)\n\n\(paragraph)"
            if candidate.count <= targetCharacters {
                current = candidate
            } else {
                chunks.append(current)
                current = paragraph
            }
        }
        if !current.isEmpty {
            chunks.append(current)
        }
        return chunks
    }

    private func splitLongText(_ text: String, targetCharacters: Int, overlapCharacters: Int) -> [String] {
        guard text.count > targetCharacters else { return [text] }
        var chunks: [String] = []
        var start = text.startIndex
        while start < text.endIndex {
            let end = text.index(start, offsetBy: targetCharacters, limitedBy: text.endIndex) ?? text.endIndex
            let chunk = String(text[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !chunk.isEmpty {
                chunks.append(chunk)
            }
            guard end < text.endIndex else { break }
            let nextOffset = max(1, targetCharacters - overlapCharacters)
            start = text.index(start, offsetBy: nextOffset, limitedBy: text.endIndex) ?? end
        }
        return chunks
    }

    private func stableDocumentID(for document: PendingImportDocument) -> String {
        let key = "\(document.sourcePath)\n\(document.text)"
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in key.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x1000_0000_01b3
        }
        return "import-\(String(hash, radix: 16))"
    }

    // MARK: - Results Export

    func exportData(format: ResultsExportFormat) throws -> Data {
        switch format {
        case .csv:
            return exportCSVData()
        case .json:
            return try exportJSONData()
        }
    }

    func writeExportFile(format: ResultsExportFormat) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(defaultExportFilename(format: format))
        try exportData(format: format).write(to: url, options: .atomic)
        return url
    }

    func defaultExportFilename(format: ResultsExportFormat) -> String {
        let rawStem = currentQuery.isEmpty ? "results" : currentQuery
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let sanitized = rawStem
            .lowercased()
            .unicodeScalars
            .map { allowed.contains($0) ? Character($0) : "-" }
        let compact = String(sanitized)
            .split(separator: "-")
            .joined(separator: "-")
            .prefix(40)
        let stem = compact.isEmpty ? "results" : String(compact)
        return "proxima-\(stem).\(format.fileExtension)"
    }

    private func exportCSVData() -> Data {
        var lines = [
            ["query", "id", "score", "document_title", "category", "text"]
                .map(csvEscape)
                .joined(separator: ",")
        ]
        for result in results {
            lines.append([
                currentQuery,
                result.id.uuidString,
                String(format: "%.6f", Double(result.distance)),
                result.documentTitle,
                result.category,
                result.text,
            ].map(csvEscape).joined(separator: ","))
        }
        return Data(lines.joined(separator: "\n").utf8)
    }

    private func exportJSONData() throws -> Data {
        struct ExportResult: Codable {
            let id: UUID
            let score: Float
            let documentTitle: String
            let category: String
            let text: String
            let documentID: String?
            let chunkIndex: Int?
        }
        struct ExportDocument: Codable {
            let query: String
            let resultCount: Int
            let results: [ExportResult]
        }

        let document = ExportDocument(
            query: currentQuery,
            resultCount: results.count,
            results: results.map {
                ExportResult(
                    id: $0.id,
                    score: $0.distance,
                    documentTitle: $0.documentTitle,
                    category: $0.category,
                    text: $0.text,
                    documentID: $0.documentID,
                    chunkIndex: $0.chunkIndex)
            })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(document)
    }

    private func csvEscape(_ value: String) -> String {
        guard value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r") else {
            return value
        }
        return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
    }

    private func decodeSentence(_ data: Data?) -> Sentence? {
        guard let data else { return nil }
        return try? JSONDecoder().decode(Sentence.self, from: data)
    }

    private func displayTitle(for sentence: Sentence) -> String {
        if let title = sentence.documentTitle, !title.isEmpty {
            return title
        }
        switch sentence.category {
        case "Your Notes": return "Your Notes"
        case "Images": return "Images"
        default: return "Sample Corpus"
        }
    }
}

private struct ImportReadError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
