# On-Device RAG: Private Answers Over Your Notes

*Retrieval-augmented generation in ~130 lines of Swift — no server, no API key, no internet.*

The runnable example lives in [`Examples/OnDeviceRAG/`](../Examples/OnDeviceRAG/). Three files:

| File | What it does |
|------|--------------|
| [`OnDeviceRAG.swift`](../Examples/OnDeviceRAG/OnDeviceRAG.swift) | The whole RAG flow — embed, index, retrieve, augment, answer |
| [`LanguageModel.swift`](../Examples/OnDeviceRAG/LanguageModel.swift) | The model seam: a 2-requirement protocol + two implementations |
| [`SampleNotes.swift`](../Examples/OnDeviceRAG/SampleNotes.swift) | 20 built-in "notes" — swap in your own data here |

```bash
swift run OnDeviceRAG                                            # interactive
swift run OnDeviceRAG -question "How long should I steep cold brew?"  # scripted
swift run OnDeviceRAG -llm template                              # force the deterministic stand-in model
```

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

## Why On-Device RAG?

RAG — retrieval-augmented generation — is the standard answer to "how do I make a language model answer questions about *my* data?" Instead of fine-tuning, you retrieve the most relevant passages at question time and hand them to the model as context.

Almost every RAG tutorial starts with a cloud vector database and a hosted LLM API. For personal data — notes, journals, messages, health logs — that means shipping your most private text to two different third parties before you get an answer.

The entire pipeline can run on the Mac or iPhone in your hand:

- **Embeddings**: Apple's NaturalLanguage framework ships a sentence-embedding model with the OS.
- **Vector search**: ProximaKit — pure-Swift HNSW, the part this repo exists for.
- **Generation**: Apple's FoundationModels on-device LLM (macOS 26+/iOS 26+ with Apple Intelligence), or any local model you bring.

No token leaves the device at any step. It works in airplane mode. There is no per-query cost.

## The Five Concepts

| # | Concept | In this example | Code |
|---|---------|-----------------|------|
| 1 | **Embed** | Each note becomes a 512-dim vector that captures its *meaning* | `NLEmbeddingProvider.embed` |
| 2 | **Index** | Vectors go into an HNSW graph for fast nearest-neighbour search | `HNSWIndex.add` |
| 3 | **Retrieve** | The question is embedded too; the index returns the k closest notes | `HNSWIndex.search` |
| 4 | **Augment** | The retrieved notes are packed into the model's context, numbered for citation | `LanguageModel.reply(question:context:)` |
| 5 | **Answer** | The model answers *from the provided notes only*, citing `[1]`, `[2]`, … | `TemplateLLM` / `FoundationModelsLLM` |

Steps 1–3 are classic ProximaKit semantic search (the same thing `ProximaDemo` does). RAG is just steps 4–5 bolted on top: instead of *showing* you the search results, you make a language model *read* them and answer.

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

## Walking the Code

All snippets below are verbatim from the files in `Examples/OnDeviceRAG/`.

### Steps 1 + 2 — Embed and index the notes

From `OnDeviceRAG.swift` (inside `main()`):

```swift
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
```

The note text rides along as metadata, so a search result can hand back the original passage without a side lookup. `HNSWIndex` is an actor ([ADR-002](adr/ADR-002-actor-isolation.md)), hence the `await`s.

### Steps 3 + 4 + 5 — Retrieve, augment, answer

The RAG core is one function. From `OnDeviceRAG.swift`:

```swift
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
```

Two deliberate choices here:

- **The retrieved passages are printed, with distances, before the answer.** RAG that hides its retrieval is unauditable. When you can see what the model was given, you can tell a retrieval failure from a generation failure (the [limitations](#honest-limitations) section shows a real example of each).
- **The model sees only `passages`.** Grounding is the whole point — the answer must come from your notes, and citations `[1]`–`[3]` point back at the list you just saw.

### The model seam — `LanguageModel`

Generation is behind a protocol so the retrieval code never cares what model answers. From `LanguageModel.swift`:

```swift
protocol LanguageModel: Sendable {
    /// Human-readable backend description, printed at startup so transcripts
    /// are honest about which model produced the answers.
    var name: String { get }

    /// Produces an answer to `question` grounded in `context`.
    func reply(question: String, context: [String]) async throws -> String
}
```

Two implementations ship with the example.

**`TemplateLLM` — the deterministic stand-in.** This is **not** a generative model and does not pretend to be one. It scores each retrieved passage by content-word overlap with the question and stitches the best ones into a templated, cited answer:

```swift
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
```

Why include it at all? It makes the example run on **every** supported machine with zero model downloads and zero dependencies, and it makes scripted runs reproducible. It is the placeholder that defines the socket.

**`FoundationModelsLLM` — Apple's on-device LLM.** When the OS provides a real model, use it. The whole thing is conditionally compiled and availability-guarded, so the example still builds on a macOS 14 toolchain and still runs where Apple Intelligence is off:

```swift
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
```

Selection happens once at startup, and the chosen backend's `name` is printed so every transcript is honest about what produced it:

```swift
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
```

### The question loop

EOF-safe, same pattern as `ProximaDemo` — pipe questions in and it exits cleanly when stdin runs dry:

```swift
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
```

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

## A Real Session

Everything below is a **real, unedited transcript** captured from `swift run OnDeviceRAG` in **debug mode** (SwiftPM's default; only SwiftPM's build-progress lines are omitted) on an Apple Silicon Mac running macOS 26.0.1 with Apple Intelligence enabled — so the run auto-selected the FoundationModels backend. The millisecond figures are debug-mode and illustrative, **not** benchmark claims ([ADR-005](adr/ADR-005-benchmark-methodology.md) — benchmarks are release-mode only).

```text
$ swift run OnDeviceRAG -question "How long should I steep cold brew?" \
    -question "When should I water the fiddle leaf fig?" \
    -question "What do I need to pack for a day hike?"

╔══════════════════════════════════════════════╗
║         OnDeviceRAG  ·  ProximaKit            ║
║   Private answers over your notes. No cloud.  ║
╚══════════════════════════════════════════════╝

Loading NLEmbedding model...
  Dimension: 512
Indexing 20 notes...
  Indexed 20 notes.

Answering with: FoundationModels (Apple's on-device LLM — generative)

❓ Question: How long should I steep cold brew?

📚 Retrieved notes (lower distance = more relevant, 3.5 ms):
  [1] 0.667  Cold brew recipe: 80 grams of coarsely ground coffee per litre of cold water, steep 16 hours in the fridge, then filter and dilute one-to-one before serving.
  [2] 0.732  Morning meditation: ten minutes before coffee, box breathing four counts in, four hold, four out, four hold.
  [3] 0.746  The basil on the windowsill bolts when it flowers — pinch the flower buds early and harvest from the top to keep it bushy.

💬 Answer: Steep the coffee for 16 hours. [1]

❓ Question: When should I water the fiddle leaf fig?

📚 Retrieved notes (lower distance = more relevant, 4.7 ms):
  [1] 0.575  The basil on the windowsill bolts when it flowers — pinch the flower buds early and harvest from the top to keep it bushy.
  [2] 0.702  Carbonara rules: no cream ever, whisk two egg yolks and one whole egg with pecorino, and toss the pasta off the heat so the eggs stay silky.
  [3] 0.779  The fiddle leaf fig only needs water when the top five centimetres of soil are dry, roughly every ten days, and it hates being moved away from its bright window.

💬 Answer: [3] The fiddle leaf fig only needs water when the top five centimetres of soil are dry, roughly every ten days.

❓ Question: What do I need to pack for a day hike?

📚 Retrieved notes (lower distance = more relevant, 5.0 ms):
  [1] 0.773  Hiking day-pack checklist: two litres of water, headlamp, first-aid kit, rain shell, paper map, and a power bank for the phone.
  [2] 0.786  Plant the tomato seedlings outside after the last frost in mid-May, and harden them off with a week of afternoons on the balcony first.
  [3] 0.789  Thermostat schedule: 20 degrees during the day, 17 at night; the radiators need bleeding if the upstairs ones stay cold.

💬 Answer: You need two litres of water, headlamp, first-aid kit, rain shell, paper map, and a power bank for your phone. [1]
```

Note the second question: retrieval ranked two irrelevant notes *above* the fig note — and the model still answered correctly, citing `[3]`, because the right passage was inside the top-3 context. That is RAG working as designed: retrieval only has to get the answer *into* the context window, and the printed distances let you verify it did.

The same question with the stand-in forced (`-llm template`, same machine, same debug build — the answer is deterministic given the same retrieved notes):

```text
$ swift run OnDeviceRAG -llm template -question "When should I water the fiddle leaf fig?"

╔══════════════════════════════════════════════╗
║         OnDeviceRAG  ·  ProximaKit            ║
║   Private answers over your notes. No cloud.  ║
╚══════════════════════════════════════════════╝

Loading NLEmbedding model...
  Dimension: 512
Indexing 20 notes...
  Indexed 20 notes.

Answering with: TemplateLLM (extractive stand-in — selects from retrieved notes, does not generate text)

❓ Question: When should I water the fiddle leaf fig?

📚 Retrieved notes (lower distance = more relevant, 3.8 ms):
  [1] 0.575  The basil on the windowsill bolts when it flowers — pinch the flower buds early and harvest from the top to keep it bushy.
  [2] 0.702  Carbonara rules: no cream ever, whisk two egg yolks and one whole egg with pecorino, and toss the pasta off the heat so the eggs stay silky.
  [3] 0.779  The fiddle leaf fig only needs water when the top five centimetres of soil are dry, roughly every ten days, and it hates being moved away from its bright window.

💬 Answer: From your notes: The fiddle leaf fig only needs water when the top five centimetres of soil are dry, roughly every ten days, and it hates being moved away from its bright window. [3]
```

And the EOF behaviour, for scripted pipelines (real transcript, same build):

```text
$ printf 'When are taxes due?\n' | swift run OnDeviceRAG

╔══════════════════════════════════════════════╗
║         OnDeviceRAG  ·  ProximaKit            ║
║   Private answers over your notes. No cloud.  ║
╚══════════════════════════════════════════════╝

Loading NLEmbedding model...
  Dimension: 512
Indexing 20 notes...
  Indexed 20 notes.

Answering with: FoundationModels (Apple's on-device LLM — generative)

Ask a question about your notes. Type "quit" to exit.

❓ Question: 
📚 Retrieved notes (lower distance = more relevant, 3.7 ms):
  [1] 0.767  Taxes: file by April 15, but the accountant wants every document — receipts, brokerage statements, charity letters — by March 20.
  [2] 0.870  Morning meditation: ten minutes before coffee, box breathing four counts in, four hold, four out, four hold.
  [3] 0.891  Thermostat schedule: 20 degrees during the day, 17 at night; the radiators need bleeding if the upstairs ones stay cold.

💬 Answer: [1] Taxes are due on April 15. 

❓ Question: 
Bye!
```

Exit code 0 — the piped question echoes as empty because the terminal never sees what `printf` typed.

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

## Swap In Your Own Model

The retrieval half never changes. Everything model-specific lives behind the two-requirement `LanguageModel` protocol, so "upgrade the model" means "write one struct".

**Already done for you: Apple FoundationModels.** On macOS 26+/iOS 26+ with Apple Intelligence enabled, the example auto-selects `FoundationModelsLLM` (shown above) — a real generative model, still fully on-device. There is nothing to configure; the availability guard handles machines where the model is absent, disabled, or still downloading, falling back to `TemplateLLM`.

**Any other LLM.** Conform to the protocol and return it from `makeLanguageModel` (or construct it directly in `main`). A sketch — *this is illustrative, not part of the repo*:

```swift
struct MyLocalLLM: LanguageModel {
    let name = "MyModel (MLX / llama.cpp / anything with a generate function)"

    func reply(question: String, context: [String]) async throws -> String {
        let notes = context.enumerated()
            .map { "[\($0.offset + 1)] \($0.element)" }
            .joined(separator: "\n")
        let prompt = """
            Answer using ONLY these notes, citing note numbers like [1]:
            \(notes)

            Question: \(question)
            """
        return try await myModel.generate(prompt)   // ← your inference call
    }
}
```

The contract is small and worth keeping: the model gets the passages **in retrieval order**, cites them by 1-based position so citations line up with the printed list, and is instructed to refuse when the notes don't contain the answer. Those three habits are what separate "RAG" from "an LLM that happens to have seen some pasted text".

Two adjacent swaps work the same way:

- **Better embeddings**: replace `NLEmbeddingProvider` with `CoreMLEmbeddingProvider` and a converted sentence-transformer (see [README — Use a Custom AI Model](../README.md#use-a-custom-ai-model-coreml)). Nothing else changes — both conform to `EmbeddingProvider`.
- **Better retrieval**: replace `HNSWIndex` with `HybridIndex` to add BM25 keyword recall for exact terms ([docs/HYBRID.md](HYBRID.md)).

## Honest Limitations

This example optimises for "readable in one sitting", not production retrieval quality. Know what you're getting:

- **NLEmbedding quality is modest.** It ships with the OS and needs zero setup, but it is far below modern sentence-transformers. You can see it in the real transcript above: for the fig question, the correct note ranked third (distance 0.779) behind basil and carbonara. With a small `k` and a weak embedder, the right passage can miss the context window entirely. First upgrade to make: `CoreMLEmbeddingProvider` with a MiniLM-class model.
- **`TemplateLLM` is extractive, not generative.** It can only quote whole retrieved notes. It cannot paraphrase, synthesise across passages, do arithmetic, or say anything that isn't verbatim in a note. It exists to keep the example dependency-free and deterministic — treat it as the protocol's placeholder, never as "a small LLM".
- **Small on-device models make grounded mistakes too.** In one real run (same build, FoundationModels backend), "What tire pressure should the road bike run?" was answered with `[2] The gravel bike runs best around 40 psi.` — a correct quote of the cited note, but the *road* bike answer (85 psi) sat in the same sentence. RAG narrows the model's world to your notes; it does not make the model careful. The printed passages are your audit trail.
- **No chunking.** Each note is one sentence-sized embedding. Real documents need splitting into overlapping chunks before indexing — that splitting is the caller's job; `VectorStore` then manages the per-document chunk storage (add/remove/count) for you.
- **20 notes don't need HNSW.** At this scale `BruteForceIndex` would be exact and just as fast; the example uses `HNSWIndex` because that's the API you'll keep as your corpus grows to thousands of notes (the two share the `VectorIndex` API — swap freely).
- **The timings shown are debug-mode prints**, included for flavour. For real numbers, see [docs/BENCHMARKS.md](BENCHMARKS.md).

## Where Next

- [`README.md`](../README.md) — the full API tour: quantization, persistence, filtered search
- [`docs/RAG-WRAPPER-RECIPE.md`](RAG-WRAPPER-RECIPE.md) — wrapping `HNSWIndex` yourself? crash-safe chunk records for consumers who own the chunk pipeline
- [`docs/HYBRID.md`](HYBRID.md) — BM25 + dense fusion, the usual next step for RAG retrieval quality
- [ADR-008](adr/ADR-008-filtered-search.md) — per-user / per-folder filtering on every search call
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — how the pieces fit
