# ProximaDemoApp

A macOS SwiftUI demo app showcasing ProximaKit's semantic search capabilities.

---

## What It Does

Type a natural language query and see ProximaKit find semantically similar content in real time. The app demonstrates:

- **Semantic text search** — find sentences by meaning, not keywords
- **Image similarity search** — add images and find visually similar ones
- **User notes** — add your own text to the search index on the fly
- **Live tuning** — adjust `efSearch` with a slider and see how it affects results
- **Index persistence** — the index saves to disk and survives app restart

```
┌──────────────────────────────────────────────┐
│  ProximaDemoApp                               │
│                                               │
│  ┌─────────────┐  ┌────────────────────────┐ │
│  │  Sidebar     │  │  Search Results        │ │
│  │             │  │                        │ │
│  │  efSearch ──│  │  0.12  Ocean waves     │ │
│  │  [====50==] │  │  0.18  Tropical...     │ │
│  │             │  │  0.25  Travel guide    │ │
│  │  Add Note   │  │  0.31  Beach sunset    │ │
│  │  [________] │  │                        │ │
│  │             │  │  Color = distance:     │ │
│  │  Add Image  │  │  green < orange < red  │ │
│  │  [  Pick  ] │  │                        │ │
│  │             │  └────────────────────────┘ │
│  │  Stats:     │                             │
│  │  46 vectors │                             │
│  │  384d       │                             │
│  └─────────────┘                             │
└──────────────────────────────────────────────┘
```

---

## Requirements

- macOS 14+
- Xcode 15+ (Swift 5.9)
- Apple Silicon recommended

---

## Running

### Option A: Xcode

```bash
open ProximaDemoApp.xcodeproj
```

Press **Cmd+R** to build and run.

### Option B: XcodeGen (regenerate project)

If you modify `project.yml`:

```bash
brew install xcodegen  # if needed
xcodegen generate
open ProximaDemoApp.xcodeproj
```

---

## How It Works

### Startup

1. `SearchEngine.buildIndex()` checks for a persisted index on disk.
2. If none found, it embeds 46 sample sentences (see `SampleData.swift`) using the selected embedding provider — `CoreMLEmbeddingProvider` if a model is found (see below), otherwise `NLEmbeddingProvider` — and builds an `HNSWIndex`.
3. The index is saved to `~/Library/Application Support/ProximaDemoApp/` for next launch.

### Search

1. User types a query in the search bar.
2. The query is embedded using the same provider that built the index (whichever `setupEmbedder()` selected at startup).
3. `HNSWIndex.search(query:k:)` returns the nearest neighbors.
4. Results are color-coded by distance: green (<0.55), orange (<0.68), red (>0.68).

### CoreML Model (Optional)

By default the app embeds with `NLEmbeddingProvider` — it ships with the OS and needs zero setup. Drop in a converted MiniLM model and `SearchEngine` switches to `CoreMLEmbeddingProvider` for higher-quality 384-dimensional embeddings automatically, with no rebuild.

**Where the app looks.** `SearchEngine.findModelFiles()` scans, in order, for a file named exactly `MiniLM-L6-v2.mlmodel` *and* a `vocab.txt` sitting together in the same directory:

1. The `Models/` directory at the ProximaKit repo root — the easiest option when running via `swift run ProximaDemo` or Xcode against a source checkout.
2. The current working directory, or its `Models/` subdirectory, when launching the built executable directly.
3. The directory containing the built `.app` bundle, or its `Models/` subdirectory — drop the two files next to a built app.

Both files must be present together. If either is missing, the app falls back to `NLEmbeddingProvider` silently — no error, no crash. Check which provider is active in the sidebar: it reads `"CoreML (MiniLM-L6-v2, 384d)"` or `"NLEmbedding (…d) — add Models/ for better quality"`.

**Getting a model.** ProximaKit doesn't ship a converted model or a conversion script — bring your own with [coremltools](https://github.com/apple/coremltools), the same tool the root [`README.md`](../../README.md#use-a-custom-ai-model-coreml) points to:

```bash
pip install coremltools transformers torch
```

Convert `sentence-transformers/all-MiniLM-L6-v2` (or any BERT-family sentence-transformer) to a Core ML model whose inputs are `input_ids` and `attention_mask` (`MLMultiArray<Int32>`) and whose output is a float `MLMultiArray` embedding — that is the exact contract `CoreMLEmbeddingProvider` expects (see its doc comment in `Sources/ProximaEmbeddings/CoreMLEmbeddingProvider.swift`). Save the result as `MiniLM-L6-v2.mlmodel`, and save the tokenizer's vocabulary as `vocab.txt` (one WordPiece token per line — `WordPieceTokenizer` reads it directly; `AutoTokenizer.save_vocabulary()` in the `transformers` library writes this format). Place both files together in one of the three locations above.

`CoreMLEmbeddingProvider` itself is more flexible than the demo's auto-discovery — its `init(modelAt:vocabURL:)` also accepts a `.mlpackage`, and `init(compiledModelURL:vocabURL:)` takes a pre-compiled `.mlmodelc`. The demo specifically looks for the legacy `.mlmodel` extension under that exact filename, so if you only have a `.mlpackage`, either re-export to `.mlmodel` or use `CoreMLEmbeddingProvider` directly in your own code instead of relying on the demo's file scan.

**Planned, not yet built:** an in-app model browser that lists models from the HuggingFace Hub and downloads a `.mlpackage` without leaving the app is tracked in [`docs/ROADMAP.md`](../../docs/ROADMAP.md) as a demo-app improvement. Today, installing a model means placing the two files yourself as described above.

---

## Source Files

| File | Purpose |
|------|---------|
| `ProximaDemoApp.swift` | App entry point, initializes `SearchEngine` |
| `MainView.swift` | SwiftUI interface with sidebar and results pane |
| `SearchEngine.swift` | Index lifecycle, search, persistence, embedding |
| `SampleData.swift` | 46 sample sentences across 9 categories |

---

## Sample Categories

The demo ships with sentences in these categories: Animals, Food, Technology, Nature, Sports, Science, Travel, Music, and Arts. Try queries like:

- "pets and wildlife" — finds animal sentences
- "cooking recipes" — finds food sentences
- "space exploration" — finds science sentences
- "outdoor activities" — finds sports and nature sentences
