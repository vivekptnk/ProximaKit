# ProximaDemoApp

A multiplatform SwiftUI demo app — iPhone, iPad, macOS, and visionOS from a single target — showcasing ProximaKit's semantic search, persistence, and benchmarking capabilities.

---

## What It Does

The app has three feature screens, switched via a `DemoScreen` picker (see `MainView.swift`):

### Search

The original experience:

- **Semantic text search** — find sentences by meaning, not keywords
- **Image similarity search** — add images and find visually similar ones
- **User notes** — add your own text to the search index on the fly
- **Live tuning** — adjust `efSearch` with a slider and see how it affects results
- **Index persistence** — the index saves to disk and survives app restart

### Persistence lab

Exercises ProximaKit's WAL-backed persistence end-to-end against a reproducible synthetic corpus (3,000 / 6,000 / 12,000 vectors × 384d, picked via segmented control):

- Saves the corpus as a v2 base, then reopens it journaled
- Live WAL readouts: generation, WAL bytes on disk, ops since checkpoint, needs-checkpoint, base format (v2/v3)
- Grow the WAL, then checkpoint it into a page-aligned v3 base
- Demonstrates the library's real, typed error path: opening `.paged` on an unpadded v2 base is refused with `PersistenceError`, surfaced as an honest "Paged open blocked" banner with a one-tap "Checkpoint to enable paging" recovery — nothing is faked or silently worked around
- Measures live resident-vs-paged process memory (via `task_vm_info.phys_footprint`, the same probe `PagedVectorMemoryTests` uses) and shows the delta, plus a caption on how much extra memory a batch of warm searches faults into the paged mapping

### Benchmark

Runs a seeded `efSearch` sweep (16 / 32 / 64 / 128 / 256) over a reproducible synthetic corpus (3,000 × 128d):

- Measures recall@10 against an exact `BruteForceIndex` ground truth
- Measures live per-query latency (median + p90)
- Charts recall-vs-latency with SwiftUI Charts, plus a results table
- Corpus and graph construction use fixed seeds, so recall is identical every run; latency is measured live and varies by machine/load

### Layout

- **Compact width** (iPhone, narrow iPad split): a 4-tab `TabView` — Search, Benchmark, Persistence, Index.
- **Regular width** (iPad full-screen, macOS, visionOS): a `NavigationSplitView` — the sidebar is the Index view, and the detail pane has a segmented picker switching between Search / Persistence / Benchmark.

### Search screen

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

*This mockup shows the Search screen only — Persistence and Benchmark are separate SwiftUI screens with their own controls, banners, and charts (see above).*

---

## Requirements

- macOS 14+, iOS 17+, or visionOS 1+
- Xcode 15+ (Swift 5.9)
- Apple Silicon recommended

---

## Running

### Option A: Xcode

```bash
open ProximaDemoApp.xcodeproj
```

Pick a run destination (My Mac, an iPhone/iPad simulator, or a visionOS simulator) and press **Cmd+R** to build and run.

### Option B: XcodeGen (regenerate project)

If you modify `project.yml`, regenerate the Xcode project before building:

```bash
brew install xcodegen  # if needed
xcodegen generate
open ProximaDemoApp.xcodeproj
```

### Option C: xcodebuild (CLI)

```bash
# macOS
xcodebuild -project ProximaDemoApp.xcodeproj -scheme ProximaDemoApp \
  -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO build

# iOS Simulator (iPhone)
xcodebuild -project ProximaDemoApp.xcodeproj -scheme ProximaDemoApp \
  -destination 'platform=iOS Simulator,name=iPhone 16' CODE_SIGNING_ALLOWED=NO build

# iPadOS Simulator
xcodebuild -project ProximaDemoApp.xcodeproj -scheme ProximaDemoApp \
  -destination 'platform=iOS Simulator,name=iPad Pro 13-inch (M4)' CODE_SIGNING_ALLOWED=NO build

# visionOS Simulator
xcodebuild -project ProximaDemoApp.xcodeproj -scheme ProximaDemoApp \
  -destination 'platform=visionOS Simulator,name=Apple Vision Pro 4K' CODE_SIGNING_ALLOWED=NO build
```

`CODE_SIGNING_ALLOWED=NO` is needed for these CLI/CI-style builds since there's no signing team configured; Xcode GUI builds (Cmd+R) don't need it once you've picked a personal team in project settings. The simulator device names above are examples — run `xcrun simctl list devices available` to see what's installed locally.

---

## Launch Hooks

For screenshots and UI-test automation, `MainView.swift` reads launch arguments from `UserDefaults` in its `.task` (launch args of the `-key value` form land there automatically):

| Argument | Effect |
|---|---|
| `-demoScreen <search\|persistence\|benchmark\|index>` | Selects the initial screen/tab on launch |
| `-demoQuery "<text>"` | Pre-fills the search field once the index has finished building (Search screen only; waits up to 30s for indexing to complete before filling, to avoid racing an empty index) |
| `-demoAutorun 1` | Boolean flag. Kicks off the selected screen's headline action automatically on appear, so a populated screen can be captured non-interactively: on Benchmark it runs the full sweep; on Persistence it builds the lab corpus, then either (default) grows the WAL by 50 ops and attempts a `.paged` open — surfacing the paged-blocked banner — or, when paired with `-demoFlow memory`, checkpoints and measures memory instead (the happy path) |
| `-demoFlow memory` | Paired with `-demoScreen persistence -demoAutorun 1`; selects the checkpoint-then-measure-memory flow instead of the default WAL-growth/paged-blocked-banner flow |

Example — build, boot, install, and launch straight into a populated Benchmark screen:

```bash
xcrun simctl boot "iPhone 16"   # if not already booted
xcrun simctl install "iPhone 16" /path/to/ProximaDemoApp.app
xcrun simctl launch "iPhone 16" com.vivekptnk.ProximaDemoApp \
  -demoScreen benchmark -demoAutorun 1
```

Bundle identifier: `com.vivekptnk.ProximaDemoApp`.

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
| `MainView.swift` | Root view: `DemoScreen` enum, compact `TabView` / regular `NavigationSplitView` layout, launch-hook handling |
| `SearchEngine.swift` | Index lifecycle, search, persistence, embedding (Search screen) |
| `SampleData.swift` | 46 sample sentences across 9 categories |
| `BenchmarkEngine.swift` | `@Observable` controller: builds exact + approximate indexes, computes ground truth, sweeps `efSearch`, measures recall@10 and latency (median/p90) |
| `BenchmarkView.swift` | SwiftUI view for the Benchmark screen: run button, progress, recall-vs-latency chart (SwiftUI Charts), results table |
| `PersistenceLab.swift` | `@Observable @MainActor` controller for the Persistence screen: builds the synthetic corpus, opens/reopens journaled, grows the WAL, checkpoints, measures live resident-vs-paged memory via `task_vm_info` |
| `PersistencePanel.swift` | SwiftUI view for the Persistence screen: WAL-state readout card, paged-blocked banner + recovery, mode picker + actions, memory measurement card |
| `DemoLabSupport.swift` | Shared, dependency-free helpers for the Persistence and Benchmark screens: `MemoryProbe` (the `task_vm_info.phys_footprint` probe) and `SyntheticCorpus`/`DemoRNG` (deterministic synthetic vectors, no embedder or network) |

---

## Sample Categories

The demo ships with sentences in these categories: Animals, Food, Technology, Nature, Sports, Science, Travel, Music, and Arts. Try queries like:

- "pets and wildlife" — finds animal sentences
- "cooking recipes" — finds food sentences
- "space exploration" — finds science sentences
- "outdoor activities" — finds sports and nature sentences
