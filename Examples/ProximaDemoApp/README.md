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
│  │  48 vectors │                             │
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
2. If none found, it embeds 48 sample sentences (see `SampleData.swift`) using `NLEmbeddingProvider` and builds an `HNSWIndex`.
3. The index is saved to `~/Library/Application Support/ProximaDemoApp/` for next launch.

### Search

1. User types a query in the search bar.
2. The query is embedded using the same `NLEmbeddingProvider`.
3. `HNSWIndex.search(query:k:)` returns the nearest neighbors.
4. Results are color-coded by distance: green (<0.55), orange (<0.68), red (>0.68).

### CoreML Model (Optional)

The app automatically looks for a MiniLM-L6-v2 CoreML model in these locations:

1. `Models/` directory in the ProximaKit repo root
2. `~/Documents/ProximaKit-Models/`
3. App bundle resources
4. App's Application Support directory

If found, it uses `CoreMLEmbeddingProvider` for higher-quality 384-dimensional embeddings. Otherwise it falls back to `NLEmbeddingProvider`.

---

## Source Files

| File | Purpose |
|------|---------|
| `ProximaDemoApp.swift` | App entry point, initializes `SearchEngine` |
| `MainView.swift` | SwiftUI interface with sidebar and results pane |
| `SearchEngine.swift` | Index lifecycle, search, persistence, embedding |
| `SampleData.swift` | 48 sample sentences across 9 categories |

---

## Sample Categories

The demo ships with sentences in these categories: Animals, Food, Technology, Nature, Sports, Science, Travel, Music, and Arts. Try queries like:

- "pets and wildlife" — finds animal sentences
- "cooking recipes" — finds food sentences
- "space exploration" — finds science sentences
- "outdoor activities" — finds sports and nature sentences
