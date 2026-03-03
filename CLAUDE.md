# ProximaKit

Zero-dependency, pure-Swift vector search. Accelerate/vDSP for math. HNSW from scratch.

## Quick Reference

| Need to... | Do this |
|---|---|
| Build | `swift build` |
| Test | `swift test` |
| Test specific | `swift test --filter HNSWTests` |
| Generate docs | `swift package generate-documentation` |
| See project structure | `docs/ARCHITECTURE.md` |
| See task breakdown | `docs/PRD.md` |
| Understand a design choice | `docs/adr/` |
| See vDSP patterns | `.claude/skills/accelerate-patterns.md` |
| See HNSW pseudocode | `.claude/skills/hnsw-reference.md` |
| Choose the right model | `docs/MODEL_GUIDE.md` |

## Modules
- **ProximaKit** (core): Vector math, indices, persistence. Imports: Foundation, Accelerate only.
- **ProximaEmbeddings**: Converts content → Vector. Imports: CoreML, NaturalLanguage, Vision.
- **Demo app**: SwiftUI semantic photo search. Imports everything.

Module boundaries are enforced by hooks. You'll get blocked if you violate them.

## What The Harness Enforces Automatically
These rules are NOT suggestions — hooks will block your tool calls if violated:

- ❌ Edits on main branch → create a feature branch first
- ❌ Third-party imports in Sources/ → Apple frameworks only
- ❌ ProximaKit core importing ProximaEmbeddings → wrong direction
- ❌ ProximaEmbeddings importing UI frameworks → it's a data layer
- ❌ Force unwraps in Sources/ → use guard let
- ❌ Manual float loops in core → use vDSP
- ❌ Stopping with failing tests → fix them first
- ❌ Stopping with uncommitted changes → commit your work

## What You Should Do (Not Enforced, But Expected)
- Write DocC comments on every public symbol
- Use `struct` over `class` unless reference semantics needed
- Use `guard` for preconditions
- Use `actor` for shared mutable state
- Read the relevant ADR before changing core components
- Benchmark before and after performance-critical changes

## Story Tracking
Stories are in `docs/PRD.md`, prefixed `PK-XXX`. Branch naming: `feature/PK-005-hnsw-nsw`.

## Model Routing
Read `docs/MODEL_GUIDE.md` for the full strategy. Quick version:
- **Opus** → algorithm design, ADRs, deep debugging, interview prep
- **Sonnet** → daily building, tests, reviews, bug fixes (default)
- **Haiku** → scaffolding, renaming, formatting, boilerplate

The `UserPromptSubmit` hook will suggest the right model based on your prompt.
