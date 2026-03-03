# Model Routing Guide

ProximaKit is designed for multi-model development. Each model has a role. Using the right model for the right task saves money, saves time, and produces better results.

## The Three Roles

```
┌─────────────────────────────────────────────────────────────┐
│  OPUS — The Architect                                       │
│  Deep reasoning, algorithm design, architecture decisions   │
│  claude --model opus                                        │
│                                                             │
│  Use for: HNSW implementation, ADRs, system design,         │
│  complex debugging, interview prep, performance analysis    │
│  Speed: Slow  Cost: High  Intelligence: Maximum             │
├─────────────────────────────────────────────────────────────┤
│  SONNET — The Builder                                       │
│  Daily driver, implements features, writes tests            │
│  claude --model sonnet (default)                            │
│                                                             │
│  Use for: Story implementation, bug fixes, code review,     │
│  test writing, documentation, refactoring, PR prep          │
│  Speed: Medium  Cost: Medium  Intelligence: High            │
├─────────────────────────────────────────────────────────────┤
│  HAIKU — The Sprinter                                       │
│  Fast execution, boilerplate, mechanical tasks              │
│  claude --model haiku                                       │
│                                                             │
│  Use for: Scaffolding, file renaming, formatting cleanup,   │
│  listing files, simple queries, boilerplate generation      │
│  Speed: Fast  Cost: Low  Intelligence: Good                 │
└─────────────────────────────────────────────────────────────┘
```

## Routing by Task

### Use Opus When...

| Task | Why Opus | Command |
|------|----------|---------|
| Implementing HNSW algorithm | Multi-step algorithm with subtle correctness requirements | `/story PK-005` |
| Writing an ADR | Requires weighing tradeoffs and reasoning about long-term consequences | `/adr "topic"` |
| Complex debugging | Needs to reason about state across multiple files and execution paths | Direct prompt |
| Performance analysis | Understanding why recall is low or latency is high requires deep analysis | `/bench` then follow-up |
| Interview prep | Needs to simulate an Apple interviewer with deep technical knowledge | `/interview HNSW` |
| System design review | Architectural decisions need careful reasoning about tradeoffs | `/architect "change"` |
| Understanding papers | Explaining HNSW paper, CRDT theory, etc. requires synthesis | `/explain HNSW` |

**Opus workflow:**
```bash
claude --model opus
> /explain HNSW                    # Learn the algorithm deeply
> /story PK-006                    # Implement multi-layer HNSW
> /adr "persistent index format"   # Write the ADR for persistence design
```

### Use Sonnet When...

| Task | Why Sonnet | Command |
|------|-----------|---------|
| Implementing a story | Good balance of speed and quality for feature work | `/story PK-002` |
| Fixing bugs | Understands code well, fast enough for iteration | `/fix-issue 42` |
| Writing tests | Can generate comprehensive tests with edge cases | `/test` |
| Code review | Catches issues effectively without over-thinking | `/review` |
| Documentation | Good at writing clear, accurate docs | Direct prompt |
| Refactoring | Understands patterns and can restructure safely | Direct prompt |

**Sonnet workflow (daily driver):**
```bash
claude                             # Defaults to Sonnet
> /story PK-003                    # Implement distance metrics
> /review                          # Self-review
> git push                         # Done
```

### Use Haiku When...

| Task | Why Haiku | Command |
|------|----------|---------|
| Scaffolding files | Creating boilerplate is mechanical, doesn't need deep reasoning | `/scaffold DistanceMetric` |
| Renaming/moving files | Simple find-and-replace across files | `/refactor rename X to Y` |
| Formatting/cleanup | Mechanical code tidying | `/tidy` |
| Listing/searching files | Quick codebase navigation | Direct prompt |
| Generating boilerplate tests | Template-based test generation | `/scaffold-tests VectorTests` |
| Reading and summarizing | Quick summaries of files or docs | Direct prompt |

**Haiku workflow:**
```bash
claude --model haiku
> /scaffold BruteForceIndex        # Generate the file skeleton
> /tidy Sources/ProximaKit/        # Clean up formatting
> /refactor rename magnitude to norm  # Quick rename
```

## The Multi-Model Session

For complex stories, use all three models in sequence:

```bash
# Phase 1: Understand (Opus)
claude --model opus
> /explain "heuristic neighbor selection in HNSW"
> exit

# Phase 2: Build (Sonnet)
claude
> /story PK-006
> # implement the feature
> exit

# Phase 3: Polish (Haiku)
claude --model haiku
> /tidy Sources/ProximaKit/Index/
> /scaffold-tests HNSWHeuristicTests
> exit

# Phase 4: Review (Sonnet)
claude
> /review
```

## The Prompt-Context Hook

The `UserPromptSubmit` hook (`prompt-context.py`) automatically detects task complexity and suggests the right model. If you see:

```
[Hint: This task involves deep reasoning. Consider using Opus: claude --model opus]
```

...take the hint. Switch models. It's faster than having Sonnet struggle with algorithm design, and cheaper than having Opus do boilerplate.

## Cost Estimate Per Session Type

| Session | Model | Duration | Approx Cost |
|---------|-------|----------|-------------|
| Scaffolding + cleanup | Haiku | ~10 min | $0.10-0.30 |
| Story implementation | Sonnet | ~30 min | $1-3 |
| Algorithm deep dive | Opus | ~45 min | $5-15 |
| Full story (multi-model) | Mixed | ~60 min | $3-8 |

These are rough estimates. Actual cost depends on context window usage and tool calls.
