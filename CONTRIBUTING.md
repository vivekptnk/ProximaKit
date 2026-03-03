# Contributing to ProximaKit

## The Harness Does The Enforcing

Traditional repos: read the style guide, hope people follow it. ProximaKit: hooks block your tool calls if you violate rules. You don't need to memorize the code standards — the harness tells you when you break them.

What the hooks enforce automatically:
- No edits on main branch
- No third-party imports
- No module boundary violations
- Build must compile after every write
- No force unwraps in Sources/
- No manual float loops (must use vDSP)
- Tests must pass before stopping
- Changes must be committed before stopping

What you're responsible for (not enforced):
- Writing good tests
- DocC comments on public APIs
- Reading ADRs before changing core components
- Running benchmarks for performance-critical changes

## Quick Start

```bash
git clone https://github.com/vivek/ProximaKit.git
cd ProximaKit
swift build && swift test     # verify everything works
claude                        # start Claude Code
> /onboard                    # get oriented
```

## Choosing Your Model

Read [`docs/MODEL_GUIDE.md`](docs/MODEL_GUIDE.md). Quick version:

```bash
claude --model haiku          # scaffolding, renaming, formatting
claude                        # features, bugs, tests, reviews (Sonnet default)
claude --model opus           # algorithms, ADRs, deep debugging, interviews
```

The `prompt-context.py` hook auto-suggests the right model based on your prompt.

## Commands

| Command | Best Model | What It Does |
|---------|-----------|-------------|
| `/onboard` | Any | Read architecture, run tests, show status |
| `/story PK-005` | Sonnet | Implement a PRD story |
| `/fix-issue 42` | Sonnet | Fix a GitHub issue |
| `/explain HNSW` | Opus | Deep concept explanation |
| `/architect "topic"` | Opus | Make an architecture decision |
| `/interview HNSW` | Opus | Mock Apple interview |
| `/adr "topic"` | Opus | Write Architecture Decision Record |
| `/scaffold BruteForce` | Haiku | Create file skeleton |
| `/scaffold-tests Vector` | Haiku | Create test skeletons |
| `/tidy Sources/` | Haiku | Clean up formatting |
| `/refactor rename X Y` | Haiku | Mechanical rename/move |
| `/review` | Sonnet | Pre-PR quality check |
| `/bench` | Sonnet | Run benchmarks |
| `/test` | Any | Run and diagnose tests |
| `/ship` | Sonnet | Release checklist |

## Workflow

### Fixing a Bug
```bash
claude
> /fix-issue 42
# Hook creates branch, reads issue, implements fix, runs tests, commits
> /review
# Push and open PR
```

### Implementing a Story
```bash
claude --model opus
> /explain HNSW                    # understand the concept first

claude                             # switch to Sonnet
> /story PK-006                    # implement it

claude --model haiku
> /tidy Sources/ProximaKit/Index/  # clean up

claude
> /review                          # final check
```

### Making an Architecture Decision
```bash
claude --model opus
> /architect "Should we add product quantization for memory reduction?"
# Opus analyzes options, recommends, drafts ADR
```

## PR Template

PRs auto-populate from `.github/pull_request_template.md`. Key sections:
- What changed and why
- Checklist (build, tests, docs, benchmarks)
- How to review

## Issue Templates

Use structured templates in `.github/ISSUE_TEMPLATE/`:
- **Bug Report**: affected file, reproduction steps, acceptance criteria
- **Feature Request**: proposal, affected files, ADR needed?
- **Good First Issue**: scoped for one Claude Code session, explicit file list

## License

MIT. By contributing, you agree your contributions are MIT-licensed.
