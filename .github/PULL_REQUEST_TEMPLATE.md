## What & why

<!-- What does this change, and why? Link an issue if there is one. -->

## How verified

<!--
Command(s) run and the specific test names/classes exercised, e.g.:
  swift test --filter VectorStoreTests
  swift test --skip RecallBenchmarkTests
-->

## Checklist

Per [CONTRIBUTING.md](../CONTRIBUTING.md):

- [ ] `swift build` succeeds with no warnings
- [ ] `swift test --skip RecallBenchmarkTests` passes
- [ ] `swiftlint lint --strict` passes (pinned to 0.63.2, matching CI — see [Run Lint](../CONTRIBUTING.md#run-lint))
- [ ] New public APIs have `///` documentation
- [ ] New features have corresponding tests
- [ ] No new external dependencies added to `ProximaKit`
- [ ] ADR created for significant architectural decisions (required up front for Quantization, GPU, and Filtered-Search — see [CONTRIBUTING.md](../CONTRIBUTING.md#prerequisite-for-quantization-gpu-and-filtered-search-prs))
- [ ] `docs/ARCHITECTURE.md` updated if module boundaries changed
