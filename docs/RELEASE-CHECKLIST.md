# Release Checklist

Use this checklist for every tagged ProximaKit release. A tag push triggers `.github/workflows/release.yml`, which validates the release and publishes the GitHub release automatically; do not push a tag until the exact commit is authorized.

## Prepare the release

- [ ] Finish the release changes on `main` with a clean working tree.
- [ ] Confirm `ProximaKit.version`, the changelog heading, README/DocC dependency examples, roadmap, security support table, and issue template all name the same version.
- [ ] Confirm `[Unreleased]` is empty and the release section uses the intended release date.
- [ ] Review every commit since the previous tag and ensure it is represented in the release notes or intentionally excluded.
- [ ] Dry-run the changelog extraction used by `.github/workflows/release.yml`; inspect the rendered notes for correct boundaries and no internal planning language.

## Verify the exact commit

- [ ] Record the full release candidate SHA: `git rev-parse HEAD`.
- [ ] Confirm `HEAD == origin/main`; if `main` changes afterward, discard the old evidence and verify the new SHA.
- [ ] Run the CI-equivalent functional suite:

  ```bash
  swift test --skip RecallBenchmarkTests --skip PQBenchmarkTests
  ```

- [ ] Run targeted persistence, compatibility, graph-snapshot, file-extension, and consumer-recipe tests.
- [ ] Run the opted-in PQ acceptance suite in Release mode:

  ```bash
  PROXIMA_PQ_BENCH=1 swift test -c release --filter PQBenchmarkTests
  ```

- [ ] Run build and policy gates:

  ```bash
  swift build -Xswiftc -warnings-as-errors
  swift build -c release
  swift build --target ProximaDemo
  scripts/check-imports.sh
  swiftlint lint --strict
  swift package generate-documentation --target ProximaKit
  bash .github/ci-scripts/check_version.sh
  bash .github/ci-scripts/check_version.sh <version>
  ```

- [ ] Run the iOS SDK-only compile and supported Demo app builds used by CI.
- [ ] Push `main` and wait for CI, Docs, lint, iOS, and benchmark evidence whose `headSha` exactly matches the recorded SHA.
- [ ] Trigger the benchmark workflow with the `full` slice and confirm its Release-mode PQ acceptance job is green for that SHA.

## Authorization boundary

Stop before creating a tag. Obtain explicit maintainer confirmation naming both the version and full SHA, for example:

> Confirm creation and push of tag `v1.9.0` at SHA `<full-sha>`.

Without that confirmation, do not create a local tag, push a tag, call `gh release create`, or otherwise publish a release.

## Publish and observe

After explicit confirmation:

- [ ] Create and push the authorized tag from the exact verified SHA.
- [ ] Observe `.github/workflows/release.yml`; its macOS PQ validation must pass before the publishing job can run.
- [ ] Confirm the generated GitHub release body matches the changelog section.
- [ ] Confirm the tag resolves through SwiftPM using the documented dependency floor.
- [ ] Confirm the published source reports the expected `ProximaKit.version` and security support line.
