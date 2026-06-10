#!/usr/bin/env bash
# Verifies that ProximaKit's runtime version constant and the CHANGELOG agree,
# and (optionally) that both match a release tag.
#
# Usage:
#   .github/ci-scripts/check_version.sh           # CI: constant == newest CHANGELOG entry
#   .github/ci-scripts/check_version.sh 1.4.0     # release: both must equal the tag version
#
# Context: ProximaKit.version shipped as "1.0.0" through the v1.1.0 and v1.4.0
# tags because nothing validated it. This script is wired into ci.yml (drift
# caught at PR time) and release.yml (drift blocks the tag build).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

constant="$(sed -n 's/.*public static let version = "\([^"]*\)".*/\1/p' \
    "$ROOT/Sources/ProximaKit/ProximaKit.swift")"
if [[ -z "$constant" ]]; then
    echo "::error::could not find 'public static let version' in Sources/ProximaKit/ProximaKit.swift" >&2
    exit 1
fi

# Newest released section, e.g. '## [1.4.0] — 2026-04-19' (skips [Unreleased]).
changelog="$(grep -m1 -oE '^## \[[0-9]+\.[0-9]+\.[0-9]+[^]]*\]' "$ROOT/CHANGELOG.md" \
    | sed -E 's/^## \[([^]]+)\]/\1/')"
if [[ -z "$changelog" ]]; then
    echo "::error::could not find a released '## [x.y.z]' section in CHANGELOG.md" >&2
    exit 1
fi

echo "ProximaKit.version  = $constant"
echo "CHANGELOG newest    = $changelog"

if [[ "$constant" != "$changelog" ]]; then
    echo "::error::ProximaKit.version ($constant) != newest CHANGELOG entry ($changelog)" >&2
    exit 1
fi

if [[ $# -ge 1 ]]; then
    tag_version="$1"
    echo "Release tag version = $tag_version"
    if [[ "$constant" != "$tag_version" ]]; then
        echo "::error::ProximaKit.version ($constant) != tag version ($tag_version)" >&2
        exit 1
    fi
fi

echo "Version check passed."
