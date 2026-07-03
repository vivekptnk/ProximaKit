#!/bin/sh
# Enforces ProximaKit's module import boundaries — the "Module Rules" convention
# in CONTRIBUTING.md that the build itself does not enforce (Apple's system
# frameworks link against any target on the SDK, and Package.swift declares no
# per-target framework linkage that would reject e.g. `import CoreML` inside the
# core module). This script turns that convention into an automated gate.
#
# Usage:
#   scripts/check-imports.sh            # scan the real repo (run from any cwd)
#   scripts/check-imports.sh /path/root # scan an alternate tree rooted at /path
#                                        # (that root's Sources/... is scanned;
#                                        # used by the self-test to point at a
#                                        # scratch copy without touching the repo)
#
# Exit codes:
#   0  no violations — every import is within its scope's allowlist
#   1  one or more forbidden imports found (each printed as file:line: ...)
#   2  usage / environment error (no Sources/ directory under the scan root)
#
# Detection: matches a plain `import Foo` line AND an attributed one such as
# `@preconcurrency import Foo` (leading `@attr` tokens are stripped before the
# module name is read), across an optional `import`-kind keyword (`import
# struct Foo.Bar`). Lines like `#if canImport(Metal)` are NOT import
# statements and are correctly ignored.
#
# ── Scope 1: Sources/ProximaKit/**, EXCLUDING Sources/ProximaKit/Documentation.docc/**
# The DocC catalog is a documentation resource, not a compiled target source;
# SwiftPM does not build its `.swift` tutorial snippets, and those snippets
# legitimately `import ProximaKit`/`import ProximaEmbeddings` as an external
# consumer would — so the catalog is excluded from scope, not allow-listed.
#
# Allowed imports for ProximaKit (with justification):
#   Foundation  — Swift standard library surface; used in every file.
#   Accelerate  — vDSP for all vector math (ADR-001).
#   Metal       — GPU batch-distance kernel (ADR-009); present only in
#                 Distance/MetalBatchDistance.swift behind `#if canImport(Metal)`
#                 because Metal is not available in every build environment.
#   Darwin      — POSIX surface (fcntl / F_FULLFSYNC, the strongest available
#                 flush) for WAL durability (ADR-013); behind `#if canImport(Darwin)`.
#   Glibc       — the Linux equivalent of the Darwin durability code path;
#                 behind `#elseif canImport(Glibc)`, mutually exclusive with Darwin.
#
# ── Scope 2: Sources/ProximaEmbeddings/** (all compiled source; no DocC catalog)
# ProximaEmbeddings may import everything ProximaKit may (it is the same
# Apple-platform baseline) PLUS the provider-specific frameworks below.
#
# Additional allowed imports for ProximaEmbeddings (with justification):
#   ProximaKit      — internal dependency for the `Vector` type
#                     (Package.swift: dependencies: ["ProximaKit"]).
#   CoreML          — CoreMLEmbeddingProvider (written `@preconcurrency import CoreML`).
#   NaturalLanguage — NLEmbeddingProvider (NLEmbedding-based text vectors).
#   Vision          — VisionEmbeddingProvider (image feature-print vectors).
#   CoreGraphics    — VisionEmbeddingProvider; Vision's CGImage-based APIs need it.
#
# The lists above are the authoritative allowlist; CONTRIBUTING.md and the CI
# `lint` job point here rather than duplicating them (single source of truth).
#
# POSIX sh, no bashisms. Verified clean with: shellcheck -s sh scripts/check-imports.sh

set -eu

# Scan root: explicit first argument, else the repository root (this script
# lives in <root>/scripts/), resolved independently of the caller's cwd.
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
ROOT=${1:-$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)}

if [ ! -d "$ROOT/Sources" ]; then
    echo "check-imports: no Sources/ directory under '$ROOT'" >&2
    exit 2
fi

# Allowlists (space-separated module names).
PROXIMAKIT_ALLOW="Foundation Accelerate Metal Darwin Glibc"
EMBEDDINGS_ALLOW="Foundation Accelerate Metal Darwin Glibc ProximaKit CoreML NaturalLanguage Vision CoreGraphics"

VIOLATIONS=0

# Membership test using a case glob over a space-padded haystack — avoids word
# splitting entirely (module names are [A-Za-z0-9_], so no glob metacharacters).
in_list() {
    case " $2 " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

TAB=$(printf '\t')

# Emits, for every import statement, a tab-separated: <path> <line> <module>.
# The program references awk fields ($0) and built-ins (FNR, FILENAME) that must
# NOT be expanded by the shell — hence the single quotes are deliberate.
# shellcheck disable=SC2016
IMPORT_AWK='
BEGIN { OFS = "\t" }
/^[[:blank:]]*(@[A-Za-z_][A-Za-z0-9_]*[[:blank:]]+)*import[[:blank:]]/ {
    s = $0
    sub(/^[[:blank:]]+/, "", s)
    while (s ~ /^@[A-Za-z_]/) { sub(/^@[A-Za-z_][A-Za-z0-9_]*[[:blank:]]+/, "", s) }
    sub(/^import[[:blank:]]+/, "", s)
    sub(/^(typealias|struct|class|enum|protocol|func|var|let)[[:blank:]]+/, "", s)
    sub(/[[:blank:]].*$/, "", s)
    sub(/\..*$/, "", s)
    if (s != "") print FILENAME, FNR, s
}
'

# scan_scope <label> <dir> <allowlist> <exclude-dir-or-empty>
scan_scope() {
    scope_label=$1
    scope_dir=$2
    allow=$3
    exclude=$4

    [ -d "$scope_dir" ] || return 0

    allow_display=$(printf '%s' "$allow" | sed 's/ /, /g')

    tmp=$(mktemp "${TMPDIR:-/tmp}/check-imports.XXXXXX")

    if [ -n "$exclude" ]; then
        find "$scope_dir" -type d -path "$exclude" -prune -o \
            -type f -name '*.swift' -exec awk "$IMPORT_AWK" {} + > "$tmp"
    else
        find "$scope_dir" -type f -name '*.swift' -exec awk "$IMPORT_AWK" {} + > "$tmp"
    fi

    while IFS="$TAB" read -r file lineno module; do
        [ -n "$module" ] || continue
        if ! in_list "$module" "$allow"; then
            rel=${file#"$ROOT"/}
            printf "%s:%s: forbidden import '%s' (%s may import: %s)\n" \
                "$rel" "$lineno" "$module" "$scope_label" "$allow_display" >&2
            VIOLATIONS=$((VIOLATIONS + 1))
        fi
    done < "$tmp"

    rm -f "$tmp"
}

scan_scope "ProximaKit" "$ROOT/Sources/ProximaKit" "$PROXIMAKIT_ALLOW" \
    "$ROOT/Sources/ProximaKit/Documentation.docc"
scan_scope "ProximaEmbeddings" "$ROOT/Sources/ProximaEmbeddings" "$EMBEDDINGS_ALLOW" ""

if [ "$VIOLATIONS" -gt 0 ]; then
    printf '\ncheck-imports: %d import-policy violation(s) found.\n' "$VIOLATIONS" >&2
    exit 1
fi

echo "check-imports: no violations — ProximaKit and ProximaEmbeddings imports are within policy."
exit 0
