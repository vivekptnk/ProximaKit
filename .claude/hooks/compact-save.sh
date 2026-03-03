#!/bin/bash
# compact-save.sh — PreCompact hook.
# Saves progress snapshot before context compression.

PROGRESS_FILE=".claude/progress.md"
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

mkdir -p .claude

cat > "$PROGRESS_FILE" << PROGRESS
# Progress Snapshot
Saved: $DATE
Branch: $BRANCH

## Recent Commits
$(git log --oneline -10 2>/dev/null || echo "No commits")

## Uncommitted Changes
$(git status --short 2>/dev/null || echo "No changes")

## Current Files
$(find Sources/ -name "*.swift" 2>/dev/null | sort)
$(find Tests/ -name "*.swift" 2>/dev/null | sort)
PROGRESS

echo "Progress saved to $PROGRESS_FILE"
