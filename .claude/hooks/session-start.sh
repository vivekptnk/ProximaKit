#!/bin/bash
# session-start.sh — SessionStart hook.
# Injects branch info, recent commits, and warns if on main.

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no-repo")

echo "=== ProximaKit Session ==="
echo "Branch: $BRANCH"

if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
    echo "WARNING: You're on main. Create a feature branch before editing."
fi

# Show recent commits
COMMITS=$(git log --oneline -5 2>/dev/null)
if [ -n "$COMMITS" ]; then
    echo ""
    echo "Recent commits:"
    echo "$COMMITS"
fi

# Show uncommitted changes count
CHANGES=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [ "$CHANGES" -gt 0 ]; then
    echo ""
    echo "Uncommitted changes: $CHANGES files"
fi
