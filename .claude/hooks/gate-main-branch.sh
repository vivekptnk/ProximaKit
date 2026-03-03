#!/bin/bash
# gate-main-branch.sh — PreToolUse hook for Write/Edit
# Blocks edits when on the main branch. Feature branches only.

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
    echo "BLOCKED: You're on the '$BRANCH' branch."
    echo "Create a feature branch first: git checkout -b feature/PK-XXX-description"
    exit 2
fi
