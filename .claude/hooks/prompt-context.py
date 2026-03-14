#!/usr/bin/env python3
"""prompt-context.py — UserPromptSubmit hook.
Injects branch context and guards against dependency additions.
"""

import json
import os
import subprocess
import sys
import re


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
        prompt = hook_input.get("user_prompt", hook_input.get("content", ""))
    except (json.JSONDecodeError, Exception):
        prompt = ""

    if not prompt:
        return

    prompt_lower = prompt.lower().strip()
    lines = []

    # Current branch + story ID
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    )
    branch = result.stdout.strip()
    if branch:
        lines.append(f"Branch: {branch}")
        story_match = re.search(r'PK-\d+', branch)
        if story_match:
            lines.append(f"Active story: {story_match.group()}")

    # Uncommitted changes (only show if non-zero)
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    )
    changed = [l for l in result.stdout.strip().split("\n") if l.strip()]
    if changed:
        lines.append(f"Uncommitted changes: {len(changed)} files")

    # Zero-dependency guard
    dep_keywords = ["add dependency", "add package", "install", "spm add", "pod install"]
    if any(kw in prompt_lower for kw in dep_keywords):
        lines.append("REMINDER: ProximaKit is zero-dependency. Apple frameworks only.")

    if lines:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
