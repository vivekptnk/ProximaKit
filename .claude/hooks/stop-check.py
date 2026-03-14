#!/usr/bin/env python3
"""stop-check.py — Stop hook.
Blocks stopping if there are uncommitted changes.
Does NOT run swift test (too slow — auto-test.py catches failures during development).
"""

import subprocess
import sys

def main():
    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    uncommitted = [l for l in result.stdout.strip().split("\n") if l.strip()]
    if not uncommitted:
        return

    # Write to stderr so Claude Code displays the message
    lines = [
        f"BLOCKED: {len(uncommitted)} uncommitted file(s). Commit before stopping.",
        "",
    ]
    for f in uncommitted[:10]:
        lines.append(f"  {f}")
    if len(uncommitted) > 10:
        lines.append(f"  ... and {len(uncommitted) - 10} more")

    print("\n".join(lines), file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
