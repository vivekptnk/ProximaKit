#!/usr/bin/env python3
"""stop-check.py — Stop hook.
Blocks stopping if:
  - swift test has failures
  - There are uncommitted changes
"""

import subprocess
import sys

def main():
    errors = []

    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    uncommitted = [l for l in result.stdout.strip().split("\n") if l.strip()]
    if uncommitted:
        errors.append(f"UNCOMMITTED CHANGES ({len(uncommitted)} files):")
        for f in uncommitted[:10]:
            errors.append(f"  {f}")
        if len(uncommitted) > 10:
            errors.append(f"  ... and {len(uncommitted) - 10} more")

    # Run tests
    result = subprocess.run(
        ["swift", "test"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        lines = result.stdout.split("\n") + result.stderr.split("\n")
        failures = [l for l in lines if "failed" in l.lower() or "error:" in l.lower()]
        errors.append("TESTS FAILING:")
        for f in failures[:10]:
            errors.append(f"  {f}")

    if errors:
        print("BLOCKED: Cannot stop with issues outstanding.\n")
        print("\n".join(errors))
        print("\nFix the issues above, then commit your work before stopping.")
        sys.exit(2)


if __name__ == "__main__":
    main()
