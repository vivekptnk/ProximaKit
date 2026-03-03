#!/usr/bin/env python3
"""auto-test.py — PostToolUse hook for Write/Edit on test files.
Auto-runs the relevant test target when a test file changes.
"""

import json
import sys
import subprocess

def main():
    hook_input = json.loads(sys.stdin.read())
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    if tool_name not in ("Write", "Edit"):
        return

    file_path = tool_input.get("file_path", "")
    if not file_path.endswith(".swift"):
        return
    if "/Tests/" not in file_path:
        return

    # Determine which test filter to use
    if "ProximaKitTests" in file_path:
        test_filter = "ProximaKitTests"
    elif "ProximaEmbeddingsTests" in file_path:
        test_filter = "ProximaEmbeddingsTests"
    else:
        return

    result = subprocess.run(
        ["swift", "test", "--filter", test_filter],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        # Extract failure summary
        lines = result.stdout.split("\n") + result.stderr.split("\n")
        failures = [l for l in lines if "failed" in l.lower() or "error:" in l.lower()]
        output = "\n".join(failures[:15]) if failures else result.stdout[-500:]
        print(f"TESTS FAILED ({test_filter}):\n{output}")
        sys.exit(2)
    else:
        # Count passed tests
        for line in result.stdout.split("\n"):
            if "Executed" in line and "test" in line:
                print(f"Auto-test: {line.strip()}")
                break


if __name__ == "__main__":
    main()
