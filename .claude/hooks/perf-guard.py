#!/usr/bin/env python3
"""perf-guard.py — PostToolUse hook for Write/Edit.
Reminds to benchmark when performance-critical files change.
"""

import json
import sys

PERF_CRITICAL_PATHS = [
    "/Distance/",
    "/Index/",
    "Vector.swift",
    "Persistence",
]

def main():
    hook_input = json.loads(sys.stdin.read())
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    if tool_name not in ("Write", "Edit"):
        return

    file_path = tool_input.get("file_path", "")
    if "/Sources/" not in file_path:
        return

    for pattern in PERF_CRITICAL_PATHS:
        if pattern in file_path:
            print(f"PERF REMINDER: You modified a performance-critical file ({file_path.split('/Sources/')[-1]}).")
            print("Run benchmarks before committing: swift test --filter Benchmark")
            return


if __name__ == "__main__":
    main()
