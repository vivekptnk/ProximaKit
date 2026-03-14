#!/usr/bin/env python3
"""validate-swift-write.py — PostToolUse hook for Write/Edit on .swift files
Checks:
  - No force unwraps (!) in Sources/ (except baseAddress! for vDSP)
  - No manual float loops in Sources/ProximaKit/ (must use vDSP)

Note: swift build is NOT run here — it's slow and auto-test.py already
compiles the code. This hook only does static checks.
"""

import json
import sys
import re

def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, Exception):
        return

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    if tool_name not in ("Write", "Edit"):
        return

    file_path = tool_input.get("file_path", "")
    if not file_path.endswith(".swift"):
        return
    if "/Sources/" not in file_path:
        return

    try:
        with open(file_path, "r") as f:
            lines = f.read().split("\n")
    except FileNotFoundError:
        return

    warnings = []

    # Check for force unwraps (allow baseAddress! which is the vDSP pattern)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("///"):
            continue
        if "!" in stripped:
            cleaned = re.sub(r'"[^"]*"', '', stripped)
            cleaned = cleaned.replace("!=", "")
            cleaned = re.sub(r'!\w', '', cleaned)
            cleaned = cleaned.replace("! ", "")
            cleaned = cleaned.replace("baseAddress!", "")
            if "!" in cleaned:
                warnings.append(f"WARNING: Possible force unwrap at line {i}: {stripped.strip()}")

    # Check for manual float loops in core (must use vDSP)
    if "/Sources/ProximaKit/" in file_path:
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("///"):
                continue
            if re.search(r'for\s+\w+\s+in\s+.*\{', stripped):
                context = "\n".join(lines[i:min(i + 5, len(lines))])
                if re.search(r'\[\w+\]\s*[\*\+\-/]=?\s', context):
                    warnings.append(f"WARNING: Possible manual float loop at line {i}. Use vDSP instead.")

    if warnings:
        print("\n".join(warnings), file=sys.stderr)


if __name__ == "__main__":
    main()
