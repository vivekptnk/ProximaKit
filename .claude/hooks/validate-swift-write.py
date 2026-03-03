#!/usr/bin/env python3
"""validate-swift-write.py — PostToolUse hook for Write/Edit on .swift files
Checks:
  - swift build compiles
  - No force unwraps (!) in Sources/ (except baseAddress! for vDSP)
  - No manual float loops in Sources/ProximaKit/ (must use vDSP)
"""

import json
import sys
import subprocess
import re

def main():
    hook_input = json.loads(sys.stdin.read())
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    if tool_name not in ("Write", "Edit"):
        return

    file_path = tool_input.get("file_path", "")
    if not file_path.endswith(".swift"):
        return

    # Only validate Sources/ files strictly
    is_source = "/Sources/" in file_path
    errors = []

    if is_source:
        # Read the full file to check
        try:
            with open(file_path, "r") as f:
                content = f.read()
                lines = content.split("\n")
        except FileNotFoundError:
            return

        # Check for force unwraps (allow baseAddress! which is the vDSP pattern)
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("///"):
                continue
            # Find ! that are force unwraps (not != comparisons, not string interpolation, not optional chaining)
            # Allow baseAddress! (vDSP pattern) and .first! / .last! in tests
            if "!" in stripped:
                # Remove string literals
                cleaned = re.sub(r'"[^"]*"', '', stripped)
                # Remove != operators
                cleaned = cleaned.replace("!=", "")
                # Remove negation (!condition)
                cleaned = re.sub(r'!\w', '', cleaned)
                # Remove logical not
                cleaned = cleaned.replace("! ", "")
                # Allow baseAddress! (the vDSP C-interop pattern)
                cleaned = cleaned.replace("baseAddress!", "")
                if "!" in cleaned:
                    errors.append(f"WARNING: Possible force unwrap at {file_path}:{i}: {stripped.strip()}")

        # Check for manual float loops in core (must use vDSP)
        if "/Sources/ProximaKit/" in file_path:
            in_func = False
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("///"):
                    continue
                # Detect patterns like: for i in 0..<count { ... array[i] ... }
                # or: for element in floatArray { result += element * ... }
                if re.search(r'for\s+\w+\s+in\s+.*\{', stripped):
                    # Check next few lines for float array operations
                    context = "\n".join(lines[i:min(i+5, len(lines))])
                    if re.search(r'\[\w+\]\s*[\*\+\-/]=?\s', context):
                        errors.append(f"WARNING: Possible manual float loop at {file_path}:{i}. Use vDSP instead.")

    # Run swift build
    result = subprocess.run(
        ["swift", "build"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        # Extract just the error lines, not the full build log
        error_lines = [l for l in result.stderr.split("\n") if "error:" in l.lower()]
        if error_lines:
            errors.insert(0, "BUILD FAILED:\n" + "\n".join(error_lines[:10]))
        else:
            errors.insert(0, "BUILD FAILED:\n" + result.stderr[-500:])

    if errors:
        print("\n".join(errors))
        # Build failure is a hard block, warnings are soft
        if any("BUILD FAILED" in e for e in errors):
            sys.exit(2)


if __name__ == "__main__":
    main()
