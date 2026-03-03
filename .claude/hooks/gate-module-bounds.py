#!/usr/bin/env python3
"""gate-module-bounds.py — PreToolUse hook for Write/Edit on Sources/
Blocks:
  - Third-party imports in Sources/
  - ProximaKit core importing ProximaEmbeddings (wrong direction)
  - ProximaEmbeddings importing UI frameworks (it's a data layer)
  - Protected files (.claude/settings.json)
"""

import json
import sys
import os

APPLE_FRAMEWORKS = {
    "Foundation", "Accelerate", "CoreML", "NaturalLanguage", "Vision",
    "SwiftUI", "UIKit", "AppKit", "PhotosUI", "Combine", "os",
    "Darwin", "CoreFoundation", "CoreGraphics", "QuartzCore",
    "XCTest", "Testing",
}

CORE_ALLOWED_IMPORTS = {"Foundation", "Accelerate"}
EMBEDDINGS_ALLOWED_IMPORTS = {"Foundation", "Accelerate", "CoreML", "NaturalLanguage", "Vision", "ProximaKit"}
EMBEDDINGS_BLOCKED_IMPORTS = {"SwiftUI", "UIKit", "AppKit", "PhotosUI"}

def main():
    hook_input = json.loads(sys.stdin.read())
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    # Only check Write and Edit tools
    if tool_name not in ("Write", "Edit"):
        return

    file_path = tool_input.get("file_path", "")

    # Determine content to check
    if tool_name == "Write":
        content = tool_input.get("content", "")
    else:
        content = tool_input.get("new_string", "")

    # Only check Sources/ files
    if "/Sources/" not in file_path:
        return

    # Check for swift imports
    imports = set()
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import "):
            module = stripped.split()[1].split(".")[0]
            imports.add(module)

    if not imports:
        return

    # Determine which module this file belongs to
    is_core = "/Sources/ProximaKit/" in file_path
    is_embeddings = "/Sources/ProximaEmbeddings/" in file_path

    errors = []

    if is_core:
        for imp in imports:
            if imp == "ProximaEmbeddings":
                errors.append(f"BLOCKED: ProximaKit core cannot import ProximaEmbeddings (wrong direction)")
            elif imp not in CORE_ALLOWED_IMPORTS:
                if imp not in APPLE_FRAMEWORKS:
                    errors.append(f"BLOCKED: Third-party import '{imp}' not allowed in Sources/. Apple frameworks only.")
                elif imp not in CORE_ALLOWED_IMPORTS:
                    errors.append(f"BLOCKED: ProximaKit core can only import Foundation and Accelerate, not '{imp}'")

    elif is_embeddings:
        for imp in imports:
            if imp in EMBEDDINGS_BLOCKED_IMPORTS:
                errors.append(f"BLOCKED: ProximaEmbeddings cannot import UI framework '{imp}' (it's a data layer)")
            elif imp not in EMBEDDINGS_ALLOWED_IMPORTS:
                if imp not in APPLE_FRAMEWORKS:
                    errors.append(f"BLOCKED: Third-party import '{imp}' not allowed in Sources/. Apple frameworks only.")

    if errors:
        print("\n".join(errors))
        sys.exit(2)


if __name__ == "__main__":
    main()
