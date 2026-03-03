#!/usr/bin/env python3
"""prompt-context.py — UserPromptSubmit hook.
Cost-optimized model routing + context injection per MODEL_GUIDE.md.

Routing priority: cheapest model that can do the job.
  Haiku  ($0.01-0.03/session): Mechanical tasks, scaffolding, queries
  Sonnet ($1-3/session):       Feature work, tests, reviews, bug fixes
  Opus   ($5-15/session):      Algorithm design, ADRs, deep debugging
"""

import json
import os
import subprocess
import sys
import re

# ── Model Classification ──────────────────────────────────────────────
# Each task maps to the CHEAPEST model that handles it well.

# Opus-only: tasks that need deep multi-step reasoning
OPUS_PATTERNS = [
    # Algorithm design
    r"hnsw", r"algorithm", r"implement.*search", r"beam search",
    r"neighbor selection", r"graph traversal",
    # Architecture decisions
    r"architect", r"/adr\b", r"system design", r"redesign",
    r"tradeoff", r"trade-off",
    # Deep debugging
    r"why is recall", r"why is latency", r"bottleneck",
    r"debug.*complex", r"root cause",
    # Understanding
    r"/explain\b", r"/interview\b", r"paper", r"explain.*algorithm",
    # Performance deep-dive
    r"performance analysis", r"why.*slow", r"optimize.*hnsw",
]

# Haiku: mechanical tasks that don't need reasoning
HAIKU_PATTERNS = [
    # Scaffolding
    r"/scaffold\b", r"create.*skeleton", r"boilerplate", r"stub",
    # Mechanical edits
    r"/tidy\b", r"rename", r"move file", r"delete file",
    r"format", r"clean\s*up", r"cleanup",
    # Queries (reading, not reasoning)
    r"list files", r"show.*structure", r"what files",
    r"summarize", r"summary", r"how many",
    # Simple generation
    r"add.*import", r"add.*comment", r"add.*docstring",
    r"simple", r"quick",
]

# Slash commands → fixed model (from MODEL_GUIDE.md)
COMMAND_MODELS = {
    "/explain": "opus",
    "/interview": "opus",
    "/architect": "opus",
    "/adr": "opus",
    "/story": "sonnet",
    "/fix-issue": "sonnet",
    "/review": "sonnet",
    "/test": "sonnet",
    "/bench": "sonnet",
    "/ship": "sonnet",
    "/scaffold": "haiku",
    "/tidy": "haiku",
}

# Everything else → Sonnet (default, middle ground)


def classify_model(prompt_lower):
    """Returns (model, reason) using cheapest-first routing."""

    # 1. Slash commands have fixed mappings
    for cmd, model in COMMAND_MODELS.items():
        if prompt_lower.startswith(cmd):
            return model, f"Command {cmd} → {model}"

    # 2. Try Haiku first (cheapest)
    for pattern in HAIKU_PATTERNS:
        if re.search(pattern, prompt_lower):
            return "haiku", f"Mechanical task (matched: {pattern})"

    # 3. Check if it needs Opus (most expensive)
    for pattern in OPUS_PATTERNS:
        if re.search(pattern, prompt_lower):
            return "opus", f"Deep reasoning required (matched: {pattern})"

    # 4. Default to Sonnet
    return "sonnet", "Standard development task"


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
        prompt = hook_input.get("user_prompt", hook_input.get("content", ""))
    except (json.JSONDecodeError, KeyError):
        prompt = ""

    if not prompt:
        return

    prompt_lower = prompt.lower().strip()
    lines = []

    # ── Branch & Story Context ────────────────────────────────────────

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

    # Uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    )
    changed_files = [l for l in result.stdout.strip().split("\n") if l.strip()]
    if changed_files:
        lines.append(f"Uncommitted changes: {len(changed_files)} files")

    # ── Cost-Optimized Model Routing ──────────────────────────────────

    model, reason = classify_model(prompt_lower)

    cost_hint = {
        "haiku":  "~$0.01-0.03",
        "sonnet": "~$1-3",
        "opus":   "~$5-15",
    }

    # Always show the recommendation with cost context
    if model == "opus":
        lines.append(f"Model: OPUS recommended ({cost_hint['opus']}/session) — {reason}")
        lines.append("  Switch: claude --model opus")
    elif model == "haiku":
        lines.append(f"Model: HAIKU recommended ({cost_hint['haiku']}/session) — {reason}")
        lines.append("  Switch: claude --model haiku")
    # Don't clutter for sonnet (it's the default)

    # ── Dependency Guard ──────────────────────────────────────────────

    dep_keywords = ["add dependency", "add package", "install", "spm add", "pod"]
    if any(kw in prompt_lower for kw in dep_keywords):
        lines.append("BLOCKED: ProximaKit is zero-dependency. Apple frameworks only.")

    # ── Relevant Docs ─────────────────────────────────────────────────

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", ".")

    DOC_HINTS = [
        (["vector", "vdsp", "accelerate", "simd", "distance", "dot product", "cosine", "l2", "magnitude"],
         "docs/adr/ADR-001-accelerate-for-math.md", "Accelerate patterns"),
        (["actor", "concurrency", "sendable", "thread", "async", "await"],
         "docs/adr/ADR-002-actor-isolation.md", "Actor isolation"),
        (["persist", "save", "load", "mmap", "memory-map", "disk", "binary", "cold start"],
         "docs/adr/ADR-003-binary-persistence.md", "Binary persistence"),
        (["hnsw", "neighbor", "heuristic", "graph", "nsw", "beam", "greedy"],
         "docs/adr/ADR-004-hnsw-heuristic-selection.md", "HNSW heuristic"),
        (["architecture", "module", "layer", "boundary", "dependency direction"],
         "docs/ARCHITECTURE.md", "Architecture"),
        (["benchmark", "perf", "latency", "recall", "throughput", "p50", "p95", "p99"],
         "docs/BENCHMARKS.md", "Benchmarks"),
        (["story", "pk-", "epic", "prd", "task", "acceptance"],
         "docs/PRD.md", "PRD"),
    ]

    relevant_docs = []
    for keywords, doc_path, label in DOC_HINTS:
        if any(kw in prompt_lower for kw in keywords):
            full_path = os.path.join(project_dir, doc_path)
            if os.path.exists(full_path):
                relevant_docs.append(f"  {doc_path} ({label})")

    if relevant_docs:
        lines.append("Read before proceeding:")
        lines.extend(relevant_docs)

    if lines:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
