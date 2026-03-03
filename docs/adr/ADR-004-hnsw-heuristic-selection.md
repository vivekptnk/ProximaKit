# ADR-004: HNSW Heuristic Neighbor Selection

## Status
Accepted

## Context
When inserting: simple nearest-M selection vs heuristic diverse selection.

## Decision
Use heuristic selection (Algorithm 4 from the HNSW paper).

## Rationale
Simple selection creates locally optimal but globally poor graphs. On clustered data: 88% recall. Heuristic: only add candidate if closer to target than to any already-selected neighbor. Creates diverse long-range edges. Clustered recall: 96%.

## Consequences
- O(M²) per insertion instead of O(M log M). Acceptable because M is small (16).
- More complex to implement and test
- Must benchmark with both uniform and clustered distributions
