# ProximaKit Asset Design System

Apple-grade discipline. Every SVG asset in this directory follows these rules.

## Canvas
- Pure black `#000000`, corner radius 28 (logo) / 20 (diagrams). Generous margins (>= 32px).

## Color (Apple system palette, dark)
- Primary text/figures: `#f5f5f7` · Secondary: `#86868b` · Hairlines: `#2c2c2e` · Inactive nodes: `#48484a`
- Accent (one per asset unless semantics demand two): blue `#0A84FF`.
  Semantic seconds only: green `#30D158` (keyword/BM25), purple `#BF5AF2` (fusion/quantization).

## Typography
- NO `<text>` elements with font-family stacks — they render with the viewer's fonts.
  ALL labels are genuine SF outlines generated via the CoreText tool (`tools/outline.swift`,
  regenerate with: `swift tools/outline.swift "<text>" <size> <semibold|regular> <x> <baselineY>`).
- Sizes: titles 20 semibold, labels 13 regular, captions 12 regular (`#86868b`).

## Motion
- ONE primary animation per asset. 6–9s loop, `cubic-bezier(0.4, 0, 0.2, 1)`.
- Allowed: opacity fades, SMIL `animateMotion` along a path, gentle scale (0.92–1.0).
- Forbidden: blinking, dash-offset storms, parallel competing animations, anything under 2s.
- Traveling elements use SMIL (`animateMotion` + `keyTimes`), not CSS `offset-path` (compatibility).

### Sanctioned exception: terminal session bodies
Long monospace session transcripts may use `<text>` with a pure monospace stack
(`ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace`) when
outlining would exceed ~200 KB. Chrome labels (titles, captions) are still SF
outlines. Declare the exception in the asset's PR/report.

## Hygiene
- `xmllint --noout` must pass (a duplicate attribute once shipped a broken image).
- Render with `qlmanage -t -s 880 -o /tmp <file>` and INSPECT the PNG before calling it done.
- One idea per asset. If a label can be removed, remove it.
