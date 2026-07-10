# Security Policy

## Reporting a Vulnerability

Please report suspected security vulnerabilities **privately** — do not open a public issue.

Use GitHub Security Advisories: go to the [Security tab](https://github.com/vivekptnk/ProximaKit/security) of this repository and select **"Report a vulnerability"** (or go directly to [github.com/vivekptnk/ProximaKit/security/advisories/new](https://github.com/vivekptnk/ProximaKit/security/advisories/new)). This opens a private channel visible only to the maintainer and you, and lets us coordinate a fix and disclosure timeline before any details go public.

We'll acknowledge new reports as promptly as we can and keep you updated as we investigate.

## Supported Versions

Only the latest minor release line receives security fixes.

| Version | Supported |
| ------- | --------- |
| 1.8.x   | ✅        |
| < 1.8   | ❌        |

## Scope

ProximaKit is an on-device, pure-Swift vector search library — it has no network layer and no server component, so the realistic attack surface is narrower than a typical service. The most interesting class of report is a **malformed or adversarially-crafted on-disk file** (a `.pxkt` index snapshot, `.pxwal` write-ahead-log sidecar, `.pxbm` sparse-index snapshot, or a quantized-index file) that, when loaded, causes:

- a crash (including a Swift trap / precondition failure) instead of a typed `PersistenceError`,
- memory unsafety (out-of-bounds read/write, use-after-free) — for example via the memory-mapped read paths, or
- any other undefined behavior on load.

**Corruption and fuzzing reports are explicitly welcome.** The project already maintains an internal corruption-test matrix for these formats (e.g. `PersistenceCorruptionTests`, `PQHWFormatV3CorruptionTests`, `PagedOriginalsCorruptionTests`, `ScalarQuantizationPersistenceTests`) with the invariant that every corruption case must throw a typed error, never crash — a report that finds a gap in that invariant is a genuine security finding, not noise.

Out of scope: vulnerabilities that require an attacker to already have arbitrary code execution on the device, or that depend on a compromised/malicious host application deliberately misusing the API against itself.

## Disclosure

We'll credit reporters (unless you prefer to stay anonymous) once a fix ships and coordinate a disclosure timeline with you through the advisory thread.
