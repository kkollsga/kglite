# Platform and artifact support

KGLite's Python distribution is a native PyO3 extension. Support claims are
tiered by evidence: running a test suite is stronger than producing an
artifact, and producing an artifact is stronger than a plausible source build.
The active workflows and generated [project facts](../_generated/project-facts.md)
are the machine-readable authority.

## CPython and wheel policy

Published extension wheels use CPython's stable ABI with a Python 3.10 floor
(`cp310-abi3`). One platform wheel can therefore serve CPython 3.10 and newer
on the same OS/architecture/libc target. Normal installation is:

```bash
pip install kglite
```

The base wheel has no required Python packages. Optional integrations declare
their own dependencies: `kglite[pandas]` for DataFrame workflows,
`kglite[networkx]` for the NetworkX bridge (including pandas), and
`kglite[neo4j]` for the Neo4j driver.

## Evidence tiers

### Runtime-tested paths

- Linux x86_64 source builds on CPython 3.10, 3.12, 3.13, and 3.14 run the
  Python suite.
- Linux x86_64 CPython 3.14t builds without abi3 and runs the dedicated
  free-threading concurrency suite. This proves a source-build configuration;
  a free-threaded wheel is not currently published.
- A Linux x86_64 CPython 3.12 job builds a wheel, installs its `networkx` extra
  into a clean environment outside the checkout, and executes a bridge
  round-trip.
- At release time, the macOS arm64 wheel is installed on CPython 3.14 with
  current pyarrow for the allocator-coexistence canary.

### Release-blocking wheel builds

These artifacts must build before publication can proceed:

| Target | Compatibility floor |
|---|---|
| `x86_64-pc-windows-msvc` | 64-bit Windows |
| `aarch64-apple-darwin` | macOS 11+ arm64 |
| `x86_64-apple-darwin` | macOS 11+ x86_64 |
| `x86_64-unknown-linux-gnu` | manylinux2014 / glibc 2.17+ |
| `x86_64-unknown-linux-musl` | musllinux 1.2+ |

A release-blocking build is not the same as a full runtime test on that target.
Windows and cross-built macOS x86_64 artifacts are build-verified unless a
separate smoke job is listed above.

### Best-effort wheel builds

Linux aarch64 artifacts are published when their cross-build succeeds, but the
job is explicitly non-blocking:

| Target | Compatibility floor |
|---|---|
| `aarch64-unknown-linux-gnu` | manylinux 2.28 / glibc 2.28+ |
| `aarch64-unknown-linux-musl` | musllinux 1.2+ |

Do not plan a deployment around a best-effort wheel without checking that the
desired release actually contains it.

## PyPy

PyPy is not a supported published-artifact target. The released macOS wheel,
for example, is tagged `cp310-abi3-macosx_11_0_arm64`; a probe with PyPy 3.10
rejects it because PyPy requires a `pp310`-compatible artifact. The project
therefore does not publish the PyPy classifier. A future PyPy claim requires a
dedicated build and runtime test, not only a source-level PyO3 capability.

## Wheel-first distribution; no supported sdist

PyPI publication uploads platform wheels. It does not currently publish or
test a source distribution, so pip has no supported PyPI sdist fallback when a
wheel is unavailable.

Developers on another target can try a source checkout with a Rust toolchain:

```bash
git clone https://github.com/kkollsga/kglite.git
cd kglite
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```

That is a source-build route, not a promise that an unlisted platform is
release-tested. Rust-only consumers should depend on the `kglite` crate and do
not need the Python extension.

## Bundled entry points

The same wheel installs both `kglite` and `kglite-mcp-server`. Each console
script is a thin Python shim over its bundled Rust library inside the extension,
sharing one graph engine. There is no second CLI/server wheel dependency; the
standalone `kglite-cli` distribution is an alternative for CLI-only users.
