# Platform and artifact support

KGLite's Python distribution is a native PyO3 extension. Support claims are tiered by evidence:

1. Running a test suite is the strongest evidence.
2. Producing an artifact is weaker.
3. A plausible source build is weaker still.

The active workflows and generated [project facts](../_generated/project-facts.md)
are the machine-readable authority.

## CPython and wheel policy

Published extension wheels use CPython's stable ABI with a Python 3.12 floor
(`cp312-abi3`). One platform wheel therefore serves CPython 3.12 and newer on
the same OS/architecture/libc target. That wheel does not load on a
free-threaded interpreter.

Free-threaded CPython 3.15 and newer use a separate wheel (`cp315-abi3.abi3t`,
PEP 803). It loads on free-threaded and on GIL-enabled 3.15 and newer. On 3.15,
`pip` may therefore choose it over the `cp312-abi3` wheel. Free-threaded 3.14t
has no wheel: build from source. Normal installation is:

```bash
pip install kglite
```

The base wheel has no required Python packages. Optional integrations declare
their own dependencies:

- `kglite[pandas]` for DataFrame workflows
- `kglite[networkx]` for the NetworkX bridge (including pandas)
- `kglite[neo4j]` for the Neo4j driver

## Evidence tiers

### Runtime-tested paths

- Linux x86_64 source builds on CPython 3.12, 3.13, 3.14, and 3.15 run the
  Python suite.
- Linux x86_64 CPython 3.14t and 3.15t build without abi3 and run the dedicated
  free-threading concurrency suite. This proves a source-build configuration.
- A Linux x86_64 job builds the `cp315-abi3.abi3t` wheel with CPython 3.15t.
  It installs the wheel outside the checkout and runs the free-threading,
  concurrency and session suites against it. The GIL must stay off.
- A Linux x86_64 CPython 3.12 job builds a wheel, installs its `networkx` extra
  into a clean environment outside the checkout, and executes a bridge
  round-trip.
- At release time, the macOS arm64 wheel is installed on CPython 3.14 with
  current pyarrow for the allocator-coexistence canary.
- Windows x86_64 and macOS arm64 run the Rust engine's unit suite. The suite
  covers the storage backends, including disk mode, whose memory-mapped file
  lifetimes behave differently on Windows than on POSIX. This is a
  native-engine test, not a Python-level one. The Python API surface on Windows
  remains build-verified rather than runtime-tested.

### Release-blocking wheel builds

These artifacts must build before publication can proceed:

| Target | Compatibility floor |
|---|---|
| `x86_64-pc-windows-msvc` | 64-bit Windows |
| `aarch64-apple-darwin` | macOS 11+ arm64 |
| `x86_64-apple-darwin` | macOS 11+ x86_64 |
| `x86_64-unknown-linux-gnu` | manylinux2014 / glibc 2.17+ |
| `x86_64-unknown-linux-musl` | musllinux 1.2+ |

The free-threaded wheel (`cp315-abi3.abi3t`) is release-blocking on three
targets. Each is installed on free-threaded 3.15 and smoke-tested before upload:

| Target | Compatibility floor |
|---|---|
| `x86_64-pc-windows-msvc` | 64-bit Windows |
| `aarch64-apple-darwin` | macOS 11+ arm64 |
| `x86_64-unknown-linux-gnu` | manylinux_2_28 / glibc 2.28+ |

macOS x86_64 is cross-built on an arm64 runner with no x86_64 free-threaded
interpreter. The musl and aarch64 Linux targets are not built for this wheel.
The publish job fails when any of these, or any abi3 wheel above, is missing
(`scripts/check_wheel_set.py`).

A release-blocking build is not a full runtime test on that target. Windows and
cross-built macOS x86_64 artifacts are build-verified unless a separate smoke
job is listed above.

### Best-effort wheel builds

Linux aarch64 artifacts are published when their cross-build succeeds. The job
is explicitly non-blocking:

| Target | Compatibility floor |
|---|---|
| `aarch64-unknown-linux-gnu` | manylinux 2.28 / glibc 2.28+ |
| `aarch64-unknown-linux-musl` | musllinux 1.2+ |

Do not plan a deployment around a best-effort wheel without checking that the
desired release actually contains it.

## PyPy

PyPy is not a supported published-artifact target. The released macOS wheel,
for example, is tagged `cp312-abi3-macosx_11_0_arm64`. A probe with PyPy 3.12
rejects it, because PyPy requires a `pp312`-compatible artifact. The project
therefore does not publish the PyPy classifier. A future PyPy claim requires a
dedicated build and runtime test, not only a source-level PyO3 capability.

## Wheel-first distribution, with a tested source fallback

PyPI publication uploads platform wheels for the targets listed above, and a
source distribution for everything else. When pip finds no matching wheel, it
falls back to the sdist and builds from source. That requires a Rust toolchain
on the installing machine.

The sdist is not a courtesy upload. Every release does the following with it:

1. Builds it.
2. Unpacks it into an empty directory, to confirm it resolves with no sibling
   checkouts present.
3. Runs `pip install` on the tarball and imports the result.

A source fallback that cannot build is worse than none. It turns a clear "no
matching distribution" into a compile error deep in someone else's install log.

This does *not* promise that an unlisted platform is release-tested. The sdist
is verified to build on the CI runner. It is not exercised on every target it
might reach. Wheels remain the supported path, and a platform in the table
above has been built and smoke-tested there.

Developers can also work from a source checkout with a Rust toolchain:

```bash
git clone https://github.com/kkollsga/kglite.git
cd kglite
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```

That is a source-build route, not a promise that an unlisted platform is
release-tested. Rust-only consumers depend on the `kglite` crate and do not
need the Python extension.

## Bundled entry points

The same wheel installs both `kglite` and `kglite-mcp-server`. Each console
script is a thin Python shim over its bundled Rust library inside the
extension, and both share one graph engine. There is no second CLI/server wheel
dependency. The standalone `kglite-cli` distribution is an alternative for
CLI-only users.
