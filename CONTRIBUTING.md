# Contributing to KGLite

## Development Setup

```bash
# Clone and set up
git clone https://github.com/kkollsga/kglite.git
cd kglite
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest pandas

# Build the Rust extension into the venv
make dev          # or: maturin develop

# Run tests
make test         # Rust + Python
make test-rust    # Rust only
make test-py      # Python only
```

The code tree parser ships with the default install — the
tree-sitter grammars are bundled into the Rust extension, so a plain
`pip install .` is all you need to work on it.

## Project Structure

```
src/                          # Rust core (PyO3 bindings)
  graph/
    mod.rs                    # KnowledgeGraph struct + Python methods
    schema.rs                 # DirGraph, NodeData, EdgeData
    cypher/                   # Cypher query engine (parser, AST, executor)
    introspection.rs          # describe() output
    ...
kglite/                       # Python package
  __init__.pyi                # Type stubs for the Rust extension
  code_tree/                  # Tree-sitter based code parser
    builder.py                # Graph construction from parsed results
    parsers/                  # Language-specific parsers (rust.py, python.py, ...)
examples/
  mcp_server.py               # MCP server exposing the graph to AI agents
tests/                        # Python test suite
Cargo.toml                    # Rust dependencies + version (single source of truth)
pyproject.toml                # Python packaging (version is dynamic from Cargo.toml)
```

## Making Changes

1. **Create a branch** from `main`
2. **Make your changes** — prefer editing existing files over creating new ones
3. **Run tests** — `make test` (or at minimum `make check` for a quick compile check)
4. **Update the changelog** — see below
5. **Commit** with a descriptive message — see below
6. **Open a PR** against `main`

## Commit Messages

Use this format:

```
type: short description

Optional longer explanation.
```

Types:

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or updating tests |
| `chore` | Build, CI, dependency updates |

Examples:

```
feat: add toc() method for file table of contents
fix: resolve qualified_name lookup for structs with :: separator
docs: update README code tree examples
refactor: extract resolve_code_entity helper from find/source/context
```

This is a convention, not enforced by tooling. The goal is readable git history.

## Updating the Changelog

**Rule**: if your change is visible to users (new feature, bug fix, breaking change, deprecation), add an entry to the `[Unreleased]` section of `CHANGELOG.md` before committing.

Use the appropriate category:

- **Added** — new features or capabilities
- **Changed** — changes to existing functionality
- **Fixed** — bug fixes
- **Removed** — removed features
- **Deprecated** — features that will be removed in a future version
- **Security** — vulnerability fixes

You do **not** need a changelog entry for:

- Internal refactors with no user-visible effect
- CI/CD pipeline changes
- Test-only changes
- Code style / formatting fixes

## Release Process

1. Update the version in `Cargo.toml` (this is the single source of truth — `pyproject.toml` reads it dynamically)
2. Move the `[Unreleased]` section in `CHANGELOG.md` to a new version heading:
   ```markdown
   ## [0.5.33] - 2025-06-01

   ### Added
   - ...
   ```
3. Add a new empty `[Unreleased]` section at the top
4. Update the comparison links at the bottom of `CHANGELOG.md`
5. Commit: `chore: release v0.5.33`
6. Push to `main` — CI will build wheels, publish to PyPI, and create a GitHub Release

## Code Style

- **Rust**: `cargo fmt` and `cargo clippy -- -D warnings` must pass (enforced by CI)
- **Python**: no enforced formatter currently — follow existing code style
- **Type stubs**: update `kglite/__init__.pyi` when adding or changing Python-facing methods
