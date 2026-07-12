# Dependency license policy

KGLite's source code and independently authored Cypher contract cases are MIT
licensed. The project does not copy, translate, vendor, execute, or derive its
tests from the Apache-licensed openCypher TCK. Compatibility behavior is
implemented from public language descriptions and independently written cases.

That clean-room boundary is separate from ordinary software dependencies.
KGLite retains several reviewed permissive dependencies whose own metadata is
Apache-2.0 or MPL-2.0, including the MCP protocol implementation. Their use
does not change KGLite's MIT license, but their notices and license metadata
must remain visible in distributed SBOMs and source packages.

`python scripts/check_dependency_licenses.py` audits the locked, all-feature
Cargo graph offline. It fails on missing metadata, unknown license expressions,
strong copyleft/non-commercial terms without an independently selectable MIT
branch, new Apache/MPL-only packages, stale reviewed exceptions, non-SPDX
Python metadata, or missing KGLite LICENSE files in publishable crate roots.
The reviewed package list lives in
`tests/api-baselines/dependency-licenses.json`; changes require an explicit
license review rather than an automatic refresh.
