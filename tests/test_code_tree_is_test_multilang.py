"""Cross-language is_test detection.

0.9.37: replaced the loose `rel_path.to_lowercase().contains("test")`
check in HTML / CSS / Swift / PHP with a path-segment-aware helper. The
old check false-positived on names like `latest.html`, `contest.css`,
and `protest.swift`. TypeScript also gained `/test/` and `/tests/`
directory recognition (previously only `__tests__/` was honoured).

These tests pin both the new positive cases and the regressions we
deliberately stopped flagging.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path: pathlib.Path, rel: str, content: str) -> pathlib.Path:
    fp = tmp_path / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    return fp


def _file_is_test(g) -> dict[str, bool]:
    rows = g.cypher("MATCH (f:File) RETURN f.path AS path, f.is_test AS is_test").to_list()
    return {r["path"]: bool(r["is_test"]) for r in rows}


class TestTypeScript:
    def test_tests_directory_marks_is_test(self, tmp_path):
        _write(tmp_path, "src/app.ts", "export function main() {}\n")
        _write(tmp_path, "tests/app.test.ts", "import {main} from '../src/app'; test('x', () => main());\n")
        _write(tmp_path, "test/legacy.ts", "// legacy mocha\n")
        flags = _file_is_test(build(str(tmp_path)))
        assert flags["src/app.ts"] is False
        assert flags["tests/app.test.ts"] is True
        assert flags["test/legacy.ts"] is True

    def test_suffix_and_underscored_dir_still_work(self, tmp_path):
        _write(tmp_path, "lib/foo.spec.ts", "// vitest\n")
        _write(tmp_path, "__tests__/bar.ts", "// jest\n")
        flags = _file_is_test(build(str(tmp_path)))
        assert flags["lib/foo.spec.ts"] is True
        assert flags["__tests__/bar.ts"] is True

    def test_latest_substring_is_not_test(self, tmp_path):
        # Regression: pre-0.9.37 the html/css/swift/php parsers used a
        # loose substring check. TS already segmented, but keep this
        # case pinned so the new shared helper inherits the guarantee.
        _write(tmp_path, "src/latest_release.ts", "export const VERSION = '1.0';\n")
        _write(tmp_path, "src/contest.ts", "export function enter() {}\n")
        flags = _file_is_test(build(str(tmp_path)))
        assert flags["src/latest_release.ts"] is False
        assert flags["src/contest.ts"] is False


class TestHtmlCss:
    def test_html_in_test_dir(self, tmp_path):
        _write(tmp_path, "site/index.html", "<!doctype html><html><body><h1>Hi</h1></body></html>")
        _write(tmp_path, "tests/fixture.html", "<!doctype html><html><body><h1>T</h1></body></html>")
        flags = _file_is_test(build(str(tmp_path)))
        assert flags["site/index.html"] is False
        assert flags["tests/fixture.html"] is True

    def test_latest_html_no_longer_test(self, tmp_path):
        # The pre-0.9.37 check would have flagged this as a test because
        # "latest" contains "test" as a substring. Strict segment check
        # now rejects it.
        _write(tmp_path, "site/latest.html", "<!doctype html><html><body></body></html>")
        _write(tmp_path, "site/protest_page.html", "<!doctype html><html><body></body></html>")
        flags = _file_is_test(build(str(tmp_path)))
        assert flags["site/latest.html"] is False
        assert flags["site/protest_page.html"] is False

    def test_css_contest_no_longer_test(self, tmp_path):
        _write(tmp_path, "styles/main.css", ".btn { color: red; }\n")
        _write(tmp_path, "styles/contest.css", ".contest { display: block; }\n")
        _write(tmp_path, "tests/snapshots.css", ".snap { color: green; }\n")
        flags = _file_is_test(build(str(tmp_path)))
        assert flags["styles/main.css"] is False
        assert flags["styles/contest.css"] is False
        assert flags["tests/snapshots.css"] is True


class TestSwiftPhp:
    def test_swift_tests_suffix(self, tmp_path):
        _write(tmp_path, "Sources/Foo.swift", "public func main() {}\n")
        _write(tmp_path, "Tests/FooTests.swift", "import XCTest; class FooTests {}\n")
        _write(tmp_path, "Sources/LatestRelease.swift", "public func go() {}\n")
        flags = _file_is_test(build(str(tmp_path)))
        assert flags["Sources/Foo.swift"] is False
        assert flags["Tests/FooTests.swift"] is True
        # Regression: pre-0.9.37 this was True due to substring match.
        assert flags["Sources/LatestRelease.swift"] is False

    def test_php_test_suffix(self, tmp_path):
        _write(tmp_path, "src/UserService.php", "<?php\nclass UserService {}\n")
        _write(tmp_path, "tests/UserServiceTest.php", "<?php\nclass UserServiceTest {}\n")
        _write(tmp_path, "src/LatestRelease.php", "<?php\nclass LatestRelease {}\n")
        flags = _file_is_test(build(str(tmp_path)))
        assert flags["src/UserService.php"] is False
        assert flags["tests/UserServiceTest.php"] is True
        # Regression: pre-0.9.37 this was True due to substring match.
        assert flags["src/LatestRelease.php"] is False
