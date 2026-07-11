"""Offline contract tests for the MCP GitHub response validators.

These use canned envelopes; they do not claim to mock the HTTP client, which
lives in the external ``mcp-methods`` crate. Keeping them outside the binary
smoke module ensures they run even when the server binary is not built.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SMOKE_PATH = Path(__file__).with_name("test_mcp_server_smoke.py")
_SPEC = importlib.util.spec_from_file_location("_kglite_mcp_smoke_contract", _SMOKE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SMOKE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SMOKE)
_validate_github_issues_response = _SMOKE._validate_github_issues_response
_validate_github_user_response = _SMOKE._validate_github_user_response


def test_github_api_canned_response_contract() -> None:
    result = {
        "content": [
            {
                "type": "text",
                "text": json.dumps({"login": "octocat", "id": 1, "html_url": "https://github.com/octocat"}),
            }
        ]
    }
    assert _validate_github_user_response(result)["login"] == "octocat"


def test_github_issues_canned_response_contract() -> None:
    result = {
        "content": [
            {
                "type": "text",
                "text": "#123 Fix parser regression\nhttps://github.com/rust-lang/rust/issues/123",
            }
        ]
    }
    assert "#123" in _validate_github_issues_response(result)
