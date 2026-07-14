"""Contracts for temporary RustSec advisory exceptions."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_rustsec_advisories.py"


def test_committed_rustsec_exception_policy_is_current() -> None:
    subprocess.run([sys.executable, CHECKER, "--policy-only"], cwd=ROOT, check=True)


def test_expired_rustsec_exception_is_rejected(tmp_path: Path) -> None:
    policy = tmp_path / "rustsec.json"
    policy.write_text(
        json.dumps(
            {
                "ignored": [
                    {
                        "id": "RUSTSEC-2000-0001",
                        "reason": "A concrete but deliberately expired test justification.",
                        "reviewed": "2000-01-01",
                        "expires": "2000-01-02",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, CHECKER, "--policy", policy, "--policy-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "expired on 2000-01-02" in result.stdout
