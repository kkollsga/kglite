#!/usr/bin/env python3
"""Wait for the four push-triggered release workflows to finish, and say
whether they PASSED — not merely that they completed.

Two recorded failures shape this script:

1. **The `--commit` query silently returns nothing.** During the 0.15.0
   release `gh run list --commit <sha>` reported `runs=0` for a full
   hour while all four workflows on that exact SHA were green;
   `gh run list --branch main` showed them immediately. An earlier note
   records the same filter returning `[]` for 5-10 s right after a push.
   So the primary query here is **by branch**, with a client-side
   `head_sha` filter, and the head_sha-filtered query is only a
   secondary source unioned in. If the two disagree, the branch query
   wins and the script says so.

2. **A monitor that reports `status` is not a monitor.** A previous
   poller announced "completed" without saying whether anything passed.
   Every line this script prints carries the `conclusion`, and a
   non-`success` conclusion is a non-zero exit.

Two structural requirements follow from those:

* **Require the expected run count before concluding anything.** A naive
  "no incomplete runs left" loop exits instantly and green on an empty
  array — which is exactly what the `runs=0` hour would have produced.
  Nothing is decided until all `--expect` workflow names are present.
* **A timeout is a failure.** Running out of wall-clock exits non-zero
  and names what was still missing or pending. It never degrades to
  success.

The script is read-only: a single `gh api` GET per poll. It never
reruns, cancels, approves, or comments.

Usage:
    python scripts/wait_for_release_ci.py                    # HEAD, on main
    python scripts/wait_for_release_ci.py --sha <sha> --timeout 3600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]

# The four workflows a push to `main` triggers for a release. Names must
# match `.github/workflows/*.yml` `name:` exactly.
RELEASE_WORKFLOWS = (
    "CI",
    "Publish to crates.io",
    "Build and Publish Python Wheels",
    "Build & Publish kglite-cli wheels",
)

TERMINAL_STATUS = "completed"


#: Empty polls tolerated before probing whether the API is reachable at all.
EMPTY_POLLS_BEFORE_PROBE = 3


class PollError(RuntimeError):
    """The poll cannot proceed, or finished unsuccessfully."""


def git_output(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PollError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def gh_api(path: str) -> dict:
    proc = subprocess.run(["gh", "api", path], cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PollError(f"gh api {path} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


def fetch_runs(repo: str, branch: str, sha: str) -> tuple[list[dict], str]:
    """Return the workflow runs for `sha`, and which query found them.

    Branch-scoped first: it is the query that demonstrably works when the
    head_sha filter goes blind. The head_sha query is unioned in so a run
    on a SHA that is no longer branch-tip is still seen.
    """
    by_branch = [
        run
        for run in gh_api(f"repos/{repo}/actions/runs?branch={branch}&per_page=100").get("workflow_runs", [])
        if run.get("head_sha") == sha
    ]
    by_sha = gh_api(f"repos/{repo}/actions/runs?head_sha={sha}&per_page=100").get("workflow_runs", [])

    merged: dict[int, dict] = {run["id"]: run for run in by_sha}
    merged.update({run["id"]: run for run in by_branch})
    source = f"branch={branch} matched {len(by_branch)}, head_sha matched {len(by_sha)}"
    return list(merged.values()), source


def api_is_reachable(repo: str) -> tuple[bool, str]:
    """Whether the Actions API answers at all, independent of this SHA.

    `fetch_runs` returning nothing is ambiguous: the workflows may not have
    registered yet, or we may be unable to see them. Those want opposite
    responses -- keep waiting versus stop and say so -- and a poll loop that
    cannot tell them apart will report "no runs" right up to its timeout
    while the real problem is that nothing can reach GitHub.

    Observed 2026-07-27: local ephemeral-port exhaustion made every outbound
    connection fail with EADDRNOTAVAIL while a release was in flight.
    """
    try:
        gh_api(f"repos/{repo}/actions/runs?per_page=1")
    except PollError as exc:
        return False, str(exc)
    return True, "reachable"


def latest_per_workflow(runs: list[dict], expected: tuple[str, ...]) -> dict[str, dict]:
    """Newest run per expected workflow name (a rerun supersedes)."""
    newest: dict[str, dict] = {}
    for run in runs:
        name = run.get("name")
        if name not in expected:
            continue
        current = newest.get(name)
        if current is None or run.get("run_number", 0) >= current.get("run_number", 0):
            newest[name] = run
    return newest


def evaluate(runs: list[dict], expected: tuple[str, ...]) -> tuple[bool, list[str], list[str], list[str]]:
    """(all_terminal, missing, pending, failed).

    `all_terminal` is False whenever any expected workflow is missing —
    an absent run is *unknown*, never *fine*. This is the guard against
    the empty-array instant-green bug.
    """
    newest = latest_per_workflow(runs, expected)
    missing = [name for name in expected if name not in newest]
    pending = [name for name, run in newest.items() if run.get("status") != TERMINAL_STATUS]
    failed = [
        f"{name}: {run.get('conclusion')}"
        for name, run in newest.items()
        if run.get("status") == TERMINAL_STATUS and run.get("conclusion") != "success"
    ]
    return (not missing and not pending), missing, pending, failed


def describe(runs: list[dict], expected: tuple[str, ...]) -> str:
    newest = latest_per_workflow(runs, expected)
    lines = []
    for name in expected:
        run = newest.get(name)
        if run is None:
            lines.append(f"    {name:<34}  (no run seen)")
        else:
            status = run.get("status")
            conclusion = run.get("conclusion") or "-"
            lines.append(f"    {name:<34}  status={status} conclusion={conclusion}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sha", help="Commit to wait on (default: HEAD).")
    parser.add_argument("--branch", default="main", help="Branch the push landed on (default: main).")
    parser.add_argument("--repo", help="owner/name (default: the `origin` remote).")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds before giving up (default 3600).")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between polls (default 30).")
    parser.add_argument(
        "--expect",
        action="append",
        help="Workflow name that must be present. Repeatable; defaults to the four release workflows.",
    )
    args = parser.parse_args()

    try:
        sha = args.sha or git_output("rev-parse", "HEAD")
        repo = args.repo or gh_api("repos/{owner}/{repo}")["full_name"]
        expected = tuple(args.expect) if args.expect else RELEASE_WORKFLOWS

        print(f"waiting on {len(expected)} workflows for {repo}@{sha[:12]} (branch {args.branch})")
        print(f"timeout {args.timeout}s, poll every {args.interval}s\n")

        deadline = time.monotonic() + args.timeout
        empty_polls = 0
        while True:
            runs, source = fetch_runs(repo, args.branch, sha)
            all_terminal, missing, pending, failed = evaluate(runs, expected)
            print(f"[{time.strftime('%H:%M:%S')}] {source}")
            print(describe(runs, expected))

            # Zero runs is ambiguous -- not registered yet, or not visible.
            # Those want opposite responses, so after a few empty polls stop
            # guessing and ask the API a question whose answer does not
            # depend on this SHA.
            if not runs:
                empty_polls += 1
                if empty_polls == EMPTY_POLLS_BEFORE_PROBE:
                    ok, detail = api_is_reachable(repo)
                    if not ok:
                        raise PollError(
                            "cannot see any workflow runs, and the Actions API is "
                            f"unreachable:\n  {detail}\n"
                            "This is NOT evidence that the release failed to trigger -- "
                            "it is evidence that this machine cannot observe it. Resolve "
                            "connectivity, then re-run; the push already happened."
                        )
                    print("   (API reachable — runs genuinely have not registered yet)")
            else:
                empty_polls = 0

            if all_terminal:
                if failed:
                    raise PollError("release CI FAILED:\n  " + "\n  ".join(failed))
                print(f"\nall {len(expected)} workflows reached conclusion=success")
                return 0

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = []
                if missing:
                    detail.append("never appeared: " + ", ".join(missing))
                if pending:
                    detail.append("still running: " + ", ".join(pending))
                raise PollError(
                    f"timed out after {args.timeout}s without a verdict — " + "; ".join(detail) + "\n"
                    "This is a FAILURE, not a pass. Check the runs by branch in the web UI; "
                    "`gh run list --commit` has returned nothing for a green SHA before."
                )
            time.sleep(min(args.interval, remaining))
    except PollError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — no verdict reached", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
