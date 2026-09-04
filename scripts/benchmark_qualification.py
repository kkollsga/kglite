#!/usr/bin/env python3
"""Bind benchmark reference eligibility to captured bytes and platform.

Raw current.json remains the latest workload/capture artifact. The manifest's
reference pointer is the qualified comparison baseline; capture alone cannot
advance it. Fresh external candidates may be evaluated, but known rejected
bytes cannot be made usable by renaming them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

BASELINES_DIR = Path(__file__).resolve().parent.parent / "tests/benchmarks/baselines"
MANIFEST = "qualifications.json"
STATUSES = {"accepted", "pending", "rejected"}


class QualificationError(ValueError):
    """No trustworthy comparison verdict can be produced."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def platform(path: Path) -> str | None:
    machine = json.loads(path.read_text(encoding="utf-8")).get("machine_info", {})
    system, arch = machine.get("system"), machine.get("machine")
    return f"{system}/{arch}" if system and arch else None


def _filename(name: str) -> bool:
    return isinstance(name, str) and Path(name).name == name and name.endswith(".json")


class Registry:
    def __init__(self, directory: Path, *, required: bool = False):
        local = directory / MANIFEST
        self.path = local if local.exists() or required else BASELINES_DIR / MANIFEST
        self.directory = self.path.parent.resolve()
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise QualificationError(f"qualification manifest unavailable: {self.path}: {error}") from error
        self._validate()

    def _validate(self) -> None:
        data = self.data
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise QualificationError("unsupported qualification manifest schema")
        captures, references = data.get("captures"), data.get("references")
        if not isinstance(captures, dict) or not isinstance(references, dict):
            raise QualificationError("qualification manifest requires captures and references objects")
        for name, record in captures.items():
            if not _filename(name) or not isinstance(record, dict):
                raise QualificationError(f"invalid capture record: {name}")
            if (
                record.get("status") not in STATUSES
                or not isinstance(record.get("platform"), str)
                or not record["platform"]
                or not isinstance(record.get("evidence"), str)
                or not record["evidence"]
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
            ):
                raise QualificationError(f"invalid qualification fields: {name}")
        for identity, name in references.items():
            record = captures.get(name) if isinstance(name, str) else None
            if not record or record["platform"] != identity or record["status"] != "accepted":
                raise QualificationError(f"reference {identity} must identify an accepted capture")

    def status(self, path: Path) -> str | None:
        sha = digest(path)
        record = self.data["captures"].get(path.name) if path.parent.resolve() == self.directory else None
        if record:
            if record["sha256"] != sha:
                raise QualificationError(f"{path.name}: digest differs from its qualification")
            if record["platform"] != platform(path):
                raise QualificationError(f"{path.name}: platform differs from its qualification")
        matching = [(name, item) for name, item in self.data["captures"].items() if item["sha256"] == sha]
        rejected = next(((name, item) for name, item in matching if item["status"] == "rejected"), None)
        if rejected:
            name, item = rejected
            print(f"qualification: {path.name} matches rejected {name}: {item['evidence']}")
            return "rejected"
        return record["status"] if record else None

    def candidate(self, path: Path) -> None:
        if self.status(path) == "rejected":
            raise QualificationError(
                f"{path.name}: rejected candidate; no valid verdict (candidate is never substituted)"
            )

    def reference(self, path: Path) -> Path:
        status = self.status(path)
        managed = path.parent.resolve() == self.directory
        if managed and path.name in {"current.json", "current.linux.json"}:
            selected = self.data["references"].get(platform(path))
            if not selected:
                raise QualificationError(f"{path.name}: no qualified reference for its platform")
            resolved = self.directory / selected
            if self.status(resolved) != "accepted":
                raise QualificationError(f"{selected}: no qualified reference")
            print(f"qualification: {path.name} resolves to accepted {selected}")
            print(f"  evidence: {self.data['captures'][selected]['evidence']}")
            return resolved
        if status is not None and status != "accepted" or managed and status is None:
            raise QualificationError(f"{path.name}: no qualified reference ({status or 'unregistered'})")
        return path


def compatible(reference: Path, candidate: Path) -> None:
    left, right = platform(reference), platform(candidate)
    if not left or not right:
        raise QualificationError("benchmark platform identity missing (system and machine are required)")
    if left != right:
        raise QualificationError(f"incompatible benchmark platforms: {left} vs {right}")


def validate_measurements(stats: dict, metric: str) -> None:
    if not stats:
        raise QualificationError("empty benchmark capture; no valid verdict")
    for name, values in stats.items():
        value = values.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise QualificationError(f"{name}: {metric} must be finite and positive")


def record_capture(path: Path, *, evidence: str, alias: Path | None = None) -> None:
    """Register a new raw capture as pending; a retry preserves its decision."""
    manifest = path.parent / MANIFEST
    if not manifest.exists():
        manifest.write_text(json.dumps({"schema_version": 1, "captures": {}, "references": {}}), encoding="utf-8")
    registry = Registry(path.parent, required=True)
    identity = platform(path)
    if not identity:
        raise QualificationError(f"{path.name}: capture has no platform identity")
    old = registry.data["captures"].get(path.name)
    if old:
        registry.status(path)  # Digest/platform must still match even for rejected captures.
    else:
        registry.data["captures"][path.name] = {
            "sha256": digest(path),
            "platform": identity,
            "status": "pending",
            "evidence": evidence,
        }
    if alias:
        if alias.parent.resolve() != path.parent.resolve() or alias.name not in {"current.json", "current.linux.json"}:
            raise QualificationError("capture alias must be current.json or current.linux.json in the same directory")
        if digest(alias) != digest(path):
            raise QualificationError("capture alias digest differs from its versioned capture")
        registry.data["captures"][alias.name] = dict(registry.data["captures"][path.name])
    _write(registry)


def qualify(path: Path, status: str, evidence: str, *, promote: bool = False) -> None:
    """An explicit evidence-backed qualification; never rewrite capture bytes."""
    if status not in STATUSES or not evidence.strip():
        raise QualificationError("qualification requires a status and evidence")
    if promote and status != "accepted":
        raise QualificationError("only accepted captures can be promoted as references")
    record_capture(path, evidence=evidence)
    registry = Registry(path.parent, required=True)
    record = registry.data["captures"][path.name]
    # All aliases of these exact bytes must agree after an explicit decision.
    for item in registry.data["captures"].values():
        if item["sha256"] == record["sha256"]:
            item.update(status=status, evidence=evidence)
    if status != "accepted":
        registry.data["references"] = {
            key: name
            for key, name in registry.data["references"].items()
            if registry.data["captures"][name]["sha256"] != record["sha256"]
        }
    if promote:
        if path.name.startswith("current."):
            raise QualificationError("promote a versioned capture, not the mutable current alias")
        registry.data["references"][record["platform"]] = path.name
    _write(registry)


def _write(registry: Registry) -> None:
    registry._validate()
    temporary = registry.path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(registry.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(registry.path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--status", choices=sorted(STATUSES), required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    try:
        qualify(args.capture, args.status, args.evidence, promote=args.promote)
    except (QualificationError, OSError, ValueError) as error:
        print(f"qualification failed: {error}")
        return 2
    print(f"{args.capture.name}: {args.status}; raw capture unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
