"""One-shot guard for the held-out test evaluation.

The test region may be evaluated once. ``acquire_lock`` writes a lock file *before* any test data is
loaded, recording a fingerprint (SHA-256) of the exact code, config and protocol document in
force; a second run is refused unless the caller explicitly acknowledges that it burns the test
set, in which case the re-run is counted in the lock and must be disclosed in any report.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path


class TestSetAlreadyUsedError(RuntimeError):
    """Raised when a second test-set evaluation is attempted without acknowledgement."""

    __test__ = False  # not a pytest test class


def fingerprint(paths: Iterable[str | Path], *, root: str | Path | None = None) -> str:
    """SHA-256 over root-relative names and LF-normalised contents of the selected files.

    Experiment scripts pass the repository root explicitly. Other callers default to the common
    parent directory of the selected files. Files outside an explicit root are rejected.
    Historical locks used absolute paths; their hashes are preserved, not migrated.
    """
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    files = sorted({f.resolve() for f in files})
    if not files:
        raise ValueError("fingerprint requires at least one file")
    base = Path(root).resolve() if root is not None else Path(
        os.path.commonpath([str(f.parent) for f in files])
    )
    named = sorted((f.relative_to(base).as_posix(), f) for f in files)
    h = hashlib.sha256()
    h.update(b"spectrum-occupancy-fingerprint-v2\0")
    for name, f in named:
        content = f.read_bytes().replace(b"\r\n", b"\n")
        encoded = name.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
        h.update(len(content).to_bytes(8, "big"))
        h.update(content)
    return h.hexdigest()


def _git(cwd, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_state(cwd: str | Path = ".") -> dict:
    """Current commit and whether the working tree has uncommitted changes (None if no git)."""
    try:
        return {"commit": _git(cwd, "rev-parse", "HEAD"),
                "dirty": bool(_git(cwd, "status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def acquire_lock(lock_path: str | Path, fp: str, acknowledge_rerun: bool = False) -> dict:
    """Create the lock (status ``started``) or refuse; returns the lock record."""
    lock_path = Path(lock_path)
    prior = json.loads(lock_path.read_text()) if lock_path.exists() else None
    if prior is not None and not acknowledge_rerun:
        raise TestSetAlreadyUsedError(
            f"{lock_path} exists (status {prior.get('status')!r}, started "
            f"{prior.get('started')}). The test region was already evaluated or an evaluation "
            "was started. Re-running burns the test set; pass acknowledge_rerun=True and disclose "
            "it in the report."
        )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "started",
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "fingerprint": fp,
        "fingerprint_scheme": "root-relative-v2",
        "git": git_state(lock_path.parent),
        "reruns": (prior.get("reruns", 0) + 1) if prior else 0,
        "previous": prior,
    }
    lock_path.write_text(json.dumps(record, indent=1))
    return record


def complete_lock(lock_path: str | Path, results_path: str | Path) -> dict:
    lock_path, results_path = Path(lock_path), Path(results_path)
    record = json.loads(lock_path.read_text())
    record["status"] = "completed"
    record["completed"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    record["results_sha256"] = hashlib.sha256(results_path.read_bytes()).hexdigest()
    lock_path.write_text(json.dumps(record, indent=1))
    return record
