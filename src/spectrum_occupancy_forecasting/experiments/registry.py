"""Durable, append-only scientific attempt registry (local filesystem only)."""

import ast
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def implementation_digest(files, root, selected_functions=None):
    """Hash executable ASTs: comments/docstrings/formatting do not create new science."""
    selected_functions = selected_functions or {}
    records = {}
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        if path.suffix != ".py":
            records[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if path.name in selected_functions:
            tree.body = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in selected_functions[path.name]
            ]
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body = node.body[1:]
        records[relative] = ast.dump(tree, include_attributes=False)
    return digest(records)


class Registry:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("incomplete registry record; explicit recovery required")
        rows = [json.loads(line) for line in raw.splitlines()]
        previous = None
        for row in rows:
            payload = {k: v for k, v in row.items() if k != "hash"}
            if payload["previous"] != previous or row["hash"] != digest(payload):
                raise ValueError("registry integrity failure")
            previous = row["hash"]
        return rows

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        deadline = time.monotonic() + 10
        while True:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, json.dumps({"pid": os.getpid(), "time": time.time()}).encode())
                os.fsync(descriptor)
                os.close(descriptor)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "registry locked; inspect owner before explicit recovery"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            lock.unlink()

    def _append(self, event, rows):
        payload = event | {
            "timestamp": datetime.now(UTC).isoformat(),
            "previous": rows[-1]["hash"] if rows else None,
        }
        row = payload | {"hash": digest(payload)}
        with self.path.open("ab") as stream:
            stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        return row

    def append(self, event):
        with self.locked():
            return self._append(event, self.read())

    def declare(
        self,
        specification,
        implementation,
        *,
        hypothesis,
        kind="model",
        parent=None,
        evidence=None,
        reproduction=None,
    ):
        with self.locked():
            rows = self.read()
            declarations = [r for r in rows if r["event"] == "declaration"]
            spec_hash = digest(specification)
            same = [
                r
                for r in declarations
                if r["specification_hash"] == spec_hash
                and r["implementation_hash"] == implementation
            ]
            if same and reproduction not in {r["attempt"] for r in same}:
                raise ValueError("duplicate scientific attempt; declare a reproduction")
            same_specification = [r for r in declarations if r["specification_hash"] == spec_hash]
            if (
                same_specification
                and not same
                and not reproduction
                and kind != "implementation-maintenance"
            ):
                raise ValueError(
                    "same specification with changed code requires reproduction "
                    "or explicit implementation maintenance"
                )
            originals = [r for r in declarations if r["attempt"] == reproduction]
            if reproduction and (not originals or originals[0]["specification_hash"] != spec_hash):
                raise ValueError("reproduction must match original scientific specification")
            if (
                reproduction
                and originals[0]["implementation_hash"] != implementation
                and not evidence
            ):
                raise ValueError(
                    "changed-implementation reproduction needs an explicit explanation"
                )
            if kind in ("parameter-revision", "implementation-maintenance") and (
                not parent or not evidence
            ):
                raise ValueError("parameter-only revision needs parent and motivating evidence")
            related = [
                r
                for r in declarations
                if all(
                    r["specification"].get(k) == specification.get(k)
                    for k in ("model_family", "inputs", "protocol", "phase")
                )
            ]
            parameter_changes = [
                r
                for r in related
                if r["specification"].get("parameters") != specification.get("parameters")
            ]
            if parameter_changes and not reproduction and (not parent or not evidence):
                raise ValueError(
                    "related parameter-only revision needs parent and motivating evidence"
                )
            if parent and parent not in {
                r["attempt"] for r in rows if r["event"] in ("declaration", "historical-import")
            }:
                raise ValueError("unknown parent attempt")

            def without_seeds(value):
                if isinstance(value, dict):
                    return {k: without_seeds(v) for k, v in value.items() if "seed" not in k}
                if isinstance(value, list):
                    return [without_seeds(v) for v in value]
                return value

            seed_only = [
                r
                for r in related
                if r["specification_hash"] != spec_hash
                and without_seeds(r["specification"]) == without_seeds(specification)
            ]
            if seed_only and kind != "seed-replication" and not reproduction:
                raise ValueError("a new seed is a seed replication, not a new learning idea")
            attempt = uuid.uuid4().hex
            self._append(
                {
                    "event": "declaration",
                    "attempt": attempt,
                    "specification": specification,
                    "specification_hash": spec_hash,
                    "implementation_hash": implementation,
                    "hypothesis": hypothesis,
                    "kind": "reproduction" if reproduction else kind,
                    "parent": parent,
                    "evidence": evidence,
                    "reproduction_of": reproduction,
                    "related_attempts": [r["attempt"] for r in related],
                    "original_implementation_hash": originals[0]["implementation_hash"]
                    if originals
                    else None,
                },
                rows,
            )
            return attempt

    def stale(self, hours=24):
        states = {}
        for row in self.read():
            if row["event"] == "status":
                states[row["attempt"]] = row
        now = datetime.now(UTC)
        return [
            r
            for r in states.values()
            if r["status"] == "running"
            and (now - datetime.fromisoformat(r["timestamp"])).total_seconds() > hours * 3600
        ]

    def recover(self, reason, attempt=None, unlock=False):
        if not reason:
            raise ValueError("explicit recovery reason required")
        if unlock:
            lock = self.path.with_suffix(self.path.suffix + ".lock")
            if lock.exists():
                owner = json.loads(lock.read_text())
                # Never automatically evict a lock. Caller explicitly authorizes recovery.
                lock.unlink()
            else:
                owner = None
        else:
            owner = None
        with self.locked():
            raw = self.path.read_bytes() if self.path.exists() else b""
            fragment = None
            if raw and not raw.endswith(b"\n"):
                position = raw.rfind(b"\n") + 1
                fragment = raw[position:].hex()
                backup = self.path.with_name(self.path.name + ".recovery-" + uuid.uuid4().hex)
                backup.write_bytes(raw)
                with self.path.open("r+b") as stream:
                    stream.truncate(position)
                    stream.flush()
                    os.fsync(stream.fileno())
            event = self._append(
                {
                    "event": "recovery",
                    "attempt": attempt,
                    "reason": reason,
                    "discarded_fragment_hex": fragment,
                    "lock_owner": owner,
                },
                self.read(),
            )
            if attempt:
                self._append(
                    {
                        "event": "status",
                        "attempt": attempt,
                        "status": "interrupted",
                        "reason": reason,
                    },
                    self.read(),
                )
            return event

    def export(self, out):
        import pandas as pd

        out = Path(out)
        rows = self.read()
        declarations = [r for r in rows if r["event"] in ("declaration", "historical-import")]
        table = []
        for declaration in declarations:
            attempt = declaration["attempt"]
            events = [r for r in rows if r.get("attempt") == attempt]
            statuses = [r["status"] for r in events if r["event"] == "status"]
            table.append(
                {
                    "attempt": attempt,
                    "kind": declaration.get("kind", "historical"),
                    "hypothesis": declaration.get("hypothesis"),
                    "status": statuses[-1] if statuses else declaration.get("status", "declared"),
                    "specification_hash": declaration.get("specification_hash"),
                    "implementation_hash": declaration.get("implementation_hash"),
                    "parent": declaration.get("parent"),
                    "source_reference": declaration.get("source_reference"),
                    "outcome": next(
                        (
                            r["findings"]["advancement.json"]["status"]
                            for r in reversed(events)
                            if r["event"] == "findings" and "advancement.json" in r["findings"]
                        ),
                        None,
                    ),
                    "review": next(
                        (r.get("decision") for r in reversed(events) if r["event"] == "review"),
                        None,
                    ),
                }
            )
        pd.DataFrame(table).to_csv(out / "experiment_history.csv", index=False)
        comparisons = []
        for row in rows:
            if row["event"] == "historical-import":
                comparisons.append(
                    {
                        "attempt": row["attempt"],
                        "source_reference": row["source_reference"],
                        **row["historical_metadata"],
                    }
                )
            elif row["event"] == "findings":
                for name in ("summary.csv", "protocol_comparison.csv", "controller_comparison.csv"):
                    comparisons.extend(
                        {"attempt": row["attempt"], "source_table": name, **result}
                        for result in row["findings"].get(name, [])
                    )
        pd.DataFrame(comparisons).to_csv(out / "experiment_comparison.csv", index=False)
        (out / "experiment_history.md").write_text(
            "# Experiment history\n\nGenerated from the append-only registry. Historical missing "
            "metadata stays missing. All recording comparisons are exploratory.\n\n"
            + "\n".join(
                f"- `{r['attempt'][:12]}` — {r['hypothesis']} ({r['status']})" for r in table
            )
            + "\n",
            encoding="utf-8",
        )


def migrate(registry, root):
    """Import immutable ledger rows and their source bytes, without guessing identities."""
    import csv

    root = Path(root)
    known = {
        r.get("source_reference") for r in registry.read() if r["event"] == "historical-import"
    }
    for path in sorted((root / "results").rglob("experiment_ledger.csv")):
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream), 2):
                reference = f"{path.relative_to(root).as_posix()}:{index}"
                if reference in known:
                    continue
                registry.append(
                    {
                        "event": "historical-import",
                        "attempt": digest([reference, source_hash]),
                        "source_reference": reference,
                        "source_sha256": source_hash,
                        "historical_metadata": row,
                        "hypothesis": row.get("experiment", row.get("method", reference)),
                        "missing_metadata": [
                            "complete_resolved_specification",
                            "scored_origin_hash",
                            "scientific_implementation_hash",
                        ],
                        "status": row.get("status", "historical-outcome-recorded"),
                    }
                )
    provenance_known = {
        r.get("source_reference") for r in registry.read() if r["event"] == "historical-provenance"
    }
    # Archive inventories identify preserved development code; no raw files are touched.
    archives = root / "reproductions/adaptive_archive_initial"
    provenance = list((root / "results").rglob("manifest.json"))
    provenance += list((archives / "snapshot/configs").glob("*.yaml"))
    provenance += list(archives.glob("*inventory*.json"))
    for path in sorted(provenance):
        reference = path.relative_to(root).as_posix()
        if reference in provenance_known:
            continue
        import yaml

        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        registry.append(
            {
                "event": "historical-provenance",
                "source_reference": reference,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "historical_metadata": content,
                "original_fingerprints_preserved": True,
            }
        )
