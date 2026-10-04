"""Freeze bundles and verification of independently timestamped confirmation artifacts.

No CC2 file may be opened before this verifier succeeds. Merely supplying a URL or a local
timestamp is insufficient. Supported archives: immutable GitHub releases and published Zenodo
records containing the exact frozen bundle. This module never publishes or downloads RF data.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from spectrum_occupancy_forecasting.experiments.protocol import fingerprint

PACKAGE_NAME = "followup_v2_freeze.zip"


def normalized_bytes(path):
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_manifest(root, config, protocol):
    root = Path(root).resolve()
    source = [
        root / "src",
        root / "scripts" / "run_followup.py",
        root / "scripts" / "freeze_followup.py",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    paths = []
    for path in source:
        paths.extend(path.rglob("*.py") if path.is_dir() else [path])
    return {
        "scheme": "root-relative-v2",
        "source_fingerprint": fingerprint(source, root=root),
        "config_sha256": hashlib.sha256(normalized_bytes(config)).hexdigest(),
        "protocol_sha256": hashlib.sha256(normalized_bytes(protocol)).hexdigest(),
        "source_files": {
            p.relative_to(root).as_posix(): hashlib.sha256(normalized_bytes(p)).hexdigest()
            for p in sorted(paths)
        },
    }


def create_bundle(root, config, protocol, out):
    manifest = freeze_manifest(root, config, protocol)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing a package already sent to an archive.
    with out.open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr("followup_v2.yaml", normalized_bytes(config))
        archive.writestr("followup_v2_protocol.md", normalized_bytes(protocol))
        for name in manifest["source_files"]:
            archive.writestr(name, normalized_bytes(Path(root) / name))
    return manifest


def fetch(url, *, json_response=False):
    if urlparse(url).scheme != "https":
        raise ValueError("archive verification requires HTTPS")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "spectrum-followup-v2",
            "Accept": "application/json" if json_response else "application/octet-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read(25 * 1024 * 1024 + 1)
    if len(payload) > 25 * 1024 * 1024:
        raise ValueError("archive response exceeds the freeze-package size limit")
    return json.loads(payload) if json_response else payload


def verify_bundle(payload, expected):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(info.file_size for info in archive.infolist()) > 25 * 1024 * 1024:
            raise ValueError("expanded freeze package exceeds size limit")
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("duplicate entries in freeze package")
        actual = json.loads(archive.read("manifest.json"))
        if actual != expected:
            raise ValueError(
                "archived source, configuration or protocol differs from this checkout"
            )
        for filename, digest in expected["source_files"].items():
            if hashlib.sha256(archive.read(filename)).hexdigest() != digest:
                raise ValueError(f"archived source checksum mismatch: {filename}")
        for filename, key in (
            ("followup_v2.yaml", "config_sha256"),
            ("followup_v2_protocol.md", "protocol_sha256"),
        ):
            if hashlib.sha256(archive.read(filename)).hexdigest() != expected[key]:
                raise ValueError(f"archived artifact checksum mismatch: {filename}")


def verify_archive(url, expected, *, fetcher=fetch):
    """Fetch a trusted archive's server timestamp and verify its frozen package contents."""
    parsed = urlparse(url or "")
    if parsed.scheme != "https" or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("an independent HTTPS archive API reference is required")
    if parsed.netloc == "api.github.com" and re.fullmatch(
        r"/repos/[^/]+/[^/]+/releases/\d+", parsed.path
    ):
        record = fetcher(url, json_response=True)
        if record.get("draft") or not record.get("immutable") or not record.get("published_at"):
            raise ValueError("GitHub reference must be a published immutable release")
        assets = [a for a in record["assets"] if a["name"] == PACKAGE_NAME]
        timestamp = record["published_at"]
        asset_url = assets[0]["browser_download_url"] if len(assets) == 1 else None
        if asset_url and urlparse(asset_url).netloc != "github.com":
            raise ValueError("unexpected GitHub artifact host")
    elif parsed.netloc == "zenodo.org" and re.fullmatch(r"/api/records/\d+", parsed.path):
        record = fetcher(url, json_response=True)
        if not record.get("doi") or not record.get("id") or record.get("is_published") is False:
            raise ValueError("Zenodo reference must be a published record with a DOI")
        assets = [a for a in record["files"] if a["key"] == PACKAGE_NAME]
        timestamp = record["updated"]
        asset_url = assets[0]["links"]["self"] if len(assets) == 1 else None
        if asset_url and urlparse(asset_url).netloc != "zenodo.org":
            raise ValueError("unexpected Zenodo artifact host")
    else:
        raise ValueError(
            "use an immutable GitHub release API URL or published Zenodo record API URL"
        )
    if asset_url is None:
        raise ValueError(f"archive must contain exactly one {PACKAGE_NAME}")
    archived = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if archived.tzinfo is None or archived > datetime.now(UTC):
        raise ValueError("invalid independent archive timestamp")
    payload = fetcher(asset_url)
    verify_bundle(payload, expected)
    return {
        "archive_url": url,
        "archived_at": timestamp,
        "bundle_sha256": hashlib.sha256(payload).hexdigest(),
        "verified": True,
        "limits": "archive timestamps do not prove that no earlier activity inspection occurred",
    }
