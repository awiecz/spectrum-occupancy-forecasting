"""Create a private freeze bundle; does not upload, publish or inspect CC2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectrum_occupancy_forecasting.experiments.followup_gate import create_bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/followup_v2.yaml")
    ap.add_argument("--protocol", type=Path, default=ROOT / "docs/technical/followup_v2_protocol.md")
    ap.add_argument("--out", type=Path, default=ROOT / "reproductions/followup_v2_freeze.zip")
    a = ap.parse_args()
    manifest = create_bundle(ROOT, a.config, a.protocol, a.out)
    print(json.dumps(manifest, indent=2))
    print(
        f"Private bundle: {a.out}. CC2 confirmation remains blocked until independently archived."
    )


if __name__ == "__main__":
    main()
