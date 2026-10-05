"""Byte-preserving historical checks and descriptive diagnosis before adaptation."""

import json

import numpy as np
import pandas as pd

from .fine_spectrum import FINE_COLUMNS, sha, write_json


def audit(root, nodes, out):
    archive = root / "reproductions/adaptive_archive_initial"
    if not (archive / "inventory.json").exists():
        raise FileNotFoundError(
            f"{archive} is a local, Git-ignored archive that is not present in this checkout. "
            "The historical audit was already applied and its outputs are saved under "
            "results/adaptive_learning/audit; rebuild the report with --mode report "
            "--predictions-dir results/adaptive_learning/run."
        )
    inventory = json.loads((archive / "inventory.json").read_text())
    protected = json.loads((archive / "protected_hashes.json").read_text())
    for name, digest in inventory.items():
        if sha(archive / "snapshot" / name) != digest:
            raise ValueError(f"archive checksum mismatch: {name}")
    checked = []
    for name, digest in protected.items():
        path = root / name
        if not path.exists():
            path = archive / "snapshot" / name
        if sha(path) != digest:
            raise ValueError(f"protected artifact changed: {name}")
        checked.append(name)
    reference = pd.read_parquet(archive / "feature_reference.parquet")
    for name, old in reference.groupby("node"):
        node = nodes[f"{name}_first"]
        X, base, valid = node.features(node.data.grid[500:505], 0.2, "history_fine")
        X["base"], X["quality"], X["node"] = base, valid, name
        pd.testing.assert_frame_equal(
            X[old.columns].reset_index(drop=True), old.reset_index(drop=True)
        )
    comparisons, diagnosis = [], []
    for part in ("development", "later_assessment"):
        folder = root / "results/shared_learning" / part
        if not folder.exists():
            folder = archive / "snapshot/results/shared_learning" / part
        predictions = pd.read_parquet(folder / "predictions.parquet")
        summary = pd.read_csv(folder / "summary.csv")
        for (node, scenario, method), frame in predictions.groupby(["node", "scenario", "method"]):
            match = summary[
                (summary.node == node) & (summary.scenario == scenario) & (summary.method == method)
            ]
            if len(match) != 1:
                raise ValueError("saved comparison is not uniquely specified")
            mse = float(np.mean((frame.prediction - frame.y) ** 2))
            if not np.isclose(mse, match.mse.iloc[0], rtol=1e-10, atol=1e-14):
                raise ValueError("historical metrics disagree with saved predictions")
            comparisons.append(
                {
                    "study": part,
                    "node": node,
                    "scenario": scenario,
                    "method": method,
                    "mse": mse,
                    "n": len(frame),
                }
            )
            if scenario == "clean":
                bias = float(np.mean(frame.prediction - frame.y))
                diagnosis.append(
                    {
                        "study": part,
                        "node": node,
                        "method": method,
                        "mean_y": float(frame.y.mean()),
                        "prediction_bias": bias,
                        "mse": mse,
                        "n": len(frame),
                    }
                )
    inputs = []
    for recording, node in nodes.items():
        for period, keep in (
            ("calibration", node.frame.time_s + 0.001 <= node.data.t0 + 86400),
            ("post_calibration", node.frame.time_s + 0.001 > node.data.t0 + 86400),
        ):
            power = node.frame.loc[keep, "target_dbm"]
            inputs.append(
                {
                    "recording": recording,
                    "period": period,
                    "n": len(power),
                    "floor_dbm": node.data.floors[2437.0],
                    "threshold_dbm": node.data.thresholds[2437.0],
                    "power_p10": float(power.quantile(0.1)),
                    "power_median": float(power.median()),
                    "power_p90": float(power.quantile(0.9)),
                    "busy_share": float((power > node.data.thresholds[2437.0]).mean()),
                    "fine_normalized_median": np.median(node.normalized[keep], axis=0).tolist(),
                    "fine_columns": FINE_COLUMNS,
                }
            )
    pd.DataFrame(comparisons).to_csv(out / "previous_comparisons.csv", index=False)
    pd.DataFrame(diagnosis).to_csv(out / "previous_bias.csv", index=False)
    write_json(out / "input_distributions.json", inputs)
    write_json(
        out / "historical_audit.json",
        {
            "archive_files_verified": len(inventory),
            "protected_artifacts_verified": len(checked),
            "comparisons_reproduced": len(comparisons),
            "neutral_features_identical": True,
        },
    )
