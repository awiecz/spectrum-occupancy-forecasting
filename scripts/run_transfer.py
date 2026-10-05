"""Cross-node transfer analysis (CC1 <-> LW1). Runs once; see docs/technical/transfer_protocol.md.

    uv run python scripts/run_transfer.py data/processed/aerpaw_CC1_blocks5MHz.npz \
        data/processed/aerpaw_LW1_blocks5MHz.npz

Writes results/transfer_results.json and a one-shot lock results/TRANSFER_RUN_LOCK.json. The CC1
test region is removed before any computation. ``--rehearsal NPZ`` runs the whole pipeline with the
given table as BOTH nodes (transfer must then equal in-node), skipping the stored-result checks;
use it with scratch ``--out`` / ``--lock`` paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectrum_occupancy_forecasting.data.aerpaw import band_table_from_blocks
from spectrum_occupancy_forecasting.evaluation.splits import development_region_end
from spectrum_occupancy_forecasting.experiments import window_forecast as wf
from spectrum_occupancy_forecasting.experiments.protocol import (
    acquire_lock,
    complete_lock,
    fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_FILES = [
    ROOT / "src",
    ROOT / "scripts" / "run_transfer.py",
    ROOT / "configs" / "frozen_experiment.yaml",
    ROOT / "docs" / "technical" / "transfer_protocol.md",
]
X_DB = 3.0
WINDOWS = (300.0, 900.0, 3600.0)
PRIMARY_W = 900.0
N_BOOT_ARM = 300
N_BOOT_RATIO = 1000
CHECK_TOL = 1e-6  # in-node arm must reproduce stored RMSEs (floating-point summation order)
HISTORIES = ("recent_1800s", "recent_7200s", "decayed")
LEARNED = ("ridge", "hgb", "resid_ridge", "resid_hgb")


def lw1_folds(t0, t_last):
    """The LW1 protocol's fold rule: 1-day folds from day 2, last partial fold if >= 0.5 d."""
    span_d = (t_last - t0) / wf.DAY
    out = []
    for k in range(2, int(span_d) + 1):
        lo = t0 + k * wf.DAY
        hi = min(lo + wf.DAY, t_last)
        if hi - lo >= 0.5 * wf.DAY:
            out.append((lo, hi))
    return out


def cc1_dev_folds(t0, region_end):
    return [
        (t0 + 2 * wf.DAY, t0 + 3 * wf.DAY),
        (t0 + 3 * wf.DAY, t0 + 4 * wf.DAY),
        (t0 + 4 * wf.DAY, region_end),
    ]


def run_direction(nodes, source, target, folds_def, name):
    arms_def = {
        "in_node": ([target], None),
        "transfer": ([source], None),
        "joint": ([source, target], None),
        "transfer_abs": ([source], source),
    }
    out = {"source": source, "target": target, "folds_days": None, "windows": {}}
    for W in WINDOWS:
        folds = {}
        for arm, (train_nodes, abs_from) in arms_def.items():
            folds[arm] = [
                wf.run_transfer_fold(
                    nodes,
                    train_nodes,
                    target,
                    W,
                    X_DB,
                    lo,
                    hi,
                    wf.CHANNELS,
                    wf.TARGET_MHZ,
                    abs_from,
                )
                for lo, hi in folds_def
            ]
        res = {
            "n_eval": [len(f["y"]) for f in folds["in_node"]],
            "n_train": {a: [f["n_train"] for f in fl] for a, fl in folds.items()},
            "arms": {a: wf.evaluate_folds(fl, False, n_boot=N_BOOT_ARM) for a, fl in folds.items()},
            "ratios": {},
        }
        names = list(folds["in_node"][0]["preds"]["clean"])
        for arm in ("transfer", "joint"):  # identical windows/labels as in-node, so paired
            res["ratios"][arm] = {
                m: list(wf.paired_mse_ratio(folds[arm], folds["in_node"], m, n_boot=N_BOOT_RATIO))
                for m in names
            }
        out["windows"][f"{W:.0f}s"] = res
        print(
            f"\n=== {name}: {source} -> {target}   W={W:.0f}s  windows {res['n_eval']}  "
            f"train windows {res['n_train']['in_node']} (in-node) {res['n_train']['transfer']} "
            "(transfer)",
            flush=True,
        )
        print(
            f"{'model':<14}{'RMSE in-node':>13}{'transfer':>10}{'joint':>8}{'abs-thr':>9}"
            f"   {'penalty transfer [95% CI]':<28}{'penalty joint':>14}"
        )
        for m in names:
            r = {a: res["arms"][a]["clean"][m]["rmse"] for a in res["arms"]}
            pt, pj = res["ratios"]["transfer"][m], res["ratios"]["joint"][m]
            print(
                f"{m:<14}{r['in_node']:>13.4f}{r['transfer']:>10.4f}{r['joint']:>8.4f}"
                f"{r['transfer_abs']:>9.4f}   {pt[0]:>6.2f} [{pt[1]:.2f}, {pt[2]:.2f}]"
                f"{'':<8}{pj[0]:>6.2f}",
                flush=True,
            )
    return out


def check_in_node(direction, stored_json, label):
    """In-node arm must reproduce the stored in-node RMSEs (non-augmented models)."""
    stored = json.loads(Path(stored_json).read_text())["windows"]
    worst = 0.0
    for wkey, res in direction["windows"].items():
        for m, rep in res["arms"]["in_node"]["clean"].items():
            ref = stored[wkey]["clean"].get(m)
            if ref is None:
                continue
            worst = max(worst, abs(rep["rmse"] - ref["rmse"]))
    print(f"consistency check ({label}): max |RMSE difference| vs stored = {worst:.2e}", flush=True)
    if worst > CHECK_TOL:
        raise RuntimeError(f"in-node arm does not reproduce stored results ({label}): {worst:.3e}")
    return worst


def score_hypotheses(d1):
    """T1-T4 of the protocol, at W = 900 for CC1 -> LW1."""
    r = d1["windows"][f"{PRIMARY_W:.0f}s"]["ratios"]
    tr = {m: v[0] for m, v in r["transfer"].items()}
    jt = {m: v[0] for m, v in r["joint"].items()}
    t1 = all(0.95 <= tr[m] <= 1.10 for m in HISTORIES)
    t2 = tr["ridge"] >= 1.10 and tr["hgb"] >= 1.10
    t3 = tr["resid_ridge"] < tr["ridge"] and tr["resid_hgb"] < tr["hgb"]
    t4 = all(jt[m] >= 0.95 for m in LEARNED)
    out = {
        "T1": bool(t1),
        "T2": bool(t2),
        "T3": bool(t3),
        "T4": bool(t4),
        "transfer_penalty": tr,
        "joint_penalty": jt,
    }
    print(
        "\nRegistered hypotheses (W=900, CC1 -> LW1): "
        + ", ".join(
            f"{k} {'SUPPORTED' if out[k] else 'REJECTED'}" for k in ("T1", "T2", "T3", "T4")
        )
    )
    print("  history transfer penalties:", {m: round(tr[m], 3) for m in HISTORIES})
    print("  learned transfer penalties:", {m: round(tr[m], 3) for m in LEARNED})
    print("  learned joint penalties:   ", {m: round(jt[m], 3) for m in LEARNED})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cc1_npz", nargs="?")
    ap.add_argument("lw1_npz", nargs="?")
    ap.add_argument("--rehearsal", default=None, help="use this table as both nodes")
    ap.add_argument("--out", default=str(ROOT / "results" / "transfer_results.json"))
    ap.add_argument("--lock", default=str(ROOT / "results" / "TRANSFER_RUN_LOCK.json"))
    ap.add_argument(
        "--acknowledge-rerun",
        action="store_true",
        help="burns the run: counted in the lock, must be disclosed",
    )
    a = ap.parse_args()
    if not a.rehearsal and not (a.cc1_npz and a.lw1_npz):
        ap.error("give both npz files (or --rehearsal NPZ)")

    fp = fingerprint(FROZEN_FILES, root=ROOT)
    lock = acquire_lock(a.lock, fp, a.acknowledge_rerun)
    print(f"fingerprint {fp}\nlock written ({lock['status']})", flush=True)

    if a.rehearsal:
        base = band_table_from_blocks(a.rehearsal, wf.CHANNELS, 20.0, session=0)
        cc1, lw1 = base.copy(), base.copy()
        cc1["site"], lw1["site"] = "CC1", "LW1"
    else:
        cc1 = band_table_from_blocks(a.cc1_npz, wf.CHANNELS, 20.0, session=0)
        lw1 = band_table_from_blocks(a.lw1_npz, wf.CHANNELS, 20.0, session=0)
    t0c, tlc = float(cc1["t_start_s"].min()), float(cc1["t_start_s"].max())
    # (rehearsal data is already truncated to the development region, so use its own end)
    region_end = tlc if a.rehearsal else development_region_end(t0c, tlc)
    cc1 = cc1[cc1["t_start_s"] <= region_end].reset_index(drop=True)  # test rows removed here
    print(
        f"CC1 test region removed (after {(region_end - t0c) / wf.DAY:.2f} d); "
        f"LW1 session {(lw1['t_start_s'].max() - lw1['t_start_s'].min()) / wf.DAY:.2f} d",
        flush=True,
    )
    nodes = {"CC1": cc1, "LW1": lw1}
    t0l, tll = float(lw1["t_start_s"].min()), float(lw1["t_start_s"].max())

    results = {"fingerprint": fp, "x_db": X_DB, "rehearsal": bool(a.rehearsal), "directions": {}}
    d1 = run_direction(nodes, "CC1", "LW1", lw1_folds(t0l, tll), "primary")
    d1["folds_days"] = [
        [(lo - t0l) / wf.DAY, (hi - t0l) / wf.DAY] for lo, hi in lw1_folds(t0l, tll)
    ]
    d2 = run_direction(nodes, "LW1", "CC1", cc1_dev_folds(t0c, region_end), "exploratory")
    d2["folds_days"] = [
        [(lo - t0c) / wf.DAY, (hi - t0c) / wf.DAY] for lo, hi in cc1_dev_folds(t0c, region_end)
    ]
    results["directions"] = {"CC1_to_LW1": d1, "LW1_to_CC1": d2}

    if not a.rehearsal:
        results["consistency"] = {
            "LW1_in_node_vs_stored": check_in_node(
                d1, ROOT / "results" / "node_LW1_window_forecast.json", "LW1"
            ),
            "CC1_dev_in_node_vs_stored": check_in_node(
                d2, ROOT / "results" / "dev_window_forecast.json", "CC1 dev"
            ),
        }
        results["hypotheses"] = score_hypotheses(d1)
    else:  # rehearsal sanity: identical source/target data must transfer exactly
        for wkey, res in d1["windows"].items():
            for m, v in res["ratios"]["transfer"].items():
                assert abs(v[0] - 1.0) < 1e-9, (wkey, m, v)
        print("rehearsal: transfer == in-node for identical data (all penalties = 1)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    done = complete_lock(a.lock, out)
    print(f"\nwrote {out}\nlock completed; results sha256 {done['results_sha256']}")


if __name__ == "__main__":
    np.seterr(all="warn")
    main()
