"""Sanity checks on the measurements and the headline numbers. Writes results/sanity_checks.json.

    uv run python scripts/sanity_checks.py

Descriptive controls only (no forecasting): they use whole-period noise floors and the CC1
development region (the held-out CC1 test region is never read here). Section 5 reads stored
result files; it computes nothing new on the test region.

1. Negative control: the busy rule applied to bands that are idle all week should almost never fire.
2. Receiver drift: level of idle bands by hour of day, against the 3 dB busy threshold.
3. Spectral shape of the busy state: a localised bump matching a 20 MHz channel, or not.
4. Dependence: autocorrelation of 15-min busy fractions and an AR(1) effective sample size.
5. Value of old evidence: skill versus evidence delay and its half-life, from the stored results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectrum_occupancy_forecasting.data.aerpaw import sessions
from spectrum_occupancy_forecasting.evaluation.diagnostics import (
    effective_n,
    half_life,
    lag_autocorr,
)
from spectrum_occupancy_forecasting.evaluation.splits import development_region_end

ROOT = Path(__file__).resolve().parents[1]
X_DB = 3.0
TARGET = 2437.0
CONTROLS = (5200.0, 5240.0, 5280.0, 5745.0, 5785.0)  # 5 GHz U-NII: idle all week on both nodes
WINDOW_S = 900.0
DELAY_SCENARIOS = (("clean", 0), ("delay_60s", 1), ("delay_600s", 10), ("delay_1800s", 30),
                   ("delay_3600s", 60), ("delay_7200s", 120))
MODELS = ("recent_7200s", "decayed", "hgb")
NOMINAL_WINDOWS = {"CC1 dev": 307, "CC1 test": 169, "LW1": 261}  # windows scored at W = 15 min


def load(node):
    z = np.load(ROOT / "data" / "processed" / f"aerpaw_{node}_blocks5MHz.npz")
    t, f, m = z["times_s"], z["centres_mhz"], z["mean_dbm"].astype(float)
    k = sessions(t) == 0
    t, m = t[k], m[k]
    if node == "CC1":  # development region only
        keep = t <= development_region_end(t[0], t[-1])
        t, m = t[keep], m[keep]
    return t, f, m


def band_mean(f, m, centre, bw=20.0):
    sel = np.abs(f - centre) <= bw / 2
    return 10 * np.log10((10 ** (m[:, sel] / 10)).mean(axis=1))


def controls(node):
    t, f, m = load(node)
    out = {"sweeps": int(len(t)), "days": float((t[-1] - t[0]) / 86400)}
    hour = ((t - 5 * 3600) % 86400) // 3600  # local time (UTC-5 in February)

    neg = {}
    for c in CONTROLS:
        p = band_mean(f, m, c)
        neg[f"{c:.0f}"] = float(np.mean(p > np.percentile(p, 5) + X_DB))
    p6 = band_mean(f, m, TARGET)
    busy = p6 > np.percentile(p6, 5) + X_DB
    out["negative_control_prevalence"] = neg
    out["target_prevalence"] = float(busy.mean())

    idle = np.mean([band_mean(f, m, c) for c in CONTROLS[:3]], axis=0)
    hourly = np.array([np.median(idle[hour == h]) for h in range(24)])
    out["idle_band_hourly_median_span_db"] = float(hourly.max() - hourly.min())
    q10 = np.array([np.percentile(p6[hour == h], 10) for h in range(24)])
    out["target_quiet_level_span_db"] = float(q10.max() - q10.min())

    sel = (f >= 2395) & (f <= 2495)
    floor = np.percentile(m[:, sel], 5, axis=0)
    diff = (m[busy][:, sel] - floor).mean(axis=0) - (m[~busy][:, sel] - floor).mean(axis=0)
    inband = (f[sel] >= 2430) & (f[sel] <= 2445)
    outband = (f[sel] >= 2455) & (f[sel] <= 2490)
    out["spectral_shape_db"] = {f"{fc:.0f}": float(d) for fc, d in zip(f[sel], diff, strict=True)}
    out["spectral_in_channel_mean_db"] = float(diff[inband].mean())
    out["spectral_out_of_channel_mean_db"] = float(diff[outband].mean())

    edges = np.arange(t[0], t[-1], WINDOW_S)
    idx = np.searchsorted(edges, t, side="right") - 1
    cnt = np.bincount(idx, minlength=len(edges)).astype(float)
    frac = np.bincount(idx, weights=busy.astype(float), minlength=len(edges)) / np.maximum(cnt, 1)
    frac = frac[cnt > 0.8 * WINDOW_S / np.median(np.diff(t))]
    out["window_autocorr_15_30_60_120min"] = [lag_autocorr(frac, k) for k in (1, 2, 4, 8)]
    out["n_windows"] = int(len(frac))
    out["n_effective_ar1"] = effective_n(frac, 1)
    return out


def evidence_age():
    sources = {"CC1 dev": "dev_window_forecast.json", "CC1 test": "test_window_forecast.json",
               "LW1": "node_LW1_window_forecast.json"}
    out = {}
    for name, fname in sources.items():
        rob = json.loads((ROOT / "results" / fname).read_text())["windows"]["900s"]["robustness"]
        out[name] = {}
        for m in MODELS:
            skill = [rob[s][m]["mss"] for s, _ in DELAY_SCENARIOS]
            out[name][m] = {"skill_by_delay_min": dict(zip((d for _, d in DELAY_SCENARIOS), skill,
                                                           strict=True)),
                            "half_life_min": half_life([d for _, d in DELAY_SCENARIOS], skill)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "sanity_checks.json")
    a = ap.parse_args()
    res = {"controls": {n: controls(n) for n in ("CC1", "LW1")}, "evidence_age": evidence_age()}
    for node, c in res["controls"].items():
        factor = (1 - c["window_autocorr_15_30_60_120min"][0]) / (1 + c["window_autocorr_15_30_60_120min"][0])
        print(f"\n== {node}: {c['sweeps']} sweeps, {c['days']:.2f} d (CC1: development region only)")
        print("1. negative control (busy prevalence in idle 5 GHz bands): "
              + ", ".join(f"{k} {v:.4f}" for k, v in c["negative_control_prevalence"].items())
              + f"   | target ch6: {c['target_prevalence']:.3f}")
        print(f"2. drift: idle-band hourly medians span {c['idle_band_hourly_median_span_db']:.2f} dB; "
              f"target quiet level spans {c['target_quiet_level_span_db']:.2f} dB (threshold 3 dB)")
        print(f"3. spectral shape: busy-minus-idle excess {c['spectral_in_channel_mean_db']:+.1f} dB in "
              f"2430-2445 MHz vs {c['spectral_out_of_channel_mean_db']:+.1f} dB in 2455-2490 MHz")
        print(f"4. dependence: autocorr at 15/30/60/120 min = "
              f"{[round(x, 2) for x in c['window_autocorr_15_30_60_120min']]}; {c['n_windows']} windows "
              f"-> ~{c['n_effective_ar1']:.0f} effective (factor {factor:.2f})")
        res.setdefault("effective_fraction", {})[node] = factor
    print("\n5. value of old evidence (W = 15 min; MSE skill vs train mean at delay 0/1/10/30/60/120 min):")
    for name, models in res["evidence_age"].items():
        for m, v in models.items():
            sk = "  ".join(f"{x:+.2f}" for x in v["skill_by_delay_min"].values())
            print(f"   {name:<9}{m:<14}{sk}   half-life {v['half_life_min']:.0f} min")
    print("\nNominal windows scored at W = 15 min and the indicative effective numbers "
          "(dependence measured on CC1 dev / LW1):")
    for name, n in NOMINAL_WINDOWS.items():
        fac = res["effective_fraction"]["LW1" if name == "LW1" else "CC1"]
        res.setdefault("nominal_vs_effective_windows", {})[name] = [n, n * fac]
        print(f"   {name}: {n} windows -> ~{n * fac:.0f} effective")
    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {out.as_posix()}")


if __name__ == "__main__":
    main()
