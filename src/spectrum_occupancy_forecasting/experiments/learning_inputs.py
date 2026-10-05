"""Neutral calibrated recording inputs and training-only preprocessing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum_occupancy_forecasting.robustness.degrade import Degradation, visibility_mask

from .fine_spectrum import FINE_COLUMNS, raw_table, sha
from .followup_data import NodeData
from .followup_models import history_prediction
from .mixture_support import history_candidates

FIXED = {"id": "recent_7200s", "window": 7200}


def cached_frame(cache, folder):
    allowed = {
        "CC1_session0_development": ("CC1", 0, "development"),
        "LW1_session0_development": ("LW1", 0, "development"),
        "LW1_session2_assessment": ("LW1", 2, "assessment"),
    }
    if folder not in allowed:
        raise ValueError("only already-inspected CC1/LW1 caches are permitted")
    path = Path(cache) / folder
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("incomplete input cache")
    if (manifest["node"], manifest["session"], manifest["purpose"]) != allowed[folder]:
        raise ValueError("cache recording identity mismatch")
    for name, digest in manifest["shards"].items():
        if Path(name).name != name or sha(path / name) != digest:
            raise ValueError("corrupted or unsupported input shard")
    return pd.concat(
        [pd.read_parquet(path / name) for name in sorted(manifest["shards"])], ignore_index=True
    )


class LearningNode:
    def __init__(self, frame, name, cfg):
        if np.any(np.diff(frame.time_s) <= 0):
            raise ValueError("fine observations must be unique and ordered")
        self.frame, self.cfg = frame.copy(), cfg
        self.data = NodeData(raw_table(frame, name), cfg, 3.0)
        cal = frame.time_s.to_numpy() + 0.001 <= self.data.t0 + 86400
        self.fine_floors = np.quantile(frame.loc[cal, FINE_COLUMNS], 0.05, axis=0)
        self.normalized = frame[FINE_COLUMNS].to_numpy() - self.fine_floors
        self._fine_cache = {}

    def features(self, origins, mean, variant, deg=None, seed=0, *, calibration_clean=False):
        deg = deg or Degradation()
        X = self.data.features(origins, deg, seed, calibration_clean=calibration_clean)
        out = pd.DataFrame(
            {h["id"]: history_prediction(h, X, mean) for h in history_candidates(self.cfg)}
        )
        out["diff_5_30"] = X.frac_300s - X.frac_1800s
        out["diff_15_120"] = X.frac_900s - X.frac_7200s
        out["last_occ"], out["age_s"] = X.last_occ, X.age_s
        for w in self.cfg["history_windows_s"]:
            out[f"coverage_{w}s"] = X[f"n_{w}s"] / (w / self.data.dt)
        if variant == "history_fine":
            key = (deg, seed, calibration_clean)
            if key not in self._fine_cache:
                times = self.frame.time_s.to_numpy()
                visible = visibility_mask(times, deg, seed)
                if calibration_clean:
                    visible |= times + 0.001 <= self.data.t0 + 86400
                ends = times[visible] + 0.001
                values = self.normalized[visible]
                total = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
                self._fine_cache[key] = (ends, total)
            ends, total = self._fine_cache[key]
            query = np.asarray(origins) - deg.delay_s
            for w in (300, 900, 1800):
                hi = np.searchsorted(ends, query, side="left")
                lo = np.searchsorted(ends, query - w, side="right")
                count = hi - lo
                values = np.divide(
                    total[hi] - total[lo],
                    count[:, None],
                    out=np.full((len(query), len(FINE_COLUMNS)), np.nan),
                    where=count[:, None] > 0,
                )
                for j, name in enumerate(FINE_COLUMNS):
                    out[f"fine_{name}_{w}s"] = values[:, j]
        base = history_prediction(FIXED, X, mean)
        valid = (X.age_s.to_numpy() <= 900) & (X.n_900s.to_numpy() / (900 / self.data.dt) >= 0.5)
        return out, base, valid

    def training(self, cutoff, spacing):
        end = min(cutoff, self.data.last)
        origins = self.data.origins(self.data.t0, end, 900, spacing)
        o, y, count = self.data.dataset(origins, 900)
        if not len(y):
            raise ValueError("no eligible training labels")
        assert np.max(o + 1800) <= cutoff + 1e-6
        return o, y, count


class StableScaler:
    """Training-only site-weighted scaling with nonnegative centered variance."""

    def fit(self, values, sample_weight):
        self.mean_ = np.average(values, axis=0, weights=sample_weight)
        self.var_ = np.average((values - self.mean_) ** 2, axis=0, weights=sample_weight)
        eps = np.finfo(float).eps
        bound = len(values) * eps * self.var_ + (len(values) * self.mean_ * eps) ** 2
        self.scale_ = np.sqrt(self.var_)
        self.scale_[self.var_ <= bound] = 1.0
        return self

    def transform(self, values):
        return (values - self.mean_) / self.scale_
