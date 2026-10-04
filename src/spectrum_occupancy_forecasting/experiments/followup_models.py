"""Earlier-only selection of history and learned forecasts for the v2 study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectrum_occupancy_forecasting.experiments.followup_data import (
    feature_columns,
    maximum_lookback,
)
from spectrum_occupancy_forecasting.robustness.degrade import Degradation


def candidates(cfg):
    out = [
        {"id": f"recent_{w}s", "family": "history", "window": w} for w in cfg["history_windows_s"]
    ]
    out += [{"id": f"decay_{tau}s", "family": "history", "tau": tau} for tau in cfg["decay_taus_s"]]
    for family in ("ridge", "hgb", "resid_ridge", "resid_hgb"):
        if "ridge" in family:
            out += [
                {"id": f"{family}_a{a}", "family": family, "alpha": a} for a in cfg["ridge_alphas"]
            ]
        else:
            out += [
                {"id": f"{family}_d{d}_l{n}", "family": family, "depth": d, "leaf": n}
                for d in cfg["boosting_depths"]
                for n in cfg["boosting_leaf_sizes"]
            ]
    return out


def history_prediction(candidate, X, mean):
    if "window" in candidate:
        w = candidate["window"]
        return X[f"frac_{w}s"].fillna(mean).to_numpy()
    tau = candidate["tau"]
    weight = X[f"decay_weight_{tau}s"].to_numpy()
    fraction = X[f"decay_fraction_{tau}s"].fillna(0).to_numpy()
    return (fraction * weight + mean) / (weight + 1.0)


class ForecastModel:
    def __init__(self, candidate, cfg, variant="full", base=None):
        self.candidate, self.cfg, self.variant, self.base = candidate, cfg, variant, base

    def fit(self, X, y):
        self.mean = float(np.mean(y))
        family = self.candidate["family"]
        if family == "history":
            return self
        self.columns = feature_columns(X, self.variant)
        if not self.columns:
            raise ValueError("no usable learned features")
        if "ridge" in family:
            self.est = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=self.candidate["alpha"]),
            )
        else:
            residual = family.startswith("resid_")
            self.est = HistGradientBoostingRegressor(
                max_iter=self.cfg["residual_boosting_iterations"]
                if residual
                else self.cfg["boosting_iterations"],
                learning_rate=self.cfg["residual_boosting_learning_rate"]
                if residual
                else self.cfg["boosting_learning_rate"],
                max_depth=self.candidate["depth"],
                min_samples_leaf=self.candidate["leaf"],
                random_state=self.cfg["model_seed"],
                early_stopping=False,
            )
        target = np.asarray(y)
        if family.startswith("resid_"):
            target = target - history_prediction(self.base, X, self.mean)
        self.est.fit(X[self.columns].to_numpy(), target)
        return self

    def predict(self, X):
        family = self.candidate["family"]
        if family == "history":
            return history_prediction(self.candidate, X, self.mean)
        p = self.est.predict(X[self.columns].to_numpy())
        if family.startswith("resid_"):
            p = p + history_prediction(self.base, X, self.mean)
        return np.clip(p, 0.0, 1.0)


def training(node, cutoff, lead):
    origins = node.origins(node.t0, cutoff, lead, node.cfg["training_stride_s"])
    o, y, _ = node.dataset(origins, lead)
    if not len(y):
        raise ValueError("no eligible training targets")
    assert np.max(o + lead + node.cfg["window_s"]) <= cutoff + 1e-6
    return node.features(o), y


def select_models(node, eval_lo, lead, variant):
    """Fit every candidate on earlier targets, score fixed chronological validation blocks.

    History is selected first. Residual validation conditions on that same validation-selected
    history; these scores are selection diagnostics, not independent estimates of performance.
    """
    cfg = node.cfg
    lookback = maximum_lookback(cfg)
    cutoff = eval_lo - lookback
    start = cutoff - cfg["validation_blocks"] * cfg["validation_block_s"]
    if start < node.t0 + cfg["calibration_s"]:
        raise ValueError("validation starts before initial calibration finishes")
    blocks = []
    for i in range(cfg["validation_blocks"]):
        lo = start + i * cfg["validation_block_s"]
        hi = lo + cfg["validation_block_s"]
        Xtr, ytr = training(node, lo - lookback, lead)
        o, y, _ = node.dataset(node.origins(lo, hi, lead, cfg["window_s"], common=True), lead)
        if not len(y):
            raise ValueError("no eligible validation targets")
        blocks.append((Xtr, ytr, node.features(o), y, o))
    all_candidates = candidates(cfg)
    records, predictions = [], {}

    def score(candidate, base=None):
        pp = []
        for Xtr, ytr, Xv, yv, ov in blocks:
            m = ForecastModel(candidate, cfg, variant, base).fit(Xtr, ytr)
            p = m.predict(Xv)
            pp.append(pd.DataFrame({"origin_s": ov, "y": yv, "prediction": p}))
        frame = pd.concat(pp, ignore_index=True)
        predictions[candidate["id"]] = frame
        records.append(
            {
                **candidate,
                "validation_mse": float(np.mean((frame["prediction"] - frame["y"]) ** 2)),
                "validation_n": len(frame),
            }
        )

    for candidate in all_candidates:
        if candidate["family"] == "history":
            score(candidate)
    history = min(records, key=lambda r: r["validation_mse"])
    for candidate in all_candidates:
        if candidate["family"] != "history":
            score(candidate, history)
    selected = {"selected_history": history}
    for family in ("ridge", "hgb", "resid_ridge", "resid_hgb"):
        selected[family] = min(
            (r for r in records if r["family"] == family), key=lambda r: r["validation_mse"]
        )
    selected["selected_learned"] = min(
        (r for r in selected.values() if r["family"] != "history"),
        key=lambda r: r["validation_mse"],
    )
    selected["fixed_history"] = next(r for r in records if r["id"] == "recent_7200s")
    selected["climatology"] = {"id": "climatology", "family": "climatology"}
    validation = {}
    for name, choice in selected.items():
        if name == "climatology":
            parts = [
                pd.DataFrame({"origin_s": ov, "y": yv, "prediction": np.full(len(yv), ytr.mean())})
                for _, ytr, _, yv, ov in blocks
            ]
            validation[name] = pd.concat(parts, ignore_index=True)
        else:
            validation[name] = predictions[choice["id"]]
    return selected, records, validation


def augment_training(node, X, y):
    cfg = node.cfg
    aug = [
        Degradation(dropout_p=0.5),
        Degradation(dropout_p=0.9),
        Degradation(delay_s=600),
        Degradation(delay_s=3600),
        Degradation(outage_len_s=3600, outage_frac=0.25),
    ]
    origins = X["t_start_s"].to_numpy()
    X = pd.concat(
        [X]
        + [
            node.features(origins, deg, seed)
            for deg, seed in zip(aug, cfg["augmentation_seeds"], strict=True)
        ],
        ignore_index=True,
    )
    return X, np.tile(y, len(aug) + 1)


def fit_selected(node, eval_lo, lead, variant, selected, *, augmented=False, names=None):
    X, y = training(node, eval_lo - maximum_lookback(node.cfg), lead)
    if augmented:
        X, y = augment_training(node, X, y)
    models = {}
    for name, choice in selected.items():
        if names is not None and name not in names:
            continue
        if name == "climatology":
            models[name] = float(y.mean())
        else:
            models[name] = ForecastModel(
                choice, node.cfg, variant, selected["selected_history"]
            ).fit(X, y)
    return models


def augmented_validation(node, eval_lo, lead, variant, selected):
    cfg = node.cfg
    lookback = maximum_lookback(cfg)
    start = eval_lo - lookback - cfg["validation_blocks"] * cfg["validation_block_s"]
    output = {name + "_aug": [] for name in ("ridge", "hgb")}
    for i in range(cfg["validation_blocks"]):
        lo = start + i * cfg["validation_block_s"]
        X, y = training(node, lo - lookback, lead)
        Xa, ya = augment_training(node, X, y)
        o, yv, _ = node.dataset(
            node.origins(lo, lo + cfg["validation_block_s"], lead, cfg["window_s"], common=True),
            lead,
        )
        Xv = node.features(o)
        for name in ("ridge", "hgb"):
            model = ForecastModel(selected[name], cfg, variant, selected["selected_history"]).fit(
                Xa, ya
            )
            output[name + "_aug"].append(
                pd.DataFrame({"origin_s": o, "y": yv, "prediction": model.predict(Xv)})
            )
    return {name: pd.concat(parts, ignore_index=True) for name, parts in output.items()}


def operating_threshold(y, score, budget, cfg):
    """Lowest score threshold whose VALIDATION FPR meets the budget, including ties."""
    event = np.asarray(y) >= cfg["event_cutoff"]
    score = np.asarray(score)
    npos, nneg = int(event.sum()), int((~event).sum())
    result = {"n_events": npos, "n_nonevents": nneg, "budget": budget}
    if npos < cfg["minimum_validation_events"] or nneg < cfg["minimum_validation_nonevents"]:
        return result | {"status": "insufficient_calibration", "threshold": None}
    for threshold in np.append(np.unique(score), np.nextafter(float(score.max()), np.inf)):
        fpr = float(np.mean(score[~event] >= threshold))
        if fpr <= budget:
            return result | {
                "status": "calibrated",
                "threshold": float(threshold),
                "validation_false_busy": fpr,
                "validation_false_idle": float(np.mean(score[event] < threshold)),
            }
    raise AssertionError("all-negative threshold must satisfy the budget")
