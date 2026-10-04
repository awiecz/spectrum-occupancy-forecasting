"""Binary occupancy metrics. Positive class = busy (occupied).

Denominators are explicit:
* false-idle rate  = FN / (FN + TP)   -- fraction of truly busy targets predicted idle
* false-busy rate  = FP / (FP + TN)   -- fraction of truly idle targets predicted busy
"""

from __future__ import annotations

import numpy as np


def confusion(y, pred) -> dict[str, int]:
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    return {
        "tp": int(np.sum(y & pred)),
        "fn": int(np.sum(y & ~pred)),
        "fp": int(np.sum(~y & pred)),
        "tn": int(np.sum(~y & ~pred)),
    }


def _div(a, b):
    return float(a / b) if b > 0 else float("nan")


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def reliability(y, p, n_bins: int = 10):
    """Equal-width reliability table: (bin_center, mean_pred, frac_pos, count) arrays."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.bincount(idx, weights=p, minlength=n_bins) / count
        frac = np.bincount(idx, weights=y, minlength=n_bins) / count
    centers = (np.arange(n_bins) + 0.5) / n_bins
    return centers, mean_p, frac, count


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error (equal-width bins)."""
    _, mean_p, frac, count = reliability(y, p, n_bins)
    ok = count > 0
    return float(np.sum(count[ok] * np.abs(mean_p[ok] - frac[ok])) / count.sum())


def binary_report(y, p, threshold: float = 0.5) -> dict[str, float]:
    """Full report for probabilistic predictions ``p`` of busy targets ``y``."""
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    c = confusion(y, p >= threshold)
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    recall = _div(tp, tp + fn)
    precision = _div(tp, tp + fp)
    specificity = _div(tn, tn + fp)
    f1 = _div(2 * precision * recall, precision + recall) if tp > 0 else 0.0
    return {
        **c,
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "precision": precision,
        "recall": recall,
        "false_idle_rate": _div(fn, fn + tp),
        "false_busy_rate": _div(fp, fp + tn),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "f1": f1,
        "brier": brier(y, p),
        "ece": ece(y, p),
    }


def fraction_report(y, p, climatology: float, theta: float) -> dict[str, float]:
    """Metrics for forecasts ``p`` of a busy FRACTION ``y`` in [0, 1].

    * ``mse``/``rmse``/``mae``; ``mss`` = 1 - mse / mse(climatology) (mean-squared-error skill
      score vs the *training* mean; >0 beats climatology); ``r2`` = 1 - mse / var(y).
    * Event metrics for "high-activity window" = ``y >= theta``, predicted when ``p >= theta``;
      ``false_idle_rate`` = FN / (FN + TP): high-activity windows forecast as below ``theta``.
    * ``bias`` = mean(p) - mean(y); ``calib_slope`` = OLS slope of y on p (1 = calibrated).
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    mse = float(np.mean(err**2))
    mse_clim = float(np.mean((y - climatology) ** 2))
    var = float(np.var(y))
    slope = float(np.cov(p, y, bias=True)[0, 1] / np.var(p)) if np.var(p) > 0 else float("nan")
    ev = confusion(y >= theta, p >= theta)
    tp, fn, fp, tn = ev["tp"], ev["fn"], ev["fp"], ev["tn"]
    return {
        "n": int(len(y)),
        "mean_y": float(y.mean()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "mss": 1.0 - mse / mse_clim if mse_clim > 0 else float("nan"),
        "r2": 1.0 - mse / var if var > 0 else float("nan"),
        "bias": float(err.mean()),
        "calib_slope": slope,
        "event_prevalence": float(np.mean(y >= theta)),
        "event_false_idle": _div(fn, fn + tp),
        "event_false_busy": _div(fp, fp + tn),
        "event_recall": _div(tp, tp + fn),
        "event_precision": _div(tp, tp + fp),
    }


def event_tradeoff(y_frac, score, theta: float, false_busy_budgets=(0.1, 0.2)) -> dict[str, float]:
    """Compare forecasters on the event ``y_frac >= theta`` at EQUAL false-busy rates.

    For each budget the decision threshold on ``score`` is set so that this fraction of non-event
    windows is flagged busy. This uses the supplied evaluation labels: it is an oracle ROC
    comparison, not a threshold selected on development data or a deployable false-busy guarantee.
    The false-idle rate is the fraction of event windows scored below it. Also returns ROC AUC.
    """
    from sklearn.metrics import roc_auc_score

    ev = np.asarray(y_frac, dtype=float) >= theta
    s = np.asarray(score, dtype=float)
    out: dict[str, float] = {"event_prevalence": float(ev.mean())}
    if ev.all() or not ev.any():
        return out | {"auc": float("nan")}
    out["auc"] = float(roc_auc_score(ev, s))
    neg, pos = s[~ev], s[ev]
    for fb in false_busy_budgets:
        thr = np.quantile(neg, 1.0 - fb)
        out[f"false_idle_at_fb{int(fb * 100)}"] = float(np.mean(pos < thr))
    return out
