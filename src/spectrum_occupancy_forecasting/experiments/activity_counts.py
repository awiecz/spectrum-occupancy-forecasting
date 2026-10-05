"""Delivered five-minute counts and a deterministic latent-logit grid filter."""

import pickle
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import betaln, expit, gammaln, logsumexp, ndtr

from spectrum_occupancy_forecasting.robustness.degrade import Degradation, visibility_mask


@dataclass(frozen=True)
class CountCandidate:
    family: str = "history"
    half_life: float = 0
    sd: float = 0
    rho: float = 0
    strength: float = 0

    @property
    def key(self):
        return self.half_life, self.sd, self.rho

    @property
    def id(self):
        return (
            "count_history"
            if self.family == "history"
            else "count_" + "_".join(str(v) for v in (*self.key, self.strength))
        )

    @property
    def tie(self):
        return self.family != "history", self.strength, -self.half_life, self.sd, self.rho


def candidates(cfg):
    return [CountCandidate()] + [
        CountCandidate("count", h, s, rho, w)
        for h in cfg["count_half_lives_h"]
        for s in cfg["count_standard_deviations"]
        for rho in cfg["count_dispersions"]
        for w in cfg["count_strengths"]
    ]


def count_bins(node, cfg, deg=None, seed=0):
    """Completion-time half-open bins; calibration is always clean and frozen."""
    deg = deg or Degradation()
    times = node.frame.time_s.to_numpy() + 0.001  # verified cache dwell used by LearningNode
    start = node.data.t0
    width = cfg["count_bin_s"]
    visible = visibility_mask(node.frame.time_s.to_numpy(), deg, seed)
    visible[times <= start + 86400] = True
    indices = np.floor((times - start) / width).astype(int)
    number = int(np.floor((times[-1] - start) / width))
    keep = visible & (indices >= 0) & (indices < number)
    busy = node.frame.target_dbm.to_numpy() > node.data.thresholds[2437.0]
    n = np.bincount(indices[keep], minlength=number)
    k = np.bincount(indices[keep & busy], minlength=number)
    end = start + (np.arange(number) + 1) * width
    return pd.DataFrame(
        {
            "bin_start_s": end - width,
            "event_s": end,
            "release_s": np.where(end <= start + 86400, end, end + deg.delay_s),
            "busy_count": k,
            "total_count": n,
            "usable": n >= cfg["count_coverage"] * width / node.data.dt,
        }
    )


def log_likelihood(grid, k, n, rho):
    if not (0 <= k <= n) or n < 0 or not 0 <= rho < 1:
        raise ValueError("invalid counts or dispersion")
    p = expit(grid)
    coefficient = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    if rho == 0:
        return coefficient + k * (-np.logaddexp(0, -grid)) + (n - k) * (-np.logaddexp(0, grid))
    concentration = 1 / rho - 1
    a, b = p * concentration, (1 - p) * concentration
    return coefficient + betaln(k + a, n - k + b) - betaln(a, b)


@lru_cache(maxsize=64)
def transition(nodes, limit, mu, half_life, sd, elapsed_s):
    """Cell-integrated Gaussian transition; nine-SD sparse tails < 3e-19."""
    if elapsed_s < 0:
        raise ValueError("cannot propagate backwards")
    if elapsed_s == 0:
        from scipy.sparse import eye

        return eye(nodes, format="csr")
    grid = np.linspace(-limit, limit, nodes)
    step = grid[1] - grid[0]
    phi = np.exp2(-elapsed_s / (half_life * 3600))
    sigma = sd * np.sqrt(-np.expm1(-2 * np.log(2) * elapsed_s / (half_life * 3600)))
    means = mu + phi * (grid - mu)
    rows, cols, values = [], [], []
    for source, mean in enumerate(means):
        low = max(0, int(np.floor((mean - 9 * sigma + limit) / step)) - 1)
        high = min(nodes, int(np.ceil((mean + 9 * sigma + limit) / step)) + 2)
        destinations = np.arange(low, high)
        edges_low = grid[destinations] - step / 2
        edges_high = edges_low + step
        edges_low[destinations == 0] = -np.inf
        edges_high[destinations == nodes - 1] = np.inf
        weights = ndtr((edges_high - mean) / sigma) - ndtr((edges_low - mean) / sigma)
        weights /= weights.sum()
        rows.extend(destinations)
        cols.extend([source] * len(destinations))
        values.extend(weights)
    return csr_matrix((values, (rows, cols)), shape=(nodes, nodes))


class CountFilter:
    def __init__(self, mu, half_life, sd, rho, nodes=801, limit=16):
        self.mu, self.half_life, self.sd, self.rho = mu, half_life, sd, rho
        self.grid = np.linspace(-limit, limit, nodes)
        self.limit = limit
        edges = np.r_[-np.inf, (self.grid[:-1] + self.grid[1:]) / 2, np.inf]
        self.posterior = np.diff(ndtr((edges - mu) / sd))
        self.time = None

    def distribution(self, time):
        if self.time is None:
            return self.posterior.copy()
        return (
            transition(
                len(self.grid),
                self.limit,
                self.mu,
                self.half_life,
                self.sd,
                float(time - self.time),
            )
            @ self.posterior
        )

    def assimilate(self, time, k, n):
        prior = self.distribution(time)
        logs = np.log(np.maximum(prior, np.finfo(float).tiny)) + log_likelihood(
            self.grid, k, n, self.rho
        )
        self.posterior = np.exp(logs - logsumexp(logs))
        if not np.isfinite(self.posterior).all() or not np.isclose(self.posterior.sum(), 1):
            raise ValueError("posterior normalization failure")
        self.time = time

    def predict(self, completion_times):
        return float(np.mean([self.distribution(t) @ expit(self.grid) for t in completion_times]))


class CountBank:
    """Adapter: count measurements update filters; delivered targets update safety only."""

    snapshot_features = "history"

    def __init__(self, cfg, bins, startup, active=None):
        self.cfg = cfg
        self.candidates = candidates(cfg)
        self.preprocessing = {}
        self.feedback, self.failures, self.state_records, self.update_records = [], [], [], []
        self.bins = bins.to_dict("records")
        initial = [b for b in self.bins if b["event_s"] <= startup and b["usable"]]
        self.insufficient = len(initial) < cfg["minimum_count_initialization"]
        busy = sum(b["busy_count"] for b in initial)
        total = sum(b["total_count"] for b in initial)
        p = (busy + 0.5) / (total + 1)
        mu = float(np.log(p / (1 - p)))
        self.states = {
            c.key: CountFilter(mu, *c.key, nodes=cfg["grid_nodes"], limit=cfg["grid_limit"])
            for c in self.candidates[1:]
            if active is None or c.id in active
        }
        self.cursor = 0
        self.advance(startup)

    def advance(self, time):
        while self.cursor < len(self.bins) and self.bins[self.cursor]["release_s"] <= time:
            record = self.bins[self.cursor]
            if record["usable"] and not self.insufficient:
                for key, state in self.states.items():
                    try:
                        state.assimilate(
                            record["event_s"], record["busy_count"], record["total_count"]
                        )
                    except (ValueError, FloatingPointError) as error:
                        self.failures.append(
                            {"state": list(key), "time": time, "reason": str(error)}
                        )
            self.update_records.append(
                record
                | {
                    "update_reason": "updated"
                    if record["usable"] and not self.insufficient
                    else "insufficient_coverage_or_initialization"
                }
            )
            self.cursor += 1

    def update(self, time, target_end, y, base, experts, vectors):
        if target_end > time:
            raise ValueError("feedback before target completion")
        self.feedback.append((time, target_end))

    def forecasts(self, time, base, selected, experts, vectors, age, coverage, active=None):
        self.advance(time)
        recent = [
            end
            for release, end in self.feedback
            if release <= time and time - self.cfg["feedback_window_s"] < end <= time
        ]
        reason = (
            "insufficient_initialization"
            if self.insufficient
            else "stale_input"
            if not np.isfinite(age) or age > 900
            else "low_input_coverage"
            if not np.isfinite(coverage) or coverage < 0.5
            else "insufficient_feedback"
            if len(recent) < 4
            else "stale_feedback"
            if time - max(recent) > 3600
            else "model_failure"
            if self.failures
            else ""
        )
        means = {}
        for key, state in self.states.items():
            if active is not None and not any(
                c.key == key and c.id in active for c in self.candidates[1:]
            ):
                continue
            means[key] = None if reason else state.predict([time + 1200, time + 1500, time + 1800])
            self.state_records.append(
                {
                    "origin_s": time,
                    "state": str(key),
                    "event_s": state.time,
                    "mean_logit": float(state.posterior @ state.grid),
                    "variance_logit": float(
                        state.posterior @ state.grid**2 - (state.posterior @ state.grid) ** 2
                    ),
                    "expected_activity": means[key],
                }
            )
        return [
            {
                "candidate": c.id,
                "prediction": float(
                    base
                    if reason or c.family == "history"
                    else (1 - c.strength) * base + c.strength * means[c.key]
                ),
                "fallback_reason": reason,
                "feedback_windows": len(recent),
                "correction": 0.0
                if reason or c.family == "history"
                else c.strength * (means[c.key] - base),
                "raw_correction": None if reason or c.family == "history" else means[c.key] - base,
            }
            for c in self.candidates
            if active is None or c.id in active
        ]

    def checkpoint(self):
        return pickle.dumps(self, protocol=5)

    @staticmethod
    def resume(payload):
        return pickle.loads(payload)
