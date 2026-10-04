"""First-look statistics of a block spectrogram: sessions, noise floor and activity per block.

    uv run python scripts/explore_blocks.py data/processed/aerpaw_CC1_blocks5MHz.npz
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import numpy as np

z = np.load(sys.argv[1])
t, f, mean, mx = z["times_s"], z["centres_mhz"], z["mean_dbm"], z["max_dbm"]
fmt = lambda x: datetime.fromtimestamp(x, UTC).strftime("%m-%d %H:%M")  # noqa: E731
print(f"sweeps {len(t)}  blocks {len(f)}  {f[0]:.1f}-{f[-1]:.1f} MHz")

# Recording sessions (split at gaps > 5 min)
cut = np.nonzero(np.diff(t) > 300)[0]
bounds = np.concatenate([[0], cut + 1, [len(t)]])
for a, b in zip(bounds[:-1], bounds[1:], strict=True):
    hours = (t[b - 1] - t[a]) / 3600
    print(f"session {fmt(t[a])} -> {fmt(t[b - 1])} UTC  {b - a} sweeps  {hours:.1f} h")

main = max(zip(bounds[:-1], bounds[1:], strict=True), key=lambda ab: ab[1] - ab[0])
a, b = main
tm, M, X = t[a:b], mean[a:b], mx[a:b]
print(f"main session: {len(tm)} sweeps, median dt {np.median(np.diff(tm)):.0f}s")

floor = np.percentile(M, 5, axis=0)
excess = X - floor  # how far the block's max bin rises above its own noise floor
active10 = (excess > 10).mean(axis=0)
active6 = (excess > 6).mean(axis=0)
var = M.std(axis=0)


def show(lo, hi, name, k=8):
    m = (f >= lo) & (f <= hi)
    idx = np.nonzero(m)[0]
    top = idx[np.argsort(-active10[idx])[:k]]
    print(f"\n{name} [{lo}-{hi} MHz]: blocks ranked by P(max bin > floor+10 dB)")
    for i in sorted(top, key=lambda i: -active10[i]):
        print(f"  {f[i]:7.1f} MHz  floor {floor[i]:7.1f} dBm  P>+6dB {active6[i]:.3f}  "
              f"P>+10dB {active10[i]:.3f}  std(mean) {var[i]:.2f} dB")


show(2400, 2484, "2.4 GHz ISM")
show(5150, 5850, "5 GHz U-NII")
show(700, 960, "700-960 MHz cellular")
show(1700, 2200, "1.7-2.2 GHz cellular")
