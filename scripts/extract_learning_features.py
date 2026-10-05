"""Extract first-session learning inputs and later LW1 calibration only."""

from pathlib import Path

from spectrum_occupancy_forecasting.experiments.fine_spectrum import extract

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cache = ROOT / "reproductions/shared_learning_cache"
    for node in ("CC1", "LW1"):
        extract(ROOT / f"data/raw/aerpaw/Results{node}Feb2022_SigMF.zip", node, 0, cache)
    extract(ROOT / "data/raw/aerpaw/ResultsLW1Feb2022_SigMF.zip", "LW1", 2, cache, "calibration")
