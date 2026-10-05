import pytest

from spectrum_occupancy_forecasting.experiments.adaptive_audit import audit


def test_historical_audit_explains_the_missing_local_archive(tmp_path):
    with pytest.raises(FileNotFoundError, match="--mode report"):
        audit(tmp_path, {}, tmp_path / "out")


def test_final_study_evidence_is_not_hidden_from_git():
    """Final predictions must be committable and per-fold checkpoints must not be."""
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("git") is None:
        pytest.skip("git not available")
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")

    def ignored(path):
        done = subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=root, capture_output=True, check=False
        )
        return done.returncode == 0

    for study in ("sparse_regime", "operational_learning", "residual_state_space"):
        assert not ignored(f"results/{study}/run/predictions.parquet"), study
    assert ignored("results/sparse_regime/run/_parts/CC1__clean__0.parquet")
