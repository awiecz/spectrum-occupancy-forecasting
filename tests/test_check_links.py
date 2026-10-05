import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_links", Path(__file__).resolve().parents[1] / "scripts/check_links.py"
)
check_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_links)


def tree(tmp_path, readme, other="# Title\n\n## Second part (extra)\n"):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    (tmp_path / "docs/other.md").write_text(other, encoding="utf-8")
    (tmp_path / "docs/fig.png").write_bytes(b"x")
    return tmp_path


def test_valid_links_images_and_anchors_pass(tmp_path):
    root = tree(
        tmp_path,
        "[a](docs/other.md) ![f](docs/fig.png) [b](docs/other.md#second-part-extra) "
        "[c](#local)\n\n## Local\n[ext](https://example.com/x)\n",
    )
    assert check_links.check(root) == []


def test_missing_file_and_anchor_are_reported(tmp_path):
    root = tree(tmp_path, "[a](docs/nope.md)\n[b](docs/other.md#absent)\n")
    problems = check_links.check(root)
    assert any("missing file docs/nope.md" in p for p in problems)
    assert any("missing anchor docs/other.md#absent" in p for p in problems)


def test_links_in_code_fences_and_inline_code_are_ignored(tmp_path):
    root = tree(tmp_path, "```\n[a](docs/nope.md)\n```\n`[b](docs/nope.md)`\n")
    assert check_links.check(root) == []


def test_duplicate_headings_get_numbered_anchors(tmp_path):
    root = tree(tmp_path, "[a](docs/other.md#same-1)\n", other="# Same\n\n# Same\n")
    assert check_links.check(root) == []


def test_slug_matches_github_for_punctuation():
    assert check_links.slug("Operational learning (declared in the registry)") == (
        "operational-learning-declared-in-the-registry"
    )
    assert check_links.slug("Fix `code` & Things!") == "fix-code--things"
