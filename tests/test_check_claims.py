import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_claims", ROOT / "scripts/check_claims.py")
check_claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_claims)


def tree(tmp_path, value=12.35, doc="The gain was 12.35% here."):
    (tmp_path / "docs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "results/s.csv").write_text("node,gain\nA,0.12349\nB,0.5\n", encoding="utf-8")
    (tmp_path / "docs/d.md").write_text(doc, encoding="utf-8")
    claim = {
        "id": "T1",
        "statement": "gain at A",
        "value": value,
        "unit": "%",
        "decimals": 2,
        "source": {
            "kind": "csv",
            "file": "results/s.csv",
            "column": "gain",
            "transform": "pct",
            "filter": {"node": "A"},
        },
        "cited_in": ["docs/d.md"],
    }
    (tmp_path / "docs/claims.yaml").write_text(
        yaml.safe_dump({"claims": [claim]}), encoding="utf-8"
    )
    return tmp_path, [claim]


def test_a_matching_claim_passes(tmp_path):
    root, claims = tree(tmp_path)
    assert check_claims.verify(claims, root) == []


def test_a_wrong_value_is_reported(tmp_path):
    root, claims = tree(tmp_path, value=99.0)
    assert any("file gives 12.35" in p for p in check_claims.verify(claims, root))


def test_a_number_missing_from_the_document_is_reported(tmp_path):
    root, claims = tree(tmp_path, doc="The gain was about twelve percent.")
    assert any("not found in docs/d.md" in p for p in check_claims.verify(claims, root))


def test_a_filter_matching_several_rows_is_an_error(tmp_path):
    root, claims = tree(tmp_path)
    claims[0]["source"]["filter"] = {}
    assert any("expected 1" in p for p in check_claims.verify(claims, root))


def test_the_rendered_table_lists_every_claim(tmp_path):
    root, claims = tree(tmp_path)
    rendered = check_claims.table(claims)
    assert "| T1 | gain at A | 12.35% |" in rendered


def test_every_claim_in_the_repository_holds():
    claims = yaml.safe_load((ROOT / "docs/claims.yaml").read_text(encoding="utf-8"))["claims"]
    ids = [c["id"] for c in claims]
    assert len(ids) == len(set(ids))
    assert check_claims.verify(claims, ROOT) == []


def test_the_committed_table_is_current():
    claims = yaml.safe_load((ROOT / "docs/claims.yaml").read_text(encoding="utf-8"))["claims"]
    committed = (ROOT / "docs/claims.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == check_claims.table(claims)


def test_latex_percent_minus_and_thin_space_forms_are_recognised(tmp_path):
    root, claims = tree(tmp_path, value=-12.35, doc=r"The change was $-$12.35\,\% here.")
    (root / "results/s.csv").write_text("node,gain\nA,-0.12349\n", encoding="utf-8")
    claims[0]["cited_in"] = ["docs/note.tex"]
    (root / "docs/note.tex").write_text(r"The change was $-$12.35\,\% here.", encoding="utf-8")
    assert check_claims.verify(claims, root) == []


def test_a_missing_cited_document_is_reported_not_raised(tmp_path):
    root, claims = tree(tmp_path)
    claims[0]["cited_in"] = ["docs/absent.md"]
    assert any("does not exist" in p for p in check_claims.verify(claims, root))


def test_unicode_minus_in_documents_matches_ascii_values(tmp_path):
    root, claims = tree(tmp_path, value=-12.35, doc="The change was \u221212.35% here.")
    (root / "results/s.csv").write_text("node,gain\nA,-0.12349\n", encoding="utf-8")
    assert check_claims.verify(claims, root) == []
