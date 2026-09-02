import json
from pathlib import Path

from corpusguard.audit import audit_dataset


def _write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_clean_dataset_scores_high(tmp_path: Path):
    path = tmp_path / "clean.jsonl"
    _write(
        path,
        [
            {"prompt": "Explain photosynthesis simply.", "completion": "Plants use light to turn water and carbon dioxide into stored chemical energy."},
            {"question": "Capital of Kazakhstan?", "answer": "Astana is the capital of Kazakhstan."},
        ],
    )
    report = audit_dataset(path)
    assert report.metrics.valid_rows == 2
    assert report.metrics.invalid_rows == 0
    assert report.score >= 90


def test_detects_duplicates_and_pii(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    row = {"prompt": "Contact me at test@example.com", "completion": "My phone is +7 701 123 45 67."}
    _write(path, [row, row])
    report = audit_dataset(path)
    assert report.metrics.exact_duplicate_rows == 1
    assert report.metrics.pii_hits == 2
    assert report.score < 90


def test_detects_split_leakage(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    shared = {"instruction": "Add 2 and 2", "output": "The result is 4."}
    _write(train, [shared, {"prompt": "Say hello", "completion": "Hello there! Nice to meet you."}])
    _write(eval_path, [shared])
    report = audit_dataset(train, compare_path=eval_path)
    assert report.metrics.leakage_matches == 1
    assert any(f.code == "split_leakage" for f in report.findings)
