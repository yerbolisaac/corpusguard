import json
from pathlib import Path

import pytest

from corpusguard.audit import audit_dataset


class FakeTokenizer:
    @property
    def name(self) -> str:
        return "fake:test"

    def count(self, text: str) -> int:
        return len(text.split())


def _write(path: Path, rows):
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_clean_dataset_scores_high(tmp_path: Path):
    path = tmp_path / "clean.jsonl"

    _write(
        path,
        [
            {
                "prompt": "Explain photosynthesis simply.",
                "completion": (
                    "Plants use light to turn water and carbon dioxide "
                    "into stored chemical energy."
                ),
            },
            {
                "question": "Capital of Kazakhstan?",
                "answer": "Astana is the capital of Kazakhstan.",
            },
        ],
    )

    report = audit_dataset(path)

    assert report.metrics.valid_rows == 2
    assert report.metrics.invalid_rows == 0
    assert report.score >= 90


def test_detects_duplicates_and_pii(tmp_path: Path):
    path = tmp_path / "bad.jsonl"

    row = {
        "prompt": "Contact me at test@example.com",
        "completion": "My phone is +7 701 123 45 67.",
    }

    _write(path, [row, row])

    report = audit_dataset(path)

    assert report.metrics.exact_duplicate_rows == 1
    assert report.metrics.pii_hits == 2
    assert report.score < 90


def test_detects_split_leakage(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"

    shared = {
        "instruction": "Add 2 and 2",
        "output": "The result is 4.",
    }

    _write(
        train,
        [
            shared,
            {
                "prompt": "Say hello",
                "completion": "Hello there! Nice to meet you.",
            },
        ],
    )

    _write(eval_path, [shared])

    report = audit_dataset(
        train,
        compare_path=eval_path,
    )

    assert report.metrics.leakage_matches == 1
    assert any(
        finding.code == "split_leakage"
        for finding in report.findings
    )


def test_token_metrics_are_not_populated_without_tokenizer(
    tmp_path: Path,
):
    path = tmp_path / "dataset.jsonl"

    _write(
        path,
        [
            {
                "prompt": "one two",
                "completion": "three four five",
            },
        ],
    )

    report = audit_dataset(path)

    assert report.metrics.tokenizer is None
    assert report.metrics.avg_input_tokens is None
    assert report.metrics.avg_output_tokens is None
    assert report.metrics.p95_output_tokens is None
    assert report.metrics.max_output_tokens is None
    assert report.metrics.avg_total_tokens is None
    assert report.metrics.p95_total_tokens is None
    assert report.metrics.max_total_tokens is None
    assert report.metrics.context_window is None
    assert report.metrics.context_window_exceeded_rows is None


def test_token_metrics_are_calculated(
    tmp_path: Path,
):
    path = tmp_path / "dataset.jsonl"

    _write(
        path,
        [
            {
                "prompt": "one two",
                "completion": "three four five",
            },
            {
                "prompt": "one two three four",
                "completion": "five six",
            },
        ],
    )

    report = audit_dataset(
        path,
        tokenizer=FakeTokenizer(),
    )

    assert report.metrics.tokenizer == "fake:test"

    assert report.metrics.avg_input_tokens == 3.0
    assert report.metrics.avg_output_tokens == 2.5

    assert report.metrics.p95_output_tokens == 3
    assert report.metrics.max_output_tokens == 3

    assert report.metrics.avg_total_tokens == 5.5
    assert report.metrics.p95_total_tokens == 6
    assert report.metrics.max_total_tokens == 6

    assert report.metrics.context_window is None
    assert report.metrics.context_window_exceeded_rows is None


def test_context_window_exceeded_is_detected(
    tmp_path: Path,
):
    path = tmp_path / "dataset.jsonl"

    _write(
        path,
        [
            {
                "prompt": "one two three",
                "completion": "four five six",
            },
            {
                "prompt": "one",
                "completion": "two three",
            },
        ],
    )

    report = audit_dataset(
        path,
        tokenizer=FakeTokenizer(),
        context_window=5,
    )

    assert report.metrics.context_window == 5
    assert report.metrics.context_window_exceeded_rows == 1

    findings = [
        finding
        for finding in report.findings
        if finding.code == "context_window_exceeded"
    ]

    assert len(findings) == 1
    assert findings[0].row == 1

    assert any(
        "context window" in recommendation.lower()
        for recommendation in report.recommendations
    )


def test_context_window_equal_to_total_tokens_is_allowed(
    tmp_path: Path,
):
    path = tmp_path / "dataset.jsonl"

    _write(
        path,
        [
            {
                "prompt": "one two",
                "completion": "three four",
            },
        ],
    )

    report = audit_dataset(
        path,
        tokenizer=FakeTokenizer(),
        context_window=4,
    )

    assert report.metrics.context_window_exceeded_rows == 0
    assert not any(
        finding.code == "context_window_exceeded"
        for finding in report.findings
    )


def test_context_window_requires_tokenizer(
    tmp_path: Path,
):
    path = tmp_path / "dataset.jsonl"

    _write(
        path,
        [
            {
                "prompt": "one two",
                "completion": "three four",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="requires a tokenizer",
    ):
        audit_dataset(
            path,
            context_window=4096,
        )


def test_context_window_must_be_positive(
    tmp_path: Path,
):
    path = tmp_path / "dataset.jsonl"

    _write(
        path,
        [
            {
                "prompt": "one two",
                "completion": "three four",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        audit_dataset(
            path,
            tokenizer=FakeTokenizer(),
            context_window=0,
        )
