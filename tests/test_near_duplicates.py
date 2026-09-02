from __future__ import annotations

from corpusguard import checks
from corpusguard.checks import PairRecord, near_duplicate_findings


def test_near_duplicate_exact_path_detects_similar_records() -> None:
    common = " ".join(f"token{i}" for i in range(25))

    records = [
        PairRecord(
            row=1,
            prompt="Question",
            output=f"{common} left",
        ),
        PairRecord(
            row=2,
            prompt="Question",
            output=f"{common} right",
        ),
        PairRecord(
            row=3,
            prompt="Completely unrelated prompt",
            output="alpha beta gamma delta epsilon zeta eta theta",
        ),
    ]

    findings, pairs = near_duplicate_findings(records)

    assert pairs == 1
    assert len(findings) == 1
    assert findings[0].code == "near_duplicate"
    assert findings[0].row == 1
    assert "Rows 1 and 2" in findings[0].message


def test_identical_token_sets_are_not_near_duplicates() -> None:
    records = [
        PairRecord(
            row=1,
            prompt="alpha beta gamma",
            output="delta epsilon zeta",
        ),
        PairRecord(
            row=2,
            prompt="gamma beta alpha",
            output="zeta epsilon delta",
        ),
    ]

    findings, pairs = near_duplicate_findings(records)

    assert pairs == 0
    assert findings == []


def test_max_pairs_limits_findings_but_not_pair_count() -> None:
    common = " ".join(f"token{i}" for i in range(50))

    records = [
        PairRecord(
            row=1,
            prompt="Question",
            output=f"{common} variant_one",
        ),
        PairRecord(
            row=2,
            prompt="Question",
            output=f"{common} variant_two",
        ),
        PairRecord(
            row=3,
            prompt="Question",
            output=f"{common} variant_three",
        ),
    ]

    findings, pairs = near_duplicate_findings(
        records,
        max_pairs=1,
    )

    assert pairs == 3
    assert len(findings) == 1


def test_invalid_threshold_is_rejected() -> None:
    records = [
        PairRecord(
            row=1,
            prompt="Question",
            output="one two three four five six",
        )
    ]

    try:
        near_duplicate_findings(records, threshold=1.1)
    except ValueError as exc:
        assert str(exc) == "threshold must be between 0.0 and 1.0"
    else:
        raise AssertionError("Expected ValueError")


def test_lsh_path_detects_high_similarity_pair(monkeypatch) -> None:
    # Force the scalable path without constructing thousands of records.
    monkeypatch.setattr(
        checks,
        "_EXACT_NEAR_DUPLICATE_LIMIT",
        1,
    )

    common = " ".join(f"shared_token_{i}" for i in range(100))

    records = [
        PairRecord(
            row=10,
            prompt="Shared question",
            output=f"{common} left_variant",
        ),
        PairRecord(
            row=20,
            prompt="Shared question",
            output=f"{common} right_variant",
        ),
        PairRecord(
            row=30,
            prompt="Different question",
            output=(
                "red orange yellow green blue indigo violet "
                "black white silver gold bronze"
            ),
        ),
    ]

    findings, pairs = near_duplicate_findings(records)

    assert pairs == 1
    assert len(findings) == 1
    assert findings[0].code == "near_duplicate"
    assert "Rows 10 and 20" in findings[0].message


def test_lsh_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(
        checks,
        "_EXACT_NEAR_DUPLICATE_LIMIT",
        1,
    )

    common = " ".join(f"shared_token_{i}" for i in range(100))

    records = [
        PairRecord(
            row=1,
            prompt="Question",
            output=f"{common} variant_a",
        ),
        PairRecord(
            row=2,
            prompt="Question",
            output=f"{common} variant_b",
        ),
    ]

    first_findings, first_pairs = near_duplicate_findings(records)
    second_findings, second_pairs = near_duplicate_findings(records)

    assert first_pairs == second_pairs
    assert [finding.to_dict() for finding in first_findings] == [
        finding.to_dict() for finding in second_findings
    ]