from __future__ import annotations

from pathlib import Path
from statistics import mean

from .checks import (
    PairRecord,
    exact_duplicate_findings,
    fingerprint,
    near_duplicate_findings,
    pii_types,
    repetition_ratio,
    script_counts,
)
from .io import read_jsonl
from .models import AuditMetrics, AuditReport, Finding
from .schema import extract_pair
from .tokenization import TokenCounter


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0

    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            round((len(ordered) - 1) * q),
        ),
    )
    return ordered[index]


def _grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _score(metrics: AuditMetrics) -> int:
    if metrics.rows == 0:
        return 0

    score = 100.0

    score -= min(
        35.0,
        metrics.invalid_rows / metrics.rows * 100 * 1.5,
    )

    score -= min(
        20.0,
        metrics.exact_duplicate_rows
        / max(metrics.valid_rows, 1)
        * 100,
    )

    score -= min(
        12.0,
        metrics.near_duplicate_pairs
        / max(metrics.valid_rows, 1)
        * 40,
    )

    score -= min(
        18.0,
        metrics.pii_hits
        / max(metrics.valid_rows, 1)
        * 80,
    )

    score -= min(
        10.0,
        metrics.empty_outputs
        / max(metrics.valid_rows, 1)
        * 100,
    )

    score -= min(
        8.0,
        metrics.short_outputs
        / max(metrics.valid_rows, 1)
        * 30,
    )

    score -= min(
        10.0,
        metrics.high_repetition_rows
        / max(metrics.valid_rows, 1)
        * 40,
    )

    if metrics.leakage_matches:
        score -= min(
            35.0,
            12.0 + metrics.leakage_matches * 3.0,
        )

    return max(
        0,
        min(
            100,
            round(score),
        ),
    )


def audit_dataset(
    path: str | Path,
    compare_path: str | Path | None = None,
    tokenizer: TokenCounter | None = None,
    context_window: int | None = None,
) -> AuditReport:
    dataset = Path(path)

    if context_window is not None and context_window <= 0:
        raise ValueError("context_window must be greater than zero.")

    if context_window is not None and tokenizer is None:
        raise ValueError(
            "context_window requires a tokenizer."
        )

    metrics = AuditMetrics()

    if tokenizer is not None:
        metrics.tokenizer = tokenizer.name
        metrics.context_window = context_window

        if context_window is not None:
            metrics.context_window_exceeded_rows = 0

    findings: list[Finding] = []
    records: list[PairRecord] = []

    input_lengths: list[int] = []
    output_lengths: list[int] = []

    input_token_lengths: list[int] = []
    output_token_lengths: list[int] = []
    total_token_lengths: list[int] = []

    for row_number, row, error in read_jsonl(dataset):
        metrics.rows += 1

        if error or row is None:
            metrics.invalid_rows += 1
            findings.append(
                Finding(
                    "invalid_json",
                    "error",
                    error or "invalid row",
                    row=row_number,
                )
            )
            continue

        pair = extract_pair(row)

        if pair is None:
            metrics.invalid_rows += 1
            findings.append(
                Finding(
                    "unsupported_schema",
                    "error",
                    (
                        "Could not extract an input/output "
                        "pair from this row"
                    ),
                    row=row_number,
                )
            )
            continue

        prompt, output = pair

        metrics.valid_rows += 1

        records.append(
            PairRecord(
                row_number,
                prompt,
                output,
            )
        )

        input_lengths.append(len(prompt))
        output_lengths.append(len(output))

        if tokenizer is not None:
            input_tokens = tokenizer.count(prompt)
            output_tokens = tokenizer.count(output)
            total_tokens = input_tokens + output_tokens

            input_token_lengths.append(input_tokens)
            output_token_lengths.append(output_tokens)
            total_token_lengths.append(total_tokens)

            if (
                context_window is not None
                and total_tokens > context_window
            ):
                if metrics.context_window_exceeded_rows is not None:
                    metrics.context_window_exceeded_rows += 1

                findings.append(
                    Finding(
                        "context_window_exceeded",
                        "warning",
                        (
                            f"Example uses {total_tokens} tokens, "
                            f"exceeding the configured context "
                            f"window of {context_window}"
                        ),
                        row=row_number,
                    )
                )

        if not output.strip():
            metrics.empty_outputs += 1

            findings.append(
                Finding(
                    "empty_output",
                    "error",
                    "Output is empty",
                    row=row_number,
                )
            )

        elif len(output.strip()) < 12:
            metrics.short_outputs += 1

            findings.append(
                Finding(
                    "short_output",
                    "warning",
                    (
                        f"Output is only "
                        f"{len(output.strip())} characters"
                    ),
                    row=row_number,
                )
            )

        combined = prompt + "\n" + output

        types = pii_types(combined)

        if types:
            metrics.pii_hits += 1

            findings.append(
                Finding(
                    "pii",
                    "error",
                    (
                        "Potential sensitive data: "
                        + ", ".join(types)
                    ),
                    row=row_number,
                )
            )

        rep = repetition_ratio(output)

        if rep >= 0.45:
            metrics.high_repetition_rows += 1

            findings.append(
                Finding(
                    "high_repetition",
                    "warning",
                    (
                        "High token repetition ratio "
                        f"({rep:.2f})"
                    ),
                    row=row_number,
                )
            )

        latin, cyrillic, other = script_counts(combined)

        metrics.latin_chars += latin
        metrics.cyrillic_chars += cyrillic
        metrics.other_letter_chars += other

    duplicate_findings, groups, duplicate_rows = (
        exact_duplicate_findings(records)
    )

    findings.extend(duplicate_findings)

    metrics.exact_duplicate_groups = groups
    metrics.exact_duplicate_rows = duplicate_rows

    near_findings, near_pairs = near_duplicate_findings(records)

    findings.extend(near_findings)

    metrics.near_duplicate_pairs = near_pairs

    if compare_path:
        compare_fingerprints: set[str] = set()

        for _, row, error in read_jsonl(compare_path):
            if error or row is None:
                continue

            pair = extract_pair(row)

            if pair:
                compare_fingerprints.add(
                    fingerprint(
                        pair[0]
                        + "\n<CG>\n"
                        + pair[1]
                    )
                )

        for record in records:
            fp = fingerprint(
                record.prompt
                + "\n<CG>\n"
                + record.output
            )

            if fp in compare_fingerprints:
                metrics.leakage_matches += 1

                findings.append(
                    Finding(
                        "split_leakage",
                        "critical",
                        (
                            "Exact example also appears "
                            "in comparison split"
                        ),
                        row=record.row,
                    )
                )

    metrics.avg_input_chars = (
        round(mean(input_lengths), 1)
        if input_lengths
        else 0.0
    )

    metrics.avg_output_chars = (
        round(mean(output_lengths), 1)
        if output_lengths
        else 0.0
    )

    metrics.p95_output_chars = _percentile(
        output_lengths,
        0.95,
    )

    if tokenizer is not None:
        metrics.avg_input_tokens = (
            round(mean(input_token_lengths), 1)
            if input_token_lengths
            else 0.0
        )

        metrics.avg_output_tokens = (
            round(mean(output_token_lengths), 1)
            if output_token_lengths
            else 0.0
        )

        metrics.p95_output_tokens = _percentile(
            output_token_lengths,
            0.95,
        )

        metrics.max_output_tokens = (
            max(output_token_lengths)
            if output_token_lengths
            else 0
        )

        metrics.avg_total_tokens = (
            round(mean(total_token_lengths), 1)
            if total_token_lengths
            else 0.0
        )

        metrics.p95_total_tokens = _percentile(
            total_token_lengths,
            0.95,
        )

        metrics.max_total_tokens = (
            max(total_token_lengths)
            if total_token_lengths
            else 0
        )

    score = _score(metrics)

    recommendations: list[str] = []

    if metrics.invalid_rows:
        recommendations.append(
            "Fix invalid JSON/schema rows before training."
        )

    if (
        metrics.exact_duplicate_rows
        or metrics.near_duplicate_pairs
    ):
        recommendations.append(
            "Deduplicate repeated examples "
            "to reduce memorization bias."
        )

    if metrics.pii_hits:
        recommendations.append(
            "Review and redact potential PII "
            "before publishing or training."
        )

    if metrics.leakage_matches:
        recommendations.append(
            "Rebuild train/eval splits: "
            "leakage makes evaluation unreliable."
        )

    if metrics.high_repetition_rows:
        recommendations.append(
            "Inspect repetitive outputs; they may teach "
            "degenerate generation patterns."
        )

    if metrics.context_window_exceeded_rows:
        recommendations.append(
            "Review examples exceeding the configured "
            "context window before training."
        )

    if not recommendations:
        recommendations.append(
            "No major deterministic issues found; "
            "proceed to semantic/manual review."
        )

    return AuditReport(
        dataset=str(dataset),
        score=score,
        grade=_grade(score),
        metrics=metrics,
        findings=findings,
        recommendations=recommendations,
    )
