from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Finding:
    code: str
    severity: str
    message: str
    row: int | None = None
    field: str | None = None
    sample: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuditMetrics:
    rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    empty_outputs: int = 0
    short_outputs: int = 0
    exact_duplicate_groups: int = 0
    exact_duplicate_rows: int = 0
    near_duplicate_pairs: int = 0
    pii_hits: int = 0
    high_repetition_rows: int = 0
    latin_chars: int = 0
    cyrillic_chars: int = 0
    other_letter_chars: int = 0
    avg_input_chars: float = 0.0
    avg_output_chars: float = 0.0
    p95_output_chars: int = 0
    leakage_matches: int = 0

    tokenizer: str | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    p95_output_tokens: int | None = None
    max_output_tokens: int | None = None
    avg_total_tokens: float | None = None
    p95_total_tokens: int | None = None
    max_total_tokens: int | None = None
    context_window: int | None = None
    context_window_exceeded_rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuditReport:
    dataset: str
    score: int
    grade: str
    metrics: AuditMetrics
    findings: list[Finding] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "score": self.score,
            "grade": self.grade,
            "metrics": self.metrics.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "recommendations": self.recommendations,
        }
