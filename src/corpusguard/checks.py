from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .models import Finding

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")
_IIN = re.compile(r"(?<!\d)\d{12}(?!\d)")


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()


def pii_types(text: str) -> list[str]:
    hits: list[str] = []
    if _EMAIL.search(text):
        hits.append("email")
    if _PHONE.search(text):
        hits.append("phone")
    if _IIN.search(text):
        hits.append("12-digit identifier")
    if _CARD.search(text):
        digits = re.sub(r"\D", "", _CARD.search(text).group(0))  # type: ignore[union-attr]
        if 13 <= len(digits) <= 19:
            hits.append("payment-card-like number")
    return sorted(set(hits))


def repetition_ratio(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(tokens) < 8:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(tokens)


def script_counts(text: str) -> tuple[int, int, int]:
    latin = cyrillic = other = 0
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            latin += 1
        elif "CYRILLIC" in name:
            cyrillic += 1
        else:
            other += 1
    return latin, cyrillic, other


def token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalized(text), flags=re.UNICODE))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


@dataclass(slots=True)
class PairRecord:
    row: int
    prompt: str
    output: str


def exact_duplicate_findings(records: Iterable[PairRecord]) -> tuple[list[Finding], int, int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record in records:
        groups[fingerprint(record.prompt + "\n<CG>\n" + record.output)].append(record.row)

    findings: list[Finding] = []
    duplicate_groups = 0
    duplicate_rows = 0
    for rows in groups.values():
        if len(rows) <= 1:
            continue
        duplicate_groups += 1
        duplicate_rows += len(rows) - 1
        findings.append(
            Finding(
                code="exact_duplicate",
                severity="warning",
                message=f"Exact duplicate across rows {rows}",
                row=rows[0],
            )
        )
    return findings, duplicate_groups, duplicate_rows


def near_duplicate_findings(
    records: list[PairRecord], threshold: float = 0.92, max_pairs: int = 200
) -> tuple[list[Finding], int]:
    # Deliberately dependency-free O(n^2) reference implementation for small/medium audits.
    # A future release will swap in MinHash/LSH for large corpora.
    findings: list[Finding] = []
    pairs = 0
    tokenized = [(r, token_set(r.prompt + " " + r.output)) for r in records]
    for i, (left, left_tokens) in enumerate(tokenized):
        if len(left_tokens) < 5:
            continue
        for right, right_tokens in tokenized[i + 1 :]:
            if len(right_tokens) < 5:
                continue
            score = jaccard(left_tokens, right_tokens)
            if threshold <= score < 1.0:
                pairs += 1
                if len(findings) < max_pairs:
                    findings.append(
                        Finding(
                            code="near_duplicate",
                            severity="warning",
                            message=(
                                f"Rows {left.row} and {right.row} are near-duplicates "
                                f"(Jaccard={score:.3f})"
                            ),
                            row=left.row,
                        )
                    )
    return findings, pairs
