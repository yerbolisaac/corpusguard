from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

from .models import Finding

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")
_IIN = re.compile(r"(?<!\d)\d{12}(?!\d)")

_MINHASH_PRIME = (1 << 61) - 1
_MINHASH_PERMUTATIONS = 128
_LSH_BANDS = 16
_LSH_ROWS_PER_BAND = _MINHASH_PERMUTATIONS // _LSH_BANDS

# Small datasets keep the exact v0.1 behavior.
# Larger datasets use MinHash/LSH candidate generation and exact
# Jaccard verification only for candidate pairs.
_EXACT_NEAR_DUPLICATE_LIMIT = 2_000


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

    card_match = _CARD.search(text)
    if card_match:
        digits = re.sub(r"\D", "", card_match.group(0))
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


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    if not a and not b:
        return 1.0

    union = a | b
    return len(a & b) / len(union) if union else 0.0


@dataclass(slots=True)
class PairRecord:
    row: int
    prompt: str
    output: str


def exact_duplicate_findings(
    records: Iterable[PairRecord],
) -> tuple[list[Finding], int, int]:
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


def _token_hash(token: str) -> int:
    digest = hashlib.blake2b(
        token.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(digest, "big") % _MINHASH_PRIME


@lru_cache(maxsize=1)
def _minhash_coefficients() -> tuple[tuple[int, int], ...]:
    coefficients: list[tuple[int, int]] = []

    for index in range(_MINHASH_PERMUTATIONS):
        digest = hashlib.sha256(
            f"corpusguard-minhash-{index}".encode()
        ).digest()

        a = int.from_bytes(digest[:8], "big") % (_MINHASH_PRIME - 1) + 1
        b = int.from_bytes(digest[8:16], "big") % _MINHASH_PRIME

        coefficients.append((a, b))

    return tuple(coefficients)


def _minhash_signature(tokens: frozenset[str]) -> tuple[int, ...]:
    coefficients = _minhash_coefficients()
    signature = [_MINHASH_PRIME] * _MINHASH_PERMUTATIONS

    for token in tokens:
        token_hash = _token_hash(token)

        for index, (a, b) in enumerate(coefficients):
            value = (a * token_hash + b) % _MINHASH_PRIME

            if value < signature[index]:
                signature[index] = min(signature[index], value)

    return tuple(signature)


def _lsh_candidate_pairs(
    token_sets: list[frozenset[str]],
) -> set[tuple[int, int]]:
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()

    for record_index, tokens in enumerate(token_sets):
        signature = _minhash_signature(tokens)

        for band in range(_LSH_BANDS):
            start = band * _LSH_ROWS_PER_BAND
            end = start + _LSH_ROWS_PER_BAND

            band_signature = signature[start:end]
            bucket_key = (band, band_signature)

            for previous_index in buckets[bucket_key]:
                candidates.add((previous_index, record_index))

            buckets[bucket_key].append(record_index)

    return candidates


def _can_reach_jaccard_threshold(
    left: frozenset[str],
    right: frozenset[str],
    threshold: float,
) -> bool:
    left_size = len(left)
    right_size = len(right)

    if left_size == 0 or right_size == 0:
        return False

    smaller = min(left_size, right_size)
    larger = max(left_size, right_size)

    return smaller / larger >= threshold


def near_duplicate_findings(
    records: list[PairRecord],
    threshold: float = 0.92,
    max_pairs: int = 200,
) -> tuple[list[Finding], int]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")

    # Group records that have exactly the same token set.
    #
    # Jaccard for records inside the same group is 1.0, so those pairs are
    # intentionally excluded from near-duplicate findings just as in v0.1.
    grouped_records: dict[frozenset[str], list[PairRecord]] = defaultdict(list)

    for record in records:
        tokens = frozenset(token_set(record.prompt + " " + record.output))

        if len(tokens) < 5:
            continue

        grouped_records[tokens].append(record)

    token_sets = list(grouped_records.keys())

    if len(token_sets) < 2:
        return [], 0

    if len(token_sets) <= _EXACT_NEAR_DUPLICATE_LIMIT:
        candidate_pairs: Iterable[tuple[int, int]] = combinations(
            range(len(token_sets)),
            2,
        )
    else:
        candidate_pairs = sorted(_lsh_candidate_pairs(token_sets))

    findings: list[Finding] = []
    pair_count = 0

    for left_index, right_index in candidate_pairs:
        left_tokens = token_sets[left_index]
        right_tokens = token_sets[right_index]

        if not _can_reach_jaccard_threshold(
            left_tokens,
            right_tokens,
            threshold,
        ):
            continue

        score = jaccard(left_tokens, right_tokens)

        if not threshold <= score < 1.0:
            continue

        left_records = grouped_records[left_tokens]
        right_records = grouped_records[right_tokens]

        group_pair_count = len(left_records) * len(right_records)
        pair_count += group_pair_count

        if len(findings) >= max_pairs:
            continue

        for left_record in left_records:
            for right_record in right_records:
                if len(findings) >= max_pairs:
                    break

                findings.append(
                    Finding(
                        code="near_duplicate",
                        severity="warning",
                        message=(
                            f"Rows {left_record.row} and {right_record.row} "
                            f"are near-duplicates (Jaccard={score:.3f})"
                        ),
                        row=left_record.row,
                    )
                )

            if len(findings) >= max_pairs:
                break

    return findings, pair_count