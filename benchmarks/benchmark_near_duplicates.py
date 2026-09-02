from __future__ import annotations

import argparse
import random
import time

from corpusguard.checks import PairRecord, near_duplicate_findings


def build_records(size: int, seed: int = 42) -> list[PairRecord]:
    rng = random.Random(seed)

    records: list[PairRecord] = []

    for index in range(size):
        base_tokens = [
            f"topic_{index}",
            f"group_{index % 100}",
            f"category_{index % 37}",
            f"subject_{index % 53}",
            f"concept_{index % 71}",
        ]

        noise_tokens = [
            f"noise_{rng.randint(0, 1_000_000)}"
            for _ in range(20)
        ]

        records.append(
            PairRecord(
                row=index + 1,
                prompt=" ".join(base_tokens[:2]),
                output=" ".join(base_tokens[2:] + noise_tokens),
            )
        )

    # Inject deterministic near-duplicate pairs.
    #
    # Every 500 rows, add a record that is almost identical to the
    # previous record but has one token changed.
    for index in range(499, size, 500):
        source = records[index]

        records[index] = PairRecord(
            row=source.row,
            prompt=source.prompt,
            output=source.output + " near_duplicate_variant",
        )

        if index > 0:
            previous = records[index - 1]

            shared_tokens = previous.output.split()

            records[index] = PairRecord(
                row=source.row,
                prompt=previous.prompt,
                output=" ".join(
                    shared_tokens[:-1] + ["near_duplicate_variant"]
                ),
            )

    return records


def run_benchmark(size: int) -> None:
    records = build_records(size)

    started = time.perf_counter()

    findings, pair_count = near_duplicate_findings(records)

    elapsed = time.perf_counter() - started

    print(f"Rows: {size:,}")
    print(f"Near-duplicate pairs: {pair_count:,}")
    print(f"Findings retained: {len(findings):,}")
    print(f"Elapsed: {elapsed:.4f}s")
    print(f"Rows/sec: {size / elapsed:,.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "size",
        type=int,
        help="Number of synthetic records to benchmark",
    )

    args = parser.parse_args()

    if args.size <= 0:
        raise SystemExit("size must be greater than zero")

    run_benchmark(args.size)


if __name__ == "__main__":
    main()