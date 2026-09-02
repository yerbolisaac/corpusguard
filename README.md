<p align="center">
  <h1 align="center">CorpusGuard</h1>
  <p align="center"><strong>Pre-flight checks for LLM fine-tuning data.</strong></p>
  <p align="center">Catch duplicates, PII, broken rows, repetition and train/eval leakage before they become model behavior.</p>
</p>

---

## Why CorpusGuard?

LLM fine-tuning pipelines usually validate whether training *runs*. CorpusGuard asks a different question first:

> **Should this dataset be allowed into a training run at all?**

A clean loss curve cannot tell you that your evaluation examples leaked into training, that hundreds of rows are duplicated, or that private data is sitting in a JSONL file.

CorpusGuard provides a local pre-flight audit with a deterministic quality score and evidence for every deduction.

## Quick start

Install the latest released version from PyPI:

```bash
pip install corpusguard-audit
```

Run an audit:

```bash
corpusguard scan dataset.jsonl
```

Generate machine-readable and standalone HTML reports:

```bash
corpusguard scan data/train.jsonl \
  --json-out corpusguard-report.json \
  --html-out corpusguard-report.html
```

Check train/eval leakage:

```bash
corpusguard scan data/train.jsonl --compare data/eval.jsonl
```

Use it as a CI gate:

```bash
corpusguard scan data/train.jsonl --compare data/eval.jsonl --fail-below 85
```

Exit code `2` means the quality score fell below the requested threshold.

## What CorpusGuard checks

| Check                  | What it catches                                                       |
| ---------------------- | --------------------------------------------------------------------- |
| JSON/schema validation | Malformed JSONL and unsupported SFT row shapes                        |
| Exact duplicates       | Repeated prompt/answer examples                                       |
| Near duplicates        | Highly overlapping examples using token-set Jaccard similarity        |
| PII patterns           | Emails, phone-like values, 12-digit IDs, card-like numbers            |
| Empty/short outputs    | Weak supervision and accidental blank labels                          |
| Repetition             | Degenerate repeated-token answers                                     |
| Script distribution    | Latin/Cyrillic/other letter counts for multilingual corpus visibility |
| Split leakage          | Exact examples duplicated between train and eval/test JSONL           |
| Quality score          | Deterministic 0–100 score with A–F grade                              |

## Scalable near-duplicate detection

The current development version uses a hybrid strategy for near-duplicate detection.

For small corpora, CorpusGuard performs exact pairwise token-set Jaccard comparisons.

For larger corpora, it switches to deterministic MinHash signatures and locality-sensitive hashing (LSH) to generate likely candidate pairs. Candidate pairs are then verified with the exact Jaccard similarity before they are reported.

This keeps the final similarity decision exact for every candidate that LSH surfaces while avoiding an exhaustive comparison of every possible pair in large corpora.

Current defaults:

* exact pairwise path for up to 2,000 unique token sets
* 128 deterministic MinHash permutations
* 16 LSH bands
* 8 signature rows per band
* exact Jaccard verification of candidate pairs
* default near-duplicate threshold of `0.92`

MinHash/LSH is a candidate-generation technique. On the scalable path, it can theoretically miss a similar pair that does not collide in an LSH bucket. It is therefore not described as an exhaustive exact search over every possible pair.

### Local synthetic benchmark

A benchmark utility is included at:

```text
benchmarks/benchmark_near_duplicates.py
```

Run it with:

```bash
python benchmarks/benchmark_near_duplicates.py 10000
```

One local development benchmark produced:

|   Rows | Detection path | Elapsed |
| -----: | -------------- | ------: |
|  1,000 | exact pairwise | 1.1651s |
|  2,000 | exact pairwise | 4.6805s |
|  2,001 | MinHash/LSH    | 1.7143s |
|  5,000 | MinHash/LSH    | 4.2887s |
| 10,000 | MinHash/LSH    | 8.6317s |

These figures are a synthetic development benchmark from one machine, not a universal performance guarantee. Runtime depends on corpus size, token distribution, similarity structure, hardware and Python environment.

## Supported dataset shapes

OpenAI/chat style:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Prompt/completion:

```json
{"prompt":"Question","completion":"Answer"}
```

Alpaca/instruction:

```json
{"instruction":"Task","input":"Optional context","output":"Answer"}
```

Question/answer:

```json
{"question":"Question","answer":"Answer"}
```

## Example output

The repository includes an intentionally imperfect demo dataset:

```bash
corpusguard scan examples/demo.jsonl
```

Example result:

```text
CorpusGuard  v0.1.0
Dataset: examples/demo.jsonl
Quality score: 64/100 (D)

Metric                    Value
rows                           5
valid rows                     5
invalid rows                   0
exact duplicate rows           1
near duplicate pairs           0
pii hits                       1
high repetition rows           0
leakage matches                0
avg output chars            49.4
p95 output chars             101
empty outputs                  0
short outputs                  0
exact duplicate groups         1
latin chars                  325
cyrillic chars                 0
other letter chars             0
avg input chars             36.6

Findings: 2

• pii row 4: Potential sensitive data: email, phone
• exact_duplicate row 2: Exact duplicate across rows [2, 5]
```

The demo file intentionally contains bad data so the checks are visible.

## Release status

The latest PyPI release is `v0.1.0`.

The `main` branch is currently developing functionality planned for `v0.2.0`, including scalable MinHash/LSH near-duplicate candidate generation.

Until `v0.2.0` is released, installing with:

```bash
pip install corpusguard-audit
```

installs the latest published release rather than unreleased development changes from `main`.

Developers who want to test the current repository version can install it from source as described below.

## Philosophy

CorpusGuard's default checks are **local, deterministic and model-free**. Your training data is not sent to an API.

Semantic or LLM-based review can be useful, but it should be an explicit optional layer rather than a hidden dependency in the basic audit path.

The core audit should remain inspectable: findings should explain what was detected, and scoring should remain reproducible from the same input and configuration.

## Current limitations

* PII checks are pattern-based and can produce false positives or false negatives; they are a review signal, not a compliance guarantee.
* Script counts are not language identification.
* Exact split leakage does not yet detect paraphrased or semantic leakage.
* Large-corpus MinHash/LSH candidate generation is probabilistic and can theoretically miss near-duplicate pairs.
* CorpusGuard does not yet stream multi-GB datasets and currently loads audit data into memory.
* Length metrics are character-based rather than tokenizer-aware.

See [ROADMAP.md](ROADMAP.md) for planned scalable and semantic checks.

## Development

Clone the repository and install CorpusGuard in editable mode with development dependencies:

```bash
git clone https://github.com/yerbolisaac/corpusguard.git
cd corpusguard

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Run the test suite and linter:

```bash
pytest -q
ruff check .
```

Run the synthetic near-duplicate benchmark:

```bash
python benchmarks/benchmark_near_duplicates.py 10000
```

## License

Apache-2.0. Created by [@yerbolisaac](https://github.com/yerbolisaac).
