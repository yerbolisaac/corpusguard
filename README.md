<p align="center">
  <h1 align="center">CorpusGuard</h1>
  <p align="center"><strong>Pre-flight checks for LLM fine-tuning data.</strong></p>
  <p align="center">Catch duplicates, PII, broken rows, repetition and train/eval leakage before they become model behavior.</p>
</p>

---

## Why CorpusGuard?

LLM fine-tuning pipelines usually validate whether training *runs*. CorpusGuard asks a different question first:

> **Should this dataset be allowed into a training run at all?**

A clean loss curve cannot tell you that your evaluation examples leaked into training, that hundreds of rows are duplicated, or that private data is sitting in a JSONL file. CorpusGuard provides a deterministic, local pre-flight audit with a single quality score and evidence for every deduction.

## Quick start

Install CorpusGuard from PyPI:

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

## What v0.1 checks

| Check                  | What it catches                                                       |
| ---------------------- | --------------------------------------------------------------------- |
| JSON/schema validation | Malformed JSONL and unsupported SFT row shapes                        |
| Exact duplicates       | Repeated prompt/answer examples                                       |
| Near duplicates        | Highly overlapping examples using token Jaccard similarity            |
| PII patterns           | Emails, phone-like values, 12-digit IDs, card-like numbers            |
| Empty/short outputs    | Weak supervision and accidental blank labels                          |
| Repetition             | Degenerate repeated-token answers                                     |
| Script distribution    | Latin/Cyrillic/other letter counts for multilingual corpus visibility |
| Split leakage          | Exact examples duplicated between train and eval/test JSONL           |
| Quality score          | Deterministic 0–100 score with A–F grade                              |

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

## Philosophy

CorpusGuard's default checks are **local, deterministic and model-free**. Your training data is not sent to an API. Semantic/LLM-based review can be useful, but it should be an explicit optional layer rather than a hidden dependency in the basic audit path.

## Current limitations

* Near-duplicate detection in v0.1 is an O(n²) reference implementation intended for small and medium corpora.
* PII checks are pattern-based and can produce false positives/negatives; they are a review signal, not a compliance guarantee.
* Script counts are not language identification.
* Exact split leakage does not yet detect paraphrased semantic leakage.

See [ROADMAP.md](ROADMAP.md) for the planned scalable and semantic checks.

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


## License

Apache-2.0. Created by [@yerbolisaac](https://github.com/yerbolisaac).
