# Roadmap

CorpusGuard is being developed incrementally from a deterministic local dataset auditor toward scalable corpus analysis, semantic quality checks and extensible production integrations.

## v0.1 — deterministic pre-flight audit

Released.

* [x] JSONL schema normalization
* [x] OpenAI/chat dataset support
* [x] prompt/completion dataset support
* [x] Alpaca instruction/input/output support
* [x] question/answer dataset support
* [x] exact duplicate detection
* [x] dependency-free near-duplicate detection
* [x] PII pattern checks
* [x] empty and short output checks
* [x] repetition checks
* [x] script distribution metrics
* [x] train/eval exact leakage
* [x] deterministic 0–100 quality score
* [x] A–F quality grade
* [x] JSON reports
* [x] standalone HTML reports
* [x] CI score gates with `--fail-below`
* [x] PyPI distribution
* [x] GitHub release workflow

## v0.2 — scalable corpus analysis

In development.

* [x] deterministic MinHash signatures
* [x] LSH candidate generation for scalable near-duplicate search
* [x] exact Jaccard verification of LSH candidates
* [x] hybrid exact/LSH detection path
* [x] dedicated near-duplicate test coverage
* [x] synthetic near-duplicate benchmark
* [x] tokenizer-aware input/output length analysis
* [x] optional `tiktoken` integration
* [x] optional Hugging Face `tokenizers` integration
* [x] local Hugging Face `tokenizer.json` support
* [x] tokenizer-aware total-token metrics
* [x] configurable context-window checks
* [x] context-window findings and aggregate metrics
* [ ] streaming mode for multi-GB datasets
* [ ] Hugging Face Dataset support
* [ ] configurable policies via `corpusguard.yaml`

### Near-duplicate architecture

The scalable near-duplicate implementation uses a hybrid strategy:

* small corpora use exhaustive pairwise Jaccard comparison
* large corpora use deterministic MinHash signatures and LSH for candidate generation
* candidate pairs are verified with exact token-set Jaccard similarity before reporting
* identical token sets remain handled by exact-duplicate detection rather than near-duplicate detection

The large-corpus path trades exhaustive candidate recall for scalability. LSH can theoretically miss a pair that does not collide in a candidate bucket even when its true Jaccard similarity is above the configured threshold.

### Tokenizer-aware length architecture

Tokenizer-aware analysis is an optional layer and does not add tokenizer dependencies to the base CorpusGuard installation.

The current implementation supports:

* `tiktoken` model names and encodings
* Hugging Face Hub tokenizers through the `tokenizers` package
* local Hugging Face `tokenizer.json` files
* average, p95 and maximum token-length statistics
* configurable context-window checks

CorpusGuard currently defines the total content-token length of a valid example as:

```text
input tokens + output tokens
```

Hugging Face automatically added special tokens are excluded from this calculation.

This is intentionally a content-level pre-flight metric rather than an exact reconstruction of a model's final serialized training sequence. Model-specific chat templates, BOS/EOS tokens, role markers and other formatting overhead are not currently included.

Context-window findings are therefore diagnostic. Tokenizer-aware metrics and context-window findings do not currently modify the deterministic quality score.

## v0.3 — semantic quality

Planned.

* [ ] optional local embedding checks
* [ ] semantic near-duplicate detection
* [ ] topic balance analysis
* [ ] cluster outlier detection
* [ ] semantic train/eval leakage
* [ ] answer-consistency checks
* [ ] benchmark contamination packs

## v0.4 — LLM-native review

Planned.

* [ ] opt-in local judge adapters
* [ ] rubric-based quality scoring
* [ ] hallucination/evidence checks for grounded datasets
* [ ] human-review queues
* [ ] configurable judge policies
* [ ] explicit separation between deterministic and model-based findings

## v1.0 — stable audit platform

Planned.

* [ ] stable policy schema
* [ ] stable report schema
* [ ] reproducible audit manifests
* [ ] SARIF/GitHub code-scanning output
* [ ] plugin API for custom checks
* [ ] documented extension interface
* [ ] backward-compatibility policy
