# Roadmap

## v0.1 — deterministic pre-flight audit
- JSONL schema normalization
- exact duplicates
- dependency-free near-duplicate detection
- PII pattern checks
- repetition checks
- train/eval exact leakage
- JSON + HTML reports
- CI score gates

## v0.2 — scalable corpus analysis
- MinHash/LSH duplicate search
- streaming mode for multi-GB datasets
- tokenizer-aware length analysis
- Hugging Face Dataset support
- configurable policies via `corpusguard.yaml`

## v0.3 — semantic quality
- optional local embedding checks
- topic balance and cluster outliers
- semantic train/eval leakage
- answer-consistency checks
- benchmark contamination packs

## v0.4 — LLM-native review
- opt-in local judge adapters
- rubric-based quality scoring
- hallucination/evidence checks for grounded datasets
- human-review queues

## v1.0
- stable policy schema
- SARIF/GitHub code-scanning output
- reproducible audit manifests
- plugin API for custom checks
