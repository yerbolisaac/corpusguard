# Contributing to CorpusGuard

Thanks for helping improve LLM dataset quality.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check .
```

## Design principles

1. Deterministic checks first: audits should be reproducible and inexpensive.
2. No silent mutation: CorpusGuard reports problems; it does not rewrite datasets unless a user explicitly asks.
3. Evidence over vibes: every score deduction should map to a visible metric or finding.
4. Privacy by default: never send dataset rows to a remote service in the default installation.

Please include tests for new checks and document any false-positive tradeoffs.
