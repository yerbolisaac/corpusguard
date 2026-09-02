from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class TokenizerError(RuntimeError):
    """Raised when a requested tokenizer cannot be loaded."""


class TokenCounter(Protocol):
    """Minimal interface used by CorpusGuard for token counting."""

    @property
    def name(self) -> str:
        """Human-readable tokenizer identifier."""
        ...

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""
        ...


@dataclass(slots=True)
class TiktokenCounter:
    encoding: Any
    tokenizer_name: str

    @property
    def name(self) -> str:
        return f"tiktoken:{self.tokenizer_name}"

    def count(self, text: str) -> int:
        return len(
            self.encoding.encode(
                text,
                disallowed_special=(),
            )
        )


@dataclass(slots=True)
class HuggingFaceCounter:
    tokenizer: Any
    tokenizer_name: str

    @property
    def name(self) -> str:
        return f"hf:{self.tokenizer_name}"

    def count(self, text: str) -> int:
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        return len(encoded.ids)


def _load_tiktoken(name: str) -> TiktokenCounter:
    try:
        import tiktoken
    except ImportError as exc:
        raise TokenizerError(
            "tiktoken support is not installed. "
            'Install CorpusGuard with: pip install "corpusguard-audit[tokenizers]"'
        ) from exc

    try:
        encoding = tiktoken.encoding_for_model(name)
    except KeyError:
        try:
            encoding = tiktoken.get_encoding(name)
        except ValueError as exc:
            raise TokenizerError(
                f"Unknown tiktoken model or encoding: {name}"
            ) from exc

    return TiktokenCounter(
        encoding=encoding,
        tokenizer_name=name,
    )


def _load_huggingface(name: str) -> HuggingFaceCounter:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise TokenizerError(
            "Hugging Face tokenizer support is not installed. "
            'Install CorpusGuard with: pip install "corpusguard-audit[tokenizers]"'
        ) from exc

    path = Path(name)

    try:
        if path.is_file():
            tokenizer = Tokenizer.from_file(str(path))
        else:
            tokenizer = Tokenizer.from_pretrained(name)
    except Exception as exc:
        raise TokenizerError(
            f"Could not load Hugging Face tokenizer: {name}"
        ) from exc

    return HuggingFaceCounter(
        tokenizer=tokenizer,
        tokenizer_name=name,
    )


def load_tokenizer(spec: str) -> TokenCounter:
    """
    Load a tokenizer from a CorpusGuard tokenizer specification.

    Supported forms:

    - tiktoken:<model-or-encoding>
    - hf:<repository-or-tokenizer-json>
    """
    normalized = spec.strip()

    if not normalized:
        raise TokenizerError(
            "Tokenizer specification cannot be empty."
        )

    if ":" not in normalized:
        raise TokenizerError(
            "Tokenizer must include a backend prefix, for example "
            "'tiktoken:cl100k_base' or 'hf:bert-base-uncased'."
        )

    backend, name = normalized.split(":", 1)
    backend = backend.strip().lower()
    name = name.strip()

    if not name:
        raise TokenizerError(
            "Tokenizer name cannot be empty."
        )

    if backend == "tiktoken":
        return _load_tiktoken(name)

    if backend in {"hf", "huggingface"}:
        return _load_huggingface(name)

    raise TokenizerError(
        f"Unsupported tokenizer backend: {backend}. "
        "Supported backends are 'tiktoken' and 'hf'."
    )