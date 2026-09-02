from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from corpusguard.tokenization import (
    HuggingFaceCounter,
    TiktokenCounter,
    TokenizerError,
    load_tokenizer,
)


class FakeTiktokenEncoding:
    def encode(
        self,
        text: str,
        *,
        disallowed_special: tuple[str, ...] = (),
    ) -> list[str]:
        del disallowed_special
        return text.split()


class FakeHuggingFaceEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeHuggingFaceTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> FakeHuggingFaceEncoding:
        assert add_special_tokens is False

        return FakeHuggingFaceEncoding(
            list(range(len(text.split())))
        )


def test_tiktoken_counter_counts_tokens() -> None:
    counter = TiktokenCounter(
        encoding=FakeTiktokenEncoding(),
        tokenizer_name="test",
    )

    assert counter.name == "tiktoken:test"
    assert counter.count("one two three") == 3


def test_huggingface_counter_counts_tokens() -> None:
    counter = HuggingFaceCounter(
        tokenizer=FakeHuggingFaceTokenizer(),
        tokenizer_name="test",
    )

    assert counter.name == "hf:test"
    assert counter.count("one two three four") == 4


def test_load_tokenizer_rejects_empty_spec() -> None:
    with pytest.raises(
        TokenizerError,
        match="cannot be empty",
    ):
        load_tokenizer("")


def test_load_tokenizer_requires_backend_prefix() -> None:
    with pytest.raises(
        TokenizerError,
        match="backend prefix",
    ):
        load_tokenizer("cl100k_base")


def test_load_tokenizer_rejects_empty_name() -> None:
    with pytest.raises(
        TokenizerError,
        match="name cannot be empty",
    ):
        load_tokenizer("tiktoken:")


def test_load_tokenizer_rejects_unknown_backend() -> None:
    with pytest.raises(
        TokenizerError,
        match="Unsupported tokenizer backend",
    ):
        load_tokenizer("unknown:test")


def test_load_tiktoken_model(monkeypatch: pytest.MonkeyPatch) -> None:
    encoding = FakeTiktokenEncoding()

    fake_module = SimpleNamespace(
        encoding_for_model=lambda name: encoding,
        get_encoding=lambda name: encoding,
    )

    monkeypatch.setitem(sys.modules, "tiktoken", fake_module)

    counter = load_tokenizer("tiktoken:gpt-test")

    assert counter.name == "tiktoken:gpt-test"
    assert counter.count("one two three") == 3


def test_load_tiktoken_encoding_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = FakeTiktokenEncoding()

    def encoding_for_model(name: str) -> FakeTiktokenEncoding:
        raise KeyError(name)

    fake_module = SimpleNamespace(
        encoding_for_model=encoding_for_model,
        get_encoding=lambda name: encoding,
    )

    monkeypatch.setitem(sys.modules, "tiktoken", fake_module)

    counter = load_tokenizer("tiktoken:cl100k_base")

    assert counter.name == "tiktoken:cl100k_base"
    assert counter.count("one two") == 2