from corpusguard.schema import extract_pair


def test_openai_messages_shape():
    row = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    }
    assert extract_pair(row) == ("Be concise.\nHi", "Hello!")


def test_alpaca_shape():
    row = {"instruction": "Translate", "input": "hello", "output": "bonjour"}
    assert extract_pair(row) == ("Translate\nhello", "bonjour")
