from __future__ import annotations

from typing import Any


def extract_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    """Normalize common SFT shapes into an (input, output) pair.

    Supported shapes:
    - {"messages": [{"role": ..., "content": ...}, ...]}
    - {"prompt": ..., "completion": ...}
    - {"instruction": ..., "input": ..., "output": ...}
    - {"question": ..., "answer": ...}
    """
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).lower()
            content = item.get("content")
            if not isinstance(content, str):
                continue
            if role in {"user", "human", "system"}:
                user_parts.append(content)
            elif role in {"assistant", "bot", "model"}:
                assistant_parts.append(content)
        if assistant_parts:
            return "\n".join(user_parts).strip(), "\n".join(assistant_parts).strip()

    if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        return row["prompt"].strip(), row["completion"].strip()

    if isinstance(row.get("instruction"), str) and isinstance(row.get("output"), str):
        extra = row.get("input")
        prompt = row["instruction"]
        if isinstance(extra, str) and extra.strip():
            prompt += "\n" + extra
        return prompt.strip(), row["output"].strip()

    if isinstance(row.get("question"), str) and isinstance(row.get("answer"), str):
        return row["question"].strip(), row["answer"].strip()

    return None
