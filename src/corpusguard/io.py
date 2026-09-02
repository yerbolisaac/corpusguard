from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                yield row_number, None, "blank line"
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                yield row_number, None, f"invalid JSON: {exc.msg}"
                continue
            if not isinstance(value, dict):
                yield row_number, None, "top-level JSON value must be an object"
                continue
            yield row_number, value, None
