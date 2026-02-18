from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}

    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    # Covers numpy arrays and other custom types exposing tolist().
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_jsonable(tolist())

    return str(value)

