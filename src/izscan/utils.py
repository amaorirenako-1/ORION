from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def open_text(path: str | Path, mode: str = "rt"):
    target = Path(path)
    if target.suffix == ".gz":
        return gzip.open(target, mode, encoding=None if "b" in mode else "utf-8")
    return target.open(mode, encoding=None if "b" in mode else "utf-8")


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def read_lines(path: str | Path) -> list[str]:
    with open_text(path, "rt") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

