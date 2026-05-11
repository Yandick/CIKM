from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REQUIRED_EXTRA_KEYS = {
    "candidate_ids",
    "positive_candidate_ids",
    "evidence_ids",
    "source_evidence_ids",
}


def _first_prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            return str(first.get("content") or "")
    return ""


def _extra_info(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("extra_info")
    return value if isinstance(value, dict) else {}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        return len(value) == 0
    except TypeError:
        return False


def check_file(path: Path) -> list[str]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Checking parquet requires pandas and pyarrow.") from exc

    errors: list[str] = []
    if not path.exists():
        return [f"{path}: file does not exist"]

    df = pd.read_parquet(path)
    if df.empty:
        return [f"{path}: file is empty"]

    missing_columns = {"prompt", "reward_model", "extra_info"} - set(df.columns)
    if missing_columns:
        errors.append(f"{path}: missing columns {sorted(missing_columns)}")

    prompts = [_first_prompt_text(row) for row in df.head(min(20, len(df))).to_dict("records")]
    if not all("rationale" in prompt for prompt in prompts):
        errors.append(f"{path}: prompt samples do not all request rationale; regenerate data")

    empty_positive_rows = 0
    missing_extra_rows = 0
    for row in df.to_dict("records"):
        extra = _extra_info(row)
        missing = REQUIRED_EXTRA_KEYS - set(extra)
        if missing:
            missing_extra_rows += 1
        if _is_empty(extra.get("positive_candidate_ids")):
            empty_positive_rows += 1

    if missing_extra_rows:
        errors.append(f"{path}: {missing_extra_rows} rows miss required extra_info keys")
    if empty_positive_rows:
        errors.append(f"{path}: {empty_positive_rows} rows have empty positive_candidate_ids")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RecBench RL parquet schema before training.")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    for path in args.files:
        errors.extend(check_file(path))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"checked {len(args.files)} file(s): ok")


if __name__ == "__main__":
    main()
