from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from faithrec.recbench import render_recbench_prompt, to_rl_row
from faithrec.reward import compute_score


SYSTEM_PROMPT = """You are generating supervised fine-tuning targets for a recommendation reranker.
Use the teacher-only gold labels only to choose a correct ranking.
Do not mention gold labels, supervision, training data, or hidden answers.
Follow the requested response format and end with the Final Answer JSON."""


def iter_jsonl(path: Path, limit: int | None = None):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if limit is not None and count >= limit:
                break
            yield json.loads(line)
            count += 1


def teacher_messages(instance: dict[str, Any]) -> list[dict[str, str]]:
    positives = instance.get("label", {}).get("positive_candidate_ids") or []
    candidates = {item["candidate_id"]: item for item in instance.get("candidates", [])}
    teacher_only = {
        "gold_positive_candidate_ids": positives,
        "gold_positive_items": [
            {
                "candidate_id": cid,
                "item_id": candidates.get(cid, {}).get("item_id"),
                "title": candidates.get(cid, {}).get("title"),
            }
            for cid in positives
        ],
    }
    content = (
        render_recbench_prompt(instance)
        + "\n\nTeacher-only supervision:\n"
        + json.dumps(teacher_only, ensure_ascii=False)
        + "\n\nGenerate the assistant response for SFT. Put the gold positive candidate first, "
        "then rank the remaining candidates by semantic relevance."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def chat_with_retries(args: argparse.Namespace, messages: list[dict[str, str]]) -> str:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            return chat_completion(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(args.retry_sleep * (attempt + 1))
    raise RuntimeError(f"teacher request failed: {last_error}") from last_error


def validate(response: str, instance: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rl_row = to_rl_row(instance)
    score = compute_score(
        response,
        positive_candidate_ids=rl_row["extra_info"]["positive_candidate_ids"],
        candidate_ids=rl_row["extra_info"]["candidate_ids"],
        evidence_ids=rl_row["extra_info"]["evidence_ids"],
        k=1,
    )
    accepted = (
        score.get("parse_success") == 1.0
        and score.get("format", 0.0) >= args.min_format
        and score.get("ndcg@1", 0.0) >= args.min_ndcg1
        and score.get("evidence", 0.0) >= args.min_evidence
        and not score.get("validation_errors")
    )
    return {"accepted": accepted, **score}


def sft_row(
    instance: dict[str, Any], response: str, model: str, validation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": render_recbench_prompt(instance)},
            {"role": "assistant", "content": response},
        ],
        "instance_id": instance["instance_id"],
        "domain": instance["domain"],
        "task_type": instance["task_type"],
        "teacher_model": model,
        "validation": validation,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Writing parquet requires pandas and pyarrow.") from exc
        pd.DataFrame(rows).to_parquet(path, index=False)
        return
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build teacher-generated RecBench SFT data.")
    parser.add_argument("--input", type=Path, required=True, help="Canonical RecBench JSONL.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Accepted SFT .parquet or .jsonl."
    )
    parser.add_argument("--rejected-output", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--api-base", default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-ndcg1", type=float, default=1.0)
    parser.add_argument("--min-format", type=float, default=1.0)
    parser.add_argument("--min-evidence", type=float, default=1.0)
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api-key.")

    rejected_output = args.rejected_output or args.output.with_name(
        args.output.stem + ".rejected.jsonl"
    )
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for instance in iter_jsonl(args.input, args.limit):
        try:
            response = chat_with_retries(args, teacher_messages(instance))
            validation = validate(response, instance, args)
        except Exception as exc:
            rejected_rows.append({"instance_id": instance.get("instance_id"), "error": str(exc)})
            continue
        if validation["accepted"]:
            accepted_rows.append(sft_row(instance, response, args.model, validation))
        else:
            rejected_rows.append(
                {
                    "instance_id": instance["instance_id"],
                    "response": response,
                    "validation": validation,
                }
            )

    write_rows(args.output, accepted_rows)
    write_rows(rejected_output, rejected_rows)
    print(f"accepted={len(accepted_rows)} rejected={len(rejected_rows)}")


if __name__ == "__main__":
    main()
