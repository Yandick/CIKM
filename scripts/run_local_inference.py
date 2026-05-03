from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from faithrec.local_inference import (
    batched,
    generate_batch,
    is_placeholder_model_path,
    load_hf_model,
    read_model_config,
)


def read_prompt_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def dry_run_output(record: dict[str, Any]) -> dict[str, Any]:
    candidate_ids = record.get("candidate_ids") or []
    selected = candidate_ids[0] if candidate_ids else ""
    ranking = candidate_ids
    if record.get("prompt_style") == "evidence_rerank":
        text = (
            "Evidence Selection:\n"
            "E1: [H01] Dry-run placeholder evidence.\n\n"
            "Candidate Reasoning:\n"
            f"{selected}: support=[E1], conflict=[], brief_reason=\"Dry-run placeholder.\"\n\n"
            "Final Answer:\n"
            + json.dumps(
                {
                    "ranking": ranking,
                    "selected_candidate_id": selected,
                    "evidence_refs": ["H01"],
                },
                ensure_ascii=False,
            )
        )
    else:
        text = "Final Answer:\n" + json.dumps(
            {"ranking": ranking, "selected_candidate_id": selected},
            ensure_ascii=False,
        )

    return {
        "output_text": text,
        "latency_ms": 0,
        "input_tokens": None,
        "output_tokens": None,
    }


def write_result(
    output_file: Any,
    record: dict[str, Any],
    generation: dict[str, Any],
    model_name: str,
    dry_run: bool,
) -> None:
    output_file.write(
        json.dumps(
            {
                "instance_id": record["instance_id"],
                "dataset": record.get("dataset"),
                "prompt_style": record.get("prompt_style"),
                "model": model_name,
                "dry_run": dry_run,
                "candidate_ids": record.get("candidate_ids"),
                "evidence_ids": record.get("evidence_ids"),
                "label": record.get("label"),
                "output_text": generation["output_text"],
                "latency_ms": generation["latency_ms"],
                "input_tokens": generation["input_tokens"],
                "output_tokens": generation["output_tokens"],
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Hugging Face inference on prompts.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/qwen2_5_1_5b_instruct.yaml"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = read_model_config(args.model_config)
    model_name = str(config.get("name", args.model_config))
    records = read_prompt_records(args.prompts, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run or is_placeholder_model_path(config.get("model_path")):
        if not args.dry_run:
            raise SystemExit(
                "Model path is still a placeholder. Set model_path or rerun with --dry-run."
            )
        with args.output.open("w", encoding="utf-8") as f:
            for record in records:
                write_result(f, record, dry_run_output(record), model_name, dry_run=True)
        print(f"Wrote {len(records)} dry-run generations to {args.output}")
        return

    tokenizer, model = load_hf_model(config)
    runtime = config.get("runtime", {})
    generation_config = dict(config.get("generation", {}))
    if args.max_new_tokens is not None:
        generation_config["max_new_tokens"] = args.max_new_tokens

    batch_size = args.batch_size or int(runtime.get("batch_size", 1))
    use_chat_template = bool(runtime.get("use_chat_template", True))
    max_input_tokens = int(runtime.get("max_input_tokens", 4096))

    written = 0
    start = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as f:
        for batch in batched(records, batch_size):
            prompts = [record["prompt"] for record in batch]
            generations = generate_batch(
                tokenizer,
                model,
                prompts,
                use_chat_template=use_chat_template,
                max_input_tokens=max_input_tokens,
                generation_config=generation_config,
            )
            for record, generation in zip(batch, generations, strict=True):
                write_result(
                    f,
                    record,
                    {
                        "output_text": generation.output_text,
                        "latency_ms": generation.latency_ms,
                        "input_tokens": generation.input_tokens,
                        "output_tokens": generation.output_tokens,
                    },
                    model_name,
                    dry_run=False,
                )
                written += 1

    elapsed = time.perf_counter() - start
    print(f"Wrote {written} generations to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
