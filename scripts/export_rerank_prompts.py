from __future__ import annotations

import argparse
import json
from pathlib import Path

from faithrec.io import read_instances
from faithrec.prompts import (
    render_direct_prompt,
    render_evidence_rerank_prompt,
    render_free_form_cot_prompt,
)


PROMPT_RENDERERS = {
    "direct": render_direct_prompt,
    "free_form_cot": render_free_form_cot_prompt,
    "evidence_rerank": render_evidence_rerank_prompt,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reranking prompts from pilot JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-style", choices=sorted(PROMPT_RENDERERS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer = PROMPT_RENDERERS[args.prompt_style]

    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for instance in read_instances(args.input):
            prompt = renderer(instance)
            record = {
                "instance_id": instance.instance_id,
                "dataset": instance.dataset,
                "prompt_style": args.prompt_style,
                "prompt": prompt,
                "candidate_ids": instance.candidate_ids,
                "evidence_ids": sorted(instance.evidence_ids),
                "label": {
                    "target_candidate_id": instance.target_candidate_id,
                    "target_item_id": instance.metadata.get("target_item_id"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if args.limit is not None and count >= args.limit:
                break

    print(f"Wrote {count} {args.prompt_style} prompts to {args.output}")


if __name__ == "__main__":
    main()
