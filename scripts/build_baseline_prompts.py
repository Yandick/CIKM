"""Build prompt-only and weak-SFT baseline datasets from Amazon Food examples."""

from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.prompts import BaselinePromptStyle
from llm4rec.prompts import load_baseline_prompt_template
from llm4rec.training import (
    iter_baseline_prompt_records,
    prompt_record_to_sft_record,
    write_prompt_records_jsonl,
    write_sft_records_jsonl,
)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = Path("configs/data/amazon_food.yaml"),
    split: str = "validation",
    style: BaselinePromptStyle = BaselinePromptStyle.ANSWER_ONLY,
    prompt_config: Path | None = None,
    output_dir: Path = Path("data/processed/baselines"),
    max_examples: int = 200,
) -> None:
    """Render baseline prompt records and weak-SFT records for one split."""

    prompt_template = None if prompt_config is None else load_baseline_prompt_template(prompt_config)
    prompt_records = list(
        iter_baseline_prompt_records(
            config,
            split=split,
            style=style,
            template=prompt_template,
            limit=max_examples,
        )
    )
    prompt_path = output_dir / f"{split}_{style.value}_prompts.jsonl"
    prompt_count = write_prompt_records_jsonl(prompt_path, prompt_records)
    lines = [
        f"style={style.value}",
        f"split={split}",
        f"prompt_records={prompt_count}",
        f"prompt_config={prompt_config}",
        f"prompt_path={prompt_path}",
    ]

    if style == BaselinePromptStyle.ANSWER_ONLY:
        sft_path = output_dir / f"{split}_{style.value}_sft.jsonl"
        sft_count = write_sft_records_jsonl(
            sft_path,
            (prompt_record_to_sft_record(record) for record in prompt_records),
        )
        lines.extend(
            [
                f"sft_records={sft_count}",
                f"sft_path={sft_path}",
            ]
        )
    else:
        lines.append("sft_records=0")
        lines.append("sft_note=free_form_cot prompts are exported without SFT targets")

    typer.echo("\n".join(lines))


if __name__ == "__main__":
    app()
