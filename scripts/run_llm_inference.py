"""Run local-HF baseline inference and write prediction JSONL."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation import (
    build_example_index,
    score_offline_prediction_records,
    summarize_scored_predictions,
    write_offline_prediction_records_jsonl,
)
from llm4rec.inference import (
    generate_prediction_records_with_local_hf,
    load_inference_run_config,
    load_local_hf_causal_lm,
    load_local_hf_model_config,
)
from llm4rec.prompts import load_baseline_prompt_template
from llm4rec.training import iter_baseline_prompt_records


app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_config: Path = Path("configs/data/amazon_food.yaml"),
    model_config: Path = Path("configs/model/base_llm.yaml"),
    inference_config: Path = Path("configs/inference/answer_only.yaml"),
    output_dir: Path = Path("data/processed/predictions"),
) -> None:
    """Run one local-model baseline inference job and export prediction JSONL."""

    llm_config = load_local_hf_model_config(model_config)
    run_config = load_inference_run_config(inference_config)
    prompt_template = (
        None
        if run_config.prompt_config_path is None
        else load_baseline_prompt_template(run_config.prompt_config_path)
    )
    prompt_version = (
        run_config.prompt_version
        if prompt_template is None
        else prompt_template.name
    )
    prompt_records = list(
        iter_baseline_prompt_records(
            data_config,
            split=run_config.split,
            style=run_config.prompt_style,
            template=prompt_template,
            limit=run_config.max_examples,
        )
    )

    model, tokenizer = load_local_hf_causal_lm(llm_config)
    run_id = _build_run_id(llm_config.name, run_config.name)
    prediction_records = generate_prediction_records_with_local_hf(
        prompt_records,
        model=model,
        tokenizer=tokenizer,
        model_config=llm_config,
        run_id=run_id,
        prompt_version=prompt_version,
    )

    output_path = output_dir / f"{run_id}_{run_config.prompt_style.value}_predictions.jsonl"
    record_count = write_offline_prediction_records_jsonl(output_path, prediction_records)

    lines = [
        f"run_id={run_id}",
        f"model_name={llm_config.name}",
        f"model_path={llm_config.model_path}",
        f"split={run_config.split}",
        f"prompt_style={run_config.prompt_style.value}",
        f"prompt_version={prompt_version}",
        f"prompt_config_path={run_config.prompt_config_path}",
        f"predictions={record_count}",
        f"output_path={output_path}",
    ]

    if run_config.score_with_verifier:
        example_index = build_example_index(
            data_config,
            split=run_config.split,
            limit=run_config.max_examples,
        )
        summary = summarize_scored_predictions(
            score_offline_prediction_records(example_index, prediction_records)
        )
        lines.extend(
            [
                f"hit_rate={summary.hit_rate:.4f}",
                f"mrr={summary.mean_reciprocal_rank:.4f}",
                f"mean_reward={summary.mean_reward:.4f}",
                f"schema_pass_rate={summary.schema_pass_rate:.4f}",
            ]
        )

    typer.echo("\n".join(lines))


def _build_run_id(model_name: str, run_name: str) -> str:
    """Create one deterministic-enough run id for artifact naming."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_name.lower().replace("/", "_").replace(" ", "_")
    run_slug = run_name.lower().replace("/", "_").replace(" ", "_")
    return f"{timestamp}_{model_slug}_{run_slug}"


if __name__ == "__main__":
    app()
