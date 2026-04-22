"""Build deterministic prediction JSONL files for the offline verifier."""

from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation import (
    BaselinePredictionStrategy,
    build_example_index,
    iter_heuristic_prediction_records,
    score_offline_prediction_records,
    summarize_scored_predictions,
    write_offline_prediction_records_jsonl,
)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = Path("configs/data/amazon_food.yaml"),
    split: str = "validation",
    strategy: BaselinePredictionStrategy = BaselinePredictionStrategy.HISTORY_FEATURE_OVERLAP,
    output_dir: Path = Path("data/processed/predictions"),
    max_examples: int = 200,
) -> None:
    """Export deterministic baseline predictions in the offline verifier format."""

    prediction_records = list(
        iter_heuristic_prediction_records(
            config,
            split=split,
            strategy=strategy,
            limit=max_examples,
        )
    )
    output_path = output_dir / f"{split}_{strategy.value}_predictions.jsonl"
    record_count = write_offline_prediction_records_jsonl(output_path, prediction_records)

    example_index = build_example_index(
        config,
        split=split,
        limit=max_examples,
    )
    summary = summarize_scored_predictions(
        score_offline_prediction_records(example_index, prediction_records)
    )

    typer.echo(
        "\n".join(
            [
                f"split={split}",
                f"strategy={strategy.value}",
                f"predictions={record_count}",
                f"hit_rate={summary.hit_rate:.4f}",
                f"mrr={summary.mean_reciprocal_rank:.4f}",
                f"mean_reward={summary.mean_reward:.4f}",
                f"output_path={output_path}",
            ]
        )
    )


if __name__ == "__main__":
    app()
