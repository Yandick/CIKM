"""Score prediction JSONL with the offline verifier and select best-by-reward outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation import (
    build_example_index,
    load_offline_prediction_records_jsonl,
    score_offline_prediction_records,
    select_best_predictions_by_reward,
    summarize_scored_predictions,
    write_offline_summary_json,
    write_scored_prediction_records_jsonl,
)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    predictions_path: Path,
    config: Path = Path("configs/data/amazon_food.yaml"),
    split: str = "validation",
    output_dir: Path = Path("data/processed/offline_verifier"),
    max_examples: int | None = None,
) -> None:
    """Run offline verifier scoring and reward-guided reranking for one split."""

    example_index = build_example_index(
        config,
        split=split,
        limit=max_examples,
    )
    prediction_records = load_offline_prediction_records_jsonl(predictions_path)
    scored_records = score_offline_prediction_records(
        example_index,
        prediction_records,
    )
    best_records = select_best_predictions_by_reward(scored_records)
    summary = summarize_scored_predictions(scored_records)

    stem = predictions_path.stem
    scored_path = output_dir / f"{stem}_{split}_scored.jsonl"
    best_path = output_dir / f"{stem}_{split}_best.jsonl"
    summary_path = output_dir / f"{stem}_{split}_summary.json"
    scored_count = write_scored_prediction_records_jsonl(scored_path, scored_records)
    best_count = write_scored_prediction_records_jsonl(best_path, best_records)
    write_offline_summary_json(summary_path, summary)

    typer.echo(
        "\n".join(
            [
                f"split={split}",
                f"predictions={len(prediction_records)}",
                f"indexed_examples={len(example_index)}",
                f"scored_records={scored_count}",
                f"best_records={best_count}",
                f"mean_reward={summary.mean_reward:.4f}",
                f"hit_rate={summary.hit_rate:.4f}",
                f"best_of_n_hit_rate={summary.best_of_n_hit_rate:.4f}",
                f"summary_path={summary_path}",
                f"scored_path={scored_path}",
                f"best_path={best_path}",
            ]
        )
    )


if __name__ == "__main__":
    app()
