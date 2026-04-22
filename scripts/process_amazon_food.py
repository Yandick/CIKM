"""Export processed Amazon Food examples and item records to JSONL."""

from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.data import export_prepared_amazon_food


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = Path("configs/data/amazon_food.yaml"),
    output_dir: Path = Path("data/processed/amazon-food"),
) -> None:
    """Process the configured Amazon Food raw files into canonical JSONL artifacts."""

    summary = export_prepared_amazon_food(config, output_dir)
    typer.echo(
        "\n".join(
            [
                f"output_dir={summary.output_dir}",
                f"item_records={summary.item_count}",
                f"train_examples={summary.train_count}",
                f"validation_examples={summary.validation_count}",
                f"test_examples={summary.test_count}",
            ]
        )
    )


if __name__ == "__main__":
    app()
