from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from faithrec.recbench import (
    REC_BENCH_DOMAINS,
    REC_BENCH_TASKS,
    attach_candidates,
    build_catalog_index,
    load_domain_catalog,
    normalize_query_record,
    query_file_paths,
    read_query_file,
    to_rl_row,
)


def build_domain_instances(
    *,
    recbench_root: Path,
    item_dir: Path,
    domain: str,
    tasks: list[str],
    seed: int,
    num_candidates: int,
    positive_candidates: int,
    hard_negatives: int,
    max_instances: int | None,
    keep_no_positive_candidates: bool,
) -> list[dict]:
    catalog = load_domain_catalog(recbench_root, domain, item_dir=item_dir)
    catalog_indexes = build_catalog_index(catalog)
    instances: list[dict] = []
    for _, task_name, path in query_file_paths(recbench_root, domains=[domain], tasks=tasks):
        for row_idx, record in enumerate(read_query_file(path)):
            source_name = f"{domain}/{path.name}:{row_idx}"
            for instance in normalize_query_record(
                record,
                domain=domain,
                task_name=task_name,
                source_name=source_name,
            ):
                rng = random.Random(f"{seed}:{instance['instance_id']}")
                attached = attach_candidates(
                    instance,
                    catalog,
                    num_candidates=num_candidates,
                    positive_candidates=positive_candidates,
                    hard_negatives=hard_negatives,
                    rng=rng,
                    catalog_indexes=catalog_indexes,
                )
                if (
                    not keep_no_positive_candidates
                    and not attached.get("label", {}).get("positive_candidate_ids")
                ):
                    continue
                instances.append(attached)
                if max_instances is not None and len(instances) >= max_instances:
                    break
            if max_instances is not None and len(instances) >= max_instances:
                break
        if max_instances is not None and len(instances) >= max_instances:
            break
    random.Random(f"{seed}:{domain}:split").shuffle(instances)
    return instances


def split_instances(
    instances: list[dict],
    *,
    train_size: int,
    dev_size: int,
    test_size: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    train = instances[:train_size]
    dev_start = train_size
    test_start = train_size + dev_size
    dev = instances[dev_start:test_start]
    test = instances[test_start : test_start + test_size]
    return train, dev, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: Path, rows: list[dict]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Writing parquet requires pandas and pyarrow.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ReRec-aligned RecBench+ train/test data for SFT and RL."
    )
    parser.add_argument("--recbench-root", type=Path, required=True)
    parser.add_argument("--item-dir", type=Path, default=Path("data/recbench"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recbench/processed"))
    parser.add_argument(
        "--domains", nargs="+", choices=REC_BENCH_DOMAINS, default=list(REC_BENCH_DOMAINS)
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=REC_BENCH_TASKS, default=list(REC_BENCH_TASKS)
    )
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=12000)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--positive-candidates", type=int, default=1)
    parser.add_argument("--hard-negatives", type=int, default=0)
    parser.add_argument(
        "--keep-no-positive-candidates",
        action="store_true",
        help="Keep rows where the positive item could not be placed in the candidate set.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help=(
            "Maximum normalized instances to build per domain. "
            "Defaults to train-size + dev-size + test-size."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for domain in args.domains:
        max_instances = args.max_instances
        if max_instances is None:
            max_instances = args.train_size + args.dev_size + args.test_size
        instances = build_domain_instances(
            recbench_root=args.recbench_root,
            item_dir=args.item_dir,
            domain=domain,
            tasks=args.tasks,
            seed=args.seed,
            num_candidates=args.num_candidates,
            positive_candidates=args.positive_candidates,
            hard_negatives=args.hard_negatives,
            max_instances=max_instances,
            keep_no_positive_candidates=args.keep_no_positive_candidates,
        )
        train, dev, test = split_instances(
            instances,
            train_size=args.train_size,
            dev_size=args.dev_size,
            test_size=args.test_size,
        )

        canonical_dir = args.output_dir / "canonical"
        rl_dir = args.output_dir / "rl"
        write_jsonl(canonical_dir / f"{domain}_train.jsonl", train)
        write_jsonl(canonical_dir / f"{domain}_dev.jsonl", dev)
        write_jsonl(canonical_dir / f"{domain}_test.jsonl", test)
        write_parquet(rl_dir / f"{domain}_train.parquet", [to_rl_row(row) for row in train])
        write_parquet(rl_dir / f"{domain}_dev.parquet", [to_rl_row(row) for row in dev])
        write_parquet(rl_dir / f"{domain}_test.parquet", [to_rl_row(row) for row in test])

        print(
            f"{domain}: built {len(instances)} instances, "
            f"wrote {len(train)} train, {len(dev)} dev, and {len(test)} test"
        )


if __name__ == "__main__":
    main()
