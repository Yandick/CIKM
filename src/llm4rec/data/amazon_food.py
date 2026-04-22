"""Amazon Food preprocessing and export utilities for TRACE-Rec."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from llm4rec.data.candidates import PopularityCandidateConstructor
from llm4rec.data.schema import (
    CandidateItem,
    DatasetConfig,
    InteractionEvent,
    ItemRecord,
    NextItemExample,
    load_dataset_config,
)


_CSV_FIELD_LIMIT = sys.maxsize
while True:
    try:
        csv.field_size_limit(_CSV_FIELD_LIMIT)
        break
    except OverflowError:
        _CSV_FIELD_LIMIT = _CSV_FIELD_LIMIT // 10


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared dataset bundle with item records and canonical split examples."""

    item_records: dict[str, ItemRecord]
    train_examples: tuple[NextItemExample, ...]
    validation_examples: tuple[NextItemExample, ...]
    test_examples: tuple[NextItemExample, ...]


@dataclass(frozen=True)
class ExportSummary:
    """Summary of processed-data export outputs."""

    item_count: int
    train_count: int
    validation_count: int
    test_count: int
    output_dir: Path


@dataclass(frozen=True)
class AmazonFoodSplitRow:
    """One interaction row from a precomputed Amazon Food split file."""

    user_id: str
    item_id: str
    rating: float
    timestamp: int
    history_item_ids: tuple[str, ...]
    split: str


@dataclass
class _ItemAccumulator:
    """Running aggregate for item metadata and review-derived fields."""

    meta_title: str | None = None
    main_category: str | None = None
    store: str | None = None
    categories: list[str] = field(default_factory=list)
    feature_bullets: list[str] = field(default_factory=list)
    description_snippets: list[str] = field(default_factory=list)
    metadata_average_rating: float | None = None
    metadata_rating_number: int | None = None
    review_count: int = 0
    rating_sum: float = 0.0
    verified_count: int = 0
    review_titles: list[str] = field(default_factory=list)
    review_snippets: list[str] = field(default_factory=list)


def iter_amazon_food_rows(
    config: DatasetConfig | str | Path,
    *,
    split: str,
) -> Iterator[AmazonFoodSplitRow]:
    """Yield parsed rows from a precomputed Amazon Food split file."""

    dataset_config = _ensure_dataset_config(config)
    split_path = _split_path_for(dataset_config, split)
    column_map = dataset_config.columns

    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            user_id = str(raw_row[column_map["user_id"]]).strip()
            item_id = str(raw_row[column_map["item_id"]]).strip()
            rating = float(raw_row[column_map["rating"]])
            timestamp = int(float(raw_row[column_map["timestamp"]]))
            history_raw = raw_row.get(column_map.get("history", "history"), "")
            history_item_ids = tuple(
                token for token in str(history_raw).strip().split() if token
            )

            if not user_id or not item_id:
                continue

            yield AmazonFoodSplitRow(
                user_id=user_id,
                item_id=item_id,
                rating=rating,
                timestamp=timestamp,
                history_item_ids=history_item_ids,
                split=split,
            )


def build_candidate_constructor_from_train_split(
    config: DatasetConfig | str | Path,
) -> PopularityCandidateConstructor:
    """Build the default popularity candidate constructor from the train split only."""

    dataset_config = _ensure_dataset_config(config)
    threshold = dataset_config.preprocessing.rating_threshold
    interactions: list[InteractionEvent] = []

    for row in iter_amazon_food_rows(dataset_config, split="train"):
        if threshold is not None and row.rating < threshold:
            continue
        interactions.append(
            InteractionEvent(
                item_id=row.item_id,
                timestamp=row.timestamp,
                rating=row.rating,
                context={"user_id": row.user_id},
            )
        )

    if not interactions:
        raise ValueError("train split produced no positive interactions for candidate building")

    return PopularityCandidateConstructor.from_interactions(
        interactions,
        config=dataset_config.candidate_constructor,
    )


def collect_split_item_ids(
    config: DatasetConfig | str | Path,
) -> set[str]:
    """Collect all target and history item ids used by the configured split files."""

    dataset_config = _ensure_dataset_config(config)
    threshold = dataset_config.preprocessing.rating_threshold
    item_ids: set[str] = set()

    for split in ("train", "validation", "test"):
        for row in iter_amazon_food_rows(dataset_config, split=split):
            if threshold is not None and row.rating < threshold:
                continue
            item_ids.add(row.item_id)
            item_ids.update(row.history_item_ids)

    return item_ids


def build_item_records_from_sources(
    config: DatasetConfig | str | Path,
    *,
    allowed_item_ids: set[str] | None = None,
) -> dict[str, ItemRecord]:
    """Aggregate metadata and review text into prompt-ready item records."""

    dataset_config = _ensure_dataset_config(config)
    if dataset_config.metadata_file is None and dataset_config.review_file is None:
        return {
            item_id: ItemRecord(item_id=item_id, title=None, feature_refs=(), raw={})
            for item_id in sorted(allowed_item_ids or set())
        }

    accumulators: dict[str, _ItemAccumulator] = {}

    if dataset_config.metadata_file is not None:
        metadata_path = Path(dataset_config.raw_dir) / dataset_config.metadata_file
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                item_id = str(payload.get("parent_asin") or payload.get("asin") or "").strip()
                if not item_id:
                    continue
                if allowed_item_ids is not None and item_id not in allowed_item_ids:
                    continue

                accumulator = accumulators.setdefault(item_id, _ItemAccumulator())
                title = _normalize_text(str(payload.get("title", "")))
                if title:
                    accumulator.meta_title = title
                accumulator.main_category = _normalize_text(str(payload.get("main_category", ""))) or None
                accumulator.store = _normalize_text(str(payload.get("store", ""))) or None
                accumulator.metadata_average_rating = _maybe_float(payload.get("average_rating"))
                accumulator.metadata_rating_number = _maybe_int(payload.get("rating_number"))

                for category in payload.get("categories", []) or []:
                    category_text = _normalize_text(str(category))
                    if category_text:
                        _append_unique(
                            accumulator.categories,
                            category_text,
                            limit=dataset_config.preprocessing.max_item_snippets + 2,
                        )
                for feature in payload.get("features", []) or []:
                    feature_text = _normalize_text(str(feature))
                    if feature_text:
                        _append_unique(
                            accumulator.feature_bullets,
                            _truncate_text(
                                feature_text,
                                max_chars=dataset_config.preprocessing.max_snippet_chars,
                            ),
                            limit=dataset_config.preprocessing.max_item_snippets,
                        )
                for description in payload.get("description", []) or []:
                    description_text = _normalize_text(str(description))
                    if description_text:
                        _append_unique(
                            accumulator.description_snippets,
                            _truncate_text(
                                description_text,
                                max_chars=dataset_config.preprocessing.max_snippet_chars,
                            ),
                            limit=dataset_config.preprocessing.max_item_snippets,
                        )

    if dataset_config.review_file is not None:
        review_path = Path(dataset_config.raw_dir) / dataset_config.review_file
        with review_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                item_id = str(payload.get("parent_asin") or payload.get("asin") or "").strip()
                if not item_id:
                    continue
                if allowed_item_ids is not None and item_id not in allowed_item_ids:
                    continue

                accumulator = accumulators.setdefault(item_id, _ItemAccumulator())
                rating = payload.get("rating")
                if rating is not None:
                    accumulator.review_count += 1
                    accumulator.rating_sum += float(rating)
                accumulator.verified_count += int(bool(payload.get("verified_purchase", False)))

                review_title = _normalize_text(str(payload.get("title", "")))
                review_snippet = _normalize_text(str(payload.get("text", "")))
                if review_title:
                    _append_unique(
                        accumulator.review_titles,
                        review_title,
                        limit=dataset_config.preprocessing.max_item_snippets,
                    )
                if review_snippet:
                    _append_unique(
                        accumulator.review_snippets,
                        _truncate_text(
                            review_snippet,
                            max_chars=dataset_config.preprocessing.max_snippet_chars,
                        ),
                        limit=dataset_config.preprocessing.max_item_snippets,
                    )

    item_records: dict[str, ItemRecord] = {}
    for item_id in sorted(allowed_item_ids or accumulators.keys()):
        accumulator = accumulators.get(item_id)
        if accumulator is None:
            item_records[item_id] = ItemRecord(item_id=item_id, title=None, feature_refs=(), raw={})
            continue

        mean_rating = (
            accumulator.rating_sum / accumulator.review_count
            if accumulator.review_count > 0
            else accumulator.metadata_average_rating
        )
        verified_ratio = (
            accumulator.verified_count / accumulator.review_count
            if accumulator.review_count > 0
            else None
        )
        feature_refs = tuple(
            feature
            for feature in (
                _main_category_feature(accumulator.main_category),
                _store_feature(accumulator.store),
                *(_category_features(accumulator.categories)),
                _review_count_bucket(accumulator.review_count) if accumulator.review_count > 0 else None,
                _mean_rating_bucket(mean_rating),
                _verified_ratio_bucket(verified_ratio),
            )
            if feature is not None
        )
        item_records[item_id] = ItemRecord(
            item_id=item_id,
            title=accumulator.meta_title,
            feature_refs=feature_refs,
            raw={
                "main_category": accumulator.main_category,
                "store": accumulator.store,
                "categories": tuple(accumulator.categories),
                "feature_bullets": tuple(accumulator.feature_bullets),
                "description_snippets": tuple(accumulator.description_snippets),
                "metadata_average_rating": accumulator.metadata_average_rating,
                "metadata_rating_number": accumulator.metadata_rating_number,
                "review_count": accumulator.review_count,
                "mean_rating": None if mean_rating is None else round(mean_rating, 4),
                "verified_purchase_ratio": (
                    None if verified_ratio is None else round(verified_ratio, 4)
                ),
                "review_titles": tuple(accumulator.review_titles),
                "review_snippets": tuple(accumulator.review_snippets),
            },
        )

    return item_records


def build_item_records_from_reviews(
    config: DatasetConfig | str | Path,
    *,
    allowed_item_ids: set[str] | None = None,
) -> dict[str, ItemRecord]:
    """Backward-compatible wrapper around the unified item-record builder."""

    return build_item_records_from_sources(config, allowed_item_ids=allowed_item_ids)


def iter_amazon_food_examples(
    config: DatasetConfig | str | Path,
    *,
    split: str,
    candidate_constructor: PopularityCandidateConstructor,
    item_records: dict[str, ItemRecord] | None = None,
) -> Iterator[NextItemExample]:
    """Yield canonical next-item examples from one Amazon Food split file."""

    dataset_config = _ensure_dataset_config(config)
    threshold = dataset_config.preprocessing.rating_threshold
    min_history_len = dataset_config.preprocessing.min_history_len
    max_history_len = dataset_config.preprocessing.max_history_len

    for row in iter_amazon_food_rows(dataset_config, split=split):
        if threshold is not None and row.rating < threshold:
            continue

        history_item_ids = row.history_item_ids[-max_history_len:]
        if len(history_item_ids) < min_history_len:
            continue

        base_timestamp = row.timestamp - len(history_item_ids)
        history = tuple(
            InteractionEvent(
                item_id=item_id,
                timestamp=base_timestamp + idx + 1,
            )
            for idx, item_id in enumerate(history_item_ids)
        )
        example_id = f"amazon-food-{split}-{row.user_id}-{row.timestamp}-{row.item_id}"
        candidates = candidate_constructor.construct(
            example_id=example_id,
            user_id=row.user_id,
            history_item_ids=history_item_ids,
            target_item_id=row.item_id,
        )
        example = NextItemExample(
            example_id=example_id,
            user_id=row.user_id,
            history=history,
            target_item_id=row.item_id,
            candidates=candidates,
            split=split,
            context={
                "target_timestamp": row.timestamp,
                "target_rating": row.rating,
                "raw_history_length": len(row.history_item_ids),
                "history_timestamp_mode": "synthetic_from_order",
                "available_evidence_refs": tuple(
                    f"history:{index}" for index, _ in enumerate(history_item_ids)
                ),
                "available_feature_refs_by_candidate": _feature_registry_for_candidates(
                    candidates,
                    item_records=item_records,
                ),
            },
        )
        example.validate()
        yield example


def build_next_item_splits(
    config: DatasetConfig | str | Path,
    *,
    candidate_constructor: PopularityCandidateConstructor,
    item_records: dict[str, ItemRecord] | None = None,
) -> tuple[tuple[NextItemExample, ...], tuple[NextItemExample, ...], tuple[NextItemExample, ...]]:
    """Build train, validation, and test examples from precomputed Amazon Food splits."""

    dataset_config = _ensure_dataset_config(config)
    train_examples = tuple(
        iter_amazon_food_examples(
            dataset_config,
            split="train",
            candidate_constructor=candidate_constructor,
            item_records=item_records,
        )
    )
    validation_examples = tuple(
        iter_amazon_food_examples(
            dataset_config,
            split="validation",
            candidate_constructor=candidate_constructor,
            item_records=item_records,
        )
    )
    test_examples = tuple(
        iter_amazon_food_examples(
            dataset_config,
            split="test",
            candidate_constructor=candidate_constructor,
            item_records=item_records,
        )
    )
    return train_examples, validation_examples, test_examples


def prepare_amazon_food(
    config: DatasetConfig | str | Path,
) -> PreparedDataset:
    """Run the Amazon Food preprocessing pipeline into canonical examples."""

    dataset_config = _ensure_dataset_config(config)
    candidate_constructor = build_candidate_constructor_from_train_split(dataset_config)
    item_ids = collect_split_item_ids(dataset_config)
    item_records = build_item_records_from_sources(
        dataset_config,
        allowed_item_ids=item_ids,
    )
    train_examples, validation_examples, test_examples = build_next_item_splits(
        dataset_config,
        candidate_constructor=candidate_constructor,
        item_records=item_records,
    )

    return PreparedDataset(
        item_records=item_records,
        train_examples=train_examples,
        validation_examples=validation_examples,
        test_examples=test_examples,
    )


def export_prepared_amazon_food(
    config: DatasetConfig | str | Path,
    output_dir: str | Path,
) -> ExportSummary:
    """Export Amazon Food item records and examples as processed JSONL files."""

    dataset_config = _ensure_dataset_config(config)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    candidate_constructor = build_candidate_constructor_from_train_split(dataset_config)
    item_ids = collect_split_item_ids(dataset_config)
    item_records = build_item_records_from_sources(
        dataset_config,
        allowed_item_ids=item_ids,
    )

    _write_jsonl(
        target_dir / "item_records.jsonl",
        (asdict(item_record) for item_record in item_records.values()),
    )

    train_count = _write_jsonl(
        target_dir / "train.jsonl",
        (
            asdict(example)
            for example in iter_amazon_food_examples(
                dataset_config,
                split="train",
                candidate_constructor=candidate_constructor,
                item_records=item_records,
            )
        ),
    )
    validation_count = _write_jsonl(
        target_dir / "validation.jsonl",
        (
            asdict(example)
            for example in iter_amazon_food_examples(
                dataset_config,
                split="validation",
                candidate_constructor=candidate_constructor,
                item_records=item_records,
            )
        ),
    )
    test_count = _write_jsonl(
        target_dir / "test.jsonl",
        (
            asdict(example)
            for example in iter_amazon_food_examples(
                dataset_config,
                split="test",
                candidate_constructor=candidate_constructor,
                item_records=item_records,
            )
        ),
    )

    return ExportSummary(
        item_count=len(item_records),
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        output_dir=target_dir,
    )


def _write_jsonl(path: Path, rows: Iterator[dict]) -> int:
    """Write an iterator of JSON-serializable rows to a JSONL file."""

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def _append_unique(values: list[str], candidate: str, *, limit: int) -> None:
    """Append a value once while respecting the configured maximum list size."""

    if len(values) >= limit or candidate in values:
        return
    values.append(candidate)


def _feature_registry_for_candidates(
    candidates: tuple[CandidateItem, ...],
    *,
    item_records: dict[str, ItemRecord] | None,
) -> dict[str, tuple[str, ...]]:
    """Build a candidate-scoped feature registry from available item records."""

    if item_records is None:
        return {}
    return {
        candidate.item_id: tuple(item_records.get(candidate.item_id, ItemRecord(candidate.item_id)).feature_refs)
        for candidate in candidates
    }


def _normalize_text(text: str) -> str:
    """Collapse extra whitespace so review strings are prompt-friendly."""

    return " ".join(text.split()).strip()


def _truncate_text(text: str, *, max_chars: int) -> str:
    """Hard-truncate overly long review text snippets."""

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _review_count_bucket(review_count: int) -> str:
    """Bucket review counts into a small discrete feature space."""

    if review_count >= 50:
        return "review_count:50+"
    if review_count >= 10:
        return "review_count:10-49"
    if review_count >= 3:
        return "review_count:3-9"
    return "review_count:1-2"


def _mean_rating_bucket(mean_rating: float) -> str:
    """Bucket mean item rating into a small discrete feature space."""

    if mean_rating is None:
        return None
    if mean_rating >= 4.5:
        return "mean_rating:4.5+"
    if mean_rating >= 4.0:
        return "mean_rating:4.0-4.49"
    if mean_rating >= 3.0:
        return "mean_rating:3.0-3.99"
    return "mean_rating:<3.0"


def _verified_ratio_bucket(verified_ratio: float | None) -> str | None:
    """Bucket verified-purchase ratio into a small discrete feature space."""

    if verified_ratio is None:
        return None
    if verified_ratio >= 0.8:
        return "verified_ratio:high"
    if verified_ratio >= 0.5:
        return "verified_ratio:mid"
    return "verified_ratio:low"


def _main_category_feature(main_category: str | None) -> str | None:
    """Convert main category into one compact feature ref."""

    if not main_category:
        return None
    return f"main_category:{_slugify(main_category)}"


def _store_feature(store: str | None) -> str | None:
    """Convert store/brand into one compact feature ref."""

    if not store:
        return None
    return f"store:{_slugify(store)}"


def _category_features(categories: list[str]) -> tuple[str, ...]:
    """Convert a short category path into compact feature refs."""

    features: list[str] = []
    for category in categories[:3]:
        slug = _slugify(category)
        if slug:
            features.append(f"category:{slug}")
    return tuple(features)


def _slugify(text: str) -> str:
    """Convert free text into a compact ASCII-ish feature token."""

    return (
        text.lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .strip(" _")
    )


def _maybe_float(value: object) -> float | None:
    """Safely cast a metadata scalar to float."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: object) -> int | None:
    """Safely cast a metadata scalar to int."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _split_path_for(config: DatasetConfig, split: str) -> Path:
    """Resolve the configured path for one split name."""

    raw_dir = Path(config.raw_dir)
    if split == "train":
        if config.interactions_file is None:
            raise ValueError("interactions_file is required for the train split")
        return raw_dir / config.interactions_file
    if split == "validation":
        if config.validation_file is None:
            raise ValueError("validation_file is required for the validation split")
        return raw_dir / config.validation_file
    if split == "test":
        if config.test_file is None:
            raise ValueError("test_file is required for the test split")
        return raw_dir / config.test_file
    raise ValueError(f"unsupported split: {split}")


def _ensure_dataset_config(config: DatasetConfig | str | Path) -> DatasetConfig:
    """Normalize config input into a DatasetConfig object."""

    if isinstance(config, DatasetConfig):
        return config
    return load_dataset_config(config)
