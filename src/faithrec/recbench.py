from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


REC_BENCH_DOMAINS = ("movie", "book")
REC_BENCH_TASKS = (
    "ExplicitQuery",
    "ImplicitQuery",
    "MisinformedQuery",
    "InterestBasedQuery",
    "DemographicsBasedQuery",
)
TASK_TYPE_BY_FILE = {
    "ExplicitQuery": "condition_explicit",
    "ImplicitQuery": "condition_implicit",
    "MisinformedQuery": "condition_misinformed",
    "InterestBasedQuery": "profile_interest",
    "DemographicsBasedQuery": "profile_demographics",
}
CatalogIndexes = tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _item_key(domain: str) -> str:
    if domain not in {"movie", "book"}:
        raise ValueError(f"Unsupported RecBench+ domain: {domain}")
    return domain


def _letter_id(index: int) -> str:
    if index >= 26:
        raise ValueError("candidate_id supports at most 26 candidates")
    return chr(ord("A") + index)


def _normalize_title(value: str) -> str:
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def _title_aliases(title: str) -> set[str]:
    clean = re.sub(r"\s+", " ", title.replace("_", " ")).strip()
    aliases = {_normalize_title(clean)}

    suffix_article = re.match(r"^(?P<name>.+), (?P<article>The|A|An) (?P<year>\(\d{4}\))$", clean)
    if suffix_article:
        aliases.add(
            _normalize_title(
                f"{suffix_article.group('article')} {suffix_article.group('name')} {suffix_article.group('year')}"
            )
        )

    prefix_article = re.match(r"^(?P<article>The|A|An) (?P<name>.+) (?P<year>\(\d{4}\))$", clean)
    if prefix_article:
        aliases.add(
            _normalize_title(
                f"{prefix_article.group('name')}, {prefix_article.group('article')} {prefix_article.group('year')}"
            )
        )
    return aliases


def _catalog_indexes(
    catalog: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_title: dict[str, dict[str, Any]] = {}
    for item in catalog:
        by_id[str(item["item_id"])] = item
        for alias in _title_aliases(str(item["title"])):
            by_title.setdefault(alias, item)
    return by_id, by_title


def query_file_paths(
    recbench_root: Path,
    *,
    domains: Iterable[str] = REC_BENCH_DOMAINS,
    tasks: Iterable[str] = REC_BENCH_TASKS,
) -> list[tuple[str, str, Path]]:
    dataset_dir = recbench_root / "dataset"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"RecBench+ dataset directory not found: {dataset_dir}")

    paths: list[tuple[str, str, Path]] = []
    for domain in domains:
        if domain not in REC_BENCH_DOMAINS:
            raise ValueError(f"Unsupported RecBench+ domain: {domain}")
        domain_dir = dataset_dir / domain
        if not domain_dir.is_dir():
            raise FileNotFoundError(f"RecBench+ domain directory not found: {domain_dir}")
        for task_name in tasks:
            if task_name not in REC_BENCH_TASKS:
                raise ValueError(f"Unsupported RecBench+ task: {task_name}")
            path = domain_dir / f"{task_name}.json"
            if path.exists():
                paths.append((domain, task_name, path))
    return paths


def read_query_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"RecBench+ query file must contain a JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def load_domain_catalog(
    recbench_root: Path,
    domain: str,
    *,
    item_dir: Path | None = None,
) -> list[dict[str, Any]]:
    item_dir = item_dir or Path("data/recbench")
    if domain == "movie":
        catalog_path = item_dir / "movies.dat"
        catalog = _load_movies_dat(catalog_path)
    elif domain == "book":
        catalog_path = item_dir / "books.dat"
        catalog = _load_books_dat(catalog_path)
    else:
        raise ValueError(f"Unsupported RecBench+ domain: {domain}")

    return catalog


def _load_movies_dat(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"MovieLens movies.dat not found: {path}")

    items: list[dict[str, Any]] = []
    with path.open("r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 2:
                continue
            genres = parts[2].split("|") if len(parts) >= 3 and parts[2] else []
            items.append(
                {
                    "item_id": parts[0],
                    "title": parts[1],
                    "attributes": {"Genre": genres},
                    "text": " ".join([parts[1], " ".join(genres)]),
                    "domain": "movie",
                }
            )
    return items


def _load_books_dat(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ReRec books.dat not found: {path}")

    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 2:
                continue
            items.append(
                {
                    "item_id": parts[0],
                    "title": parts[1],
                    "attributes": {},
                    "text": parts[1],
                    "domain": "book",
                }
            )
    return items


def build_catalog_index(catalog: list[dict[str, Any]]) -> CatalogIndexes:
    return _catalog_indexes(catalog)


def _record_positive_titles(record: dict[str, Any], domain: str) -> list[str]:
    item_name = _item_key(domain)
    titles = [_stringify(item) for item in _as_list(record.get(f"{item_name}Subset"))]
    for child in _profile_children(record):
        titles.extend(_stringify(item) for item in _as_list(child.get(f"{item_name} subset")))
    return [title for title in titles if title]


def _profile_children(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("query & ground true")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _record_positive_ids(record: dict[str, Any], domain: str) -> list[str]:
    if domain != "movie":
        return []
    return [_stringify(item) for item in _as_list(record.get("movieSubsetId")) if _stringify(item)]


def _normalize_conditions(value: Any) -> list[dict[str, str]]:
    conditions: list[dict[str, str]] = []
    for item in _as_list(value):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            relation = _stringify(item[0])
            target = _stringify(item[1])
        else:
            continue
        if relation and target:
            conditions.append({"relation": relation, "value": target})
    return conditions


def normalize_query_record(
    record: dict[str, Any],
    *,
    domain: str,
    task_name: str,
    source_name: str,
) -> list[dict[str, Any]]:
    if task_name == "InterestBasedQuery" or task_name == "DemographicsBasedQuery":
        return _normalize_profile_record(
            record,
            domain=domain,
            task_name=task_name,
            source_name=source_name,
        )
    return [
        _normalize_condition_record(
            record,
            domain=domain,
            task_name=task_name,
            source_name=source_name,
        )
    ]


def _normalize_condition_record(
    record: dict[str, Any],
    *,
    domain: str,
    task_name: str,
    source_name: str,
) -> dict[str, Any]:
    if task_name == "MisinformedQuery":
        query = _stringify(record.get("query"))
    else:
        query = _stringify(record.get("situational_description_query"))
    if not query:
        query = _stringify(record.get("direct_description_query"))

    conditions = _normalize_conditions(record.get("sharedRelationships"))
    evidence = _condition_evidence(record, conditions)
    positive_titles = _record_positive_titles(record, domain)
    positive_ids = _record_positive_ids(record, domain)
    return _canonical_instance(
        record,
        domain=domain,
        task_name=task_name,
        source_name=source_name,
        query=query,
        conditions=conditions,
        evidence=evidence,
        positive_titles=positive_titles,
        positive_ids=positive_ids,
        history=[],
    )


def _normalize_profile_record(
    record: dict[str, Any],
    *,
    domain: str,
    task_name: str,
    source_name: str,
) -> list[dict[str, Any]]:
    history_key = f"Shared consecutive {domain}s"
    history = [_stringify(item) for item in _as_list(record.get(history_key)) if _stringify(item)]
    shared_evidence = [
        {
            "evidence_id": f"H{idx:02d}",
            "type": "shared_history",
            "text": f"Shared consecutive {domain}: {title}",
        }
        for idx, title in enumerate(history, start=1)
    ]
    if task_name == "DemographicsBasedQuery":
        shared_evidence.extend(_demographic_evidence(record))

    instances: list[dict[str, Any]] = []
    for child_idx, child in enumerate(_profile_children(record)):
        positive_titles = [
            _stringify(item) for item in _as_list(child.get(f"{domain} subset")) if _stringify(item)
        ]
        evidence = list(shared_evidence)
        reason = _stringify(child.get("reason"))
        if reason:
            evidence.append(
                {
                    "evidence_id": "P01",
                    "type": "profile_reason",
                    "text": reason,
                }
            )
        source = f"{source_name}:child{child_idx}"
        instances.append(
            _canonical_instance(
                record,
                domain=domain,
                task_name=task_name,
                source_name=source,
                query=_stringify(child.get("query")),
                conditions=[],
                evidence=evidence,
                positive_titles=positive_titles,
                positive_ids=[],
                history=history,
            )
        )
    return instances


def _condition_evidence(
    record: dict[str, Any], conditions: list[dict[str, str]]
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for idx, condition in enumerate(conditions, start=1):
        evidence.append(
            {
                "evidence_id": f"Q{idx:02d}",
                "type": "query_condition",
                "relation": condition["relation"],
                "value": condition["value"],
                "text": f"{condition['relation']} = {condition['value']}",
            }
        )
    for idx, item in enumerate(_as_list(record.get("multihop_info")), start=1):
        if isinstance(item, dict):
            movie_or_book = _stringify(item.get("movie") or item.get("book"))
            relation = _stringify(item.get("relation"))
            person = _stringify(item.get("person"))
            text = f"{movie_or_book} -- {relation} -- {person}".strip(" -")
        else:
            text = _stringify(item)
        if text:
            evidence.append({"evidence_id": f"M{idx:02d}", "type": "multihop_hint", "text": text})
    for idx, item in enumerate(_as_list(record.get("misinformed")), start=1):
        if isinstance(item, dict):
            title = _stringify(item.get("movie") or item.get("book"))
            relation = _stringify(item.get("relation"))
            wrong = _stringify(item.get("person"))
            original = _stringify(item.get("original person"))
            text = (
                f"{title} has query misinformation: {relation} is {wrong}; original is {original}"
            )
        else:
            text = _stringify(item)
        if text:
            evidence.append({"evidence_id": f"X{idx:02d}", "type": "misinformation", "text": text})
    return evidence


def _demographic_evidence(record: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    group_type = _stringify(record.get("group_type"))
    group = _stringify(record.get("group"))
    if group_type or group:
        evidence.append(
            {
                "evidence_id": "G01",
                "type": "demographic_group",
                "text": f"{group_type}: {group}".strip(": "),
            }
        )
    summary = _stringify(record.get("summary"))
    if summary:
        evidence.append({"evidence_id": "G02", "type": "demographic_summary", "text": summary})
    return evidence


def _canonical_instance(
    record: dict[str, Any],
    *,
    domain: str,
    task_name: str,
    source_name: str,
    query: str,
    conditions: list[dict[str, str]],
    evidence: list[dict[str, Any]],
    positive_titles: list[str],
    positive_ids: list[str],
    history: list[str],
) -> dict[str, Any]:
    data_idx = _stringify(record.get("data_idx"))
    if not data_idx:
        data_idx = source_name
    source_suffix = re.sub(r"[^a-zA-Z0-9]+", "_", source_name).strip("_").lower()
    instance_id = f"recbenchplus_{domain}_{TASK_TYPE_BY_FILE[task_name]}_{data_idx}_{source_suffix}"
    instance_id = re.sub(r"[^a-zA-Z0-9_]+", "_", instance_id).strip("_").lower()
    source_key = f"{domain}:{task_name}:{data_idx}"
    return {
        "instance_id": instance_id,
        "dataset": "recbench_plus",
        "domain": domain,
        "task_name": task_name,
        "task_type": TASK_TYPE_BY_FILE[task_name],
        "source_user": record.get("source_user"),
        "query": query,
        "conditions": conditions,
        "history": history,
        "candidates": [],
        "evidence": evidence,
        "label": {
            "ground_truth_item_ids": positive_ids,
            "ground_truth_titles": positive_titles,
            "positive_candidate_ids": [],
        },
        "metadata": {
            "data_idx": data_idx,
            "source_name": source_name,
            "source_key": source_key,
            "source_user_count": len(_as_list(record.get("source_users"))),
        },
    }


def attach_candidates(
    instance: dict[str, Any],
    catalog: list[dict[str, Any]],
    *,
    num_candidates: int,
    hard_negatives: int,
    positive_candidates: int = 1,
    rng: random.Random,
    catalog_indexes: CatalogIndexes | None = None,
) -> dict[str, Any]:
    if num_candidates < 1:
        raise ValueError("--num-candidates must be positive")
    if num_candidates > 26:
        raise ValueError("--num-candidates cannot exceed 26 because candidate IDs are A-Z")
    if positive_candidates < 1:
        raise ValueError("--positive-candidates must be positive")

    all_positives = _positive_items(instance, catalog, catalog_indexes=catalog_indexes)
    rng.shuffle(all_positives)
    selected_positives = all_positives[: min(positive_candidates, num_candidates)]
    all_positive_item_ids = {item["item_id"] for item in all_positives}
    positive_ids = {item["item_id"] for item in selected_positives}

    hard: list[dict[str, Any]] = []
    if hard_negatives > 0:
        negative_pool = [item for item in catalog if item["item_id"] not in all_positive_item_ids]
        hard = _hard_negative_items(instance, negative_pool, hard_negatives)
    hard_ids = {item["item_id"] for item in hard}
    random_needed = max(0, num_candidates - len(selected_positives) - len(hard))
    random_negatives = _sample_negative_items(
        catalog,
        exclude_item_ids=all_positive_item_ids | hard_ids,
        count=random_needed,
        rng=rng,
    )
    selected = selected_positives + hard + random_negatives
    selected = selected[:num_candidates]
    rng.shuffle(selected)

    candidates: list[dict[str, Any]] = []
    positive_candidate_ids: list[str] = []
    for idx, item in enumerate(selected):
        candidate_id = _letter_id(idx)
        is_positive = item["item_id"] in positive_ids
        if is_positive:
            positive_candidate_ids.append(candidate_id)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "item_id": item["item_id"],
                "title": item["title"],
                "attributes": item.get("attributes", {}),
                "label": int(is_positive),
                "negative_type": item.get("negative_type"),
            }
        )

    actual_hard_count = sum(
        1 for item in candidates if item.get("negative_type") == "hard_condition"
    )
    updated = dict(instance)
    updated["candidates"] = candidates
    updated["label"] = dict(instance["label"])
    if not updated["label"].get("ground_truth_item_ids"):
        updated["label"]["ground_truth_item_ids"] = [item["item_id"] for item in all_positives]
    updated["label"]["positive_candidate_ids"] = positive_candidate_ids
    updated["label"]["selected_ground_truth_item_ids"] = [
        item["item_id"] for item in selected_positives
    ]
    updated["label"]["selected_ground_truth_titles"] = [
        item["title"] for item in selected_positives
    ]
    updated["metadata"] = dict(instance.get("metadata") or {})
    updated["metadata"]["candidate_policy"] = {
        "positive": len(positive_candidate_ids),
        "hard_condition": actual_hard_count,
        "random": len(candidates) - len(positive_candidate_ids) - actual_hard_count,
        "available_positive": len(all_positives),
        "dropped_positive": max(0, len(all_positives) - len(selected_positives)),
    }
    return updated


def _positive_items(
    instance: dict[str, Any],
    catalog: list[dict[str, Any]],
    *,
    catalog_indexes: CatalogIndexes | None = None,
) -> list[dict[str, Any]]:
    by_id, by_title = catalog_indexes or _catalog_indexes(catalog)
    ids = [_stringify(item) for item in instance["label"].get("ground_truth_item_ids") or []]
    titles = [_stringify(item) for item in instance["label"].get("ground_truth_titles") or []]

    positives: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_len = max(len(ids), len(titles))
    for idx in range(max_len):
        item_id = ids[idx] if idx < len(ids) else ""
        title = titles[idx] if idx < len(titles) else ""
        item = by_id.get(item_id)
        if item is None and title:
            for alias in _title_aliases(title):
                item = by_title.get(alias)
                if item is not None:
                    break
        if item is None:
            item = {
                "item_id": item_id or title,
                "title": title or item_id,
                "attributes": {},
                "text": title or item_id,
                "domain": instance.get("domain"),
            }
        if item["item_id"] not in seen:
            positives.append({**item, "negative_type": None})
            seen.add(item["item_id"])
    return positives


def _sample_negative_items(
    catalog: list[dict[str, Any]],
    *,
    exclude_item_ids: set[str],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count <= 0 or not catalog:
        return []

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    max_attempts = max(100, count * 50)
    attempts = 0
    while len(selected) < count and attempts < max_attempts:
        attempts += 1
        item = rng.choice(catalog)
        item_id = item["item_id"]
        if item_id in exclude_item_ids or item_id in selected_ids:
            continue
        selected.append({**item, "negative_type": None})
        selected_ids.add(item_id)

    if len(selected) < count:
        remaining = [
            item
            for item in catalog
            if item["item_id"] not in exclude_item_ids and item["item_id"] not in selected_ids
        ]
        rng.shuffle(remaining)
        selected.extend(
            {**item, "negative_type": None} for item in remaining[: count - len(selected)]
        )
    return selected


def _hard_negative_items(
    instance: dict[str, Any], pool: list[dict[str, Any]], hard_negatives: int
) -> list[dict[str, Any]]:
    if hard_negatives <= 0:
        return []

    terms = [
        condition["value"].lower()
        for condition in instance.get("conditions", [])
        if condition.get("value")
    ]
    if not terms:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in pool:
        text = json.dumps(
            {
                "title": item.get("title", ""),
                "attributes": item.get("attributes", {}),
                "text": item.get("text", ""),
            },
            ensure_ascii=False,
        ).lower()
        score = sum(1 for term in terms if term and term in text)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**item, "negative_type": "hard_condition"} for _, item in scored[:hard_negatives]]


def render_recbench_prompt(instance: dict[str, Any]) -> str:
    candidates = instance.get("candidates") or []
    evidence = instance.get("evidence") or []
    candidate_ids = [item["candidate_id"] for item in candidates]
    evidence_ids = [item["evidence_id"] for item in evidence] + candidate_ids
    candidate_lines = "\n".join(
        (
            f'{item["candidate_id"]}. item_id="{item["item_id"]}", title="{item["title"]}", '
            f'attributes={json.dumps(item.get("attributes", {}), ensure_ascii=False)}'
        )
        for item in candidates
    )
    evidence_lines = "\n".join(
        f'{item["evidence_id"]}. {item.get("text") or json.dumps(item, ensure_ascii=False)}'
        for item in evidence
    )
    return f"""Task:
You are a personalized recommendation assistant.
Rank the candidate items for the RecBench+ {instance.get("domain")} task.
Use only the provided query, evidence, and candidate metadata.

User Query:
{instance.get("query", "")}

Evidence:
{evidence_lines if evidence_lines else "None"}

Candidate Set:
{candidate_lines}

Allowed Candidate IDs:
{json.dumps(candidate_ids, ensure_ascii=False)}

Allowed Evidence IDs:
{json.dumps(evidence_ids, ensure_ascii=False)}

Instruction:
Select recommendations by candidate ID, not by title.
Every candidate ID in the final ranking must come from Allowed Candidate IDs.
Every evidence reference must come from Allowed Evidence IDs.
Candidate IDs may be cited as evidence when comparing candidate metadata.
Do not invent candidate IDs or evidence IDs.

Evidence Selection:
- Cite the useful query/profile/history evidence IDs.

Candidate Reasoning:
- Briefly compare the strongest candidates.

Ranking Decision:
- State the selected candidate ID and why it satisfies the query.

Final Answer:
Return exactly one JSON object with keys ranking, selected_candidate_id, evidence_refs.
ranking must include every Allowed Candidate ID exactly once.
selected_candidate_id must equal ranking[0].
"""


def to_rl_row(instance: dict[str, Any]) -> dict[str, Any]:
    candidate_id_to_item_id = {
        item["candidate_id"]: item["item_id"] for item in instance.get("candidates", [])
    }
    candidate_id_to_title = {
        item["candidate_id"]: item["title"] for item in instance.get("candidates", [])
    }
    evidence_ids = [item["evidence_id"] for item in instance.get("evidence", [])] + [
        item["candidate_id"] for item in instance.get("candidates", [])
    ]
    return {
        "prompt": [{"role": "user", "content": render_recbench_prompt(instance)}],
        "data_source": f"recbench_plus_{instance.get('domain')}",
        "reward_model": {
            "ground_truth": ";".join(
                instance.get("label", {}).get("selected_ground_truth_titles")
                or instance.get("label", {}).get("ground_truth_titles")
                or []
            ),
            "ground_truth_item_ids": instance.get("label", {}).get("selected_ground_truth_item_ids")
            or instance.get("label", {}).get("ground_truth_item_ids")
            or [],
            "positive_candidate_ids": instance.get("label", {}).get("positive_candidate_ids") or [],
        },
        "extra_info": {
            "instance_id": instance["instance_id"],
            "domain": instance.get("domain"),
            "task_type": instance.get("task_type"),
            "candidate_ids": [item["candidate_id"] for item in instance.get("candidates", [])],
            "positive_candidate_ids": instance.get("label", {}).get("positive_candidate_ids") or [],
            "candidate_id_to_item_id": candidate_id_to_item_id,
            "candidate_id_to_title": candidate_id_to_title,
            "evidence_ids": evidence_ids,
            "conditions": instance.get("conditions", []),
            "candidates": instance.get("candidates", []),
        },
    }
