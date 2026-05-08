from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path

from faithrec.recbench import (
    attach_candidates,
    load_domain_catalog,
    normalize_query_record,
    query_file_paths,
    render_recbench_prompt,
    to_rl_row,
)
from faithrec.reward import compute_score
from teacher.build_sft import sft_row, validate

VERL_REWARD_PATH = (
    Path(__file__).resolve().parents[1] / "verl" / "verl" / "utils" / "reward_score" / "recbench_json.py"
)
VERL_REWARD_SPEC = importlib.util.spec_from_file_location("recbench_json_reward", VERL_REWARD_PATH)
assert VERL_REWARD_SPEC and VERL_REWARD_SPEC.loader
VERL_REWARD_MODULE = importlib.util.module_from_spec(VERL_REWARD_SPEC)
VERL_REWARD_SPEC.loader.exec_module(VERL_REWARD_MODULE)
verl_compute_score = VERL_REWARD_MODULE.compute_score


def make_raw_explicit_query() -> dict:
    return {
        "source_user": "2326",
        "condition_num": 2,
        "movieCount": 2,
        "movieSubset": ["Forrest Gump (1994)", "Saving Private Ryan (1998)"],
        "movieSubsetId": [356, 2028],
        "sharedRelationships": [["Starring", "Tom Hanks"], ["Genre", "War"]],
        "situational_description_query": "I want a war movie starring Tom Hanks.",
        "direct_description_query": "Recommend war movies starring Tom Hanks.",
        "data_idx": 3862,
    }


def make_raw_interest_query() -> dict:
    return {
        "Shared consecutive books": ["The Hobbit"],
        "source_users": ["u1", "u2"],
        "data_idx": 11,
        "query & ground true": [
            {
                "reason": "Readers liked fantasy adventures.",
                "query": "Recommend another fantasy adventure book.",
                "book subset": ["The Fellowship of the Ring"],
            }
        ],
    }


def write_mini_recbench(root: Path) -> None:
    movie_dir = root / "dataset" / "movie"
    book_dir = root / "dataset" / "book"
    movie_dir.mkdir(parents=True)
    book_dir.mkdir(parents=True)
    (movie_dir / "movies.dat").write_text(
        "\n".join(
            [
                "356::Forrest Gump (1994)::Comedy|Drama|Romance|War",
                "2028::Saving Private Ryan (1998)::Action|Drama|War",
                "1::Toy Story (1995)::Animation|Children's|Comedy",
                "2::War Movie Without Hanks (2000)::War",
            ]
        ),
        encoding="latin-1",
    )
    (movie_dir / "ExplicitQuery.json").write_text(
        json.dumps([make_raw_explicit_query()]),
        encoding="utf-8",
    )
    (book_dir / "InterestBasedQuery.json").write_text(
        json.dumps([make_raw_interest_query()]),
        encoding="utf-8",
    )


def write_mini_rerec_data(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "movies.dat").write_text(
        "\n".join(
            [
                "356::Forrest Gump (1994)::Comedy|Drama|Romance|War",
                "2028::Saving Private Ryan (1998)::Action|Drama|War",
                "1::Toy Story (1995)::Animation|Children's|Comedy",
                "2::War Movie Without Hanks (2000)::War",
            ]
        ),
        encoding="latin-1",
    )
    (root / "books.dat").write_text(
        "\n".join(
            [
                "10::The Hobbit",
                "11::The Fellowship of the Ring",
                "12::Dune",
            ]
        ),
        encoding="utf-8",
    )


def make_catalog() -> list[dict]:
    return [
        {
            "item_id": "356",
            "title": "Forrest Gump (1994)",
            "attributes": {"Genre": ["Drama", "War"]},
            "text": "Tom Hanks Drama War",
        },
        {
            "item_id": "2028",
            "title": "Saving Private Ryan (1998)",
            "attributes": {"Genre": ["War"]},
            "text": "Tom Hanks War",
        },
        {
            "item_id": "1",
            "title": "Toy Story (1995)",
            "attributes": {"Genre": ["Animation"]},
            "text": "Animation",
        },
        {
            "item_id": "2",
            "title": "War Movie Without Hanks (2000)",
            "attributes": {"Genre": ["War"]},
            "text": "War",
        },
    ]


def make_valid_teacher_response(instance: dict) -> str:
    candidate_ids = [item["candidate_id"] for item in instance["candidates"]]
    positives = instance["label"]["positive_candidate_ids"]
    ranking = positives + [
        candidate_id for candidate_id in candidate_ids if candidate_id not in positives
    ]
    evidence_refs = [item["evidence_id"] for item in instance["evidence"][:1]]
    evidence_refs.append(ranking[0])
    final = {
        "ranking": ranking,
        "selected_candidate_id": ranking[0],
        "evidence_refs": evidence_refs,
        "rationale": [
            {
                "candidate_id": ranking[0],
                "claim": "matches_query",
                "support": evidence_refs,
            }
        ],
    }
    return (
        "Evidence Selection:\n"
        f"- [{evidence_refs[0]}] is relevant to the user's request.\n\n"
        "Candidate Reasoning:\n"
        f"{ranking[0]} best matches the query and evidence.\n\n"
        "Ranking Decision:\n"
        f"Select {ranking[0]} as the strongest candidate.\n\n"
        "Final Answer:\n" + json.dumps(final)
    )


def test_query_file_paths_and_catalog_are_recbench_specific(tmp_path: Path) -> None:
    write_mini_recbench(tmp_path)

    paths = query_file_paths(
        tmp_path,
        domains=["movie", "book"],
        tasks=["ExplicitQuery", "InterestBasedQuery"],
    )
    assert [(domain, task) for domain, task, _ in paths] == [
        ("movie", "ExplicitQuery"),
        ("book", "InterestBasedQuery"),
    ]

    rerec_data = tmp_path / "rerec_data"
    write_mini_rerec_data(rerec_data)
    movie_catalog = load_domain_catalog(tmp_path, "movie", item_dir=rerec_data)
    book_catalog = load_domain_catalog(tmp_path, "book", item_dir=rerec_data)
    assert any(item["item_id"] == "356" for item in movie_catalog)
    assert any(item["title"] == "Dune" for item in book_catalog)
    assert any(item["title"] == "The Fellowship of the Ring" for item in book_catalog)


def test_normalize_recbench_explicit_query_and_prompt() -> None:
    instance = normalize_query_record(
        make_raw_explicit_query(),
        domain="movie",
        task_name="ExplicitQuery",
        source_name="movie/ExplicitQuery.json:0",
    )[0]
    instance = attach_candidates(
        instance,
        make_catalog(),
        num_candidates=4,
        hard_negatives=1,
        rng=random.Random(7),
    )

    assert instance["query"] == "I want a war movie starring Tom Hanks."
    assert instance["label"]["ground_truth_item_ids"] == ["356", "2028"]
    assert len(instance["label"]["positive_candidate_ids"]) == 1
    assert len(instance["label"]["selected_ground_truth_titles"]) == 1

    prompt = render_recbench_prompt(instance)
    assert "Allowed Candidate IDs:" in prompt
    assert "Allowed Evidence IDs:" in prompt
    assert "rationale must be a list" in prompt
    assert "Hxx" not in prompt

    rl_row = to_rl_row(instance)
    assert rl_row["prompt"][0]["role"] == "user"
    assert rl_row["reward_model"]["positive_candidate_ids"]
    assert rl_row["extra_info"]["positive_candidate_ids"]
    assert rl_row["extra_info"]["source_evidence_ids"]


def test_profile_query_expands_children() -> None:
    instance = normalize_query_record(
        make_raw_interest_query(),
        domain="book",
        task_name="InterestBasedQuery",
        source_name="book/InterestBasedQuery.json:0",
    )[0]

    assert instance["task_type"] == "profile_interest"
    assert instance["query"] == "Recommend another fantasy adventure book."
    assert instance["history"] == ["The Hobbit"]
    assert instance["label"]["ground_truth_titles"] == ["The Fellowship of the Ring"]
    assert [item["evidence_id"] for item in instance["evidence"]] == ["H01", "P01"]


def test_recbench_reward_scores_valid_positive_ranking() -> None:
    instance = normalize_query_record(
        make_raw_explicit_query(),
        domain="movie",
        task_name="ExplicitQuery",
        source_name="movie/ExplicitQuery.json:0",
    )[0]
    instance = attach_candidates(
        instance,
        make_catalog(),
        num_candidates=4,
        hard_negatives=1,
        rng=random.Random(7),
    )
    response = make_valid_teacher_response(instance)
    rl_row = to_rl_row(instance)

    score = compute_score(
        response,
        positive_candidate_ids=rl_row["extra_info"]["positive_candidate_ids"],
        candidate_ids=rl_row["extra_info"]["candidate_ids"],
        evidence_ids=rl_row["extra_info"]["evidence_ids"],
        k=1,
    )

    assert score["parse_success"] == 1.0
    assert score["format"] == 1.0
    assert score["rationale"] == 1.0
    assert score["ndcg@1"] == 1.0


def test_recbench_reward_penalizes_missing_rationale() -> None:
    instance = normalize_query_record(
        make_raw_explicit_query(),
        domain="movie",
        task_name="ExplicitQuery",
        source_name="movie/ExplicitQuery.json:0",
    )[0]
    instance = attach_candidates(
        instance,
        make_catalog(),
        num_candidates=4,
        hard_negatives=1,
        rng=random.Random(7),
    )
    response = make_valid_teacher_response(instance)
    final = json.loads(response.split("Final Answer:\n", 1)[1])
    final.pop("rationale")
    rl_row = to_rl_row(instance)

    score = compute_score(
        json.dumps(final),
        positive_candidate_ids=rl_row["extra_info"]["positive_candidate_ids"],
        candidate_ids=rl_row["extra_info"]["candidate_ids"],
        evidence_ids=rl_row["extra_info"]["evidence_ids"],
        k=1,
    )

    assert score["parse_success"] == 1.0
    assert score["recommendation"] == 1.0
    assert score["rationale"] == 0.0
    assert 0.0 < score["faithfulness"] < 1.0
    assert score["faithful_binary"] == 0.0
    assert "empty_rationale" in score["validation_errors"]


def test_recbench_reward_is_rank_sensitive_for_reranking() -> None:
    instance = normalize_query_record(
        make_raw_explicit_query(),
        domain="movie",
        task_name="ExplicitQuery",
        source_name="movie/ExplicitQuery.json:0",
    )[0]
    instance = attach_candidates(
        instance,
        make_catalog(),
        num_candidates=4,
        hard_negatives=1,
        rng=random.Random(7),
    )
    rl_row = to_rl_row(instance)
    candidate_ids = rl_row["extra_info"]["candidate_ids"]
    positive = rl_row["extra_info"]["positive_candidate_ids"][0]
    negative = next(candidate_id for candidate_id in candidate_ids if candidate_id != positive)
    ranking = [negative, positive] + [
        candidate_id for candidate_id in candidate_ids if candidate_id not in {negative, positive}
    ]
    source_ref = rl_row["extra_info"]["source_evidence_ids"][0]
    final = {
        "ranking": ranking,
        "selected_candidate_id": negative,
        "evidence_refs": [source_ref, negative],
        "rationale": [
            {
                "candidate_id": negative,
                "claim": "supports_ranking",
                "support": [source_ref, negative],
            }
        ],
    }

    score = compute_score(
        json.dumps(final),
        positive_candidate_ids=rl_row["extra_info"]["positive_candidate_ids"],
        candidate_ids=candidate_ids,
        evidence_ids=rl_row["extra_info"]["evidence_ids"],
        source_evidence_ids=rl_row["extra_info"]["source_evidence_ids"],
        k=10,
        reward_mode="faithrl",
    )

    assert abs(score["recommendation"] - (1.0 / math.log2(3))) < 1e-9
    assert score["correctness"] == 0.0
    assert score["faithful_binary"] == 1.0
    assert score["outcome"] == "wrong_faithful"


def test_recbench_faithrl_reward_classifies_outcomes() -> None:
    instance = normalize_query_record(
        make_raw_explicit_query(),
        domain="movie",
        task_name="ExplicitQuery",
        source_name="movie/ExplicitQuery.json:0",
    )[0]
    instance = attach_candidates(
        instance,
        make_catalog(),
        num_candidates=4,
        hard_negatives=1,
        rng=random.Random(7),
    )
    response = make_valid_teacher_response(instance)
    rl_row = to_rl_row(instance)
    kwargs = {
        "positive_candidate_ids": rl_row["extra_info"]["positive_candidate_ids"],
        "candidate_ids": rl_row["extra_info"]["candidate_ids"],
        "evidence_ids": rl_row["extra_info"]["evidence_ids"],
        "source_evidence_ids": rl_row["extra_info"]["source_evidence_ids"],
        "k": 1,
        "reward_mode": "faithrl",
        "baseline_correct_rate": 0.7,
        "baseline_unfaithful_rate": 0.3,
    }

    faithful_correct = compute_score(response, **kwargs)
    assert faithful_correct["score"] == 0.3
    assert faithful_correct["outcome"] == "correct_faithful"
    assert faithful_correct["faithfulness"] == 1.0
    assert faithful_correct["faithful_binary"] == 1.0

    final = json.loads(response.split("Final Answer:\n", 1)[1])
    final.pop("rationale")
    unfaithful = compute_score(json.dumps(final), **kwargs)
    assert unfaithful["score"] == -0.7
    assert unfaithful["outcome"] == "unfaithful"
    assert 0.0 < unfaithful["faithfulness"] < 1.0
    assert unfaithful["faithful_binary"] == 0.0


def test_verl_reward_matches_local_reward_for_key_metrics() -> None:
    instance = normalize_query_record(
        make_raw_explicit_query(),
        domain="movie",
        task_name="ExplicitQuery",
        source_name="movie/ExplicitQuery.json:0",
    )[0]
    instance = attach_candidates(
        instance,
        make_catalog(),
        num_candidates=4,
        hard_negatives=1,
        rng=random.Random(7),
    )
    response = make_valid_teacher_response(instance)
    rl_row = to_rl_row(instance)
    local = compute_score(
        response,
        positive_candidate_ids=rl_row["extra_info"]["positive_candidate_ids"],
        candidate_ids=rl_row["extra_info"]["candidate_ids"],
        evidence_ids=rl_row["extra_info"]["evidence_ids"],
        k=1,
    )
    verl = verl_compute_score(response, rl_row["extra_info"], k=1)

    assert verl["parse_success"] == local["parse_success"]
    assert verl["recommendation"] == local["recommendation"]
    assert verl["format"] == local["format"]
    assert verl["evidence"] == local["evidence"]
    assert verl["rationale"] == local["rationale"]
    assert verl["faithfulness"] == local["faithfulness"]
    assert verl["faithful_binary"] == local["faithful_binary"]


def test_teacher_sft_validation_accepts_positive_top1_response() -> None:
    instance = normalize_query_record(
        make_raw_explicit_query(),
        domain="movie",
        task_name="ExplicitQuery",
        source_name="movie/ExplicitQuery.json:0",
    )[0]
    instance = attach_candidates(
        instance,
        make_catalog(),
        num_candidates=4,
        hard_negatives=1,
        rng=random.Random(7),
    )
    response = make_valid_teacher_response(instance)

    args = type(
        "Args",
        (),
        {"min_ndcg1": 1.0, "min_format": 1.0, "min_evidence": 1.0, "min_rationale": 1.0},
    )()
    validation = validate(response, instance, args)
    record = sft_row(instance, response, "teacher-model", validation)

    assert validation["accepted"] is True
    assert record["messages"][0]["role"] == "user"
    assert record["messages"][1]["role"] == "assistant"
    assert "Teacher-only supervision" not in record["messages"][0]["content"]
