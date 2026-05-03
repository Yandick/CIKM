from faithrec.metrics import irrelevant_evidence_stability, ranking_metrics
from faithrec.parsing import parse_final_answer
from faithrec.prompts import render_direct_prompt, render_evidence_rerank_prompt
from faithrec.rewards import compute_reward
from faithrec.schema import CandidateItem, HistoryItem, RerankInstance, validate_output


def make_instance() -> RerankInstance:
    return RerankInstance(
        instance_id="x1",
        user_id="u1",
        history=[
            HistoryItem(evidence_id="H01", item_id="i1", title="Dark roast coffee", rating=5),
            HistoryItem(evidence_id="H02", item_id="i2", title="Chocolate cookies", rating=2),
        ],
        candidates=[
            CandidateItem(candidate_id="A", item_id="c1", title="Coffee beans"),
            CandidateItem(candidate_id="B", item_id="c2", title="Candy box"),
        ],
        target_candidate_id="A",
    )


def test_parse_and_validate_final_answer() -> None:
    text = """
Reasoning Trace:
H01 supports A.

Final Answer:
{"ranking": ["A", "B"], "selected_candidate_id": "A", "evidence_refs": ["H01"]}
"""
    output = parse_final_answer(text)
    errors = validate_output(output, make_instance())
    assert output.selected_candidate_id == "A"
    assert errors == []


def test_ranking_metrics() -> None:
    metrics = ranking_metrics(["B", "A"], "A", ks=(1, 2))
    assert metrics["hr@1"] == 0.0
    assert metrics["hr@2"] == 1.0
    assert metrics["mrr"] == 0.5


def test_reward_penalizes_invented_evidence() -> None:
    output = parse_final_answer(
        'Final Answer:\n{"ranking": ["A", "B"], "selected_candidate_id": "A", '
        '"evidence_refs": ["H99"]}'
    )
    reward = compute_reward(output, make_instance(), completion_tokens=10)
    assert "invented_evidence_ref" in reward.errors
    assert reward.format < 1.0


def test_irrelevant_stability() -> None:
    assert irrelevant_evidence_stability("A", ["A", "B"]) == 1.0
    assert irrelevant_evidence_stability("A", ["B", "A"]) == 0.0


def test_prompt_renderers_include_candidate_contract() -> None:
    instance = make_instance()
    direct = render_direct_prompt(instance)
    evidence = render_evidence_rerank_prompt(instance)
    assert "Candidate Set:" in direct
    assert "Final Answer:" in direct
    assert "Evidence Selection:" in evidence
    assert "H01" in evidence
