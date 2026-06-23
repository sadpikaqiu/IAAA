from __future__ import annotations

import json
from pathlib import Path

from iaa_agent.data import NYCDataRepository
from iaa_agent.engine import IAAAgent, RunConfig
from iaa_agent.evaluation import evaluate_session_split
from iaa_agent.utils import haversine_km


def test_haversine_reasonable_distance() -> None:
    distance = haversine_km(40.7128, -74.0060, 40.7580, -73.9855)
    assert 5.0 < distance < 7.0


def test_agent_session_run_schema_and_guardrails() -> None:
    repo = NYCDataRepository("datasets/NYC")
    agent = IAAAgent(repo, RunConfig(llm_mode="fake"))
    user_id, trajectory_id = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="1")[0]
    query = repo.get_session_query(user_id, trajectory_id, train_ratio=0.8, min_context=1)
    result = agent.run_query(query)
    payload = result.model_dump(mode="json")

    assert payload["query_mode"] == "session_split"
    assert payload["ranked_pois"]
    assert payload["ranked_pois"][0]["poi_idx"].startswith("P")
    assert payload["dataset_capabilities"]["has_reviews"] is False
    assert payload["dataset_capabilities"]["has_images"] is False
    assert payload["dataset_capabilities"]["has_opening_hours"] is False

    top = payload["ranked_pois"][0]
    assert len(top["supporting_evidence"]) >= 3
    assert "reviews unavailable" in top["missing_evidence"]
    assert "images unavailable" in top["missing_evidence"]
    assert "opening hours unavailable" in top["missing_evidence"]
    affordance_names = {a["name"] for a in top["affordance_profile"]["affordances"]}
    assert {
        "category_match",
        "spatial_feasibility",
        "temporal_fit",
        "revisit_support",
        "transition_support",
        "peer_support",
        "popularity_support",
        "reachability_time_gap",
    }.issubset(affordance_names)


def test_prepare_summary_shape(tmp_path: Path) -> None:
    repo = NYCDataRepository("datasets/NYC")
    summary = repo.summary()
    out = tmp_path / "summary.json"
    out.write_text(json.dumps(summary), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["train"]["rows"] > loaded["test"]["rows"]
    assert loaded["dataset_capabilities"]["has_category"] is True


def test_session_split_keys_support_global_and_single_user_eval() -> None:
    repo = NYCDataRepository("datasets/NYC")
    global_keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1)
    user_keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="349")

    assert len(global_keys) > len(user_keys) > 0
    assert len(global_keys) == len(set(global_keys))
    assert len(user_keys) == 10
    assert {user_id for user_id, _ in user_keys} == {"349"}


def test_session_split_query_uses_original_trajectory() -> None:
    repo = NYCDataRepository("datasets/NYC")
    repo.use_user_chronological_split(0.8)
    user_id, trajectory_id = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="349")[0]
    query = repo.get_session_query(user_id, trajectory_id, train_ratio=0.8, min_context=1)
    rows = repo.all_events[repo.all_events["user_id"] == user_id].sort_values("UTC_time").reset_index(drop=True)
    cutoff = min(max(1, int(len(rows) * 0.8)), len(rows) - 1)

    assert query.mode == "session_split"
    assert query.history is not None
    assert query.target_index is not None
    assert query.target_index >= cutoff
    assert len(query.context) >= 1
    assert query.context["trajectory_id"].nunique() == 1
    assert str(query.context.iloc[0]["trajectory_id"]) == str(query.target["trajectory_id"])
    assert query.context["UTC_time"].max() < query.target["UTC_time"]
    assert query.history["UTC_time"].max() <= rows.iloc[cutoff - 1]["UTC_time"]


def test_single_user_session_evaluation_runs_all_user_sessions() -> None:
    repo = NYCDataRepository("datasets/NYC")
    user_keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="1")
    result = evaluate_session_split(repo, user_id="1", llm_mode="fake")
    payload = result.as_dict()
    assert payload["total"] == len(user_keys) == 2
    assert set(payload) == {"total", "Hit@1", "Hit@5", "Hit@10", "NDCG@1", "NDCG@5", "NDCG@10", "MRR"}
