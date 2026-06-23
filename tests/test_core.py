from __future__ import annotations

import json
from pathlib import Path

from iaa_agent.cli import _resolve_user_target_index
from iaa_agent.data import NYCDataRepository
from iaa_agent.engine import IAAAgent, RunConfig
from iaa_agent.utils import haversine_km


def test_haversine_reasonable_distance() -> None:
    distance = haversine_km(40.7128, -74.0060, 40.7580, -73.9855)
    assert 5.0 < distance < 7.0


def test_query_context_excludes_ground_truth() -> None:
    repo = NYCDataRepository("datasets/NYC")
    query = repo.get_query("349_52")
    assert len(query.context) >= 1
    assert str(query.context.iloc[-1]["POI_id"]) != str(query.target["POI_id"])
    assert str(query.target["POI_id"]) not in [str(x) for x in query.context["POI_id"].tail(1).tolist()]


def test_agent_run_schema_and_guardrails() -> None:
    repo = NYCDataRepository("datasets/NYC")
    agent = IAAAgent(repo, RunConfig(llm_mode="fake"))
    result = agent.run("349_52")
    payload = result.model_dump(mode="json")

    assert payload["query_id"] == "349_52"
    assert payload["query_mode"] == "trajectory"
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


def test_user_chronological_query_and_run() -> None:
    repo = NYCDataRepository("datasets/NYC")
    repo.use_user_chronological_split(0.8)
    user_id, target_index = repo.iter_user_test_events(train_ratio=0.8, min_context=1)[0]
    query = repo.get_user_query(user_id, target_index, train_ratio=0.8, context_size=5)
    assert query.mode == "user_timeline"
    assert query.history is not None
    assert len(query.history) <= target_index
    assert len(query.context) <= 5

    agent = IAAAgent(repo, RunConfig(llm_mode="fake"))
    result = agent.run_query(query)
    payload = result.model_dump(mode="json")
    assert payload["query_mode"] == "user_timeline"
    assert payload["ground_truth_poi_idx"].startswith("P")
    assert payload["ranked_pois"][0]["poi_idx"].startswith("P")


def test_default_user_target_index_uses_last_held_out_event() -> None:
    repo = NYCDataRepository("datasets/NYC")
    info = repo.user_timeline_info("349", train_ratio=0.8)
    assert _resolve_user_target_index(repo, "349", 0.8, None) == info["valid_target_index_end"]
    assert _resolve_user_target_index(repo, "349", 0.8, 576) == 576
