from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import shutil

import numpy as np
import pytest

from test_poi_evidence import fixture, A, B, C, D
from iaa_agent.data import NYCDataRepository
from iaa_agent.engine import IAAAgent, RunConfig
from iaa_agent.evidence import EvidenceStore
from scripts.mm_ablation_support import (
    ARMS, AblationAgent, Arm, agent_config, masked_store, prepare_prompt,
    digest, read, select_validation, summarize, validated_intention_attempt, write,
)
from scripts.run_multimodal_ablations import load_intention


def forbidden(*a, **kw):
    pytest.fail("An offline intervention called the model")


def test_modality_removal_preserves_joint_index_and_remaining_feature(fixture):
    root, _, _, path = fixture
    repo = NYCDataRepository(root / "NYC")
    query = repo.get_session_query("1", "1_4")
    joint = EvidenceStore(path)
    images = masked_store(joint, "images")
    assert images.matrix is joint.matrix and images.vectorizer is joint.vectorizer
    np.testing.assert_array_equal(images.score("Coffee tables"), joint.score("Coffee tables"))
    assert all(item["modality"] == "image" for item in images.get(A))
    assert not images.has(A, "review")
    assert C not in dict(images.search(images.score("TARGET_ONLY_MARKER coffee"), [A, C]))
    intention = IAAAgent(repo).run_query(query).inferred_intention
    results = {}
    for mode, store in (("both", joint), ("images", images)):
        arm = Arm(mode, mode=mode)
        agent = AblationAgent(repo, agent_config(arm, path), arm=arm, evidence_store=store,
                              frozen_intention=intention)
        agent.llm.chat_json = forbidden
        results[mode] = {p.poi_id: p.score_decomposition for p in agent.run_query(query).ranked_pois}
    assert results["images"][A]["image_intent_relevance"] > 0
    assert results["images"][A]["image_intent_relevance"] == results["both"][A]["image_intent_relevance"]
    assert "review_intent_relevance" not in results["images"][A]


def test_full_intervention_matches_production_and_rank_only_preserves_pool(fixture):
    root, _, _, path = fixture
    repo = NYCDataRepository(root / "NYC")
    query = repo.get_session_query("1", "1_4")
    store = EvidenceStore(path)
    arm = Arm("full_both")
    original = IAAAgent(repo, agent_config(arm, path), evidence_store=store).run_query(query)
    replay = AblationAgent(repo, agent_config(arm, path), evidence_store=store, arm=arm,
                           frozen_intention=original.inferred_intention)
    replay.llm.chat_json = forbidden
    result = replay.run_query(query)
    assert result.ranked_pois == original.ranked_pois
    assert result.candidate_pool_summary == original.candidate_pool_summary
    assert result.reflection == original.reflection
    text_arm = ARMS[0]
    text = AblationAgent(repo, agent_config(text_arm, None), arm=text_arm,
                         frozen_intention=original.inferred_intention)
    text_result = text.run_query(query)
    rank_arm = next(a for a in ARMS if a.name == "rank_only")
    rank = AblationAgent(repo, agent_config(rank_arm, path), evidence_store=store, arm=rank_arm,
                         frozen_intention=original.inferred_intention, text_reflection=text_result.reflection)
    rank_result = rank.run_query(query)
    assert rank_result.candidate_pool_summary["candidate_poi_ids"] == text_result.candidate_pool_summary["candidate_poi_ids"]
    assert [r["pool_ids"] for r in rank.rounds] == [r["pool_ids"] for r in text.rounds]
    assert rank_result.ranked_pois[0].score_decomposition["image_intent_relevance"] >= 0


def test_prior_and_rank_weights_are_independent(fixture):
    root, _, _, path = fixture
    repo = NYCDataRepository(root / "NYC")
    def raw():
        return {poi: dict(poi_id=poi, poi_idx=str(i), display_name=poi, category="Coffee Shop",
                          latitude=40.7, longitude=-74., distance_km=1., source_scores=scores)
                for i, (poi, scores) in enumerate([(A, {"historical": 1.}),
                    (B, {"historical": .15}), (C, {"historical": .14, "poi_evidence": .05})])}
    for prior, expected in ((.1, [A, C]), (0, [A, B])):
        arm = Arm("control", prior_weight=prior, rank_weight=.1, quota=0)
        config = replace(agent_config(arm, path), candidate_pool_size=2)
        agent = AblationAgent(repo, config, arm=arm)
        assert [c.poi_id for c in agent._select_candidates(raw(), False)] == expected
        assert agent.config.evidence_weight == .1


def test_validation_sampler_is_disjoint_and_label_independent(fixture):
    root, _, _, _ = fixture
    repo = NYCDataRepository(root / "NYC")
    first = select_validation(repo, sample_size=500, repeat_size=1)
    assert first["selected_count"] == 2
    assert first["original_test_overlap"] == 0
    assert {c["trajectory_id"] for c in first["cases"]} == {"1_3", "2_3"}
    assert sum(c["repeat"] for c in first["cases"]) == 1
    repo.all_events["POI_id"] = D
    repo.all_events["POI_catname"] = "Airport"
    assert select_validation(repo, sample_size=500, repeat_size=1) == first


def test_all_interventions_keep_target_identity_out_of_prediction(fixture):
    root, _, _, path = fixture
    repo = NYCDataRepository(root / "NYC")
    query = repo.get_session_query("1", "1_4")
    target = query.target.copy()
    target["POI_id"], target["POI_catname"], target["latitude"], target["longitude"] = D, "Airport", 0., 0.
    hidden_changed = replace(query, target=target)
    base = IAAAgent(repo, RunConfig.p4()).run_query(query)
    joint = EvidenceStore(path)
    stores = {mode: masked_store(joint, mode) for mode in ("both", "images", "reviews")}
    for arm in ARMS:
        outputs = []
        for current_query in (query, hidden_changed):
            agent = AblationAgent(repo, agent_config(arm, path), arm=arm, evidence_store=stores.get(arm.mode),
                                  frozen_intention=base.inferred_intention, text_reflection=base.reflection)
            agent.llm.chat_json = forbidden
            result = agent.run_query(current_query)
            outputs.append((result.ranked_pois, result.candidate_pool_summary, result.reflection))
        assert outputs[0] == outputs[1], arm.name
    prompt_arm = Arm("prompt")
    prompts = []
    for current_query in (query, hidden_changed):
        agent = AblationAgent(repo, agent_config(prompt_arm, path, live=True), arm=prompt_arm, evidence_store=joint)
        prompts.append(prepare_prompt(agent, current_query)[0])
    assert prompts[0] == prompts[1]
    assert "TARGET_ONLY_MARKER" not in str(prompts[0])


def test_retry_cache_records_invalid_fields_and_reuses_valid_intention(fixture, tmp_path):
    root, _, _, _ = fixture
    query = NYCDataRepository(root / "NYC").get_session_query("1", "1_4")
    valid = IAAAgent(NYCDataRepository(root / "NYC")).run_query(query).inferred_intention.model_dump(mode="json")
    replies = iter([{"summary": "missing required fields"}, valid])
    calls = []
    def chat(messages):
        calls.append(messages)
        return next(replies)
    client = SimpleNamespace(chat_json=chat, last_call_status="success", last_usage={"total_tokens": 100},
                             last_raw_content="raw", last_finish_reason="stop", last_error_type=None)
    path = tmp_path / "intent.json"
    intention, record = load_intention(client, [{"role": "user", "content": "same prompt"}], path, "protocol")
    assert len(calls) == 2 and len(record["attempts"]) == 2
    assert record["attempts"][0]["validation_errors"] and not record["attempts"][0]["valid"]
    client.chat_json = forbidden
    cached, _ = load_intention(client, [{"role": "user", "content": "same prompt"}], path, "protocol")
    assert cached == intention
    with pytest.raises(ValueError, match="mismatch"):
        load_intention(client, [{"role": "user", "content": "changed prompt"}], path, "protocol")
    client.chat_json = lambda _: valid
    client.last_usage = None
    assert not validated_intention_attempt(client, [])["valid"]
    client.last_usage, client.last_finish_reason = {"total_tokens": 1}, "length"
    assert not validated_intention_attempt(client, [])["valid"]


def test_cluster_summary_pairs_only_matching_cases_and_reports_incomplete():
    miss = dict(rank=None, in_pool=True, in_raw=True, pool_size=60, reflection={"triggered": True}, predictions=["x"])
    hit = dict(miss, rank=1, predictions=["y"])
    cases = [dict(user_id="1", trajectory_id="1_1", history_group="IH", variants={"text": miss, "full_both": hit}),
             dict(user_id="1", trajectory_id="1_2", history_group="OOH", variants={"text": hit, "full_both": miss})]
    expected = [dict(user_id="1", trajectory_id=f"1_{i}", repeat=False) for i in (1, 2, 3)]
    summary = summarize(cases, expected)
    assert not summary["complete"] and summary["n"] == 2 and summary["expected_n"] == 3
    pair = summary["contrasts"]["full_both_minus_text"]
    assert pair["delta"]["Hit@10"] == 0 and pair["users"] == 1
    assert pair["hit10_gains"] == pair["hit10_losses"] == 1
    assert not pair["complete"]


def test_runner_completes_and_resumes_without_new_requests(fixture, monkeypatch, tmp_path):
    from scripts import run_multimodal_ablations as runner
    root, _, _, snapshot = fixture
    source = tmp_path / "frozen"
    source.mkdir()
    code_root = Path(runner.__file__).resolve().parents[1]
    files = {}
    for path in (code_root / "iaa_agent").glob("*.py"):
        name = "code/iaa_agent/" + path.name
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        files[name] = digest(path.read_bytes())
    name = "evidence/NYC_fixture.json"
    (source / "evidence").mkdir()
    shutil.copyfile(snapshot, source / name)
    files[name] = digest(snapshot.read_bytes())
    write(source / "bundle_manifest.json", {"files": files})
    repo = NYCDataRepository(root / "NYC")
    value = IAAAgent(repo).run_query(repo.get_session_query("1", "1_4")).inferred_intention.model_dump(mode="json")
    calls = []
    class Client:
        last_call_status, last_usage, last_raw_content = "success", {"total_tokens": 100}, "synthetic fixture response"
        last_finish_reason, last_error_type = "stop", None
        def __init__(self, **kwargs):
            pass
        def chat_json(self, messages):
            calls.append(messages)
            return value
    monkeypatch.setattr(runner, "DeepSeekClient", Client)
    output = tmp_path / "output"
    argv = ["--source-experiment", str(source), "--data-root", str(root), "--output-dir", str(output),
            "--cities", "NYC", "--sample-size", "2", "--repeat-size", "1", "--pilot-size", "1", "--concurrency", "1"]
    assert runner.main(argv) == 0
    assert len(calls) == 10  # four primary intentions each, two repeats on one case
    assert read(output / "progress.json")["status"] == "completed"
    summary = read(output / "summaries/NYC.json")
    assert summary["complete"] and summary["arms"]["text"]["n"] == 2
    assert summary["arms"]["text_repeat"]["n"] == 1
    assert summary["request_accounting"]["started_attempts"] == 10
    assert read(output / "gates/NYC_pilot.json")["passed"]
    assert runner.main(argv) == 0
    assert len(calls) == 10
    (source / "code/iaa_agent/engine.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        runner.main(argv)
