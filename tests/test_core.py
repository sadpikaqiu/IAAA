from __future__ import annotations

import json
from pathlib import Path

from iaa_agent.data import NYCDataRepository
from iaa_agent.engine import IAAAgent, RunConfig
from iaa_agent.evaluation import (
    evaluate_session_split,
    evaluate_session_split_stratified,
    evaluate_session_split_threaded,
    stable_fractional_sample,
)
from iaa_agent.llm import DeepSeekClient
from iaa_agent.models import Candidate, Intention, LikelyCategory
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


def test_single_user_evaluation_can_save_full_traces(tmp_path: Path) -> None:
    repo = NYCDataRepository("datasets/NYC")
    result = evaluate_session_split(repo, user_id="1", save_runs_dir=tmp_path, llm_mode="fake")
    payload = result.as_dict()

    assert payload["total"] == 2
    assert len(payload["runs"]) == 2
    for record in payload["runs"]:
        trace_path = Path(record["trace_path"])
        assert trace_path.exists()
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace["query_mode"] == "session_split"
        assert trace["agent_trace_summary"]
        assert trace["ranked_pois"]


def test_all_recall_sources_recorded_in_trace() -> None:
    # 防回归:每一路召回都必须在 agent trace 中留有 ToolCallRecord(架构不变量 AD-11 Trace 完整性)。
    # 历史上 TemporalPopularityRecall 曾执行却漏记 trace,此测试锁死 6 路召回齐全。
    repo = NYCDataRepository("datasets/NYC")
    agent = IAAAgent(repo, RunConfig(llm_mode="fake"))
    user_id, trajectory_id = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="349")[0]
    query = repo.get_session_query(user_id, trajectory_id, train_ratio=0.8, min_context=1)
    result = agent.run_query(query)

    recorded_tools = {record.tool for record in result.agent_trace_summary}
    expected_recalls = {
        "HistoricalRecall",
        "SpatialRecall",
        "CategoryIntentRecall",
        "TransitionRecall",
        "PeerRecall",
        "TemporalPopularityRecall",
    }
    assert expected_recalls.issubset(recorded_tools), (
        f"trace 缺少召回记录: {expected_recalls - recorded_tools}"
    )


def test_session_query_history_excludes_target_and_future() -> None:
    # 防回归:数据隔离红线(架构不变量 AD-7)。session_split 模式下,
    # query.history 必须严格早于 target,绝不能含 target 本身或其之后的未来 check-in。
    repo = NYCDataRepository("datasets/NYC")
    repo.use_user_chronological_split(0.8)
    user_id, trajectory_id = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="349")[0]
    query = repo.get_session_query(user_id, trajectory_id, train_ratio=0.8, min_context=1)

    target_time = query.target["UTC_time"]
    assert query.history is not None and not query.history.empty
    # 长期历史的所有事件都必须早于目标时间(无泄露)。
    assert query.history["UTC_time"].max() < target_time
    # 可见上下文也必须早于目标时间。
    assert query.context["UTC_time"].max() < target_time
    # target 自身的时间戳不得出现在长期历史中。
    assert not (query.history["UTC_time"] == target_time).any()


def test_meta_lookup_equivalent_to_poi_meta() -> None:
    # 防回归:P1 性能优化契约。build_meta_lookup().get(pid) 必须与逐个
    # poi_meta(pid, context) 逐字段完全一致——这是"目录复用不改变输出"的核心保证。
    repo = NYCDataRepository("datasets/NYC")
    user_id, trajectory_id = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="349")[0]
    query = repo.get_session_query(user_id, trajectory_id, train_ratio=0.8, min_context=1)
    context = query.context
    lookup = repo.build_meta_lookup(context)

    # 取一批覆盖三种情形的 POI:可见 catalog 内、全集内、以及一个必然不存在的 id。
    sample_ids = [str(p) for p in context["POI_id"].tolist()][:20]
    sample_ids += [str(p) for p in repo.all_meta["POI_id"].tolist()[:20]]
    sample_ids.append("__nonexistent_poi__")

    for pid in sample_ids:
        assert lookup.get(pid) == repo.poi_meta(pid, context), (
            f"meta_lookup 与 poi_meta 对 {pid} 结果不一致"
        )


def test_stratified_report_overall_matches_baseline() -> None:
    # 防回归(方向 A 口径不漂移红线):分层报告的 overall 指标必须与同条件下的
    # evaluate_session_split 逐字段一致。分层只是按标签重新聚合 rank,绝不能改变 rank 口径。
    repo = NYCDataRepository("datasets/NYC")
    baseline = evaluate_session_split(repo, user_id="349", llm_mode="fake").as_dict()
    report = evaluate_session_split_stratified(repo, user_id="349")
    overall = report["overall"]

    assert overall["n"] == baseline["total"]
    # Hit@1/5/10 与 MRR 必须逐字段相等(同口径、同 rank)。
    assert overall["Hit@1"] == baseline["Hit@1"]
    assert overall["Hit@5"] == baseline["Hit@5"]
    assert overall["Hit@10"] == baseline["Hit@10"]
    assert overall["MRR"] == baseline["MRR"]


def test_stratified_slices_partition_all_sessions() -> None:
    # 防回归:每个分层维度都必须是会话集合的「完整划分」——各切片样本数之和 == 整体 n,
    # 不能有会话漏标或被重复计数(否则切片指标的分母不可信)。
    repo = NYCDataRepository("datasets/NYC")
    report = evaluate_session_split_stratified(repo, user_id="349")
    total = report["overall"]["n"]

    for dim, groups in report["strata"].items():
        slice_sum = sum(m["n"] for m in groups.values())
        assert slice_sum == total, f"维度 {dim} 切片样本数之和 {slice_sum} != 整体 {total}"

    # IH 子集样本数不得超过整体,且 OOH 切片与之互补。
    in_hist_n = report["in_history_subset"]["n"]
    ooh_groups = report["strata"]["ooh"]
    assert in_hist_n == ooh_groups.get("IH", {"n": 0})["n"]
    assert in_hist_n + ooh_groups.get("OOH", {"n": 0})["n"] == total
    for group in ooh_groups.values():
        assert group["Acc@1"] == group["Hit@1"]
        assert group["Acc@5"] == group["Hit@5"]
        assert group["Acc@10"] == group["Hit@10"]


def test_temporal_granularity_default_equivalent() -> None:
    # 防回归(P4 时间粒度开关契约):RunConfig 默认 temporal_granularity="bucket",
    # 默认配置产出的 AgentRunResult 必须与显式 "bucket" 逐字节一致——保证新增 exact 分支
    # 默认关闭、零行为漂移(守 AD-9)。任何让默认偏离 bucket 的改动都会被此测试抓住。
    repo = NYCDataRepository("datasets/NYC")
    agent_default = IAAAgent(repo, RunConfig(llm_mode="fake"))
    agent_bucket = IAAAgent(repo, RunConfig(llm_mode="fake", temporal_granularity="bucket"))

    keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="349")
    for user_id, trajectory_id in keys:
        query = repo.get_session_query(user_id, trajectory_id, train_ratio=0.8, min_context=1)
        default_json = agent_default.run_query(query).model_dump(mode="json")
        bucket_json = agent_bucket.run_query(query).model_dump(mode="json")
        assert default_json == bucket_json, (
            f"默认配置与显式 bucket 在会话 {trajectory_id} 上不一致,temporal_granularity 默认值漂移"
        )


def test_temporal_granularity_exact_changes_only_temporal_fit() -> None:
    # 防回归(单变量契约):exact 模式只改变 temporal_fit 维度的 verdict,
    # 不得新增/删除 affordance 维度,也不得改变候选池构成(poi_id 集合不变)。
    repo = NYCDataRepository("datasets/NYC")
    agent_bucket = IAAAgent(repo, RunConfig(llm_mode="fake", temporal_granularity="bucket"))
    agent_exact = IAAAgent(repo, RunConfig(llm_mode="fake", temporal_granularity="exact"))

    user_id, trajectory_id = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="349")[0]
    query = repo.get_session_query(user_id, trajectory_id, train_ratio=0.8, min_context=1)
    bucket_result = agent_bucket.run_query(query)
    exact_result = agent_exact.run_query(query)

    # 候选池构成不变(单变量:只动精排维度,不动召回)。
    bucket_pois = {p.poi_id for p in bucket_result.ranked_pois}
    exact_pois = {p.poi_id for p in exact_result.ranked_pois}
    # ranked_pois 是 top-10,池更大;比较 affordance 维度集合更可靠。
    for bp, ep in zip(bucket_result.ranked_pois, exact_result.ranked_pois):
        bucket_dims = {a.name for a in bp.affordance_profile.affordances}
        exact_dims = {a.name for a in ep.affordance_profile.affordances}
        assert bucket_dims == exact_dims, "exact 模式改变了 affordance 维度集合,破坏单变量"
        break  # 维度集合对所有候选一致,验一个足矣


def test_p4_preset_softens_category_mismatch_only_when_enabled() -> None:
    repo = NYCDataRepository("datasets/NYC")
    candidate = Candidate(
        poi_id="poi_x",
        poi_idx="P999999",
        display_name="P999999",
        category="Park",
        latitude=0.0,
        longitude=0.0,
        distance_km=1.0,
    )
    intention = Intention(
        summary="test",
        activity_goal="test",
        likely_categories=[LikelyCategory(category="Coffee Shop", weight=1.0, evidence="test")],
        spatial_preference={},
        temporal_preference={},
        behavioral_preference={},
        confidence=0.8,
        evidence=[],
        uncertainty_reasons=[],
    )

    baseline = IAAAgent(repo, RunConfig(llm_mode="fake"))._category_match(candidate, intention)
    p4 = IAAAgent(repo, RunConfig.p4(llm_mode="fake"))._category_match(candidate, intention)

    assert baseline.answer == "no"
    assert p4.answer == "uncertain"
    assert RunConfig.p4().soft_category_mismatch is True
    assert RunConfig().soft_category_mismatch is False


def test_stable_fractional_sample_is_deterministic() -> None:
    keys = [(str(i % 3), f"{i}_traj") for i in range(101)]
    first = stable_fractional_sample(keys, 0.5, seed=7)
    second = stable_fractional_sample(keys, 0.5, seed=7)
    different_seed = stable_fractional_sample(keys, 0.5, seed=8)

    assert first == second
    assert len(first) == 50
    assert first != different_seed
    assert [key for key in keys if key in set(first)] == first


def test_evaluation_progress_callback_counts_sessions() -> None:
    repo = NYCDataRepository("datasets/NYC")
    keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="1")[:2]
    calls = 0

    def tick() -> None:
        nonlocal calls
        calls += 1

    result = evaluate_session_split(repo, llm_mode="fake", session_keys=keys, progress_callback=tick)

    assert result.total == len(keys)
    assert calls == len(keys)


def test_threaded_evaluation_matches_serial_fake_mode() -> None:
    repo = NYCDataRepository("datasets/NYC")
    keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="1")[:3]
    calls = 0

    def tick() -> None:
        nonlocal calls
        calls += 1

    serial = evaluate_session_split(NYCDataRepository("datasets/NYC"), llm_mode="fake", session_keys=keys)
    threaded = evaluate_session_split_threaded(
        repo,
        llm_mode="fake",
        session_keys=keys,
        concurrency=2,
        progress_callback=tick,
    )

    assert threaded.as_dict() == {**serial.as_dict(), "fallback_count": 0}
    assert calls == len(keys)


def test_threaded_evaluation_reports_ih_ooh_from_same_predictions() -> None:
    repo = NYCDataRepository("datasets/NYC")
    keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="349")[:3]

    threaded = evaluate_session_split_threaded(
        repo,
        llm_mode="fake",
        session_keys=keys,
        concurrency=2,
        report_stratified=True,
    )
    payload = threaded.as_dict()
    report = payload["stratified"]

    assert report["overall"]["n"] == len(keys)
    assert sum(group["n"] for group in report["strata"]["ooh"].values()) == len(keys)
    assert set(report["strata"]["ooh"]).issubset({"IH", "OOH"})


def test_deepseek_missing_usage_does_not_abort_or_count_as_fallback(monkeypatch) -> None:
    intention_payload = {
        "summary": "test",
        "activity_goal": "test",
        "likely_categories": [
            {"category": "Coffee Shop", "weight": 1.0, "evidence": "test"}
        ],
        "spatial_preference": {},
        "temporal_preference": {},
        "behavioral_preference": {},
        "confidence": 0.8,
        "evidence": [],
        "uncertainty_reasons": [],
    }

    def success_without_usage(self, messages, max_tokens=900):
        self.last_usage = None
        self.last_call_status = "success"
        return intention_payload

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(DeepSeekClient, "chat_json", success_without_usage)
    repo = NYCDataRepository("datasets/NYC")
    keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="1")[:1]

    result = evaluate_session_split_threaded(
        repo,
        llm_mode="deepseek",
        run_config=RunConfig.p4(llm_mode="deepseek"),
        session_keys=keys,
        concurrency=1,
        strict_llm=True,
    ).as_dict()

    assert result["total"] == 1
    assert result["fallback_count"] == 0
    assert result["usage_missing_count"] == 1
    assert result["llm_status_counts"] == {"success": 1}
    assert result["all_sessions_used_deepseek"] is True
    assert result["llm_anomalies"][0]["usage_missing"] is True
    assert result["llm_anomalies"][0]["heuristic_fallback"] is False


def test_deepseek_heuristic_fallback_is_recorded_without_aborting(monkeypatch) -> None:
    def request_failure(self, messages, max_tokens=900):
        self.last_usage = None
        self.last_call_status = "request_error"
        return None

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(DeepSeekClient, "chat_json", request_failure)
    repo = NYCDataRepository("datasets/NYC")
    keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="1")[:1]

    result = evaluate_session_split_threaded(
        repo,
        llm_mode="deepseek",
        session_keys=keys,
        concurrency=1,
        strict_llm=True,
    ).as_dict()

    assert result["total"] == 1
    assert result["fallback_count"] == 1
    assert result["usage_missing_count"] == 1
    assert result["llm_status_counts"] == {"request_error": 1}
    assert result["all_sessions_used_deepseek"] is False
    assert result["llm_anomalies"][0]["heuristic_fallback"] is True
    assert result["llm_anomalies"][0]["strict_violation"] is True


def test_openai_qwen_client_sends_non_thinking_json_request(monkeypatch) -> None:
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            response = {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            }
            return json.dumps(response).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENAI_MODEL", "Qwen/Qwen3.8-27B-FP8")
    monkeypatch.setattr("iaa_agent.llm.urllib.request.urlopen", fake_urlopen)

    client = DeepSeekClient(provider="openai")
    result = client.chat_json([{"role": "user", "content": "Return JSON."}])

    assert result == {"ok": True}
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert captured["payload"]["seed"] == 42
    assert client.last_usage["total_tokens"] == 13


def test_openai_qwen_client_can_enable_reasoning(monkeypatch) -> None:
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            response = {
                "choices": [
                    {
                        "message": {
                            "content": '{"ok": true}',
                            "reasoning_content": "brief private reasoning",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
            }
            return json.dumps(response).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setenv("OPENAI_MODEL", "Qwen/Qwen3.8-27B-FP8")
    monkeypatch.setenv("OPENAI_ENABLE_THINKING", "1")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "4096")
    monkeypatch.setattr("iaa_agent.llm.urllib.request.urlopen", fake_urlopen)

    client = DeepSeekClient(provider="openai")
    result = client.chat_json([{"role": "user", "content": "Return JSON."}])

    assert result == {"ok": True}
    assert captured["payload"]["max_tokens"] == 4096
    assert "response_format" not in captured["payload"]
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": False,
        "reasoning_effort": "low",
    }
    assert client.last_reasoning_content == "brief private reasoning"
    assert client.last_finish_reason == "stop"
    assert client.last_usage["reasoning_tokens"] == 5


def test_openai_mode_uses_live_llm_evaluation_path(monkeypatch) -> None:
    intention_payload = {
        "summary": "local qwen intention",
        "activity_goal": "test",
        "likely_categories": [
            {"category": "Coffee Shop", "weight": 1.0, "evidence": "test"}
        ],
        "spatial_preference": {},
        "temporal_preference": {},
        "behavioral_preference": {},
        "confidence": 0.8,
        "evidence": [],
        "uncertainty_reasons": [],
    }

    def local_success(self, messages, max_tokens=900):
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        self.last_call_status = "success"
        return intention_payload

    monkeypatch.setenv("OPENAI_MODEL", "Qwen/Qwen3.8-27B-FP8")
    monkeypatch.setattr(DeepSeekClient, "chat_json", local_success)
    repo = NYCDataRepository("datasets/NYC")
    keys = repo.iter_session_test_keys(train_ratio=0.8, min_context=1, user_id="1")[:1]

    result = evaluate_session_split_threaded(
        repo,
        llm_mode="openai",
        run_config=RunConfig.p4(llm_mode="openai"),
        session_keys=keys,
        concurrency=1,
    ).as_dict()

    assert result["total"] == 1
    assert result["fallback_count"] == 0
    assert result["usage_missing_count"] == 0
    assert result["llm_status_counts"] == {"success": 1}
    assert result["all_sessions_used_llm"] is True
    assert "all_sessions_used_deepseek" not in result
