from dataclasses import replace
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from test_poi_evidence import fixture, A, B, C, D
from iaa_agent.agent_types import AgentConfig, VisibleQuery, ToolRequest
from iaa_agent.agent_tools import POIToolService
from iaa_agent.agent_runtime import JournaledModel, PromptBudget, read_json, strict_response_json
from iaa_agent.autonomous import (AutonomousAgent, rerank_fixed_result, compact_observations,
                                 decode_working_selection, decision_selection_schema)
from iaa_agent.data import NYCDataRepository
from iaa_agent.engine import IAAAgent, RunConfig
from iaa_agent.evidence import EvidenceStore


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(str(text))
    def decode(self, tokens, **kwargs):
        return "".join(tokens)
    def apply_chat_template(self, messages, **kwargs):
        return list("".join(m["content"] for m in messages))


def selection_response(decision):
    value = deepcopy(decision)
    value["working_poi_selection"] = dict.fromkeys(value.pop("working_poi_ids"), True)
    return value


class ScriptedClient:
    model = "Qwen/test"
    base_url = "local-test"
    def __init__(self, source="historical", bad_rank=False):
        self.calls = []
        self.source, self.bad_rank = source, bad_rank
        self.last_error_type = None
        self.seen = []
    def chat_json(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        payload = json.loads(messages[1]["content"])
        if "candidate_facts" in payload:
            rows = payload["candidate_facts"]["rows"]
            result = {"ranked_pois": [{"poi_idx": r[0], "reason": "Supported by recorded history.",
                "affordances": {k: "uncertain" for k in ("category", "spatial", "temporal", "revisit", "transition")},
                "evidence_refs": ["invented" if self.bad_rank else r[-1]],
                "missing_evidence": [], "conflicts": []} for r in rows[:payload["top_k"]]]}
        else:
            for obs in payload["latest_observations"]:
                for row in obs.get("result", {}).get("candidate_rows", []):
                    if row[0] not in self.seen:
                        self.seen.append(row[0])
            stop = len(self.seen) >= payload["budget"]["top_k"]
            tools = [] if stop else [{"name": "recall_pois", "args": {"source": self.source, "limit": 20}}]
            result = {"intention": {"goal": "coffee", "categories": ["Coffee Shop"], "uncertainty": []},
                      "working_poi_ids": self.seen, "tools": tools, "stop": stop, "reason": "Inspect or compare known candidates."}
            if "required_tool_order" in payload:
                args = []
                for name, source in payload["required_tool_order"]:
                    params = {}
                    if name == "recall_pois":
                        params = {"categories": ["Coffee Shop"]} if source == "category" else {}
                    elif name == "search_evidence":
                        params = {"query": "coffee"}
                    else:
                        params = {"poi_ids": self.seen[:1]}
                    args.append(params)
                result = {"intention": result["intention"], "working_poi_ids": self.seen,
                          "tool_args": args, "reason": result["reason"]}
        if "working_poi_ids" in result:
            result = selection_response(result)
        self.last_raw_content = json.dumps(result)
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        self.last_call_status, self.last_finish_reason = "success", "stop"
        return result


def service(fixture, evidence=True):
    root, _, _, path = fixture
    repo = NYCDataRepository(root / "NYC")
    repo.use_user_chronological_split(.8)
    query = repo.get_session_query("1", "1_4")
    return repo, query, POIToolService(repo, VisibleQuery.from_query(query), EvidenceStore(path) if evidence else None)


def model(tmp_path, config, client=None):
    return JournaledModel(tmp_path / "calls", "test-identity", PromptBudget(CharacterTokenizer()), config,
                           client=client or ScriptedClient())


def test_hidden_target_does_not_change_tools_or_prompt(fixture):
    repo, query, tools = service(fixture)
    changed = query.target.copy()
    changed["POI_id"], changed["POI_catname"], changed["latitude"] = D, "Airport", 0
    visible = VisibleQuery.from_query(replace(query, target=changed))
    other = POIToolService(repo, visible, tools.evidence)
    assert "POI_id" not in visible.legacy_context_query().target
    assert tools.initial_context() == other.initial_context()
    req = ToolRequest(name="recall_pois", args={"source": "historical"})
    assert tools.execute(req) == other.execute(req)
    assert "TARGET_ONLY_MARKER" not in json.dumps(tools.initial_context())


def test_registry_can_recover_candidates_and_caches_calls(fixture):
    _, _, tools = service(fixture)
    req = ToolRequest(name="recall_pois", args={"source": "spatial"})
    first = tools.execute(req)
    assert len(tools.registry) == 4
    assert tools.execute(req)["cached"]
    assert tools.execute(req)["new_candidate_count"] == 0
    assert tools.execute(ToolRequest(name="list_candidates", args={}))["candidates"] == first["candidates"]
    with pytest.raises(ValueError, match="not been observed"):
        tools.execute(ToolRequest(name="inspect_candidates", args={"poi_ids": ["P999999"]}))


def test_read_evidence_provenance_missing_and_trajectory_mode(fixture):
    _, _, tools = service(fixture)
    tools.execute(ToolRequest(name="recall_pois", args={"source": "spatial"}))
    a = tools.by_original[A]
    result = tools.execute(ToolRequest(name="read_evidence", args={"poi_ids": [a]}))
    assert {x["modality"] for x in result["pois"][0]["evidence"]} == {"image", "review"}
    for item in result["pois"][0]["evidence"]:
        assert tools.references[item["ref"]]["poi_idx"] == a
        assert tools.references[item["ref"]]["snapshot_id"] == tools.evidence.snapshot_id
    d = tools.by_original[D]
    assert tools.read_items(d)["missing"] == ["image", "review"]
    _, _, text = service(fixture, False)
    with pytest.raises(ValueError, match="unavailable"):
        text.execute(ToolRequest(name="search_evidence", args={"query": "coffee"}))


def test_different_model_actions_drive_real_tools_and_accumulate_usage(fixture, tmp_path):
    results = []
    for source in ("historical", "spatial"):
        _, _, tools = service(fixture)
        config = AgentConfig(top_k=2)
        client = ScriptedClient(source)
        result = AutonomousAgent(tools, model(tmp_path / source, config, client), config).run()
        assert result["accounting"]["logical_calls"] == 3
        assert result["accounting"]["usage"]["total_tokens"] == 450
        assert result["accounting"]["usage_missing_count"] == 0
        assert "ground_truth_poi_id" not in result
        assert client.calls[0][1]["max_tokens"] == 2048
        assert client.calls[-1][1]["max_tokens"] == 4096
        results.append(result)
    assert results[0]["trace"][0]["tools"][0]["request"]["args"]["source"] != results[1]["trace"][0]["tools"][0]["request"]["args"]["source"]
    assert len(results[0]["candidate_pool_summary"]["raw_retrieved_poi_ids"]) < len(results[1]["candidate_pool_summary"]["raw_retrieved_poi_ids"])


def test_resume_replays_without_network_and_rejects_identity_drift(fixture, tmp_path):
    config = AgentConfig(top_k=2)
    _, _, tools = service(fixture)
    first = AutonomousAgent(tools, model(tmp_path, config), config).run()
    _, _, tools = service(fixture)
    client = ScriptedClient()
    client.chat_json = lambda *a, **k: pytest.fail("Resume repeated a model request")
    second = AutonomousAgent(tools, model(tmp_path, config, client), config).run()
    assert first["ranked_pois"] == second["ranked_pois"]
    assert first["trace"] == second["trace"]
    journal = model(tmp_path, config, client)
    with pytest.raises(ValueError, match="identity mismatch"):
        journal.call("decision_01", [{"role": "user", "content": "changed"}], 2048, lambda x: x)


def test_invalid_references_exhaust_bounded_repair_without_fallback(fixture, tmp_path):
    _, _, tools = service(fixture)
    config = AgentConfig(top_k=2)
    journal = model(tmp_path, config, ScriptedClient(bad_rank=True))
    with pytest.raises(RuntimeError, match="Repair budget"):
        AutonomousAgent(tools, journal, config).run()
    assert journal.accounting()["retries"] == 2
    assert journal.accounting()["invalid_attempts"] == 3


def test_final_consolidation_and_fixed_order_are_enforced(fixture, tmp_path):
    _, _, tools = service(fixture)
    tools.execute(ToolRequest(name="recall_pois", args={"source": "spatial"}))
    config = AgentConfig(top_k=2)
    agent = AutonomousAgent(tools, model(tmp_path, config), config)
    raw = {"intention": {"goal": "coffee"}, "working_poi_ids": list(tools.registry),
           "tools": [{"name": "recall_pois", "args": {"source": "historical"}}], "stop": False, "reason": "More evidence"}
    with pytest.raises(ValueError, match="consolidation"):
        agent._validate_decision(raw, 5, None)
    with pytest.raises(ValueError, match="Fixed schedule"):
        agent._validate_decision(raw, 0, [("recall_pois", "spatial")])


def test_reranker_preserves_original_pool_and_intention(fixture, tmp_path):
    repo, query, tools = service(fixture, False)
    legacy = IAAAgent(repo, RunConfig.p4()).run_query(query).model_dump(mode="json")
    config = AgentConfig(engine="fixed_llm_rank", top_k=2)
    result = rerank_fixed_result(tools, model(tmp_path, config), legacy, config)
    assert result["candidate_pool_summary"]["candidate_poi_ids"] == legacy["candidate_pool_summary"]["candidate_poi_ids"]
    assert result["fixed_source_intention"] == legacy["inferred_intention"]


def test_long_multilingual_excerpts_are_packed_without_losing_ids():
    budget = PromptBudget(CharacterTokenizer(), limit=800)
    data = {"candidates": [{"poi_idx": f"P{i:06d}", "text": "静かなカフェ，图片与评论。" * 500} for i in range(5)]}
    messages = budget.messages("instructions", data)
    assert budget.count(messages) <= 800
    loaded = json.loads(messages[1]["content"])
    assert [x["poi_idx"] for x in loaded["candidates"]] == [x["poi_idx"] for x in data["candidates"]]
    assert all(x["excerpt_truncated"] for x in loaded["candidates"])


def test_inspection_details_survive_prompt_compaction(fixture):
    _, _, tools = service(fixture)
    tools.execute(ToolRequest(name="recall_pois", args={"source": "historical"}))
    ids = list(tools.registry)[:2]
    request = ToolRequest(name="inspect_candidates", args={"poi_ids": ids})
    result = tools.execute(request)
    packed = compact_observations([{"request": request.model_dump(), "result": result}])[0]["result"]
    restored = [dict(zip(packed["candidate_columns"], row)) for row in packed["candidate_rows"]]
    assert restored == result["candidates"]
    assert "same_hour_bucket_visits" in restored[0]
    assert "global_transition_count" in restored[0]
    assert "has_image" in restored[0]


def test_input_cap_is_checked_before_any_structured_request(tmp_path):
    config = AgentConfig()
    client = ScriptedClient()
    journal = JournaledModel(tmp_path / "calls", "test", PromptBudget(CharacterTokenizer(), limit=800), config, client=client)
    with pytest.raises(ValueError, match="context_budget_exceeded"):
        journal.call("too_long", [{"role": "user", "content": "x" * 801}], 2048, lambda x: x, schema={"type": "object"})
    assert client.calls == []


def test_evidence_views_reuse_index_and_keep_modality_boundary(fixture):
    _, _, tools = service(fixture)
    assert tools.evidence_view("both") is tools.evidence
    images = tools.evidence_view("images")
    assert tools.evidence_view("images") is images
    assert images.modalities == {"image"}
    assert all(tools.evidence.items[i][1]["modality"] == "image" for ids in images.by_poi.values() for i in ids)
    assert tools.evidence.modalities == {"image", "review"}


@pytest.mark.parametrize("failure_stage", ["decision", "ranking"])
def test_duplicate_repair_receives_previous_answer_and_specific_ids(fixture, tmp_path, failure_stage):
    class DuplicateOnce(ScriptedClient):
        bad_response = None
        def chat_json(self, messages, **kwargs):
            result = super().chat_json(messages, **kwargs)
            rank = "ranked_pois" in result
            ready = rank if failure_stage == "ranking" else bool(result.get("working_poi_selection"))
            if self.bad_response is None and ready:
                result = deepcopy(result)
                if rank:
                    result["ranked_pois"][1] = deepcopy(result["ranked_pois"][0])
                self.bad_response = deepcopy(result)
                self.last_raw_content = json.dumps(result)
                if not rank:
                    key = next(iter(result["working_poi_selection"]))
                    pair = json.dumps(key) + ": true"
                    self.last_raw_content = self.last_raw_content.replace(pair, pair + ", " + pair, 1)
            return result
    _, _, tools = service(fixture)
    config = AgentConfig(top_k=2)
    client = DuplicateOnce()
    result = AutonomousAgent(tools, model(tmp_path, config, client), config).run()
    repairs = [messages for messages, _ in client.calls if len(messages) > 2]
    assert len(repairs) == 1
    assert json.loads(repairs[0][-2]["content"]) == client.bad_response
    if failure_stage == "ranking":
        assert "1-based positions" in repairs[0][-1]["content"]
        assert "[1, 2]" in repairs[0][-1]["content"]
    else:
        assert "Duplicate JSON object key" in repairs[0][-1]["content"]
        assert next(iter(client.bad_response["working_poi_selection"])) in repairs[0][-1]["content"]
    assert result["accounting"]["retries"] == 1
    assert result["accounting"]["invalid_attempts"] == 1
    assert result["accounting"]["usage"]["total_tokens"] == 600
    assert len({p["poi_idx"] for p in result["ranked_pois"]}) == 2


def test_intermediate_working_set_cannot_drop_below_ranking_size(fixture, tmp_path):
    _, _, tools = service(fixture)
    tools.execute(ToolRequest(name="recall_pois", args={"source": "spatial"}))
    config = AgentConfig(top_k=2)
    agent = AutonomousAgent(tools, model(tmp_path, config), config)
    raw = {"intention": {"goal": "coffee"}, "working_poi_ids": list(tools.registry)[:1],
           "tools": [{"name": "recall_pois", "args": {"source": "historical"}}],
           "stop": False, "reason": "Continue investigation"}
    with pytest.raises(ValueError, match="must retain at least 2"):
        agent._validate_decision(raw, 1, None)
    client = ScriptedClient()
    _, _, tools = service(fixture)
    AutonomousAgent(tools, model(tmp_path / "run", config, client), config).run()
    schema = client.calls[1][1]["request_options"]["response_format"]["json_schema"]["schema"]
    assert all(branch["properties"]["working_poi_selection"]["minProperties"] == 2 for branch in schema["anyOf"])


def test_emitted_schema_couples_stop_with_tool_count(fixture, tmp_path):
    _, _, tools = service(fixture)
    config = AgentConfig(top_k=2)
    client = ScriptedClient()
    result = AutonomousAgent(tools, model(tmp_path, config, client), config).run()
    first_schema = client.calls[0][1]["request_options"]["response_format"]["json_schema"]["schema"]
    schema = client.calls[1][1]["request_options"]["response_format"]["json_schema"]["schema"]
    for value in (first_schema, schema):
        Draft202012Validator.check_schema(value)
    validator = Draft202012Validator(schema)
    valid = selection_response(result["trace"][1]["decision"])
    request = {"name": "recall_pois", "args": {"source": "historical"}}
    for stop, count, accepted in [(False, 0, False), (False, 1, True), (False, 3, True),
                                  (False, 4, False), (True, 0, True), (True, 1, False)]:
        raw = {**valid, "stop": stop, "tools": [request] * count}
        assert validator.is_valid(raw) is accepted, (stop, count)
    for missing in ("tools", "working_poi_selection", "stop"):
        incomplete = {k: v for k, v in valid.items() if k != missing}
        assert not validator.is_valid(incomplete), missing
    # Before enough candidates are known, stopping and a no-op are both impossible.
    initial = selection_response(result["trace"][0]["decision"])
    first = Draft202012Validator(first_schema)
    assert first.is_valid(initial)
    assert not first.is_valid({**initial, "stop": True, "tools": []})
    assert not first.is_valid({**initial, "stop": False, "tools": []})


def test_final_schema_requires_stop_without_tools(fixture, tmp_path):
    _, _, tools = service(fixture)
    config = AgentConfig(top_k=2, max_decisions=2)
    client = ScriptedClient()
    result = AutonomousAgent(tools, model(tmp_path, config, client), config).run()
    schema = client.calls[1][1]["request_options"]["response_format"]["json_schema"]["schema"]
    validator = Draft202012Validator(schema)
    valid = selection_response(result["trace"][1]["decision"])
    assert validator.is_valid(valid)
    assert not validator.is_valid({**valid, "stop": False})
    assert not validator.is_valid({**valid, "tools": [{"name": "list_candidates", "args": {}}]})


@pytest.mark.parametrize("engine", ["autonomous", "fixed_schedule"])
def test_model_selection_is_explicit_bounded_and_registered(fixture, tmp_path, engine):
    _, _, tools = service(fixture)
    config = AgentConfig(top_k=2, max_candidates=4, engine=engine)
    client = ScriptedClient()
    # Use a schema captured during an ordinary run, with the real registered pool.
    result = AutonomousAgent(tools, model(tmp_path, config, client), config).run()
    schema = client.calls[1][1]["request_options"]["response_format"]["json_schema"]["schema"]
    branches = schema.get("anyOf", [schema])
    selection_schema = branches[0]["properties"]["working_poi_selection"]
    check = Draft202012Validator(selection_schema)
    ids = list(selection_schema["properties"])
    chosen = {p: True for p in ids[:2]}
    assert check.is_valid(chosen)
    assert not check.is_valid({})
    assert not check.is_valid({"P999999": True, **chosen})
    assert not check.is_valid({ids[0]: False, ids[1]: True})
    assert not check.is_valid({ids[0]: 1, ids[1]: True})
    assert not check.is_valid(ids[:2])
    decoded = decode_working_selection({"working_poi_selection": chosen})
    assert decoded["working_poi_ids"] == list(chosen)
    with pytest.raises(ValueError, match="map selected"):
        decode_working_selection({"working_poi_selection": {ids[0]: 1}})
    with pytest.raises(ValueError, match="not working_poi_ids"):
        decode_working_selection({"working_poi_ids": ids, "working_poi_selection": chosen})
    assert len(result["ranked_pois"]) == 2


def test_selection_cardinality_cap_and_initial_empty_schema():
    from iaa_agent.agent_types import AgentDecision
    original = AgentDecision.model_json_schema()
    ids = [f"P{i:06d}" for i in range(80)]
    schema = decision_selection_schema(original, ids, minimum=10, maximum=60)
    check = Draft202012Validator(schema["properties"]["working_poi_selection"])
    assert check.is_valid(dict.fromkeys(ids[:10], True))
    assert check.is_valid(dict.fromkeys(ids[:60], True))
    assert not check.is_valid(dict.fromkeys(ids[:9], True))
    assert not check.is_valid(dict.fromkeys(ids[:61], True))
    empty = decision_selection_schema(original, [], minimum=0, maximum=60)
    first = Draft202012Validator(empty["properties"]["working_poi_selection"])
    assert first.is_valid({})
    assert not first.is_valid({ids[0]: True})
    assert "working_poi_ids" in original["properties"]  # Internal contract is unchanged.


def test_duplicate_raw_selection_is_rejected_even_after_standard_json_collapses_it():
    raw = '{"working_poi_selection":{"P000221":true,"P000221":true}}'
    assert json.loads(raw)["working_poi_selection"] == {"P000221": True}
    with pytest.raises(ValueError, match="Duplicate JSON object key: P000221"):
        strict_response_json(raw)
    with pytest.raises(ValueError, match="Duplicate JSON object key: stop"):
        strict_response_json('{"stop":false,"stop":true}')
    assert strict_response_json('```json\n{"working_poi_selection":{"P000221":true}}\n```') == json.loads(raw)


def test_resume_rejects_duplicate_raw_keys_without_new_request(fixture, tmp_path):
    config = AgentConfig(top_k=2)
    _, _, tools = service(fixture)
    AutonomousAgent(tools, model(tmp_path, config), config).run()
    path = tmp_path / "calls/decision_02.json"
    record = read_json(path)
    attempt = record["attempts"][0]
    key = next(iter(attempt["parsed"]["working_poi_selection"]))
    pair = json.dumps(key) + ": true"
    attempt["raw_content"] = attempt["raw_content"].replace(pair, pair + ", " + pair, 1)
    path.write_text(json.dumps(record), encoding="utf-8")
    client = ScriptedClient()
    client.chat_json = lambda *a, **k: pytest.fail("Corrupt accepted journal must not trigger a request")
    _, _, tools = service(fixture)
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        AutonomousAgent(tools, model(tmp_path, config, client), config).run()


def test_noop_repair_can_choose_early_stop_without_extra_tool(fixture, tmp_path):
    class NoopOnce(ScriptedClient):
        sent_noop = False
        def chat_json(self, messages, **kwargs):
            result = super().chat_json(messages, **kwargs)
            if result.get("stop") and not self.sent_noop:
                self.sent_noop = True
                result["stop"] = False
                result["reason"] = "No further investigation is needed."
                self.last_raw_content = json.dumps(result)
            return result
    _, _, tools = service(fixture)
    config = AgentConfig(top_k=2)
    client = NoopOnce()
    result = AutonomousAgent(tools, model(tmp_path, config, client), config).run()
    repairs = [messages for messages, _ in client.calls if len(messages) > 2]
    assert len(repairs) == 1
    assert "early stopping is allowed" in repairs[0][-1]["content"]
    assert result["trace"][-1]["decision"]["stop"]
    assert result["tool_calls"] == 1
    assert result["accounting"]["retries"] == 1
    assert len({p["poi_idx"] for p in result["ranked_pois"]}) == 2


def test_large_repair_preserves_original_facts_and_stays_in_input_budget(tmp_path):
    journal = model(tmp_path, AgentConfig())
    messages = [{"role": "system", "content": "instruction"},
                {"role": "user", "content": json.dumps({"mandatory_facts": "x" * 10000, "poi_idx": "P000001"})}]
    previous = {"parsed": {"working_poi_ids": ["P000001", "P000001"], "reason": "x" * 10000},
                "error": "working_poi_ids: duplicate P000001 at [1, 2]"}
    repaired, mode = journal.repair_messages(messages, [previous], 2048, structured=True)
    assert mode == "identifier_summary"
    assert repaired[:2] == messages
    assert "P000001" in repaired[-1]["content"]
    assert journal.budget.count(repaired) <= journal.budget.limit
    next_repaired, _ = journal.repair_messages(messages, [previous, previous], 2048, structured=True)
    assert next_repaired != repaired


def test_future_history_rejected(fixture):
    repo, query, _ = service(fixture)
    with pytest.raises(ValueError, match="strictly before"):
        VisibleQuery.from_query(replace(query, history=repo.all_events))


def test_context_timestamp_tie_keeps_predefined_order_and_is_flagged(fixture):
    _, query, _ = service(fixture)
    context = query.context.copy()
    context.loc[context.index[-1], "UTC_time"] = query.target["UTC_time"]
    visible = VisibleQuery.from_query(replace(query, context=context))
    assert visible.equal_time_context_count == 1
    assert visible.context["POI_id"].tolist() == query.context["POI_id"].tolist()


def test_fixed_schedule_is_dispatched_by_program_and_has_equal_caps(fixture, tmp_path):
    _, _, tools = service(fixture)
    config = AgentConfig(engine="fixed_schedule", top_k=2)
    client = ScriptedClient()
    result = AutonomousAgent(tools, model(tmp_path, config, client), config).run()
    assert len(result["trace"]) == 6
    assert result["tool_calls"] == 11
    first = result["trace"][0]["tools"]
    assert [t["request"]["args"]["source"] for t in first] == ["historical", "spatial"]
    assert all(t["request"]["args"]["radius_km"] == 20 for t in result["trace"][4]["tools"])
    assert client.calls[0][1]["request_options"]["response_format"]["type"] == "json_schema"
    assert result["accounting"]["requests"] == 7


def test_explicit_generation_options_override_env_without_mutating_it(monkeypatch):
    import os
    import urllib.request
    from iaa_agent.llm import DeepSeekClient
    captured = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}).encode()
    def request(req, timeout):
        captured.append((json.loads(req.data), timeout))
        return Response()
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "4096")
    monkeypatch.setenv("OPENAI_ENABLE_THINKING", "0")
    monkeypatch.setattr(urllib.request, "urlopen", request)
    client = DeepSeekClient(provider="openai")
    client.chat_json([{"role": "user", "content": "test"}], request_options={"max_tokens": 2048}, timeout_seconds=17)
    client.chat_json([{"role": "user", "content": "test"}])
    assert captured[0][0]["max_tokens"] == 2048 and captured[0][1] == 17
    assert captured[1][0]["max_tokens"] == 4096
    assert os.environ["OPENAI_MAX_TOKENS"] == "4096"
