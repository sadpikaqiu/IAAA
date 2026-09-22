from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from iaa_agent.agent_runtime import atomic_json, read_json, digest
from iaa_agent.agent_types import AgentConfig
from scripts.autonomous_failure_policy import (attempt_kind, classify_failure, failure_row,
    ServiceCircuit, records_and_accounting)
from scripts.autonomous_eval_support import quality_errors, summarize
from scripts.evaluate_autonomous import Experiment, rolling_results


def good(rank=1):
    return {"rank": rank, "predictions": [str(i) for i in range(10)], "valid": True,
            "heuristic_fallback": False, "in_pool": True, "in_raw": True, "in_observed": True,
            "pool_size": 30, "raw_size": 50, "tool_calls": 0, "elapsed_seconds": 2,
            "accounting": {"requests": 1, "retries": 0, "usage_missing_count": 0,
                           "usage": {"total_tokens": 100}, "model_seconds": 2}}


def failed(tmp_path):
    return failure_row(TimeoutError("Agent case deadline exceeded"), tmp_path)


@pytest.mark.parametrize("code,kind", [(408, "infrastructure"), (429, "infrastructure"),
    (500, "infrastructure"), (503, "infrastructure"), (400, "fatal"), (401, "fatal"), (None, "fatal")])
def test_http_failure_classification_preserves_config_errors(code, kind):
    assert attempt_kind({"status": "request_error", "http_status": code, "error_type": "HTTPError"}) == kind


def test_unknown_exceptions_and_cache_integrity_are_never_soft_failures(tmp_path):
    for exc in (KeyError("missing"), ValueError("Prediction identity mismatch"),
                ValueError("Validation target index changed"), RuntimeError("Unexpected program error")):
        assert failure_row(exc, tmp_path)["failure_kind"] == "fatal"
    record = {"attempts": [{"status": "success", "error": "unknown failure", "exception_type": "KeyError"}]}
    assert classify_failure(RuntimeError("Repair budget exhausted at final_ranking"), [record]) == "fatal"


def test_exhaustion_uses_actual_attempt_causes_and_preserves_missing_usage(tmp_path):
    attempts = [{"status": "request_error", "error_type": "URLError", "usage": None, "accepted": False,
                 "started_at": str(i), "elapsed_seconds": 3} for i in range(3)]
    atomic_json(tmp_path / "calls" / "final_ranking.json", {"attempts": attempts})
    row = failure_row(RuntimeError("Repair budget exhausted at final_ranking"), tmp_path)
    assert row["failure_kind"] == "infrastructure"
    assert row["accounting"]["requests"] == 3
    assert row["accounting"]["retries"] == 2
    assert row["accounting"]["usage_missing_count"] == 3
    assert row["accounting"]["usage"] == {}
    assert row["predictions"] == [] and not row["valid"]


def test_circuit_latches_on_consecutive_or_window_errors():
    bad = {"status": "request_error", "error_type": "URLError"}
    okay = {"accepted": True}
    c = ServiceCircuit()
    for _ in range(4):
        assert c.observe(bad) is None
    assert c.observe(bad)
    assert c.observe(okay) and c.opened
    c = ServiceCircuit()
    for _ in range(7):
        assert c.observe(bad) is None
        assert c.observe(okay) is None
    assert c.observe(bad)
    c = ServiceCircuit()
    for _ in range(100):
        c.observe({"status": "length_truncated"})
    assert not c.opened


def test_failure_scores_use_all_sessions_and_conditional_pairs_are_separate(tmp_path):
    a, b = "autonomous__text", "autonomous__both"
    cases = [{"user_id": str(i), "trajectory_id": str(i), "history_group": "IH",
              "status": "completed", "variants": {a: good(), b: good()}} for i in range(2)]
    cases[1]["status"] = "completed_with_failures"
    cases[1]["variants"][b] = failed(tmp_path)
    assert quality_errors(cases, cases, [a, b])
    assert quality_errors(cases, cases, [a, b], allow_failures=True) == []
    s = summarize(cases, cases, [a, b], allow_failures=True)
    arm = s["arms"][b]
    assert s["quality"]["valid"]
    assert arm["overall"]["Hit@10"] == .5
    assert arm["failure_rate"] == .5 and arm["n"] == 2
    assert arm["overall"]["CandidateRecall"] == 1 and arm["candidate_metric_n"] == 1
    assert arm["metric_denominators"]["Hit@10"] == 2
    name = b + "_minus_" + a
    assert s["contrasts"][name]["delta"]["Hit@10"] == -.5
    assert s["conditional_contrasts"][name]["success_pair_n"] == 1
    assert s["conditional_contrasts"][name]["excluded_n"] == 1
    assert s["conditional_contrasts"][name]["statistics"]["delta"]["Hit@10"] == 0
    # A missing session is incomplete work, never a zero-scored model failure.
    partial = summarize(cases[:1], cases, [a, b], allow_failures=True)
    assert not partial["quality"]["valid"] and not partial["contrasts"]


def test_failed_rows_cannot_smuggle_predictions_or_unknown_errors(tmp_path):
    row = failed(tmp_path)
    case = {"user_id": "u", "trajectory_id": "t", "status": "completed_with_failures", "variants": {"a": row}}
    for key, value in (("predictions", ["x"]), ("valid", True), ("failure_kind", "fatal"), ("heuristic_fallback", True)):
        bad = deepcopy(case)
        bad["variants"]["a"][key] = value
        assert "invalid_failure_record" in quality_errors([bad], [bad], ["a"], allow_failures=True)


def experiment(tmp_path, monkeypatch, fail_arm):
    import scripts.evaluate_autonomous as runner
    args = SimpleNamespace(output_dir=tmp_path, concurrency=2)
    exp = Experiment(args, {"protocol_sha256": "test", "protocol": {}}, None, AgentConfig())
    exp.stores["NYC"] = None
    target = pd.Series({"POI_id": "target", "UTC_time": pd.Timestamp("2026-01-01")})
    query = SimpleNamespace(target_index=3, target=target, context=pd.DataFrame({"UTC_time": [pd.Timestamp("2025-01-01")]}),
                            history=pd.DataFrame({"POI_id": ["previous"]}))
    exp.repository = lambda *_: SimpleNamespace(get_session_query=lambda *a, **kw: query)
    calls = []
    def run(engine, mode, *args, **kwargs):
        calls.append(engine + "__" + mode)
        if engine == fail_arm:
            raise TimeoutError("Agent case deadline exceeded")
        return {"engine": engine, "accounting": good()["accounting"]}
    monkeypatch.setattr(runner, "run_variant", run)
    monkeypatch.setattr(runner, "prediction_row", lambda result, query: good())
    return exp, calls


def test_failed_A_only_blocks_dependent_B_other_arms_run_and_resume_is_terminal(tmp_path, monkeypatch):
    exp, calls = experiment(tmp_path, monkeypatch, "fixed")
    arms = [e + "__text" for e in ("fixed", "fixed_llm_rank", "fixed_schedule", "autonomous")]
    case = {"user_id": "u", "trajectory_id": "t", "target_index": 3, "repeat": False}
    row = exp.run_case("validation", "NYC", case, arms, .7)
    assert row["status"] == "completed_with_failures"
    assert calls == ["fixed__text", "fixed_schedule__text", "autonomous__text"]
    assert row["variants"]["fixed_llm_rank__text"]["failure_kind"] == "dependency"
    assert quality_errors([row], [case], arms, allow_failures=True) == []
    assert exp.run_case("validation", "NYC", case, arms, .7) == row
    assert len(calls) == 3  # No new attempts, no reset of failure budget.


def test_fatal_program_error_stops_case_and_cannot_be_resumed_as_success(tmp_path, monkeypatch):
    import scripts.evaluate_autonomous as runner
    exp, _ = experiment(tmp_path, monkeypatch, None)
    monkeypatch.setattr(runner, "run_variant", lambda *a, **kw: (_ for _ in ()).throw(KeyError("bug")))
    case = {"user_id": "u", "trajectory_id": "t", "target_index": 3}
    row = exp.run_case("validation", "NYC", case, ["autonomous__text"], .7)
    assert row["fatal_error"] and exp.fatal.is_set()
    with pytest.raises(ValueError, match="fatal"):
        exp.run_case("validation", "NYC", case, ["autonomous__text"], .7)


def test_tolerant_queue_does_not_stop_at_four_soft_failures():
    rows = list(rolling_results(lambda i: {"status": "completed_with_failures", "id": i},
                                range(20), concurrency=3, failure_limit=None))
    assert len(rows) == 20


def test_report_handles_all_failed_candidate_metrics(tmp_path):
    from scripts.report_autonomous import render_report
    case = {"user_id": "u", "trajectory_id": "t", "history_group": "OOH",
            "status": "completed_with_failures", "variants": {"autonomous__both": failed(tmp_path)}}
    summary = summarize([case], [case], ["autonomous__both"], allow_failures=True)
    assert summary["arms"]["autonomous__both"]["overall"]["CandidateRecall"] is None
    atomic_json(tmp_path / "summaries" / "NYC_validation.json", summary)
    text = render_report(tmp_path).read_text(encoding="utf-8")
    assert "未观测" in text and "100.00%" in text


def test_compatibility_allows_only_exact_failure_observability_patch(tmp_path):
    from scripts.inherit_autonomous_results import audit_compatibility, OBSERVABILITY_PATCH
    root = Path(__file__).resolve().parents[1]
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_hashes, new_hashes = {}, {}
    for name in [*OBSERVABILITY_PATCH, "iaa_agent/autonomous.py", "scripts/mm_ablation_support.py"]:
        new = (root / name).read_text(encoding="utf-8")
        old = new
        for fragment in OBSERVABILITY_PATCH.get(name, []):
            assert old.count(fragment) == 1
            old = old.replace(fragment, "", 1)
        for directory, value, hashes in ((old_root, old, old_hashes), (new_root, new, new_hashes)):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
            hashes[name] = digest(path.read_bytes())
    def manifest(hashes):
        p = {"code_sha256": hashes, "model": "m", "agent_config": {"retry_budget": 2}}
        return {"protocol": p, "protocol_sha256": digest(p)}
    assert len(audit_compatibility(manifest(old_hashes), manifest(new_hashes), old_root, new_root)) == 2
    changed = new_root / "iaa_agent/autonomous.py"
    changed.write_text(changed.read_text(encoding="utf-8") + "\n# unreviewed prediction change\n", encoding="utf-8")
    new_hashes["iaa_agent/autonomous.py"] = digest(changed.read_bytes())
    with pytest.raises(ValueError, match="Prediction logic changed"):
        audit_compatibility(manifest(old_hashes), manifest(new_hashes), old_root, new_root)


def test_client_records_http_status_without_response_body_or_credentials(monkeypatch):
    from urllib.error import HTTPError
    from iaa_agent.llm import DeepSeekClient
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    def error(*a, **kw):
        raise HTTPError("http://local-test", 503, "Unavailable", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", error)
    client = DeepSeekClient(model="test", base_url="http://local-test", provider="openai")
    assert client.chat_json([]) is None
    assert client.last_http_status == 503
    assert client.last_call_status == "request_error"


def test_runtime_does_not_retry_program_exception(tmp_path):
    from test_ranking_topup import run
    _, client, execute = run(tmp_path / "calls", [KeyError("program bug")])
    with pytest.raises(KeyError):
        execute()
    assert len(client.calls) == 1
    journal = read_json(tmp_path / "calls" / "final_ranking.json")
    assert journal["attempts"][0]["exception_type"] == "KeyError"
    assert failure_row(KeyError("program bug"), tmp_path)["failure_kind"] == "fatal"


def test_known_duplicate_failure_uses_shared_budget_once_across_resumes(tmp_path):
    from test_ranking_topup import run, answer
    bad = answer("P1", "P1", "P1")
    _, client, execute = run(tmp_path / "calls", [bad, bad, bad])
    with pytest.raises(RuntimeError) as error:
        execute()
    assert len(client.calls) == 3
    row = failure_row(error.value, tmp_path)
    assert row["failure_kind"] == "model" and row["accounting"]["retries"] == 2
    _, client2, resume = run(tmp_path / "calls", [])
    with pytest.raises(RuntimeError):
        resume()
    assert client2.calls == []
