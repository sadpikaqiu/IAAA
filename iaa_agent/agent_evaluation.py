"""Common evaluation adapter for fixed and autonomous engines."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from .agent_runtime import atomic_json, read_json, digest, JournaledModel, PromptBudget
from .agent_types import VisibleQuery
from .agent_tools import POIToolService
from .autonomous import AutonomousAgent, rerank_fixed_result
from .engine import IAAAgent, RunConfig
from .models import Intention


def prepare_prompt(agent, query):
    messages = []
    def capture(value, **kwargs):
        messages.append(value)
        return None
    agent.llm.chat_json = capture
    context = agent._build_context(query)
    profile = agent._build_user_profile(query)
    peers = agent._find_peer_users(query, profile)
    agent._infer_intention(context, profile, peers, query)
    if len(messages) != 1:
        raise RuntimeError("Expected exactly one production intention prompt")
    return messages[0], (profile, peers)


class FrozenIntentionAgent(IAAAgent):
    def __init__(self, *args, frozen_intention, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_intention = frozen_intention
    def _infer_intention(self, *args, **kwargs):
        self.last_intention_source = "openai"
        self.last_llm_status = "success"
        return self.frozen_intention.model_copy(deep=True)


def legacy_prediction(repo, query, store, model):
    config = RunConfig.p4(llm_mode="openai")
    config.intention_context_size = 5
    if store:
        config.evidence_mode = store.mode
    probe = IAAAgent(repo, config, evidence_store=store)
    messages, _ = prepare_prompt(probe, query)
    intention = model.call("initial_intention", messages, 4096, Intention.model_validate)
    agent = FrozenIntentionAgent(repo, config, evidence_store=store, frozen_intention=intention)
    result = agent.run_query(query).model_dump(mode="json")
    result.update(engine="fixed", accounting=model.accounting(), heuristic_fallback=False,
                  elapsed_seconds=model.accounting()["model_seconds"], tool_calls=None)
    return result


def combined_accounting(a, b):
    out = {k: a.get(k, 0) + b.get(k, 0) for k in ("logical_calls", "requests", "retries", "invalid_attempts",
                                                 "usage_missing_count", "model_seconds")}
    out["usage"] = {k: a.get("usage", {}).get(k, 0) + b.get("usage", {}).get(k, 0)
                    for k in set(a.get("usage", {})) | set(b.get("usage", {}))}
    return out


def prediction_row(result, query):
    gt = str(query.target["POI_id"])
    predictions = [str(p["poi_id"]) for p in result["ranked_pois"]]
    if len(predictions) != 10 or len(set(predictions)) != 10:
        raise ValueError("Formal evaluation requires exactly ten unique predictions")
    pool = result["candidate_pool_summary"]
    accounting = result["accounting"]
    trace = result.get("trace", [])
    calls = [t for s in trace for t in s["tools"]]
    external_refs = {r for p in result["ranked_pois"] for r in p.get("evidence_refs", []) if r.startswith("E")}
    modalities = {result.get("references", {}).get(r, {}).get("modality") for r in external_refs}
    return {"predictions": predictions, "rank": predictions.index(gt) + 1 if gt in predictions else None,
            "in_pool": gt in pool["candidate_poi_ids"], "in_raw": gt in pool["raw_retrieved_poi_ids"],
            "in_observed": gt in pool.get("observed_poi_ids", pool["candidate_poi_ids"]),
            "observed_definition": "algorithm_candidate_pool" if result["engine"] == "fixed" else "tool_candidates_presented_to_llm",
            "pool_size": len(pool["candidate_poi_ids"]), "raw_size": len(pool["raw_retrieved_poi_ids"]),
            "candidate_poi_ids": pool["candidate_poi_ids"], "raw_poi_ids": pool["raw_retrieved_poi_ids"],
            "accounting": accounting, "tool_calls": result.get("tool_calls", 0),
            "tool_errors": sum(t["status"] != "success" for t in calls),
            "tool_counts": {name: sum(t["request"]["name"] == name for t in calls)
                            for name in sorted({t["request"]["name"] for t in calls})},
            "stop_reason": result.get("stop_reason", "fixed_pipeline"),
            "decision_count": len(trace), "image_cited": "image" in modalities, "review_cited": "review" in modalities,
            "elapsed_seconds": result.get("elapsed_seconds", accounting["model_seconds"]),
            "heuristic_fallback": bool(result.get("heuristic_fallback")), "valid": True}


def run_variant(engine, mode, repo, query, store, directory, identity, tokenizer, config, heartbeat=None,
                baseline_result=None):
    directory = Path(directory)
    output = directory / "prediction.json"
    expected = {"identity": identity, "engine": engine, "mode": mode, "query_id": query.traj_id}
    if output.exists():
        cached = read_json(output)
        if cached["identity"] != expected:
            raise ValueError("Prediction identity mismatch")
        return cached["prediction"]
    model = JournaledModel(directory / "calls", identity, PromptBudget(tokenizer, config.prompt_tokens),
                           config, heartbeat=heartbeat)
    if engine == "fixed":
        result = legacy_prediction(repo, query, store, model)
    else:
        visible = VisibleQuery.from_query(query)
        tools = POIToolService(repo, visible, store)
        current_config = replace(config, engine=engine)
        if engine == "fixed_llm_rank":
            if baseline_result is None:
                raise ValueError("B requires the corresponding A prediction")
            safe_fixed = {k: baseline_result[k] for k in ("candidate_pool_summary", "inferred_intention", "reflection")}
            result = rerank_fixed_result(tools, model, safe_fixed, current_config)
            result["exclusive_accounting"] = result["accounting"]
            result["accounting"] = combined_accounting(baseline_result["accounting"], result["accounting"])
            result["shared_intention_identity"] = digest(baseline_result["inferred_intention"])
            result["elapsed_seconds"] += baseline_result["elapsed_seconds"]
            result["tool_calls"] = None  # Legacy front end has no comparable instrumented tool-call counter.
        else:
            result = AutonomousAgent(tools, model, current_config).run()
    atomic_json(output, {"identity": expected, "prediction": result})
    return result
