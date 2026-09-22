"""Frozen experiment identities, legacy adapters, and paired agent statistics."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path

import numpy as np

from iaa_agent.agent_runtime import atomic_json, read_json, digest, now, JournaledModel, PromptBudget
from iaa_agent.agent_types import AgentConfig, VisibleQuery
from iaa_agent.agent_tools import POIToolService
from iaa_agent.autonomous import AutonomousAgent, rerank_fixed_result
from iaa_agent.engine import IAAAgent, RunConfig
from iaa_agent.models import Intention
from scripts.mm_ablation_support import prepare_prompt, select_validation
from scripts.autonomous_failure_policy import SOFT_KINDS


ENGINES = ("fixed", "fixed_llm_rank", "fixed_schedule", "autonomous")
MODES = ("text", "both")
ARMS = tuple(f"{engine}__{mode}" for engine in ENGINES for mode in MODES)
AUTONOMOUS_ARMS = tuple(f"autonomous__{mode}" for mode in MODES)


def select_new_validation(repo, previous, development_size=50, validation_size=500, repeat_size=100):
    all_cases = select_validation(repo, sample_size=10**9, repeat_size=0, seed="iaaa-autonomous-v1")
    old = {(c["user_id"], c["trajectory_id"]) for c in previous}
    remaining = [c for c in all_cases["cases"] if (c["user_id"], c["trajectory_id"]) not in old]
    if len(remaining) < development_size + validation_size:
        raise ValueError("Not enough unused validation cases")
    dev = [dict(c, repeat=False) for c in remaining[:development_size]]
    val = [dict(c, repeat=i < repeat_size) for i, c in enumerate(remaining[development_size:development_size + validation_size])]
    return {"remaining_count": len(remaining), "eligible_count": all_cases["eligible_count"],
            "excluded_previous_count": len(old), "development": dev, "validation": val,
            "selection_policy": "ID hash seed iaaa-autonomous-v1; no labels, ranks, or evidence coverage"}


from iaa_agent.agent_evaluation import (FrozenIntentionAgent, legacy_prediction, combined_accounting, prediction_row, run_variant)


def metrics(row):
    if row.get("status") == "failed":
        return {**dict.fromkeys(("Hit@1", "Hit@5", "Hit@10", "NDCG@10", "MRR@10"), 0.),
                **dict.fromkeys(("CandidateRecall", "RawCandidateRecall", "ObservedCandidateRecall",
                                 "mean_pool_size", "mean_raw_size"), None)}
    rank = row["rank"]
    return {"Hit@1": float(rank == 1), "Hit@5": float(rank is not None and rank <= 5),
            "Hit@10": float(rank is not None and rank <= 10),
            "NDCG@10": 1 / math.log2(rank + 1) if rank else 0., "MRR@10": 1 / rank if rank else 0.,
            "CandidateRecall": float(row["in_pool"]), "RawCandidateRecall": float(row["in_raw"]),
            "ObservedCandidateRecall": float(row["in_observed"]), "mean_pool_size": float(row["pool_size"]),
            "mean_raw_size": float(row["raw_size"]) if row.get("raw_size") is not None else None}


def paired_contrast(cases, a, b, draws=2000, permutations=10000):
    pairs = [c for c in cases if a in c["variants"] and b in c["variants"]]
    if not pairs:
        return None
    names = [k for k in metrics(pairs[0]["variants"][a])
             if all(metrics(c["variants"][arm])[k] is not None for c in pairs for arm in (a, b))]
    delta = np.array([[metrics(c["variants"][b])[k] - metrics(c["variants"][a])[k] for k in names] for c in pairs])
    users = sorted({c["user_id"] for c in pairs})
    lookup = {u: i for i, u in enumerate(users)}
    sums, counts = np.zeros((len(users), len(names))), np.zeros(len(users))
    for c, d in zip(pairs, delta):
        i = lookup[c["user_id"]]
        sums[i] += d
        counts[i] += 1
    rng = np.random.default_rng(42)
    boot = []
    # Bounded batches avoid a large city x all bootstrap draws allocation.
    for start in range(0, draws, 100):
        sample = rng.integers(0, len(users), size=(min(100, draws - start), len(users)))
        boot.append(sums[sample].sum(axis=1) / counts[sample].sum(axis=1)[:, None])
    intervals = np.quantile(np.concatenate(boot), [.025, .975], axis=0)
    hit = names.index("Hit@10")
    observed = abs(float(sums[:, hit].sum()))
    extreme = 0
    for start in range(0, permutations, 100):
        signs = rng.integers(0, 2, size=(min(100, permutations - start), len(users))) * 2 - 1
        extreme += int((np.abs(signs @ sums[:, hit]) >= observed - 1e-12).sum())
    return {"n": len(pairs), "users": len(users), "delta": dict(zip(names, delta.mean(axis=0).tolist())),
            "user_cluster_bootstrap_95ci": {k: intervals[:, i].tolist() for i, k in enumerate(names)},
            "hit10_cluster_permutation_p": (extreme + 1) / (permutations + 1),
            "hit10_gains": sum(c["variants"][a]["rank"] is None and c["variants"][b]["rank"] is not None for c in pairs),
            "hit10_losses": sum(c["variants"][a]["rank"] is not None and c["variants"][b]["rank"] is None for c in pairs),
            "same_top10": sum(bool(c["variants"][a]["predictions"]) and
                              c["variants"][a]["predictions"] == c["variants"][b]["predictions"] for c in pairs)}


def quality_errors(cases, expected, arms, *, repeats=False, allow_failures=False):
    expected_ids = {(c["user_id"], c["trajectory_id"]) for c in expected}
    actual = [(c["user_id"], c["trajectory_id"]) for c in cases]
    errors = []
    if len(actual) != len(set(actual)) or set(actual) != expected_ids:
        errors.append("session_identity_or_count_mismatch")
    expected_by_id = {(c["user_id"], c["trajectory_id"]): c for c in expected}
    for c in cases:
        names = list(arms)
        if repeats and expected_by_id.get((c["user_id"], c["trajectory_id"]), {}).get("repeat"):
            names += [a + "__repeat" for a in AUTONOMOUS_ARMS]
        terminal = {"completed", "completed_with_failures"} if allow_failures else {"completed"}
        if c["status"] not in terminal or set(c["variants"]) != set(names):
            errors.append("incomplete_case")
        for name in names:
            row = c["variants"].get(name)
            if row is None:
                continue
            if row.get("status") == "failed":
                if (not allow_failures or row.get("failure_kind") not in SOFT_KINDS
                        or row.get("predictions") != [] or row.get("rank") is not None
                        or row.get("valid") is not False or row.get("heuristic_fallback")
                        or not row.get("error_type") or not row.get("error")):
                    errors.append("invalid_failure_record")
                if row.get("failure_kind") == "dependency":
                    upstream = c["variants"].get(row.get("dependency"), {})
                    expected_dependency = "fixed__" + name.split("__")[1] if name.startswith("fixed_llm_rank__") else None
                    if row.get("dependency") != expected_dependency or upstream.get("status") != "failed":
                        errors.append("invalid_dependency_failure")
                continue
            if not row.get("valid") or row.get("heuristic_fallback"):
                errors.append("invalid_or_fallback_prediction")
            if row["accounting"].get("usage_missing_count") and not allow_failures:
                errors.append("usage_missing")
            if row["accounting"].get("requests", 0) < 1:
                errors.append("no_llm_request")
            if not row["accounting"].get("usage", {}).get("total_tokens") and row.get("cost_granularity") != "historical_aggregate_only":
                errors.append("usage_missing")
            if len(row["predictions"]) != 10 or len(set(row["predictions"])) != 10:
                errors.append("invalid_top10")
    return sorted(set(errors))


def summarize(cases, expected, arms, *, repeats=False, full=False, allow_failures=False):
    errors = quality_errors(cases, expected, arms, repeats=repeats, allow_failures=allow_failures)
    result = {"n": len(cases), "expected_n": len(expected), "quality": {"valid": not errors, "errors": errors},
              "arms": {}, "contrasts": {}, "created_at": now()}
    result.update(fallback_count=sum(r.get("heuristic_fallback", False) for c in cases for r in c["variants"].values()),
                  usage_missing_count=sum(r["accounting"].get("usage_missing_count", 0) for c in cases for r in c["variants"].values()),
                  all_sessions_used_llm=all(r["accounting"].get("requests", 0) > 0
                      for c in cases for r in c["variants"].values()) and not errors)
    result["evaluation_policy"] = "terminal_arm_failures_v1" if allow_failures else "strict_all_success"
    result["accounting_complete"] = result["usage_missing_count"] == 0
    result["conditional_contrasts"] = {}
    all_arms = list(arms) + ([a + "__repeat" for a in AUTONOMOUS_ARMS] if repeats else [])
    for arm in all_arms:
        subset = [c for c in cases if arm in c["variants"]]
        if not subset:
            continue
        rows = [c["variants"][arm] for c in subset]
        successful = [r for r in rows if r.get("status") != "failed"]
        failures = [r for r in rows if r.get("status") == "failed"]
        def average(group):
            if not group:
                return {}
            result = {}
            for k in metrics(rows[0]):
                values = [metrics(c["variants"][arm])[k] for c in group if metrics(c["variants"][arm])[k] is not None]
                result[k] = float(np.mean(values)) if values else None
            return result
        result["arms"][arm] = {"n": len(rows), "overall": average(subset),
            "expected_n": sum(not arm.endswith("__repeat") or c.get("repeat", False) for c in expected),
            "success_n": len(successful), "failure_n": len(failures), "failure_rate": len(failures) / len(rows),
            "failure_kinds": {kind: sum(r["failure_kind"] == kind for r in failures)
                              for kind in sorted({r["failure_kind"] for r in failures})},
            "candidate_metric_n": len(successful),
            "metric_denominators": {k: sum(metrics(r)[k] is not None for r in rows) for k in metrics(rows[0])},
            "by_history": {h: {"n": len(g := [c for c in subset if c["history_group"] == h]), **average(g)} for h in ("IH", "OOH")},
            "cost": {"requests": sum(r["accounting"]["requests"] for r in rows),
                     "total_tokens": sum(r["accounting"]["usage"].get("total_tokens", 0) for r in rows),
                     "retries": sum(r["accounting"]["retries"] for r in rows),
                     "usage_missing_count": sum(r["accounting"].get("usage_missing_count", 0) for r in rows),
                     "tokens_complete": all(not r["accounting"].get("usage_missing_count", 0) for r in rows),
                     "mean_tool_calls": (float(np.mean([r["tool_calls"] for r in rows if r["tool_calls"] is not None]))
                                         if any(r["tool_calls"] is not None for r in rows) else None),
                     "elapsed_p50": float(np.quantile([r["elapsed_seconds"] for r in rows], .5)),
                     "elapsed_p95": float(np.quantile([r["elapsed_seconds"] for r in rows], .95))},
            "errors": {"not_retrieved": sum(not r["in_raw"] for r in successful),
                       "retrieved_not_selected": sum(r["in_raw"] and not r["in_pool"] for r in successful),
                       "selected_not_top10": sum(r["in_pool"] and r["rank"] is None for r in successful)},
            "image_cited": sum(r.get("image_cited", False) for r in rows),
            "review_cited": sum(r.get("review_cited", False) for r in rows),
            "tool_errors": sum(r.get("tool_errors", 0) for r in rows),
            "tool_counts": {name: sum(r.get("tool_counts", {}).get(name, 0) for r in rows)
                            for name in sorted({n for r in rows for n in r.get("tool_counts", {})})},
            "stop_reasons": {name: sum(r.get("stop_reason", "historical_unrecorded").split(":", 1)[0] == name for r in rows)
                             for name in sorted({r.get("stop_reason", "historical_unrecorded").split(":", 1)[0] for r in rows})}}
    comparisons = [("fixed__" + m, "fixed_llm_rank__" + m) for m in MODES]
    if errors:
        result["comparisons_suppressed"] = "Incomplete or invalid paired matrix; per-arm values are diagnostics only."
        return result
    comparisons += [("fixed_schedule__" + m, "autonomous__" + m) for m in MODES]
    comparisons += [("autonomous__text", "autonomous__both")]
    comparisons += [("fixed__" + m, "autonomous__" + m) for m in MODES]
    if repeats:
        comparisons += [(a, a + "__repeat") for a in AUTONOMOUS_ARMS]
    for a, b in comparisons:
        comparison = paired_contrast(cases, a, b)
        if comparison:
            result["contrasts"][f"{b}_minus_{a}"] = comparison
            if allow_failures:
                complete = [c for c in cases if a in c["variants"] and b in c["variants"]
                            and all(c["variants"][n].get("status") != "failed" for n in (a, b))]
                conditional = paired_contrast(complete, a, b)
                result["conditional_contrasts"][f"{b}_minus_{a}"] = {
                    "eligible_n": comparison["n"], "success_pair_n": len(complete),
                    "excluded_n": comparison["n"] - len(complete), "statistics": conditional,
                    "note": "Conditioned on both arms succeeding; selection may be biased. Secondary only."}
    return result


def holm_adjust(summaries, primary_names):
    entries = [(city, name, summary["contrasts"][name]) for city, summary in summaries.items()
               for name in primary_names if name in summary["contrasts"]]
    entries.sort(key=lambda x: x[2]["hit10_cluster_permutation_p"])
    previous = 0.
    for i, (city, name, entry) in enumerate(entries):
        previous = min(1., max(previous, (len(entries) - i) * entry["hit10_cluster_permutation_p"]))
        entry["hit10_holm_p"] = previous
        entry["holm_family_size"] = len(entries)
    return summaries
