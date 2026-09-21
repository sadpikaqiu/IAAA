"""Controlled validation interventions around the unchanged fixed pipeline.

No labels are read by an intervention. Labels are used only by the runner after
prediction. Single-modality views retain the joint TF-IDF vocabulary/IDF and
the original per-modality ranking coefficient.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from iaa_agent.engine import IAAAgent, RunConfig
from iaa_agent.evidence import MODES, EvidenceStore
from iaa_agent.models import Intention


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def masked_store(joint: EvidenceStore, mode: str) -> EvidenceStore:
    if joint.mode != "both":
        raise ValueError("Masking requires the unchanged joint evidence index")
    view = copy.copy(joint)
    view.mode, view.modalities = mode, MODES[mode]
    view.by_poi = {poi: [i for i in indices if joint.items[i][1]["modality"] in view.modalities]
                   for poi, indices in joint.by_poi.items()}
    return view


@dataclass(frozen=True)
class Arm:
    name: str
    intent: str = "both_0"
    mode: str | None = "both"
    prior_weight: float = .1
    quota: int = 5
    rank_weight: float = .1
    recall: bool = True
    reflection: str = "auto"  # auto, text_budget, or mobility_scores


ARMS = (
    Arm("text", "text_0", None),
    Arm("intent_only_both", mode=None, reflection="text_budget"),
    Arm("intent_only_images", "images_0", None, reflection="text_budget"),
    Arm("intent_only_reviews", "reviews_0", None, reflection="text_budget"),
    Arm("recall_only", "text_0", rank_weight=0, reflection="text_budget"),
    Arm("rank_only", "text_0", prior_weight=0, quota=0, recall=False, reflection="text_budget"),
    Arm("full_both"),
    Arm("no_quota", quota=0),
    Arm("no_prior_boost", prior_weight=0),
    Arm("no_rank_bonus", rank_weight=0),
    Arm("mobility_reflection", reflection="mobility_scores"),
    Arm("fixed_reflection", reflection="text_budget"),
    Arm("no_prior_fixed_reflection", prior_weight=0, reflection="text_budget"),
    Arm("no_prior_no_quota_fixed_reflection", prior_weight=0, quota=0, reflection="text_budget"),
    Arm("images_only", "images_0", "images", reflection="text_budget"),
    Arm("reviews_only", "reviews_0", "reviews", reflection="text_budget"),
)
REPEAT_ARMS = (Arm("text_repeat", "text_1", None), Arm("full_both_repeat", "both_1"))

# A priori contrasts; no winner is chosen on the original test results.
CONTRASTS = (
    ("text", "full_both"), ("text", "fixed_reflection"),
    ("text", "intent_only_both"), ("text", "intent_only_images"),
    ("text", "intent_only_reviews"), ("text", "recall_only"), ("text", "rank_only"),
    ("full_both", "no_quota"), ("full_both", "no_prior_boost"),
    ("full_both", "no_rank_bonus"), ("full_both", "mobility_reflection"),
    ("full_both", "fixed_reflection"),
    ("fixed_reflection", "no_prior_fixed_reflection"),
    ("no_prior_fixed_reflection", "no_prior_no_quota_fixed_reflection"),
    ("text", "images_only"), ("text", "reviews_only"),
    ("images_only", "fixed_reflection"), ("reviews_only", "fixed_reflection"),
    ("text", "text_repeat"), ("full_both", "full_both_repeat"),
)


def select_validation(repo, *, sample_size=500, repeat_size=100, seed="mm-validation-20260917"):
    """Keep session endpoints in each user's [70%,80%) interval, then hash IDs.

    Uses the repository's exact event ordering/cutoff conventions. No POI label,
    rank, evidence availability, or historical experiment outcome affects sampling.
    """
    eligible, excluded_ties = [], 0
    test_keys = set(repo.iter_session_test_keys(train_ratio=.8))
    for uid, rows in repo.all_events.groupby("user_id", sort=False):
        ordered = rows.sort_values("UTC_time").reset_index(drop=True)
        lo, hi = (repo._user_cutoff_index(ordered, r) for r in (.7, .8))
        for tid, session in ordered.groupby("trajectory_id", sort=False):
            session = session.sort_values("UTC_time")
            idx = int(session.index[-1])
            if len(session) <= 1 or not lo <= idx < hi:
                continue
            stamp = session.iloc[-1]["UTC_time"]
            if (ordered.iloc[:lo]["UTC_time"] >= stamp).any() or (session.iloc[:-1]["UTC_time"] >= stamp).any():
                excluded_ties += 1
                continue
            key = (str(uid), str(tid))
            if key in test_keys:
                raise AssertionError("Validation and original test overlap")
            eligible.append({"city": repo.city, "user_id": key[0], "trajectory_id": key[1],
                             "target_index": idx, "history_cutoff": lo, "test_cutoff": hi,
                             "target_time": str(stamp)})
    eligible.sort(key=lambda c: digest(f"{seed}/{repo.city}/{c['user_id']}/{c['trajectory_id']}".encode()))
    selected = [dict(c, repeat=i < repeat_size) for i, c in enumerate(eligible[:sample_size])]
    return {"eligible_count": len(eligible), "excluded_non_strict_time_count": excluded_ties,
            "eligible_keys_sha256": digest(canonical(eligible)), "original_test_count": len(test_keys),
            "original_test_overlap": 0, "selected_count": len(selected), "cases": selected}


class AblationAgent(IAAAgent):
    def __init__(self, *args, arm: Arm, frozen_intention=None, prepared=None,
                 text_reflection=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm, self.frozen_intention, self.prepared = arm, frozen_intention, prepared
        self.text_reflection = text_reflection
        self.profile_saved = self.peers_saved = None
        self.rounds = []

    def _build_user_profile(self, query):
        self.profile_saved = (self.prepared[0].model_copy(deep=True) if self.prepared
                              else super()._build_user_profile(query))
        return self.profile_saved

    def _find_peer_users(self, query, profile):
        self.peers_saved = (copy.deepcopy(self.prepared[1]) if self.prepared
                            else super()._find_peer_users(query, profile))
        return self.peers_saved

    def _infer_intention(self, context, profile, peers, query):
        if self.frozen_intention is None:
            return super()._infer_intention(context, profile, peers, query)
        self.last_intention_source = "cached_validated_llm"
        self.last_llm_status = "offline_replay"
        return self.frozen_intention.model_copy(deep=True)

    def _select_candidates(self, raw, expanded):
        original = self.config
        self.config = replace(original, evidence_weight=self.arm.prior_weight)
        try:
            selected = super()._select_candidates(raw, expanded)
        finally:
            self.config = original
        self.rounds.append({"expanded": expanded, "raw_ids": sorted(raw),
                            "pool_ids": [c.poi_id for c in selected],
                            "evidence_recalled_ids": sorted(k for k, v in raw.items() if "poi_evidence" in v["source_scores"])})
        return selected

    def _candidate_affordance(self, *args, **kwargs):
        profile = super()._candidate_affordance(*args, **kwargs)
        if self.evidence is not None:
            # Use the source relevance, not an already rounded/scaled score.
            # Removing a modality must not double the coefficient of the other.
            for verdict in profile.affordances:
                if verdict.name in {"image_intent_relevance", "review_intent_relevance"}:
                    profile.score_decomposition[verdict.name] = round(self.arm.rank_weight * verdict.relevance_score / 2, 6)
            profile.alignment_score = round(sum(profile.score_decomposition.values()), 6)
        return profile

    def _maybe_reflect(self, ranked, candidates, intention, context):
        natural = super()._maybe_reflect(ranked, candidates, intention, context)
        self.rounds[-1]["natural_reflection"] = natural.model_dump(mode="json")
        if self.arm.reflection == "text_budget":
            if self.text_reflection is None:
                raise ValueError("A text-budget control requires the text reflection record")
            return self.text_reflection.model_copy(deep=True)
        if self.arm.reflection == "mobility_scores":
            stripped = []
            for item in ranked:
                item = item.model_copy(deep=True)
                item.alignment_score = round(sum(v for k, v in item.score_decomposition.items()
                                                if k not in {"image_intent_relevance", "review_intent_relevance"}), 6)
                stripped.append(item)
            return super()._maybe_reflect(self._rank_profiles(stripped), candidates, intention, context)
        return natural


def agent_config(arm: Arm, snapshot: Path | None, *, live=False):
    config = RunConfig.p4(llm_mode="openai" if live else "fake")
    return replace(config, intention_context_size=5,
                   evidence_snapshot=str(snapshot) if arm.mode is not None else None,
                   evidence_mode=arm.mode or "both", evidence_quota=arm.quota,
                   evidence_top_n=30 if arm.recall else 0, evidence_weight=arm.rank_weight)


def prepare_prompt(agent, query):
    """Capture the production prompt without allowing any network request."""
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
        raise RuntimeError("Production intention prompt was not captured exactly once")
    return messages[0], (profile, peers)


def validated_intention_attempt(client, messages):
    parsed = client.chat_json(messages)
    errors, intention = [], None
    try:
        intention = Intention.model_validate(parsed)
    except Exception as exc:
        errors = exc.errors(include_input=False, include_url=False) if hasattr(exc, "errors") else [{"type": type(exc).__name__}]
    valid = (intention is not None and client.last_call_status == "success"
             and client.last_usage is not None and client.last_finish_reason == "stop")
    return {"valid": valid, "intention": intention.model_dump(mode="json") if valid else None,
            "parsed": parsed, "raw_content": client.last_raw_content,
            "finish_reason": client.last_finish_reason, "status": client.last_call_status,
            "error_type": client.last_error_type, "usage": client.last_usage,
            "validation_errors": errors}


def compact_result(agent, result, elapsed):
    predictions = [p.poi_id for p in result.ranked_pois]
    target = result.ground_truth_poi_id
    return {"predictions": predictions, "rank": predictions.index(target) + 1 if target in predictions else None,
            "in_pool": target in result.candidate_pool_summary["candidate_poi_ids"],
            "in_raw": target in result.candidate_pool_summary["raw_retrieved_poi_ids"],
            "pool_size": result.candidate_pool_summary["candidate_count"],
            "reflection": result.reflection.model_dump(mode="json"), "rounds": agent.rounds,
            "top10_scores": [{"poi_id": p.poi_id, "score": p.alignment_score,
                              "decomposition": p.score_decomposition} for p in result.ranked_pois],
            "elapsed_seconds": elapsed, "api_calls": 0}


def metrics(row):
    rank = row["rank"]
    return {"Hit@1": float(rank == 1), "Hit@5": float(rank is not None and rank <= 5),
            "Hit@10": float(rank is not None and rank <= 10),
            "NDCG@10": 1 / math.log2(rank + 1) if rank else 0., "MRR@10": 1 / rank if rank else 0.,
            "CandidateRecall": float(row["in_pool"]), "RawCandidateRecall": float(row["in_raw"]),
            "mean_pool_size": float(row["pool_size"]), "reflection_rate": float(row["reflection"]["triggered"])}


def summarize(cases, expected, *, bootstrap_samples=2000):
    """Session-weighted point estimates with paired user-cluster bootstrap CIs."""
    expected_ids = {(c["user_id"], c["trajectory_id"]) for c in expected}
    actual_ids = {(c["user_id"], c["trajectory_id"]) for c in cases}
    if not actual_ids <= expected_ids or len(actual_ids) != len(cases):
        raise ValueError("Unexpected/duplicate result identities")
    arms = {}
    for arm in ARMS + REPEAT_ARMS:
        subset = [c for c in cases if arm.name in c["variants"]]
        target_n = sum(c["repeat"] for c in expected) if arm in REPEAT_ARMS else len(expected)
        if not subset:
            continue
        def average(rows):
            vals = [metrics(c["variants"][arm.name]) for c in rows]
            return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]} if vals else {}
        arms[arm.name] = {"n": len(subset), "expected_n": target_n, "complete": len(subset) == target_n,
                          "overall": average(subset),
                          "by_history": {h: {"n": len(rows := [c for c in subset if c["history_group"] == h]),
                                              **average(rows)} for h in ("IH", "OOH")}}
    contrasts = {}
    for a, b in CONTRASTS:
        pairs = [c for c in cases if a in c["variants"] and b in c["variants"]]
        if not pairs:
            continue
        names = list(metrics(pairs[0]["variants"][a]))
        deltas = np.array([[metrics(c["variants"][b])[k] - metrics(c["variants"][a])[k] for k in names] for c in pairs])
        users = sorted({c["user_id"] for c in pairs})
        user_index = {u: i for i, u in enumerate(users)}
        sums, counts = np.zeros((len(users), len(names))), np.zeros(len(users))
        for c, delta in zip(pairs, deltas):
            i = user_index[c["user_id"]]
            sums[i] += delta
            counts[i] += 1
        rng = np.random.default_rng(42)
        draws = rng.integers(0, len(users), size=(bootstrap_samples, len(users)))
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
        ci = np.quantile(boot, [.025, .975], axis=0)
        target_n = sum(c["repeat"] for c in expected) if b.endswith("_repeat") else len(expected)
        contrasts[f"{b}_minus_{a}"] = {
            "n": len(pairs), "expected_n": target_n, "complete": len(pairs) == target_n, "users": len(users),
            "delta": dict(zip(names, deltas.mean(axis=0).tolist())),
            "user_cluster_bootstrap_95ci": {k: [float(ci[0, i]), float(ci[1, i])] for i, k in enumerate(names)},
            "hit10_gains": sum(c["variants"][a]["rank"] is None and c["variants"][b]["rank"] is not None for c in pairs),
            "hit10_losses": sum(c["variants"][a]["rank"] is not None and c["variants"][b]["rank"] is None for c in pairs),
            "same_top10": sum(c["variants"][a]["predictions"] == c["variants"][b]["predictions"] for c in pairs),
        }
    return {"scope": "independent_validation_sample_not_original_test", "n": len(cases),
            "expected_n": len(expected), "complete": actual_ids == expected_ids,
            "uncertainty": "Paired user-cluster bootstrap, 2000 draws, exploratory unadjusted intervals; no automatic winner selection.",
            "arms": arms, "contrasts": contrasts}
