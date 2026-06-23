from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data import NYCDataRepository, QueryExample
from .llm import DeepSeekClient, parse_intention_or_none
from .models import (
    AffordanceProfile,
    AffordanceVerdict,
    AgentRunResult,
    Candidate,
    ContextSnapshot,
    Intention,
    LikelyCategory,
    RankedPOI,
    ReflectionRecord,
    ToolCallRecord,
    ToolPlan,
    ToolPlanItem,
    UserProfile,
)
from .utils import (
    category_family,
    circular_minute_diff,
    cosine_dict,
    entropy,
    haversine_km,
    normalize_scores,
    safe_div,
    time_bucket_label,
)


MISSING_EVIDENCE = [
    "reviews unavailable",
    "images unavailable",
    "opening hours unavailable",
    "price unavailable",
    "ratings unavailable",
]


@dataclass
class RunConfig:
    candidate_pool_size: int = 30
    spatial_top_n: int = 50
    category_top_n: int = 50
    transition_top_n: int = 30
    peer_top_n: int = 30
    peer_users: int = 5
    peer_window_minutes: int = 30
    max_reflection_rounds: int = 1
    llm_mode: str = "fake"


class IAAAgent:
    def __init__(self, repo: NYCDataRepository, config: RunConfig | None = None) -> None:
        self.repo = repo
        self.config = config or RunConfig()
        self.llm = DeepSeekClient()

    def run(self, traj_id: str) -> AgentRunResult:
        query = self.repo.get_query(traj_id)
        trace: list[ToolCallRecord] = []

        context = self._build_context(query)
        trace.append(
            ToolCallRecord(
                state="S0_CONTEXT_OBSERVED",
                tool="ObserveContext",
                reason="Build visible query context and target-time snapshot.",
                observations=[
                    f"{len(context.query_trajectory)} visible check-ins",
                    f"target hour {context.target_hour}",
                    f"last category {context.last_known_category}",
                ],
            )
        )

        profile = self._build_user_profile(query)
        trace.append(
            ToolCallRecord(
                state="S0_CONTEXT_OBSERVED",
                tool="BuildUserProfile",
                reason="Summarize long-term train+val behavior plus visible query context.",
                observations=[
                    f"{profile.num_checkins} visible historical check-ins",
                    f"top categories: {', '.join(x['category'] for x in profile.top_categories[:3])}",
                ],
            )
        )

        peers = self._find_peer_users(query, profile)
        trace.append(
            ToolCallRecord(
                state="S0_CONTEXT_OBSERVED",
                tool="FindPeerUsers",
                reason="Use category-time and spatial overlap to find peer evidence.",
                params={"top_k": self.config.peer_users},
                observations=[f"{len(peers)} peers selected"],
            )
        )

        intention = self._infer_intention(context, profile, peers, query)
        trace.append(
            ToolCallRecord(
                state="S1_INTENTION_INFERRED",
                tool="InferIntention",
                reason="Infer structured user intention before selecting candidates.",
                observations=[
                    intention.summary,
                    f"confidence={intention.confidence:.2f}",
                ],
            )
        )

        plan = self._build_tool_plan(intention, context, profile)
        trace.append(
            ToolCallRecord(
                state="S2_TOOL_PLAN_READY",
                tool="PlanTools",
                reason="Select bounded recall tools from the inferred intention.",
                observations=[f"{len(plan.items)} tools planned"],
            )
        )

        candidates, retrieval_trace = self._retrieve_candidates(query, context, profile, intention, peers, expanded=False)
        trace.extend(retrieval_trace)

        profiles = self._build_affordances(candidates, query, context, profile, intention, peers)
        ranked = self._rank_profiles(profiles)
        reflection = self._maybe_reflect(ranked, candidates, intention, context)

        if reflection.triggered and self.config.max_reflection_rounds > 0:
            expanded_candidates, expanded_trace = self._retrieve_candidates(
                query, context, profile, intention, peers, expanded=True
            )
            trace.extend(expanded_trace)
            profiles = self._build_affordances(expanded_candidates, query, context, profile, intention, peers)
            ranked = self._rank_profiles(profiles)
            candidates = expanded_candidates
            trace.append(
                ToolCallRecord(
                    state="S7_REFLECTION_DONE",
                    tool="ReflectAndExpand",
                    reason="Reflection triggers required a broader candidate scan.",
                    observations=reflection.triggers + reflection.actions,
                )
            )

        ranked_pois = self._render_ranked_pois(ranked[:10], intention)
        trace.append(
            ToolCallRecord(
                state="S8_FINAL_OUTPUT",
                tool="RenderOutput",
                reason="Create final top-10 ranking with evidence and missing evidence.",
                observations=[f"{len(ranked_pois)} ranked POIs returned"],
            )
        )

        return AgentRunResult(
            query_id=query.traj_id,
            user_id=str(query.target["user_id"]),
            target_time=pd.Timestamp(query.target["local_time"]).isoformat(),
            ground_truth_poi_id=str(query.target["POI_id"]),
            dataset_capabilities=self.repo.capabilities,
            context_snapshot=context,
            user_profile=profile,
            inferred_intention=intention,
            tool_plan=plan,
            candidate_pool_summary=self._candidate_summary(candidates),
            ranked_pois=ranked_pois,
            reflection=reflection,
            agent_trace_summary=trace,
        )

    def _build_context(self, query: QueryExample) -> ContextSnapshot:
        context = query.context.sort_values("UTC_time").reset_index(drop=True)
        target = query.target
        recent = context.tail(5)
        last = context.iloc[-1]
        target_ts = pd.Timestamp(target["local_time"])
        last_ts = pd.Timestamp(last["local_time"])
        gap_minutes = max((target_ts - last_ts).total_seconds() / 60.0, 0.0)
        movement = 0.0
        if len(recent) >= 2:
            for a, b in zip(recent.iloc[:-1].itertuples(index=False), recent.iloc[1:].itertuples(index=False)):
                movement += haversine_km(float(a.latitude), float(a.longitude), float(b.latitude), float(b.longitude))
        movement_summary = "stationary or short local movement"
        if movement > 10:
            movement_summary = "long cross-city movement"
        elif movement > 2:
            movement_summary = "moderate neighborhood movement"
        return ContextSnapshot(
            query_id=query.traj_id,
            user_id=str(target["user_id"]),
            target_timestamp=target_ts.isoformat(),
            target_hour=int(target["hour"]),
            target_day_of_week=int(target["day_of_week"]),
            is_weekend=int(target["day_of_week"]) >= 5,
            time_of_day_bucket=time_bucket_label(int(target["hour"])),
            query_trajectory=self.repo.to_checkins(context),
            recent_poi_sequence=[str(x) for x in recent["POI_id"].tolist()],
            recent_category_sequence=[str(x) for x in recent["POI_catname"].tolist()],
            last_known_poi=str(last["POI_id"]),
            last_known_category=str(last["POI_catname"]),
            last_known_location={"latitude": float(last["latitude"]), "longitude": float(last["longitude"])},
            time_gap_since_last_checkin_minutes=float(gap_minutes),
            recent_spatial_movement_km=float(movement),
            movement_summary=movement_summary,
            dataset_capabilities=self.repo.capabilities,
        )

    def _build_user_profile(self, query: QueryExample) -> UserProfile:
        rows = self.repo.history_for_user(query.target["user_id"], query.context)
        if rows.empty:
            rows = query.context.copy()
        top_pois = [
            {"poi_id": str(k), "count": int(v)}
            for k, v in rows["POI_id"].value_counts().head(10).items()
        ]
        top_categories = [
            {"category": str(k), "count": int(v)}
            for k, v in rows["POI_catname"].value_counts().head(10).items()
        ]
        hourly = {str(int(k)): int(v) for k, v in rows["hour"].value_counts().sort_index().items()}
        days = {str(int(k)): int(v) for k, v in rows["day_of_week"].value_counts().sort_index().items()}
        cat_hour: dict[str, dict[str, int]] = {}
        for (cat, hb), count in rows.groupby(["POI_catname", "hour_bucket"]).size().items():
            cat_hour.setdefault(str(cat), {})[str(int(hb))] = int(count)
        distances = self._movement_distances(rows)
        typical = float(np.median(distances)) if distances else 1.5
        p75 = float(np.percentile(distances, 75)) if distances else max(typical, 2.0)
        visits = len(rows)
        unique = rows["POI_id"].nunique()
        revisit = 1.0 - safe_div(unique, visits, 1.0)
        transitions = self._user_category_transitions(rows)
        evidence = [
            f"User has {visits} visible historical check-ins.",
            f"Top category is {top_categories[0]['category'] if top_categories else 'unknown'}.",
            f"Median movement distance is {typical:.2f} km.",
        ]
        return UserProfile(
            user_id=str(query.target["user_id"]),
            num_checkins=int(visits),
            num_trajectories=int(rows["trajectory_id"].nunique()),
            top_pois=top_pois,
            top_categories=top_categories,
            hourly_distribution=hourly,
            day_distribution=days,
            category_hour_distribution=cat_hour,
            revisit_ratio=float(max(0.0, min(1.0, revisit))),
            exploration_ratio=float(max(0.0, min(1.0, 1.0 - revisit))),
            typical_movement_radius_km=max(typical, 0.25),
            p75_movement_radius_km=max(p75, 0.5),
            frequent_category_transitions=transitions,
            evidence_summary=evidence,
        )

    def _find_peer_users(self, query: QueryExample, profile: UserProfile) -> list[tuple[str, float]]:
        vectors, cells = self.repo.user_peer_inputs()
        user_id = str(query.target["user_id"])
        user_vec = vectors.get(user_id) or self._profile_vector_from_user_profile(profile)
        user_cells = cells.get(user_id, set())
        scored: list[tuple[str, float]] = []
        for other_id, vec in vectors.items():
            if other_id == user_id:
                continue
            sem = cosine_dict(user_vec, vec)
            other_cells = cells.get(other_id, set())
            union = len(user_cells | other_cells)
            geo = 0.0 if union == 0 else len(user_cells & other_cells) / union
            score = 0.5 * sem + 0.5 * geo
            if score > 0:
                scored.append((other_id, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.config.peer_users]

    def _infer_intention(
        self,
        context: ContextSnapshot,
        profile: UserProfile,
        peers: list[tuple[str, float]],
        query: QueryExample,
    ) -> Intention:
        heuristic = self._heuristic_intention(context, profile, peers, query)
        if self.config.llm_mode != "deepseek" or not self.llm.available:
            return heuristic
        prompt = {
            "context": context.model_dump(),
            "user_profile": profile.model_dump(),
            "allowed_categories": [x["category"] for x in profile.top_categories[:10]],
            "instruction": "Return a strict JSON Intention object. Use only available dataset evidence.",
        }
        data = self.llm.chat_json(
            [
                {
                    "role": "system",
                    "content": "You infer structured next-POI intention from Foursquare mobility evidence. Return JSON only.",
                },
                {"role": "user", "content": str(prompt)},
            ]
        )
        return parse_intention_or_none(data) or heuristic

    def _heuristic_intention(
        self,
        context: ContextSnapshot,
        profile: UserProfile,
        peers: list[tuple[str, float]],
        query: QueryExample,
    ) -> Intention:
        scores: dict[str, float] = defaultdict(float)
        target_hb = context.target_hour // 3
        for item in profile.top_categories:
            scores[item["category"]] += float(item["count"])
        for cat, by_hour in profile.category_hour_distribution.items():
            scores[cat] += 2.0 * by_hour.get(str(target_hb), 0)
        for cat in context.recent_category_sequence[-3:]:
            scores[cat] += 2.5
        for trans in profile.frequent_category_transitions[:10]:
            if trans["from_category"] == context.last_known_category:
                scores[trans["to_category"]] += 2.0 * trans["count"]
        peer_rows = self._peer_rows_near_target(peers, pd.Timestamp(query.target["local_time"]))
        for cat, count in peer_rows["POI_catname"].value_counts().head(10).items() if not peer_rows.empty else []:
            scores[str(cat)] += 1.5 * int(count)
        global_cats = self.repo.global_category_counts_at(context.target_hour // 3, context.target_day_of_week)
        for _, row in global_cats.head(8).iterrows():
            scores[str(row["POI_catname"])] += 0.05 * float(row["score"])
        normalized = normalize_scores(dict(scores))
        likely = [
            LikelyCategory(
                category=cat,
                weight=float(weight),
                evidence="Weighted from user category history, target time, recent trajectory, transitions, peers, and global popularity.",
            )
            for cat, weight in sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        if not likely and profile.top_categories:
            likely = [
                LikelyCategory(
                    category=profile.top_categories[0]["category"],
                    weight=1.0,
                    evidence="Fallback to user's top historical category.",
                )
            ]
        concentration = likely[0].weight if likely else 0.2
        history_factor = min(profile.num_checkins / 30.0, 1.0)
        context_factor = min(len(context.query_trajectory) / 5.0, 1.0)
        confidence = max(0.35, min(0.9, 0.35 + 0.35 * concentration + 0.2 * history_factor + 0.1 * context_factor))
        top = likely[0].category if likely else "unknown"
        goal = f"{context.time_of_day_bucket} {category_family(top)} activity"
        evidence = profile.evidence_summary + [
            f"Recent trajectory ends at {context.last_known_category}.",
            f"Target time is {context.target_hour}:00 on day {context.target_day_of_week}.",
        ]
        uncertainty = []
        if profile.num_checkins < 10:
            uncertainty.append("User history is sparse.")
        if len(likely) > 1 and likely[0].weight - likely[1].weight < 0.15:
            uncertainty.append("Multiple categories have similar support.")
        return Intention(
            summary=f"User is likely seeking a {goal} aligned with {top}.",
            activity_goal=goal,
            likely_categories=likely,
            spatial_preference={
                "anchor": "last_known_location",
                "preferred_radius_km": round(profile.p75_movement_radius_km, 3),
                "allow_long_distance": profile.p75_movement_radius_km > 5,
                "evidence": f"User p75 movement radius is {profile.p75_movement_radius_km:.2f} km.",
            },
            temporal_preference={
                "target_hour": context.target_hour,
                "target_day_of_week": context.target_day_of_week,
                "preferred_time_bucket": context.time_of_day_bucket,
                "evidence": "Derived from explicit target timestamp and user category-hour history.",
            },
            behavioral_preference={
                "revisit_tendency": round(profile.revisit_ratio, 3),
                "exploration_tendency": round(profile.exploration_ratio, 3),
                "peer_dependency": 0.7 if profile.num_checkins < 10 else 0.35,
            },
            confidence=float(confidence),
            evidence=evidence,
            uncertainty_reasons=uncertainty or ["Structured data lacks reviews, images, opening hours, price, and ratings."],
        )

    def _build_tool_plan(self, intention: Intention, context: ContextSnapshot, profile: UserProfile) -> ToolPlan:
        items = [
            ToolPlanItem(tool="HistoricalRecall", reason="User revisit behavior is a strong next-POI signal."),
            ToolPlanItem(
                tool="SpatialRecall",
                reason="Last known location is available.",
                params={"top_n": self.config.spatial_top_n, "anchor": context.last_known_poi},
            ),
            ToolPlanItem(
                tool="CategoryIntentRecall",
                reason="Inferred intention provides likely categories.",
                params={"categories": [x.category for x in intention.likely_categories]},
            ),
            ToolPlanItem(tool="TransitionRecall", reason="Recent POI/category transition can indicate next stop."),
            ToolPlanItem(tool="PeerRecall", reason="Peer behavior helps sparse or ambiguous contexts."),
        ]
        if profile.num_checkins < 10:
            items.append(ToolPlanItem(tool="TemporalPopularityRecall", reason="Sparse user fallback uses global temporal patterns."))
        return ToolPlan(items=items)

    def _retrieve_candidates(
        self,
        query: QueryExample,
        context: ContextSnapshot,
        profile: UserProfile,
        intention: Intention,
        peers: list[tuple[str, float]],
        expanded: bool,
    ) -> tuple[list[Candidate], list[ToolCallRecord]]:
        raw: dict[str, dict] = {}
        trace: list[ToolCallRecord] = []
        limit = self.config.candidate_pool_size * (3 if expanded else 2)
        spatial_top = 100 if expanded else self.config.spatial_top_n

        def add_candidate(poi_id: str, score: float, source: str, distance: float | None = None) -> None:
            meta = self.repo.poi_meta(poi_id, query.context)
            if not meta["category"] or meta["category"] == "Unknown":
                return
            if distance is None:
                distance = haversine_km(
                    context.last_known_location["latitude"],
                    context.last_known_location["longitude"],
                    meta["latitude"],
                    meta["longitude"],
                )
            item = raw.setdefault(
                poi_id,
                {
                    "poi_id": poi_id,
                    "display_name": meta["display_name"],
                    "category": meta["category"],
                    "latitude": meta["latitude"],
                    "longitude": meta["longitude"],
                    "distance_km": float(distance),
                    "source_scores": {},
                },
            )
            item["source_scores"][source] = max(float(score), item["source_scores"].get(source, 0.0))

        user_rows = self.repo.history_for_user(query.target["user_id"], query.context)
        target_hb = context.target_hour // 3
        hist_scores = self._historical_scores(user_rows, context)
        for poi_id, score in hist_scores[:limit]:
            add_candidate(poi_id, score, "historical")
        trace.append(_tool_record("S3_CANDIDATES_RETRIEVED", "HistoricalRecall", len(hist_scores[:limit])))

        nearest = self.repo.nearest_pois(
            context.last_known_location["latitude"],
            context.last_known_location["longitude"],
            limit=spatial_top,
            context=query.context,
        )
        for _, row in nearest.iterrows():
            add_candidate(str(row["POI_id"]), safe_div(10.0, max(float(row["distance_km"]), 0.05)), "spatial", float(row["distance_km"]))
        trace.append(_tool_record("S3_CANDIDATES_RETRIEVED", "SpatialRecall", len(nearest), {"top_n": spatial_top}))

        likely_categories = [x.category for x in intention.likely_categories]
        family_set = {category_family(x) for x in likely_categories}
        catalog = self.repo.runtime_catalog(query.context)
        cat_pool = catalog[
            catalog["category"].isin(likely_categories)
            | catalog["category"].map(category_family).isin(family_set)
        ].copy()
        if not cat_pool.empty:
            cat_pool["score"] = cat_pool["visit_count"].astype(float)
            for _, row in cat_pool.sort_values("score", ascending=False).head(self.config.category_top_n * (2 if expanded else 1)).iterrows():
                add_candidate(str(row["POI_id"]), float(row["score"]), "category_intent")
        trace.append(_tool_record("S3_CANDIDATES_RETRIEVED", "CategoryIntentRecall", len(cat_pool), {"categories": likely_categories}))

        trans_scores = self._transition_scores(context, user_rows)
        for poi_id, score in trans_scores[: self.config.transition_top_n * (2 if expanded else 1)]:
            add_candidate(poi_id, score, "transition")
        trace.append(_tool_record("S3_CANDIDATES_RETRIEVED", "TransitionRecall", len(trans_scores[: self.config.transition_top_n])))

        peer_rows = self._peer_rows_near_target(peers, pd.Timestamp(query.target["local_time"]))
        peer_counts = peer_rows["POI_id"].value_counts() if not peer_rows.empty else pd.Series(dtype=int)
        for poi_id, count in peer_counts.head(self.config.peer_top_n * (2 if expanded else 1)).items():
            add_candidate(str(poi_id), float(count), "peer")
        trace.append(_tool_record("S3_CANDIDATES_RETRIEVED", "PeerRecall", int(len(peer_counts))))

        temporal = self.repo.history[
            (self.repo.history["hour_bucket"] == target_hb)
            & (self.repo.history["day_of_week"] == context.target_day_of_week)
        ]["POI_id"].value_counts()
        for poi_id, count in temporal.head(30 if expanded else 15).items():
            add_candidate(str(poi_id), float(count), "temporal_popularity")

        candidates = self._select_candidates(raw, expanded=expanded)
        trace.append(
            ToolCallRecord(
                state="S4_CANDIDATES_FILTERED",
                tool="FilterCandidates",
                reason="Merge recall sources and keep highest prior-score candidates.",
                params={"B": self.config.candidate_pool_size, "expanded": expanded},
                observations=[f"raw={len(raw)}", f"filtered={len(candidates)}"],
            )
        )
        return candidates, trace

    def _select_candidates(self, raw: dict[str, dict], expanded: bool) -> list[Candidate]:
        source_weights = {
            "historical": 0.30,
            "spatial": 0.20,
            "category_intent": 0.20,
            "transition": 0.15,
            "temporal_popularity": 0.10,
            "peer": 0.05,
        }
        max_by_source: dict[str, float] = defaultdict(float)
        for item in raw.values():
            for source, score in item["source_scores"].items():
                max_by_source[source] = max(max_by_source[source], float(score))
        candidates: list[Candidate] = []
        for item in raw.values():
            prior = 0.0
            for source, score in item["source_scores"].items():
                prior += source_weights.get(source, 0.05) * safe_div(float(score), max_by_source.get(source, 1.0))
            item["prior_score"] = float(prior)
            item["source_labels"] = sorted(item["source_scores"])
            candidates.append(Candidate.model_validate(item))
        candidates.sort(key=lambda c: (c.prior_score, -c.distance_km), reverse=True)
        size = self.config.candidate_pool_size * (2 if expanded else 1)
        return candidates[:size]

    def _build_affordances(
        self,
        candidates: list[Candidate],
        query: QueryExample,
        context: ContextSnapshot,
        profile: UserProfile,
        intention: Intention,
        peers: list[tuple[str, float]],
    ) -> list[AffordanceProfile]:
        user_rows = self.repo.history_for_user(query.target["user_id"], query.context)
        peer_rows = self._peer_rows_near_target(peers, pd.Timestamp(query.target["local_time"]))
        profiles = [
            self._candidate_affordance(candidate, user_rows, peer_rows, context, profile, intention)
            for candidate in candidates
        ]
        return profiles

    def _candidate_affordance(
        self,
        candidate: Candidate,
        user_rows: pd.DataFrame,
        peer_rows: pd.DataFrame,
        context: ContextSnapshot,
        profile: UserProfile,
        intention: Intention,
    ) -> AffordanceProfile:
        verdicts = [
            self._category_match(candidate, intention),
            self._spatial_feasibility(candidate, profile),
            self._temporal_fit(candidate, user_rows, context),
            self._revisit_support(candidate, user_rows),
            self._transition_support(candidate, user_rows, context),
            self._peer_support(candidate, peer_rows),
            self._popularity_support(candidate, context),
            self._reachability(candidate, context),
        ]
        weights = self._alignment_weights(intention, profile)
        score_decomp: dict[str, float] = {}
        for verdict in verdicts:
            value = _verdict_value(verdict)
            score_decomp[verdict.name] = round(weights.get(verdict.name, 0.0) * value, 6)
        score = float(sum(score_decomp.values()))
        positive = [v.confidence for v in verdicts if v.answer == "yes"]
        conflicts = [v.conflict for v in verdicts if v.conflict]
        conf = min(0.95, max(0.2, (sum(positive) / len(positive) if positive else 0.3) * (0.85 if conflicts else 1.0)))
        return AffordanceProfile(
            poi_id=candidate.poi_id,
            display_name=candidate.display_name,
            category=candidate.category,
            distance_km=round(candidate.distance_km, 3),
            affordances=verdicts,
            missing_evidence=MISSING_EVIDENCE.copy(),
            conflicts=conflicts,
            score_decomposition=score_decomp,
            alignment_score=round(score, 6),
            confidence=round(conf, 6),
        )

    def _rank_profiles(self, profiles: list[AffordanceProfile]) -> list[AffordanceProfile]:
        ranked = sorted(profiles, key=lambda p: (p.alignment_score, p.confidence, -p.distance_km), reverse=True)
        return ranked

    def _maybe_reflect(
        self,
        ranked: list[AffordanceProfile],
        candidates: list[Candidate],
        intention: Intention,
        context: ContextSnapshot,
    ) -> ReflectionRecord:
        triggers: list[str] = []
        if len(candidates) < self.config.candidate_pool_size:
            triggers.append("candidate_pool_smaller_than_B")
        top_cats = {x.category for x in intention.likely_categories[:3]}
        candidate_cats = {c.category for c in candidates}
        if not top_cats & candidate_cats:
            triggers.append("top_intention_categories_uncovered")
        if len(ranked) >= 2 and ranked[0].alignment_score - ranked[1].alignment_score < 0.05:
            triggers.append("top_scores_close")
        if intention.confidence < 0.6:
            triggers.append("low_intention_confidence")
        cat_entropy = entropy([c.category for c in candidates])
        if candidates and cat_entropy < 1.0:
            triggers.append("candidate_category_entropy_low")
        if candidates and all(c.distance_km > 10 for c in candidates):
            triggers.append("all_candidates_far_from_last_location")
        if not triggers:
            return ReflectionRecord(triggered=False, stop_reason="No reflection trigger fired.")
        actions = ["expand_spatial_radius", "add_category_transition_peer_candidates", "rerank_with_affordances"]
        return ReflectionRecord(triggered=True, triggers=triggers, actions=actions, stop_reason="Completed one bounded reflection round.")

    def _render_ranked_pois(self, ranked: list[AffordanceProfile], intention: Intention) -> list[RankedPOI]:
        out = []
        for idx, profile in enumerate(ranked, 1):
            positive = [e for v in profile.affordances if v.answer == "yes" for e in v.evidence]
            uncertain = [e for v in profile.affordances if v.answer == "uncertain" for e in v.evidence]
            evidence = (positive + uncertain)[:5]
            if len(evidence) < 3:
                evidence += ["Recommendation is based only on structured mobility evidence."] * (3 - len(evidence))
            reason = (
                f"Matches the inferred {intention.activity_goal} with score {profile.alignment_score:.2f}; "
                f"main evidence: {evidence[0]}"
            )
            out.append(
                RankedPOI(
                    rank=idx,
                    poi_id=profile.poi_id,
                    display_name=profile.display_name,
                    category=profile.category,
                    distance_km=profile.distance_km,
                    alignment_score=profile.alignment_score,
                    confidence=profile.confidence,
                    reason=reason,
                    supporting_evidence=evidence,
                    missing_evidence=profile.missing_evidence,
                    conflicts=profile.conflicts,
                    score_decomposition=profile.score_decomposition,
                    affordance_profile=profile,
                )
            )
        return out

    def _candidate_summary(self, candidates: list[Candidate]) -> dict:
        source_counts = Counter(source for c in candidates for source in c.source_labels)
        return {
            "candidate_count": len(candidates),
            "source_counts": dict(source_counts),
            "category_counts": dict(Counter(c.category for c in candidates).most_common(20)),
            "avg_distance_km": round(float(np.mean([c.distance_km for c in candidates])) if candidates else 0.0, 3),
        }

    def _historical_scores(self, rows: pd.DataFrame, context: ContextSnapshot) -> list[tuple[str, float]]:
        if rows.empty:
            return []
        trans = self._user_poi_transition_lookup(rows, context.last_known_poi)
        target_hb = context.target_hour // 3
        scores = []
        for poi_id, group in rows.groupby("POI_id"):
            freq = len(group)
            same_day = int((group["day_of_week"] == context.target_day_of_week).sum())
            same_hb = int((group["hour_bucket"] == target_hb).sum())
            score = freq + 2 * same_day + same_hb + 0.5 * trans.get(str(poi_id), 0)
            scores.append((str(poi_id), float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _transition_scores(self, context: ContextSnapshot, user_rows: pd.DataFrame) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        user_poi = self._user_poi_transition_lookup(user_rows, context.last_known_poi)
        for poi_id, count in user_poi.items():
            scores[poi_id] += 3.0 * count
        global_poi = self.repo.global_poi_transitions()
        for (src, dst), count in global_poi.items():
            if src == context.last_known_poi:
                scores[dst] += count
        global_cat = self.repo.global_category_transitions()
        catalog = self.repo.runtime_catalog()
        by_cat = catalog.groupby("category")
        for (src, dst_cat), count in global_cat.items():
            if src == context.last_known_category and dst_cat in by_cat.groups:
                for poi_id in by_cat.get_group(dst_cat).sort_values("visit_count", ascending=False)["POI_id"].head(5):
                    scores[str(poi_id)] += 0.2 * count
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def _peer_rows_near_target(self, peers: list[tuple[str, float]], target_time: pd.Timestamp) -> pd.DataFrame:
        if not peers:
            return pd.DataFrame(columns=self.repo.history.columns)
        rows = self.repo.rows_near_target_time(target_time, self.config.peer_window_minutes)
        peer_ids = {uid for uid, _ in peers}
        return rows[rows["user_id"].isin(peer_ids)].copy()

    def _category_match(self, candidate: Candidate, intention: Intention) -> AffordanceVerdict:
        weights = {x.category: x.weight for x in intention.likely_categories}
        top_families = {category_family(x.category): x.weight for x in intention.likely_categories}
        if candidate.category in weights:
            confidence = min(0.95, 0.65 + weights[candidate.category] * 0.35)
            return _verdict("category_match", "Candidate category should match inferred intention.", "yes", confidence, [f"Candidate category {candidate.category} is in likely categories."])
        fam = category_family(candidate.category)
        if fam in top_families:
            return _verdict("category_match", "Candidate category family should match inferred intention.", "uncertain", 0.62, [f"Candidate category family {fam} matches an inferred category family."])
        return _verdict("category_match", "Candidate category should match inferred intention.", "no", 0.75, [f"Candidate category {candidate.category} is not supported by inferred intention."])

    def _spatial_feasibility(self, candidate: Candidate, profile: UserProfile) -> AffordanceVerdict:
        if candidate.distance_km <= profile.p75_movement_radius_km:
            return _verdict("spatial_feasibility", "Candidate should be within user's typical movement range.", "yes", 0.85, [f"Distance {candidate.distance_km:.2f} km is within p75 movement radius {profile.p75_movement_radius_km:.2f} km."])
        if candidate.distance_km <= max(profile.p75_movement_radius_km * 2, 3.0):
            return _verdict("spatial_feasibility", "Candidate should not be too far from last location.", "uncertain", 0.55, [f"Distance {candidate.distance_km:.2f} km is above typical radius but still plausible."])
        return _verdict("spatial_feasibility", "Candidate should be reachable from last location.", "no", 0.8, [f"Distance {candidate.distance_km:.2f} km exceeds usual movement range."])

    def _temporal_fit(self, candidate: Candidate, rows: pd.DataFrame, context: ContextSnapshot) -> AffordanceVerdict:
        target_hb = context.target_hour // 3
        user_same_cat_hour = rows[(rows["POI_catname"] == candidate.category) & (rows["hour_bucket"] == target_hb)]
        user_same_poi_hour = rows[(rows["POI_id"] == candidate.poi_id) & (rows["hour_bucket"] == target_hb)]
        global_same = self.repo.history[(self.repo.history["POI_id"] == candidate.poi_id) & (self.repo.history["hour_bucket"] == target_hb)]
        if len(user_same_poi_hour) > 0:
            return _verdict("temporal_fit", "POI should fit target time.", "yes", 0.85, [f"User visited this POI {len(user_same_poi_hour)} times in the same hour bucket."])
        if len(user_same_cat_hour) > 0:
            return _verdict("temporal_fit", "Category should fit target time.", "yes", 0.75, [f"User visited category {candidate.category} {len(user_same_cat_hour)} times in the same hour bucket."])
        if len(global_same) > 0:
            return _verdict("temporal_fit", "POI should have historical activity near target time.", "uncertain", 0.55, [f"Global history contains {len(global_same)} same-hour-bucket visits for this POI."])
        return _verdict("temporal_fit", "POI/category should fit target time.", "uncertain", 0.35, ["No strong same-hour evidence; temporal fit is weak."])

    def _revisit_support(self, candidate: Candidate, rows: pd.DataFrame) -> AffordanceVerdict:
        poi_count = int((rows["POI_id"] == candidate.poi_id).sum())
        cat_count = int((rows["POI_catname"] == candidate.category).sum())
        if poi_count:
            return _verdict("revisit_support", "User may revisit known POIs.", "yes", 0.9, [f"User visited this POI {poi_count} times before."])
        if cat_count:
            return _verdict("revisit_support", "User may revisit same-style POIs.", "uncertain", 0.62, [f"User visited category {candidate.category} {cat_count} times before."])
        return _verdict("revisit_support", "User may need exploration evidence for unseen POIs.", "no", 0.55, ["User has no visible history with this POI or category."])

    def _transition_support(self, candidate: Candidate, rows: pd.DataFrame, context: ContextSnapshot) -> AffordanceVerdict:
        user_cat = self._user_category_transition_lookup(rows)
        global_cat = self.repo.global_category_transitions()
        user_count = user_cat.get((context.last_known_category, candidate.category), 0)
        global_count = global_cat.get((context.last_known_category, candidate.category), 0)
        if user_count:
            return _verdict("transition_support", "Recent category transition should be plausible.", "yes", 0.82, [f"User made {user_count} transitions from {context.last_known_category} to {candidate.category}."])
        if global_count:
            return _verdict("transition_support", "Recent category transition should be globally plausible.", "uncertain", 0.58, [f"Global data has {global_count} transitions from {context.last_known_category} to {candidate.category}."])
        return _verdict("transition_support", "Recent transition should be supported.", "uncertain", 0.32, ["No direct transition evidence for this category pair."])

    def _peer_support(self, candidate: Candidate, peer_rows: pd.DataFrame) -> AffordanceVerdict:
        if peer_rows.empty:
            return _verdict("peer_support", "Similar users can support sparse contexts.", "not_available", 0.0, [], ["No peer visits in target time window."])
        poi_count = int((peer_rows["POI_id"] == candidate.poi_id).sum())
        cat_count = int((peer_rows["POI_catname"] == candidate.category).sum())
        if poi_count:
            return _verdict("peer_support", "Peers should support candidate POI.", "yes", 0.78, [f"Similar users visited this POI {poi_count} times near target time."])
        if cat_count:
            return _verdict("peer_support", "Peers should support candidate category.", "uncertain", 0.55, [f"Similar users visited category {candidate.category} {cat_count} times near target time."])
        return _verdict("peer_support", "Peers should support candidate.", "no", 0.5, ["No peer support for this POI or category near target time."])

    def _popularity_support(self, candidate: Candidate, context: ContextSnapshot) -> AffordanceVerdict:
        same_hb = self.repo.history[
            (self.repo.history["POI_id"] == candidate.poi_id)
            & (self.repo.history["hour_bucket"] == context.target_hour // 3)
        ]
        cat_same_hb = self.repo.history[
            (self.repo.history["POI_catname"] == candidate.category)
            & (self.repo.history["hour_bucket"] == context.target_hour // 3)
        ]
        if len(same_hb) >= 3:
            return _verdict("popularity_support", "POI should be historically active at target time.", "yes", 0.72, [f"This POI has {len(same_hb)} global visits in the same hour bucket."])
        if len(cat_same_hb) >= 20:
            return _verdict("popularity_support", "Category should be historically active at target time.", "uncertain", 0.58, [f"Category {candidate.category} has {len(cat_same_hb)} global visits in the same hour bucket."])
        return _verdict("popularity_support", "POI/category should have global support.", "uncertain", 0.35, ["Weak global temporal popularity evidence."])

    def _reachability(self, candidate: Candidate, context: ContextSnapshot) -> AffordanceVerdict:
        gap_h = context.time_gap_since_last_checkin_minutes / 60.0
        if gap_h <= 0:
            return _verdict("reachability_time_gap", "Target time should be after last check-in.", "uncertain", 0.3, ["No positive time gap available."])
        feasible = max(0.5, gap_h * 12.0)
        if candidate.distance_km <= feasible:
            return _verdict("reachability_time_gap", "Distance should be reachable before target time.", "yes", 0.82, [f"{candidate.distance_km:.2f} km is reachable within {context.time_gap_since_last_checkin_minutes:.0f} minutes under urban mobility assumptions."])
        if candidate.distance_km <= feasible * 2:
            return _verdict("reachability_time_gap", "Distance may be reachable with faster transit.", "uncertain", 0.5, [f"{candidate.distance_km:.2f} km may require fast transit within {context.time_gap_since_last_checkin_minutes:.0f} minutes."])
        return _verdict("reachability_time_gap", "Distance should be reachable before target time.", "no", 0.78, [f"{candidate.distance_km:.2f} km is unlikely within {context.time_gap_since_last_checkin_minutes:.0f} minutes."])

    def _alignment_weights(self, intention: Intention, profile: UserProfile) -> dict[str, float]:
        weights = {
            "category_match": 0.22,
            "temporal_fit": 0.16,
            "spatial_feasibility": 0.14,
            "revisit_support": 0.14,
            "transition_support": 0.12,
            "peer_support": 0.08,
            "popularity_support": 0.08,
            "reachability_time_gap": 0.06,
        }
        if profile.num_checkins < 10 or intention.confidence < 0.6:
            weights["peer_support"] += 0.04
            weights["popularity_support"] += 0.03
            weights["revisit_support"] -= 0.04
            weights["category_match"] -= 0.03
        if profile.revisit_ratio > 0.65:
            weights["revisit_support"] += 0.04
            weights["peer_support"] -= 0.02
            weights["popularity_support"] -= 0.02
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _movement_distances(self, rows: pd.DataFrame) -> list[float]:
        distances: list[float] = []
        for _, group in rows.groupby("trajectory_id", sort=False):
            ordered = group.sort_values("UTC_time")
            for a, b in zip(ordered.iloc[:-1].itertuples(index=False), ordered.iloc[1:].itertuples(index=False)):
                distances.append(haversine_km(float(a.latitude), float(a.longitude), float(b.latitude), float(b.longitude)))
        return distances

    def _user_category_transitions(self, rows: pd.DataFrame) -> list[dict]:
        counts = self._user_category_transition_lookup(rows)
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
        return [{"from_category": a, "to_category": b, "count": int(c)} for (a, b), c in items]

    def _user_category_transition_lookup(self, rows: pd.DataFrame) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        if rows.empty:
            return counts
        for _, group in rows.groupby("trajectory_id", sort=False):
            vals = [str(v) for v in group.sort_values("UTC_time")["POI_catname"].tolist()]
            for a, b in zip(vals, vals[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    def _user_poi_transition_lookup(self, rows: pd.DataFrame, last_poi: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        if rows.empty:
            return counts
        for _, group in rows.groupby("trajectory_id", sort=False):
            vals = [str(v) for v in group.sort_values("UTC_time")["POI_id"].tolist()]
            for a, b in zip(vals, vals[1:]):
                if a == last_poi:
                    counts[b] = counts.get(b, 0) + 1
        return counts

    def _profile_vector_from_user_profile(self, profile: UserProfile) -> dict[str, float]:
        return {
            f"{cat}|{hb}": float(count)
            for cat, by_hour in profile.category_hour_distribution.items()
            for hb, count in by_hour.items()
        }


def _verdict(
    name: str,
    requirement: str,
    answer: str,
    confidence: float,
    evidence: list[str] | None = None,
    missing: list[str] | None = None,
    conflict: str | None = None,
) -> AffordanceVerdict:
    return AffordanceVerdict(
        name=name,
        requirement=requirement,
        answer=answer,  # type: ignore[arg-type]
        confidence=round(float(confidence), 6),
        evidence=evidence or [],
        source_tools=[f"Check{''.join(part.title() for part in name.split('_'))}"],
        missing_evidence=missing or [],
        conflict=conflict,
    )


def _verdict_value(verdict: AffordanceVerdict) -> float:
    if verdict.answer == "yes":
        return verdict.confidence
    if verdict.answer == "uncertain":
        return 0.5 * verdict.confidence
    return 0.0


def _tool_record(state: str, tool: str, count: int, params: dict | None = None) -> ToolCallRecord:
    return ToolCallRecord(
        state=state,
        tool=tool,
        reason=f"Run {tool} as part of bounded tool plan.",
        params=params or {},
        observations=[f"{count} records/candidates observed"],
    )

