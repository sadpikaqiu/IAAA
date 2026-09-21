"""Read-only POI tools. No evaluation target is accepted by this module."""
from __future__ import annotations

import copy
import json

import pandas as pd

from .agent_types import (VisibleQuery, RecallArgs, PageArgs, InspectArgs, ReadArgs, SearchArgs, ToolRequest)
from .engine import IAAAgent, RunConfig
from .evidence import MODES, POLICY
from .utils import category_family, haversine_km


TOOL_HELP = """All poi_ids are compact P000001 identifiers, never original Foursquare IDs.
get_history(offset=0,limit=20): visible user visits, newest first.
recall_pois(source,limit=20,offset=0,radius_km=10,categories=[],anchor_poi_idx=null):
 sources historical, spatial, category, transition, peer, temporal. Max limit 50.
 category needs 1-5 category names (exact names or same category family).
 spatial/category radius 1-50km; spatial anchor must be a visible historical POI.
search_evidence(query,mode='both',radius_km=10,limit=20,offset=0): lexical TF-IDF search,
 mode both/images/reviews. Scores measure lexical relevance, not visit probability.
read_evidence(poi_ids,mode='both',query=''): at most 5 known POIs, one excerpt per modality.
inspect_candidates(poi_ids): at most 10 known POIs; factual mobility features, not recommendation scores.
list_candidates(offset=0,limit=20): revisit registered candidates in discovery order.
Search/recall results register candidates. Inspection may register visible historical POIs.
Evidence tools are unavailable in trajectory-only mode. They cannot browse or call vision APIs.
""".strip()


class POIToolService:
    def __init__(self, repo, query: VisibleQuery, evidence=None):
        self.query, self.repo, self.evidence = query, repo, evidence
        self.safe_query = query.legacy_context_query()
        self.helper = IAAAgent(repo, RunConfig.p4())
        self.context = self.helper._build_context(self.safe_query)
        self.profile = self.helper._build_user_profile(self.safe_query)
        self.rows = self.helper._history_for_query(self.safe_query)
        self.catalog = repo.runtime_catalog(query.context, include_unvisited=True)
        self.meta = {str(r.POI_idx): r._asdict() for r in self.catalog.itertuples(index=False)}
        self.by_original = {str(v["POI_id"]): k for k, v in self.meta.items()}
        self.known_history = {self.by_original[str(p)] for p in self.rows["POI_id"] if str(p) in self.by_original}
        self.registry: dict[str, dict] = {}
        self.references: dict[str, dict] = {}
        self.evidence_refs: dict[str, list[str]] = {}
        self.observed_ids: set[str] = set()
        self.tool_cache: dict[str, dict] = {}
        self.fact_cache: dict[str, dict] = {}
        self._peers = None
        self._peer_counts = None
        self._evidence_views = {}
        self.user_counts = self.rows["POI_id"].value_counts().to_dict()
        self.hour_counts = self.rows[self.rows["hour_bucket"] == self.context.target_hour // 3]["POI_id"].value_counts().to_dict()
        self.day_counts = self.rows[self.rows["day_of_week"] == self.context.target_day_of_week]["POI_id"].value_counts().to_dict()
        self.temporal_counts = repo.history[(repo.history["hour_bucket"] == self.context.target_hour // 3)
                                           & (repo.history["day_of_week"] == self.context.target_day_of_week)]["POI_id"].value_counts()
        self.user_transitions = self.helper._user_poi_transition_lookup(self.rows, self.context.last_known_poi)
        self.global_transitions = repo.global_poi_transitions()
        self.distances = {idx: self._distance(row) for idx, row in self.meta.items()}

    def _distance(self, row, anchor=None):
        anchor = anchor or self.context.last_known_location
        return haversine_km(float(anchor["latitude"]), float(anchor["longitude"]),
                            float(row["latitude"]), float(row["longitude"]))

    def initial_context(self):
        p = self.profile
        return {
            "query_id": self.query.query_id, "target_time": self.query.target_time,
            "weekday": self.context.target_day_of_week,
            "last_location": self.context.last_known_location,
            "minutes_since_last_visit": self.context.time_gap_since_last_checkin_minutes,
            "equal_timestamp_context_events": self.query.equal_time_context_count,
            "context_order_policy": "provided_session_sequence; equal timestamps do not establish physical event order",
            "recent_visits": self._visits(self.query.context.tail(5)),
            "profile": {"visible_visits": p.num_checkins, "top_pois": p.top_pois,
                        "top_categories": p.top_categories, "hours": p.hourly_distribution,
                        "weekdays": p.day_distribution, "revisit_ratio": p.revisit_ratio,
                        "median_movement_km": p.typical_movement_radius_km,
                        "p75_movement_km": p.p75_movement_radius_km,
                        "category_transitions": p.frequent_category_transitions[:10]},
            "external_evidence": self.evidence.mode if self.evidence else "none",
            "knowledge_policy": POLICY,
        }

    def _visits(self, rows):
        return [{"poi_idx": self.by_original[str(r.POI_id)], "category": str(r.POI_catname),
                 "time": str(r.local_time)} for r in rows.itertuples(index=False)]

    def _known(self, idx):
        if idx not in self.registry and idx not in self.known_history:
            raise ValueError(f"POI {idx} has not been observed; use recall/search first")
        if idx not in self.meta:
            raise ValueError("Unknown catalog POI")

    def fact(self, idx):
        if idx not in self.fact_cache:
            m = self.meta[idx]
            poi = str(m["POI_id"])
            ref = f"F{idx[1:]}"
            facts = {"poi_idx": idx, "category": str(m["category"]),
                     "distance_km": round(self.distances[idx], 3),
                     "user_visits": int(self.user_counts.get(poi, 0)),
                     "same_hour_bucket_visits": int(self.hour_counts.get(poi, 0)),
                     "same_weekday_visits": int(self.day_counts.get(poi, 0)),
                     "user_transition_count": int(self.user_transitions.get(poi, 0)),
                     "global_transition_count": int(self.global_transitions.get((self.context.last_known_poi, poi), 0)),
                     "global_target_time_visits": int(self.temporal_counts.get(poi, 0)),
                     "catalog_visible_visits": int(m["visit_count"]),
                     "has_review": bool(self.evidence and self.evidence.has(poi, "review")),
                     "has_image": bool(self.evidence and self.evidence.has(poi, "image")), "ref": ref}
            self.fact_cache[idx] = facts
            self.references[ref] = {"kind": "mobility_facts", "poi_idx": idx,
                                    "facts": facts, "history_policy": self.repo._active_history_mode}
        return self.fact_cache[idx]

    def register(self, idx, source, score=None):
        if idx not in self.meta or self.meta[idx]["category"] in {"", "Unknown"}:
            return None
        item = self.registry.setdefault(idx, {"poi_idx": idx, "sources": [], "source_scores": {}})
        if source not in item["sources"]:
            item["sources"].append(source)
        if score is not None:
            item["source_scores"][source] = float(score)
        self.fact(idx)
        self.observed_ids.add(idx)
        return self.card(idx)

    def card(self, idx, *, detailed=False):
        f = self.fact(idx)
        keys = (list(f) if detailed else ["poi_idx", "category", "distance_km", "user_visits",
                                           "user_transition_count", "ref"])
        return {**{k: f[k] for k in keys}, "sources": self.registry.get(idx, {}).get("sources", [])}

    def evidence_view(self, mode):
        if self.evidence is None:
            raise ValueError("External evidence is unavailable in trajectory-only mode")
        if mode == self.evidence.mode:
            return self.evidence
        if mode in self._evidence_views:
            return self._evidence_views[mode]
        modalities = MODES[mode] & self.evidence.modalities
        view = copy.copy(self.evidence)
        view.modalities = modalities
        view.by_poi = {p: [i for i in ids if self.evidence.items[i][1]["modality"] in modalities]
                       for p, ids in self.evidence.by_poi.items()}
        self._evidence_views[mode] = view
        return view

    def read_items(self, idx, mode="both", query=""):
        self._known(idx)
        view = self.evidence_view(mode)
        poi = str(self.meta[idx]["POI_id"])
        scores = view.score(query) if query else None
        out = []
        for item in view.get(poi, scores, per_modality=1, max_chars=400):
            # Stable aliases are local to this query, with the complete source in the ledger.
            existing = next((k for k, v in self.references.items() if v.get("source_id") == item["id"]), None)
            ref = existing or f"E{sum(v.get('kind') == 'external_evidence' for v in self.references.values()) + 1:05d}"
            self.references[ref] = {"kind": "external_evidence", "poi_idx": idx,
                                    "source_id": item["id"], "snapshot_id": self.evidence.snapshot_id, **item}
            ids = self.evidence_refs.setdefault(idx, [])
            if ref not in ids:
                ids.append(ref)
            out.append({"ref": ref, "modality": item["modality"], "text": item["text"],
                        "observed_at": item.get("observed_at"), "excerpt_truncated": item["excerpt_truncated"]})
        return {"poi_idx": idx, "evidence": out, "missing": [m for m in sorted(view.modalities) if not view.has(poi, m)]}

    def _peer(self):
        if self._peer_counts is None:
            self._peers = self.helper._find_peer_users(self.safe_query, self.profile)
            rows = self.helper._peer_rows_near_target(self._peers, pd.Timestamp(self.query.target_time))
            self._peer_counts = rows["POI_id"].value_counts() if not rows.empty else pd.Series(dtype=int)
        return self._peer_counts

    def recall(self, a: RecallArgs):
        source = a.source
        if a.anchor_poi_idx is not None and a.anchor_poi_idx not in self.known_history:
            raise ValueError("Spatial anchor must be a visible historical POI")
        if source == "historical":
            scored = self.helper._historical_scores(self.rows, self.context)
        elif source == "transition":
            scored = self.helper._transition_scores(self.context, self.rows)
        elif source == "peer":
            scored = list(self._peer().items())
        elif source == "temporal":
            scored = list(self.temporal_counts.items())
        else:
            anchor = self.meta[a.anchor_poi_idx] if a.anchor_poi_idx else None
            families = {category_family(c) for c in a.categories}
            if source == "category" and not a.categories:
                raise ValueError("Category recall requires categories")
            scored = []
            for idx, m in self.meta.items():
                if source == "category" and m["category"] not in a.categories and category_family(m["category"]) not in families:
                    continue
                d = self._distance(m, anchor) if anchor else self.distances[idx]
                if d > a.radius_km:
                    continue
                score = (float(m["visit_count"]) + 5 / max(d, .05)) if source == "category" else 10 / max(d, .05)
                scored.append((str(m["POI_id"]), score))
        scored = sorted(scored, key=lambda x: (-float(x[1]), str(x[0])))
        eligible = [(self.by_original[str(p)], float(s)) for p, s in scored if str(p) in self.by_original]
        rows = [self.register(idx, source, score) for idx, score in eligible[a.offset:a.offset + a.limit]]
        return {"candidates": [x for x in rows if x], "available": len(eligible),
                "next_offset": a.offset + len(rows) if a.offset + len(rows) < len(eligible) else None,
                "score_policy": "source-local retrieval order, never a combined recommendation score"}

    def search(self, a: SearchArgs):
        view = self.evidence_view(a.mode)
        scores = view.score(a.query)
        rows = []
        for poi, relevance in view.search(scores, self.by_original):
            idx = self.by_original[poi]
            d = self.distances[idx]
            if d <= a.radius_km:
                rows.append((idx, relevance / (1 + d / a.radius_km)))
        rows.sort(key=lambda x: (-x[1], x[0]))
        selected = rows[a.offset:a.offset + a.limit]
        result = []
        for idx, score in selected:
            card = self.register(idx, "evidence_" + a.mode, score)
            if card:
                card["lexical_relevance"] = round(score, 5)
                result.append(card)
        return {"candidates": result, "available": len(rows), "query": a.query,
                "next_offset": a.offset + len(selected) if a.offset + len(selected) < len(rows) else None,
                "method": "char_ngram_tfidf_lexical", "snapshot_id": self.evidence.snapshot_id}

    def execute(self, request: ToolRequest):
        key = json.dumps(request.model_dump(), sort_keys=True, ensure_ascii=False)
        if key in self.tool_cache:
            return {**copy.deepcopy(self.tool_cache[key]), "cached": True, "new_candidate_count": 0}
        before = len(self.registry)
        name, args = request.name, request.args
        if name == "recall_pois":
            result = self.recall(RecallArgs.model_validate(args))
        elif name == "search_evidence":
            result = self.search(SearchArgs.model_validate(args))
        elif name == "get_history":
            a = PageArgs.model_validate(args)
            rows = self.rows.iloc[::-1].iloc[a.offset:a.offset + a.limit]
            result = {"visits": self._visits(rows), "available": len(self.rows),
                      "next_offset": a.offset + len(rows) if a.offset + len(rows) < len(self.rows) else None}
        elif name == "list_candidates":
            a = PageArgs.model_validate(args)
            ids = list(self.registry)[a.offset:a.offset + a.limit]
            result = {"candidates": [self.card(i) for i in ids], "available": len(self.registry),
                      "next_offset": a.offset + len(ids) if a.offset + len(ids) < len(self.registry) else None}
        elif name == "inspect_candidates":
            a = InspectArgs.model_validate(args)
            for idx in a.poi_ids:
                self._known(idx)
            for idx in a.poi_ids:
                self.register(idx, "inspection")
            result = {"candidates": [self.card(idx, detailed=True) for idx in a.poi_ids]}
        elif name == "read_evidence":
            a = ReadArgs.model_validate(args)
            for idx in a.poi_ids:
                self._known(idx)
            result = {"pois": [self.read_items(idx, a.mode, a.query) for idx in a.poi_ids], "knowledge_policy": POLICY}
        else:
            raise ValueError("Unknown tool")
        result.update(new_candidate_count=len(self.registry) - before, cached=False)
        self.tool_cache[key] = copy.deepcopy(result)
        return result
