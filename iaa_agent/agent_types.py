"""Contracts for target-blind, bounded next-POI agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data import QueryExample


@dataclass(frozen=True)
class VisibleQuery:
    query_id: str
    user_id: str
    trajectory_id: str
    target_time: str
    target_utc: str
    context: pd.DataFrame
    history: pd.DataFrame
    equal_time_context_count: int = 0

    @classmethod
    def from_query(cls, query: QueryExample, repo=None) -> "VisibleQuery":
        target = query.target
        history = query.history
        if history is None:
            if repo is None:
                raise ValueError("An explicit visible history is required")
            history = repo.history_for_user(str(target["user_id"]))
        stamp = pd.Timestamp(target["UTC_time"])
        if query.context.empty:
            raise ValueError("Empty visible context")
        if (query.context["UTC_time"] > stamp).any() or (history["UTC_time"] >= stamp).any():
            raise ValueError("History must be strictly before target time; context must not follow it")
        return cls(query.traj_id, str(target["user_id"]), str(target["trajectory_id"]),
                   str(target["local_time"]), str(stamp), query.context.copy(), history.copy(),
                   int((query.context["UTC_time"] == stamp).sum()))

    def legacy_context_query(self) -> QueryExample:
        """Adapter for audited statistical helpers; deliberately has no POI label."""
        stamp = pd.Timestamp(self.target_time)
        return QueryExample(self.query_id, self.context, pd.Series({
            "user_id": self.user_id, "trajectory_id": self.trajectory_id,
            "local_time": stamp, "UTC_time": pd.Timestamp(self.target_utc),
            "hour": stamp.hour, "day_of_week": stamp.dayofweek,
        }), mode="visible_query", history=self.history)


@dataclass(frozen=True)
class AgentConfig:
    engine: str = "autonomous"
    max_decisions: int = 6
    max_tool_calls: int = 12
    max_tools_per_decision: int = 3
    max_candidates: int = 60
    top_k: int = 10
    prompt_tokens: int = 11500
    decision_tokens: int = 2048
    ranking_tokens: int = 4096
    retry_budget: int = 2
    request_timeout: float = 180
    case_timeout: float = 1200

    def __post_init__(self):
        if self.engine not in {"autonomous", "fixed_schedule", "fixed_llm_rank"}:
            raise ValueError("Unknown new engine")
        if not 1 <= self.top_k <= self.max_candidates <= 60:
            raise ValueError("Invalid candidate limits")
        if not 2 <= self.max_decisions <= 6 or not 1 <= self.max_tool_calls <= 12:
            raise ValueError("Invalid decision/tool budget")
        if not 1 <= self.max_tools_per_decision <= 3 or not 0 <= self.retry_budget <= 2:
            raise ValueError("Invalid call/repair budget")
        if min(self.request_timeout, self.case_timeout, self.prompt_tokens,
               self.decision_tokens, self.ranking_tokens) <= 0:
            raise ValueError("Budgets must be positive")
        if self.prompt_tokens + max(self.decision_tokens, self.ranking_tokens) > 16384:
            raise ValueError("Generation exceeds model context length")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def validate_distinct_pois(ids, field):
    positions = {}
    for position, poi in enumerate(ids, 1):
        positions.setdefault(poi, []).append(position)
    duplicates = {poi: indexes for poi, indexes in positions.items() if len(indexes) > 1}
    if duplicates:
        details = "; ".join(f"{poi} at {indexes}" for poi, indexes in list(duplicates.items())[:8])
        raise ValueError(f"{field}: {len(ids)} entries but only {len(positions)} distinct POIs. "
                         f"Duplicate IDs (1-based positions): {details}. "
                         "Each POI may appear only once; choose additional allowed POIs if needed.")


class AgentIntention(StrictModel):
    goal: str = Field(min_length=1, max_length=300)
    categories: list[str] = Field(default_factory=list, max_length=5)
    uncertainty: list[str] = Field(default_factory=list, max_length=5)


class ToolRequest(StrictModel):
    name: Literal["get_history", "recall_pois", "search_evidence", "read_evidence",
                  "inspect_candidates", "list_candidates"]
    args: dict = Field(default_factory=dict)


class AgentDecision(StrictModel):
    intention: AgentIntention
    working_poi_ids: list[str] = Field(default_factory=list, max_length=60)
    tools: list[ToolRequest] = Field(default_factory=list, max_length=3)
    stop: bool
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def consistent_stop(self):
        if self.stop and self.tools:
            raise ValueError("A stop decision cannot request tools")
        validate_distinct_pois(self.working_poi_ids, "working_poi_ids")
        return self


class FixedScheduleDecision(StrictModel):
    intention: AgentIntention
    working_poi_ids: list[str] = Field(default_factory=list, max_length=60)
    tool_args: list[dict] = Field(max_length=3)
    reason: str = Field(min_length=1, max_length=500)


Verdict = Literal["yes", "no", "uncertain", "not_available"]


class CoreAffordances(StrictModel):
    category: Verdict
    spatial: Verdict
    temporal: Verdict
    revisit: Verdict
    transition: Verdict


class AgentRankedPOI(StrictModel):
    poi_idx: str
    reason: str = Field(min_length=1, max_length=300)
    affordances: CoreAffordances
    evidence_refs: list[str] = Field(min_length=1, max_length=8)
    missing_evidence: list[str] = Field(default_factory=list, max_length=5)
    conflicts: list[str] = Field(default_factory=list, max_length=5)


class AgentRanking(StrictModel):
    ranked_pois: list[AgentRankedPOI] = Field(min_length=1, max_length=10)


class RecallArgs(StrictModel):
    source: Literal["historical", "spatial", "category", "transition", "peer", "temporal"]
    limit: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0, le=1000)
    radius_km: float = Field(default=10, ge=1, le=50, allow_inf_nan=False)
    categories: list[str] = Field(default_factory=list, max_length=5)
    anchor_poi_idx: str | None = None


class PageArgs(StrictModel):
    offset: int = Field(default=0, ge=0, le=10000)
    limit: int = Field(default=20, ge=1, le=50)


class InspectArgs(StrictModel):
    poi_ids: list[str] = Field(min_length=1, max_length=10)


class ReadArgs(StrictModel):
    poi_ids: list[str] = Field(min_length=1, max_length=5)
    mode: Literal["both", "images", "reviews"] = "both"
    query: str = Field(default="", max_length=1000)


class SearchArgs(StrictModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["both", "images", "reviews"] = "both"
    radius_km: float = Field(default=10, ge=1, le=50, allow_inf_nan=False)
    limit: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0, le=1000)
