from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


VerdictAnswer = Literal["yes", "no", "uncertain", "not_available"]


class DatasetCapabilities(BaseModel):
    has_reviews: bool = False
    has_images: bool = False
    has_opening_hours: bool = False
    has_price: bool = False
    has_ratings: bool = False
    has_category: bool = True
    has_coordinates: bool = True
    has_timestamps: bool = True
    has_trajectory_id: bool = True
    notes: list[str] = Field(default_factory=list)


class CheckIn(BaseModel):
    user_id: str
    poi_id: str
    category: str
    latitude: float
    longitude: float
    utc_time: str
    local_time: str
    day_of_week: int
    hour: int
    hour_bucket: int
    trajectory_id: str


class ContextSnapshot(BaseModel):
    query_id: str
    user_id: str
    target_timestamp: str
    target_hour: int
    target_day_of_week: int
    is_weekend: bool
    time_of_day_bucket: str
    query_trajectory: list[CheckIn]
    recent_poi_sequence: list[str]
    recent_category_sequence: list[str]
    last_known_poi: str
    last_known_category: str
    last_known_location: dict[str, float]
    time_gap_since_last_checkin_minutes: float
    recent_spatial_movement_km: float
    movement_summary: str
    dataset_capabilities: DatasetCapabilities


class UserProfile(BaseModel):
    user_id: str
    num_checkins: int
    num_trajectories: int
    top_pois: list[dict]
    top_categories: list[dict]
    hourly_distribution: dict[str, int]
    day_distribution: dict[str, int]
    category_hour_distribution: dict[str, dict[str, int]]
    revisit_ratio: float
    exploration_ratio: float
    typical_movement_radius_km: float
    p75_movement_radius_km: float
    frequent_category_transitions: list[dict]
    evidence_summary: list[str]


class LikelyCategory(BaseModel):
    category: str
    weight: float
    evidence: str


class Intention(BaseModel):
    summary: str
    activity_goal: str
    likely_categories: list[LikelyCategory]
    spatial_preference: dict
    temporal_preference: dict
    behavioral_preference: dict
    confidence: float
    evidence: list[str]
    uncertainty_reasons: list[str]


class ToolPlanItem(BaseModel):
    tool: str
    reason: str
    params: dict = Field(default_factory=dict)


class ToolPlan(BaseModel):
    items: list[ToolPlanItem]


class ToolCallRecord(BaseModel):
    state: str
    tool: str
    reason: str
    params: dict = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    poi_id: str
    display_name: str
    category: str
    latitude: float
    longitude: float
    distance_km: float
    source_scores: dict[str, float] = Field(default_factory=dict)
    source_labels: list[str] = Field(default_factory=list)
    prior_score: float = 0.0


class AffordanceVerdict(BaseModel):
    name: str
    requirement: str
    answer: VerdictAnswer
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    source_tools: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    conflict: str | None = None


class AffordanceProfile(BaseModel):
    poi_id: str
    display_name: str
    category: str
    distance_km: float
    affordances: list[AffordanceVerdict]
    missing_evidence: list[str]
    conflicts: list[str] = Field(default_factory=list)
    score_decomposition: dict[str, float]
    alignment_score: float
    confidence: float


class RankedPOI(BaseModel):
    rank: int
    poi_id: str
    display_name: str
    category: str
    distance_km: float
    alignment_score: float
    confidence: float
    reason: str
    supporting_evidence: list[str]
    missing_evidence: list[str]
    conflicts: list[str]
    score_decomposition: dict[str, float]
    affordance_profile: AffordanceProfile


class ReflectionRecord(BaseModel):
    triggered: bool
    triggers: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    stop_reason: str


class AgentRunResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    query_id: str
    user_id: str
    target_time: str
    ground_truth_poi_id: str | None
    dataset_capabilities: DatasetCapabilities
    context_snapshot: ContextSnapshot
    user_profile: UserProfile
    inferred_intention: Intention
    tool_plan: ToolPlan
    candidate_pool_summary: dict
    ranked_pois: list[RankedPOI]
    reflection: ReflectionRecord
    agent_trace_summary: list[ToolCallRecord]

