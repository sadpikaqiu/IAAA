from __future__ import annotations

import math
from dataclasses import dataclass

from .data import NYCDataRepository
from .engine import IAAAgent, RunConfig


@dataclass
class EvaluationResult:
    total: int
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    ndcg_at_1: float
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "Hit@1": self.hit_at_1,
            "Hit@5": self.hit_at_5,
            "Hit@10": self.hit_at_10,
            "NDCG@1": self.ndcg_at_1,
            "NDCG@5": self.ndcg_at_5,
            "NDCG@10": self.ndcg_at_10,
            "MRR": self.mrr,
        }


def evaluate_session_split(
    repo: NYCDataRepository,
    train_ratio: float = 0.8,
    min_context: int = 1,
    smoke_limit: int | None = None,
    user_id: str | int | None = None,
    llm_mode: str = "fake",
) -> EvaluationResult:
    repo.use_user_chronological_split(train_ratio)
    agent = IAAAgent(repo, RunConfig(llm_mode=llm_mode))
    keys = repo.iter_session_test_keys(train_ratio=train_ratio, min_context=min_context, user_id=user_id)
    if smoke_limit is not None:
        keys = keys[:smoke_limit]
    ranks: list[int | None] = []
    for user_id, trajectory_id in keys:
        query = repo.get_session_query(
            user_id=user_id,
            trajectory_id=trajectory_id,
            train_ratio=train_ratio,
            min_context=min_context,
        )
        result = agent.run_query(query)
        gt = result.ground_truth_poi_id
        predicted = [item.poi_id for item in result.ranked_pois]
        rank = predicted.index(gt) + 1 if gt in predicted else None
        ranks.append(rank)
    return _metrics(ranks)


def _metrics(ranks: list[int | None]) -> EvaluationResult:
    n = len(ranks)
    if n == 0:
        return EvaluationResult(0, 0, 0, 0, 0, 0, 0, 0)

    def hit(k: int) -> float:
        return sum(1 for rank in ranks if rank is not None and rank <= k) / n

    def ndcg(k: int) -> float:
        total = 0.0
        for rank in ranks:
            if rank is not None and rank <= k:
                total += 1.0 / math.log2(rank + 1)
        return total / n

    mrr = sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n
    return EvaluationResult(
        total=n,
        hit_at_1=round(hit(1), 6),
        hit_at_5=round(hit(5), 6),
        hit_at_10=round(hit(10), 6),
        ndcg_at_1=round(ndcg(1), 6),
        ndcg_at_5=round(ndcg(5), 6),
        ndcg_at_10=round(ndcg(10), 6),
        mrr=round(mrr, 6),
    )
