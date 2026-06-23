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


def evaluate(repo: NYCDataRepository, limit: int | None = 50, llm_mode: str = "fake") -> EvaluationResult:
    agent = IAAAgent(repo, RunConfig(llm_mode=llm_mode))
    traj_ids = repo.iter_test_traj_ids()
    if limit is not None:
        traj_ids = traj_ids[:limit]
    ranks: list[int | None] = []
    for traj_id in traj_ids:
        result = agent.run(traj_id)
        gt = result.ground_truth_poi_id
        predicted = [item.poi_id for item in result.ranked_pois]
        rank = predicted.index(gt) + 1 if gt in predicted else None
        ranks.append(rank)
    return _metrics(ranks)


def evaluate_user_split(
    repo: NYCDataRepository,
    limit: int | None = 50,
    train_ratio: float = 0.8,
    context_size: int = 5,
    llm_mode: str = "fake",
) -> EvaluationResult:
    repo.use_user_chronological_split(train_ratio)
    agent = IAAAgent(repo, RunConfig(llm_mode=llm_mode))
    keys = repo.iter_user_test_events(train_ratio=train_ratio, min_context=1)
    if limit is not None:
        keys = keys[:limit]
    ranks: list[int | None] = []
    for user_id, target_index in keys:
        query = repo.get_user_query(
            user_id=user_id,
            target_index=target_index,
            train_ratio=train_ratio,
            context_size=context_size,
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
