from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from collections.abc import Callable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

from .data import NYCDataRepository
from .engine import IAAAgent, RunConfig
from .llm import is_live_llm_mode
from .utils import write_json


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
    run_records: list[dict] | None = None
    llm_usage: dict[str, int] | None = None
    fallback_count: int | None = None
    usage_missing_count: int | None = None
    llm_status_counts: dict[str, int] | None = None
    llm_anomalies: list[dict] | None = None
    all_sessions_used_llm: bool | None = None
    all_sessions_used_deepseek: bool | None = None
    stratified_report: dict | None = None

    def as_dict(self) -> dict:
        payload = {
            "total": self.total,
            "Hit@1": self.hit_at_1,
            "Hit@5": self.hit_at_5,
            "Hit@10": self.hit_at_10,
            "NDCG@1": self.ndcg_at_1,
            "NDCG@5": self.ndcg_at_5,
            "NDCG@10": self.ndcg_at_10,
            "MRR": self.mrr,
        }
        if self.run_records is not None:
            payload["runs"] = self.run_records
        if self.llm_usage is not None:
            payload["llm_usage"] = self.llm_usage
        if self.fallback_count is not None:
            payload["fallback_count"] = self.fallback_count
        if self.usage_missing_count is not None:
            payload["usage_missing_count"] = self.usage_missing_count
        if self.llm_status_counts is not None:
            payload["llm_status_counts"] = self.llm_status_counts
        if self.llm_anomalies is not None:
            payload["llm_anomalies"] = self.llm_anomalies
        if self.all_sessions_used_llm is not None:
            payload["all_sessions_used_llm"] = self.all_sessions_used_llm
        if self.all_sessions_used_deepseek is not None:
            payload["all_sessions_used_deepseek"] = self.all_sessions_used_deepseek
        if self.stratified_report is not None:
            payload["stratified"] = self.stratified_report
        return payload


def _llm_run_outcome(
    agent: IAAAgent,
    config: RunConfig,
) -> tuple[dict[str, int] | None, bool, bool, str, str, str | None, str | None]:
    if not is_live_llm_mode(config.llm_mode):
        return None, False, False, "not_requested", "heuristic", None, None
    usage = agent.llm.last_usage
    fallback = agent.last_intention_source == "heuristic_fallback"
    usage_missing = not bool(usage and usage.get("total_tokens"))
    status = agent.last_llm_status or "unknown"
    return (
        usage,
        fallback,
        usage_missing,
        status,
        agent.last_intention_source,
        agent.llm.last_error_type,
        agent.llm.last_finish_reason,
    )


def evaluate_session_split(
    repo: NYCDataRepository,
    train_ratio: float = 0.8,
    min_context: int = 1,
    smoke_limit: int | None = None,
    user_id: str | int | None = None,
    save_runs_dir: str | Path | None = None,
    llm_mode: str = "fake",
    workers: int = 1,
    run_config: RunConfig | None = None,
    session_keys: list[tuple[str, str]] | None = None,
    progress_callback: Callable[[], None] | None = None,
    strict_llm: bool = False,
    report_stratified: bool = False,
) -> EvaluationResult:
    config = run_config or RunConfig(llm_mode=llm_mode)
    repo.use_user_chronological_split(train_ratio)
    keys = (
        list(session_keys)
        if session_keys is not None
        else repo.iter_session_test_keys(train_ratio=train_ratio, min_context=min_context, user_id=user_id)
    )
    if smoke_limit is not None:
        keys = keys[:smoke_limit]

    # 并行路径(P2):仅在 workers>1、纯指标(不保存 trace)、fake 模式下启用。
    # save_runs 需 pickle 大体量的 AgentRunResult,且当前用途是快速指标基线 → 回退串行。
    # Live LLM modes use the dedicated threaded evaluation path.
    use_parallel = (
        workers
        and workers > 1
        and save_runs_dir is None
        and config.llm_mode == "fake"
        and not report_stratified
    )

    if use_parallel:
        ranks = _evaluate_parallel(
            data_dir=str(repo.data_dir),
            keys=keys,
            train_ratio=train_ratio,
            min_context=min_context,
            run_config=config,
            workers=int(workers),
        )
        metrics = _metrics(ranks)
        return metrics

    # 串行路径(原始行为,workers<=1 或需保存 trace 时):预热一次全局结构(P3),再顺序评估。
    repo.prewarm_global_structures()
    agent = IAAAgent(repo, config)
    ranks: list[int | None] = []
    labels: list[dict[str, str]] = []
    run_records: list[dict] | None = [] if save_runs_dir is not None else None
    fallback_count = 0
    usage_missing_count = 0
    llm_status_counts: dict[str, int] = defaultdict(int)
    llm_anomalies: list[dict] = []
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
        if report_stratified:
            labels.append(_session_strata_labels(query, gt))
        (
            llm_usage,
            fallback,
            usage_missing,
            llm_status,
            intention_source,
            error_type,
            finish_reason,
        ) = _llm_run_outcome(agent, config)
        if is_live_llm_mode(config.llm_mode):
            llm_status_counts[llm_status] += 1
        if fallback:
            fallback_count += 1
        if usage_missing:
            usage_missing_count += 1
        if fallback or usage_missing:
            llm_anomalies.append(
                {
                    "user_id": str(user_id),
                    "trajectory_id": str(trajectory_id),
                    "status": llm_status,
                    "error_type": error_type,
                    "finish_reason": finish_reason,
                    "intention_source": intention_source,
                    "usage_missing": usage_missing,
                    "heuristic_fallback": fallback,
                    "strict_violation": bool(strict_llm and fallback),
                }
            )
        if save_runs_dir is not None and run_records is not None:
            trace_path = Path(save_runs_dir) / f"user_{user_id}_session_{_safe_filename(trajectory_id)}.json"
            write_json(trace_path, result.model_dump(mode="json"))
            run_records.append(
                {
                    "user_id": str(user_id),
                    "trajectory_id": str(trajectory_id),
                    "query_id": result.query_id,
                    "rank": rank,
                    "ground_truth_poi_id": result.ground_truth_poi_id,
                    "ground_truth_poi_idx": result.ground_truth_poi_idx,
                    "top1_poi_id": result.ranked_pois[0].poi_id if result.ranked_pois else None,
                    "top1_poi_idx": result.ranked_pois[0].poi_idx if result.ranked_pois else None,
                    "llm_usage": llm_usage,
                    "deepseek_usage": llm_usage if config.llm_mode == "deepseek" else None,
                    "trace_path": str(trace_path),
                }
            )
        if progress_callback is not None:
            progress_callback()
    metrics = _metrics(ranks)
    metrics.run_records = run_records
    if any(agent.llm.usage_totals.values()):
        metrics.llm_usage = agent.llm.usage_totals
    if is_live_llm_mode(config.llm_mode):
        metrics.fallback_count = fallback_count
        metrics.usage_missing_count = usage_missing_count
        metrics.llm_status_counts = dict(llm_status_counts)
        metrics.llm_anomalies = sorted(
            llm_anomalies,
            key=lambda item: (item["user_id"], item["trajectory_id"]),
        )
        metrics.all_sessions_used_llm = fallback_count == 0
        if config.llm_mode == "deepseek":
            metrics.all_sessions_used_deepseek = fallback_count == 0
    if report_stratified:
        metrics.stratified_report = build_stratified_report(ranks, labels)
    return metrics


def evaluate_session_split_threaded(
    repo: NYCDataRepository,
    train_ratio: float = 0.8,
    min_context: int = 1,
    smoke_limit: int | None = None,
    user_id: str | int | None = None,
    llm_mode: str = "fake",
    run_config: RunConfig | None = None,
    session_keys: list[tuple[str, str]] | None = None,
    concurrency: int = 4,
    strict_llm: bool = False,
    stall_timeout_seconds: int = 0,
    progress_callback: Callable[[], None] | None = None,
    report_stratified: bool = False,
) -> EvaluationResult:
    config = run_config or RunConfig(llm_mode=llm_mode)
    repo.use_user_chronological_split(train_ratio)
    keys = (
        list(session_keys)
        if session_keys is not None
        else repo.iter_session_test_keys(train_ratio=train_ratio, min_context=min_context, user_id=user_id)
    )
    if smoke_limit is not None:
        keys = keys[:smoke_limit]
    if not keys:
        result = _metrics([])
        result.fallback_count = 0
        return result

    max_workers = max(1, min(int(concurrency), len(keys)))
    ranks: list[int | None] = [None] * len(keys)
    labels: list[dict[str, str] | None] = [None] * len(keys)
    usage_totals: dict[str, int] = defaultdict(int)
    fallback_count = 0
    usage_missing_count = 0
    llm_status_counts: dict[str, int] = defaultdict(int)
    llm_anomalies: list[dict] = []

    task_queue: Queue[tuple[int, tuple[str, str]] | None] = Queue()
    result_queue: Queue[tuple[str, object]] = Queue()
    stop_event = threading.Event()

    for index, key in enumerate(keys):
        task_queue.put((index, key))
    for _ in range(max_workers):
        task_queue.put(None)

    def worker() -> None:
        local_repo = NYCDataRepository(str(repo.data_dir))
        local_repo.use_user_chronological_split(train_ratio)
        local_repo.prewarm_global_structures()
        while not stop_event.is_set():
            task = task_queue.get()
            if task is None:
                return
            index, (uid, tid) = task
            try:
                query = local_repo.get_session_query(
                    user_id=uid,
                    trajectory_id=tid,
                    train_ratio=train_ratio,
                    min_context=min_context,
                )
                agent = IAAAgent(local_repo, config)
                result = agent.run_query(query)
                gt = result.ground_truth_poi_id
                predicted = [item.poi_id for item in result.ranked_pois]
                rank = predicted.index(gt) + 1 if gt in predicted else None
                label = _session_strata_labels(query, gt) if report_stratified else None
                (
                    usage,
                    fallback,
                    usage_missing,
                    llm_status,
                    intention_source,
                    error_type,
                    finish_reason,
                ) = _llm_run_outcome(agent, config)
                result_queue.put(
                    (
                        "ok",
                        (
                            index,
                            rank,
                            label,
                            usage,
                            fallback,
                            usage_missing,
                            llm_status,
                            intention_source,
                            error_type,
                            finish_reason,
                            uid,
                            tid,
                        ),
                    )
                )
            except BaseException as exc:
                result_queue.put(("error", exc))
                stop_event.set()
                return

    threads = [
        threading.Thread(target=worker, name=f"iaa-eval-worker-{i + 1}", daemon=True)
        for i in range(max_workers)
    ]
    for thread in threads:
        thread.start()

    completed = 0
    last_result_at = time.monotonic()
    try:
        while completed < len(keys):
            if stall_timeout_seconds > 0 and time.monotonic() - last_result_at > stall_timeout_seconds:
                stop_event.set()
                raise TimeoutError(
                    f"No sessions completed for {stall_timeout_seconds}s; aborting threaded evaluation"
                )
            try:
                status, payload = result_queue.get(timeout=1.0)
            except Empty:
                continue
            last_result_at = time.monotonic()
            if status == "error":
                stop_event.set()
                raise payload  # type: ignore[misc]

            (
                index,
                rank,
                label,
                usage,
                fallback,
                usage_missing,
                llm_status,
                intention_source,
                error_type,
                finish_reason,
                uid,
                tid,
            ) = payload  # type: ignore[misc]
            ranks[int(index)] = rank  # type: ignore[arg-type]
            labels[int(index)] = label  # type: ignore[assignment]
            if is_live_llm_mode(config.llm_mode):
                llm_status_counts[str(llm_status)] += 1
            if fallback:
                fallback_count += 1
            if usage_missing:
                usage_missing_count += 1
            if fallback or usage_missing:
                llm_anomalies.append(
                    {
                        "user_id": str(uid),
                        "trajectory_id": str(tid),
                        "status": str(llm_status),
                        "error_type": error_type,
                        "finish_reason": finish_reason,
                        "intention_source": str(intention_source),
                        "usage_missing": bool(usage_missing),
                        "heuristic_fallback": bool(fallback),
                        "strict_violation": bool(strict_llm and fallback),
                    }
                )
            if usage:
                for key, value in usage.items():
                    if isinstance(value, int):
                        usage_totals[key] += value
            completed += 1
            if progress_callback is not None:
                progress_callback()
    except KeyboardInterrupt:
        stop_event.set()
        raise

    metrics = _metrics(ranks)
    metrics.fallback_count = fallback_count
    if usage_totals:
        metrics.llm_usage = dict(usage_totals)
    if is_live_llm_mode(config.llm_mode):
        metrics.usage_missing_count = usage_missing_count
        metrics.llm_status_counts = dict(llm_status_counts)
        metrics.llm_anomalies = sorted(
            llm_anomalies,
            key=lambda item: (item["user_id"], item["trajectory_id"]),
        )
        metrics.all_sessions_used_llm = fallback_count == 0
        if config.llm_mode == "deepseek":
            metrics.all_sessions_used_deepseek = fallback_count == 0
    if report_stratified:
        if any(label is None for label in labels):
            raise RuntimeError("Missing stratification labels after threaded evaluation")
        metrics.stratified_report = build_stratified_report(
            ranks,
            [label for label in labels if label is not None],
        )
    return metrics


# --- 并行评估实现(P2 + P3) ------------------------------------------------
# 每个 worker 进程在 initializer 中构建一次 repo、做一次 split、预热一次全局结构,
# 之后该进程处理的所有会话复用同一 repo/agent。fake 模式确定性保证结果与会话顺序无关,
# 指标(Hit/NDCG/MRR)是按 rank 聚合的均值,顺序无关 → 与串行逐字段一致。

_WORKER_STATE: dict[str, object] = {}


def _worker_init(data_dir: str, train_ratio: float, run_config: RunConfig) -> None:
    """worker 进程初始化:重建 repo、split、预热全局结构(P3)。每进程仅执行一次。"""
    repo = NYCDataRepository(data_dir)
    repo.use_user_chronological_split(train_ratio)
    repo.prewarm_global_structures()
    _WORKER_STATE["repo"] = repo
    _WORKER_STATE["agent"] = IAAAgent(repo, run_config)
    _WORKER_STATE["train_ratio"] = train_ratio


def _worker_eval_one(task: tuple[str, str, int]) -> int | None:
    """评估单个会话,只返回轻量的 rank(避免 pickle 整个 AgentRunResult)。"""
    user_id, trajectory_id, min_context = task
    repo: NYCDataRepository = _WORKER_STATE["repo"]  # type: ignore[assignment]
    agent: IAAAgent = _WORKER_STATE["agent"]  # type: ignore[assignment]
    train_ratio: float = _WORKER_STATE["train_ratio"]  # type: ignore[assignment]
    query = repo.get_session_query(
        user_id=user_id,
        trajectory_id=trajectory_id,
        train_ratio=train_ratio,
        min_context=min_context,
    )
    result = agent.run_query(query)
    gt = result.ground_truth_poi_id
    predicted = [item.poi_id for item in result.ranked_pois]
    return predicted.index(gt) + 1 if gt in predicted else None


# --- 分层评估(可选,默认不启用;用于「方向 A:合法澄清评估口径」)-----------
# 设计原则(守 AD-9):分层只是把每个会话的 rank 按可解释标签重新聚合,
# rank 计算口径与 _worker_eval_one / 串行路径逐字一致 → 整体指标必然与基线吻合。
# 标签全部来自训练段历史(query.history,cutoff 之前),无未来泄漏。

# 分层维度定义:dim_key -> 该维度对各会话打标签的函数说明,纯文档用途。
STRATA_DIMENSIONS = ("ooh", "cold_start", "context_len")


def _session_strata_labels(query, ground_truth_poi_id: str) -> dict[str, str]:
    """为单个会话计算分层标签。仅依赖训练段历史与当前 session 上下文,无泄漏。

    - ooh:        真值 POI 是否出现在该用户训练段历史中(IH / OOH)。
                  OOH = 用户从未到访的全新 POI,基于个人历史/转移的方法结构上召回不到。
    - cold_start: 用户训练段交互次数分桶(cold<10 / warm 10-49 / rich>=50)。
    - context_len:当前 session 中目标之前可见的 check-in 数(ctx=1 / ctx=2 / ctx>=3)。
    """
    history = query.history
    n_hist = int(len(history)) if history is not None else 0
    hist_poi_ids = set(history["POI_id"].astype(str)) if history is not None else set()
    ctx_len = int(len(query.context))

    in_history = str(ground_truth_poi_id) in hist_poi_ids

    if n_hist < 10:
        cold = "cold(<10)"
    elif n_hist < 50:
        cold = "warm(10-49)"
    else:
        cold = "rich(>=50)"

    if ctx_len <= 1:
        ctxbk = "ctx=1"
    elif ctx_len == 2:
        ctxbk = "ctx=2"
    else:
        ctxbk = "ctx>=3"

    return {
        "ooh": "IH" if in_history else "OOH",
        "cold_start": cold,
        "context_len": ctxbk,
    }


def _worker_eval_one_stratified(task: tuple[str, str, int]) -> tuple[int | None, dict[str, str]]:
    """分层版 worker:同时返回 rank(口径同 _worker_eval_one)与该会话的分层标签。"""
    user_id, trajectory_id, min_context = task
    repo: NYCDataRepository = _WORKER_STATE["repo"]  # type: ignore[assignment]
    agent: IAAAgent = _WORKER_STATE["agent"]  # type: ignore[assignment]
    train_ratio: float = _WORKER_STATE["train_ratio"]  # type: ignore[assignment]
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
    labels = _session_strata_labels(query, gt)
    return rank, labels


def _limit_blas_threads_for_workers() -> None:
    """在 spawn worker 之前把 BLAS/OpenMP 线程数锁为 1,消除超额订阅(oversubscription)。

    根因:numpy/pandas 底层 BLAS 默认每进程开多线程做向量运算。多进程并行时,
    N 个 worker × 每个多线程 = 远超物理核数的线程在抢核,CPU 看似占满但大量时间
    耗在线程调度上,反而比合理配置更慢(实测 16-worker 全量 >22 分钟未完)。

    正确配置:每个 worker 单线程纯算,靠进程级并行吃满核。Windows spawn 模式下子进程
    继承父进程创建它时的环境变量,故在创建进程池之前 setdefault 即可让 worker 生效;
    用 setdefault 以尊重用户显式设置。"""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _evaluate_parallel(
    data_dir: str,
    keys: list[tuple[str, str]],
    train_ratio: float,
    min_context: int,
    run_config: RunConfig,
    workers: int,
) -> list[int | None]:
    _limit_blas_threads_for_workers()
    max_workers = max(1, min(workers, len(keys))) if keys else 1
    tasks = [(user_id, trajectory_id, min_context) for user_id, trajectory_id in keys]
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(data_dir, train_ratio, run_config),
    ) as executor:
        # 用 map 保持与 keys 一致的输出顺序(指标本身顺序无关,但保序便于调试与 trace 对齐)。
        ranks = list(executor.map(_worker_eval_one, tasks))
    return ranks


def resolve_worker_count(workers: int | None) -> int:
    """把 --workers 解析为实际进程数。0/None → 自动取 CPU 核数;负数视为 1(串行)。"""
    if workers is None or workers == 0:
        return max(1, os.cpu_count() or 1)
    return max(1, workers)


def stable_fractional_sample(
    keys: list[tuple[str, str]],
    fraction: float,
    seed: int = 42,
) -> list[tuple[str, str]]:
    if not keys:
        return []
    if fraction >= 1.0:
        return list(keys)
    if fraction <= 0.0:
        raise ValueError("sample fraction must be greater than 0")
    sample_size = max(1, int(round(len(keys) * fraction)))

    def score(item: tuple[str, str]) -> str:
        raw = f"{seed}:{item[0]}:{item[1]}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    selected = set(sorted(keys, key=score)[:sample_size])
    return [key for key in keys if key in selected]


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


def _safe_filename(value: str | int) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


# --- 分层报告聚合(方向 A)--------------------------------------------------

def _slice_metrics(ranks: list[int | None]) -> dict:
    """对一个会话切片计算 Hit@1/5/10 + MRR。口径与 _metrics 完全一致。"""
    n = len(ranks)
    if n == 0:
        return {
            "n": 0,
            "Acc@1": 0.0,
            "Acc@5": 0.0,
            "Acc@10": 0.0,
            "Hit@1": 0.0,
            "Hit@5": 0.0,
            "Hit@10": 0.0,
            "MRR": 0.0,
        }

    def hit(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / n

    mrr = sum(0.0 if r is None else 1.0 / r for r in ranks) / n
    hit_at_1 = round(hit(1), 6)
    hit_at_5 = round(hit(5), 6)
    hit_at_10 = round(hit(10), 6)
    return {
        "n": n,
        "Acc@1": hit_at_1,
        "Acc@5": hit_at_5,
        "Acc@10": hit_at_10,
        "Hit@1": hit_at_1,
        "Hit@5": hit_at_5,
        "Hit@10": hit_at_10,
        "MRR": round(mrr, 6),
    }


def build_stratified_report(
    ranks: list[int | None],
    labels: list[dict[str, str]],
) -> dict:
    """把逐会话的 rank + 分层标签聚合成可直接进报告/论文的分层结构。

    返回:
      {
        "overall": {n, Hit@1, Hit@5, Hit@10, MRR},        # 与权威基线吻合(口径自检)
        "strata": {dim: {label: {n, share, Hit@1, Hit@5, Hit@10, MRR}}},
        "in_history_subset": {n, share, ...},             # 方向 A 主报告子集
      }
    口径声明:每个切片的指标与全量基线同口径(full-candidate ranking,无负采样),
    切片仅按可解释标签拆分会话集合,不改变任何 rank 的计算方式。
    """
    total = len(ranks)
    overall = _slice_metrics(ranks)

    strata: dict[str, dict] = {}
    for dim in STRATA_DIMENSIONS:
        groups: dict[str, list[int | None]] = defaultdict(list)
        for rank, lab in zip(ranks, labels):
            groups[lab[dim]].append(rank)
        dim_out: dict[str, dict] = {}
        # 按切片样本数降序,便于阅读
        for label in sorted(groups, key=lambda x: -len(groups[x])):
            m = _slice_metrics(groups[label])
            m["share"] = round(m["n"] / total, 6) if total else 0.0
            dim_out[label] = m
        strata[dim] = dim_out

    # 方向 A 主报告子集:in_history(可预测子集)
    in_hist_ranks = [r for r, lab in zip(ranks, labels) if lab["ooh"] == "IH"]
    in_hist = _slice_metrics(in_hist_ranks)
    in_hist["share"] = round(in_hist["n"] / total, 6) if total else 0.0

    return {"overall": overall, "strata": strata, "in_history_subset": in_hist}


def evaluate_session_split_stratified(
    repo: NYCDataRepository,
    train_ratio: float = 0.8,
    min_context: int = 1,
    smoke_limit: int | None = None,
    user_id: str | int | None = None,
    workers: int = 1,
) -> dict:
    """方向 A:在评估的同时按可解释维度分层,返回 build_stratified_report 的结构。

    仅支持 fake 模式(确定性、可复现,守 AD-9)。rank 口径与 evaluate_session_split 一致,
    因此 report["overall"] 必然与同条件下的权威基线吻合——这是分层可信的自检。
    """
    repo.use_user_chronological_split(train_ratio)
    keys = repo.iter_session_test_keys(train_ratio=train_ratio, min_context=min_context, user_id=user_id)
    if smoke_limit is not None:
        keys = keys[:smoke_limit]

    tasks = [(uid, tid, min_context) for uid, tid in keys]
    ranks: list[int | None] = []
    labels: list[dict[str, str]] = []

    if workers and workers > 1:
        _limit_blas_threads_for_workers()
        max_workers = max(1, min(workers, len(tasks))) if tasks else 1
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(str(repo.data_dir), train_ratio, RunConfig(llm_mode="fake")),
        ) as executor:
            for rank, lab in executor.map(_worker_eval_one_stratified, tasks):
                ranks.append(rank)
                labels.append(lab)
    else:
        repo.prewarm_global_structures()
        agent = IAAAgent(repo, RunConfig(llm_mode="fake"))
        for uid, tid, mc in tasks:
            query = repo.get_session_query(user_id=uid, trajectory_id=tid, train_ratio=train_ratio, min_context=mc)
            result = agent.run_query(query)
            gt = result.ground_truth_poi_id
            predicted = [item.poi_id for item in result.ranked_pois]
            rank = predicted.index(gt) + 1 if gt in predicted else None
            ranks.append(rank)
            labels.append(_session_strata_labels(query, gt))

    return build_stratified_report(ranks, labels)
