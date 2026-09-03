from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .data import NYCDataRepository
from .engine import IAAAgent, RunConfig
from .evaluation import (
    evaluate_session_split,
    evaluate_session_split_stratified,
    evaluate_session_split_threaded,
    resolve_worker_count,
    stable_fractional_sample,
)
from .llm import DeepSeekClient, is_live_llm_mode
from .utils import read_json, write_json

app = typer.Typer(help="IAA-Agent NYC-first CLI")
console = Console()


@app.command()
def prepare(
    data_dir: str = typer.Option("datasets/NYC", help="Directory containing NYC_train/val/test.csv"),
    out: str = typer.Option("outputs/prepared/nyc_summary.json", help="Prepared dataset summary output"),
) -> None:
    repo = NYCDataRepository(data_dir)
    summary = repo.summary()
    write_json(out, summary)
    console.print(f"Wrote dataset summary to {out}")
    console.print(summary)


@app.command()
def run(
    traj_id: str = typer.Option(..., help="Test trajectory id, e.g. 349_52"),
    data_dir: str = typer.Option("datasets/NYC", help="Directory containing NYC_train/val/test.csv"),
    out: Optional[str] = typer.Option(None, help="JSON output path"),
    llm: str = typer.Option("fake", help="LLM mode: fake, deepseek, or openai"),
) -> None:
    _validate_llm_mode(llm)
    repo = NYCDataRepository(data_dir)
    agent = IAAAgent(repo, RunConfig(llm_mode=llm))
    result = agent.run(traj_id)
    payload = result.model_dump(mode="json")
    target = out or f"outputs/runs/{traj_id}.json"
    write_json(target, payload)
    console.print(f"Wrote run result to {target}")
    console.print({
        "traj_id": traj_id,
        "top1_poi_idx": payload["ranked_pois"][0]["poi_idx"] if payload["ranked_pois"] else None,
        "top1_poi_id": payload["ranked_pois"][0]["poi_id"] if payload["ranked_pois"] else None,
    })


@app.command("user-targets")
def user_targets(
    user_id: str = typer.Option(..., help="User id in the chronological user stream"),
    data_dir: str = typer.Option("datasets/NYC", help="Directory containing NYC_train/val/test.csv"),
    train_ratio: float = typer.Option(0.8, help="Per-user chronological train ratio"),
) -> None:
    repo = NYCDataRepository(data_dir)
    try:
        info = repo.user_timeline_info(user_id, train_ratio)
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(info)


@app.command("run-user")
def run_user(
    user_id: str = typer.Option(..., help="User id in the chronological user stream"),
    target_index: Optional[int] = typer.Option(
        None,
        help="0-based index in this user's full chronological check-in stream; defaults to the last held-out event",
    ),
    data_dir: str = typer.Option("datasets/NYC", help="Directory containing NYC_train/val/test.csv"),
    train_ratio: float = typer.Option(0.8, help="Per-user chronological train ratio"),
    context_size: int = typer.Option(5, help="Number of previous check-ins used as short-term context"),
    out: Optional[str] = typer.Option(None, help="JSON output path"),
    llm: str = typer.Option("fake", help="LLM mode: fake, deepseek, or openai"),
) -> None:
    _validate_llm_mode(llm)
    repo = NYCDataRepository(data_dir)
    repo.use_user_chronological_split(train_ratio)
    try:
        resolved_target_index = _resolve_user_target_index(repo, user_id, train_ratio, target_index)
        query = repo.get_user_query(
            user_id=user_id,
            target_index=resolved_target_index,
            train_ratio=train_ratio,
            context_size=context_size,
        )
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    agent = IAAAgent(repo, RunConfig(llm_mode=llm))
    result = agent.run_query(query)
    payload = result.model_dump(mode="json")
    target = out or f"outputs/runs/user_{user_id}_idx_{resolved_target_index}.json"
    write_json(target, payload)
    console.print(f"Wrote user-timeline run result to {target}")
    console.print({
        "query_id": payload["query_id"],
        "query_mode": payload["query_mode"],
        "target_index": resolved_target_index,
        "target_index_source": "default_tail" if target_index is None else "explicit",
        "ground_truth_poi_idx": payload["ground_truth_poi_idx"],
        "top1_poi_idx": payload["ranked_pois"][0]["poi_idx"] if payload["ranked_pois"] else None,
    })


@app.command()
def replay(
    case: str = typer.Option(..., help="Replay case JSON path"),
    out: Optional[str] = typer.Option(None, help="JSON output path"),
    llm: str = typer.Option("fake", help="LLM mode: fake, deepseek, or openai"),
) -> None:
    _validate_llm_mode(llm)
    data = read_json(case)
    data_dir = data.get("data_dir", "datasets/NYC")
    traj_id = str(data["traj_id"])
    repo = NYCDataRepository(data_dir)
    agent = IAAAgent(repo, RunConfig(llm_mode=llm))
    result = agent.run(traj_id)
    target = out or f"outputs/runs/replay_{Path(case).stem}.json"
    write_json(target, result.model_dump(mode="json"))
    console.print(f"Wrote replay result to {target}")


@app.command(name="evaluate")
def evaluate_command(
    data_dir: str = typer.Option("datasets/NYC", help="Directory containing NYC_train/val/test.csv"),
    user_id: Optional[str] = typer.Option(None, help="Optional user id; evaluates all held-out sessions for that user"),
    train_ratio: float = typer.Option(0.8, help="Per-user chronological train ratio for long-term history"),
    min_context: int = typer.Option(1, help="Minimum visible check-ins before the session target"),
    smoke_limit: int = typer.Option(0, help="Optional session sample cap for smoke runs; 0 evaluates the full split"),
    save_runs: Optional[str] = typer.Option(None, help="Optional directory for per-session full AgentRunResult traces"),
    out: str = typer.Option("outputs/evaluation/session_split_results.json", help="Metrics JSON output"),
    llm: str = typer.Option("fake", help="LLM mode: fake, deepseek, or openai"),
    variant: str = typer.Option("mainline", help="Ranking variant: mainline or p4v1"),
    workers: int = typer.Option(
        1,
        help="并行进程数(仅 fake 模式且不保存 trace 时生效);1=串行,0=自动取 CPU 核数",
    ),
    concurrency: int = typer.Option(4, help="Thread concurrency for live LLM I/O"),
    stall_timeout: int = typer.Option(
        600,
        help="Abort if no live-LLM session finishes for this many seconds; 0 disables",
    ),
    allow_fallback: bool = typer.Option(
        True,
        help="Allow heuristic fallback sessions; --no-allow-fallback marks them as strict violations without aborting",
    ),
    model: Optional[str] = typer.Option(None, help="Live LLM model name"),
    base_url: Optional[str] = typer.Option(None, help="OpenAI-compatible API base URL"),
    report_stratified: bool = typer.Option(
        False,
        "--report-stratified",
        help="Report IH/OOH, cold-start, and context-length metrics from the same predictions.",
    ),
) -> None:
    _validate_llm_mode(llm)
    if variant not in {"mainline", "p4v1"}:
        raise typer.BadParameter("--variant must be mainline or p4v1")
    if concurrency < 1:
        raise typer.BadParameter("--concurrency must be >= 1")
    if stall_timeout < 0:
        raise typer.BadParameter("--stall-timeout must be >= 0")

    resolved_model = None
    if is_live_llm_mode(llm):
        resolved_model = _configure_live_llm(llm, model, base_url)
        _preflight_llm(llm)

    repo = NYCDataRepository(data_dir)
    actual_smoke_limit = None if smoke_limit == 0 else smoke_limit
    actual_workers = resolve_worker_count(workers)
    config = RunConfig.p4(llm_mode=llm) if variant == "p4v1" else RunConfig(llm_mode=llm)

    if actual_workers > 1 and save_runs is not None:
        console.print("[yellow]保存 trace 时 fake 多进程评估回退为串行执行[/yellow]")
        actual_workers = 1

    try:
        if is_live_llm_mode(llm) and save_runs is None:
            repo.use_user_chronological_split(train_ratio)
            keys = repo.iter_session_test_keys(
                train_ratio=train_ratio,
                min_context=min_context,
                user_id=user_id,
            )
            if actual_smoke_limit is not None:
                keys = keys[:actual_smoke_limit]
            console.print(
                f"Evaluating {len(keys)} held-out sessions with {variant} {llm}; "
                f"concurrency={concurrency}; stall-timeout={stall_timeout}s."
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"{variant} {llm}", total=len(keys))
                result = evaluate_session_split_threaded(
                    repo,
                    train_ratio=train_ratio,
                    min_context=min_context,
                    llm_mode=llm,
                    run_config=config,
                    session_keys=keys,
                    concurrency=concurrency,
                    strict_llm=not allow_fallback,
                    stall_timeout_seconds=stall_timeout,
                    progress_callback=lambda: progress.advance(task),
                    report_stratified=report_stratified,
                )
            _report_llm_quality(variant, result.as_dict())
        else:
            result = evaluate_session_split(
                repo,
                train_ratio=train_ratio,
                min_context=min_context,
                smoke_limit=actual_smoke_limit,
                user_id=user_id,
                save_runs_dir=save_runs,
                llm_mode=llm,
                workers=actual_workers,
                run_config=config,
                strict_llm=is_live_llm_mode(llm) and not allow_fallback,
                report_stratified=report_stratified and actual_workers == 1,
            )
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = result.as_dict()
    payload["split"] = {
        "mode": "user_chronological_session",
        "user_id": user_id,
        "train_ratio": train_ratio,
        "min_context": min_context,
        "session_source": "original trajectory_id",
        "smoke_limit": smoke_limit,
        "workers": actual_workers,
        "concurrency": concurrency if is_live_llm_mode(llm) else None,
        "stall_timeout_seconds": stall_timeout if is_live_llm_mode(llm) else None,
        "strict_llm": is_live_llm_mode(llm) and not allow_fallback,
        "history_status_definition": (
            "IH if the target POI appears in the user's first train_ratio events; "
            "OOH otherwise. Current-session context does not change this label."
        ),
    }
    payload["variant"] = variant
    payload["llm_mode"] = llm
    if resolved_model is not None:
        payload["model"] = resolved_model

    if report_stratified and "stratified" not in payload:
        report = evaluate_session_split_stratified(
            repo,
            train_ratio=train_ratio,
            min_context=min_context,
            smoke_limit=actual_smoke_limit,
            user_id=user_id,
            workers=actual_workers,
        )
        payload["stratified"] = report
    if "stratified" in payload:
        _print_stratified_report(payload["stratified"])

    write_json(out, payload)
    if save_runs is not None:
        write_json(Path(save_runs) / "summary.json", payload)
    console.print(f"Wrote evaluation results to {out}")
    if save_runs is not None:
        console.print(f"Wrote per-session traces to {save_runs}")
    console.print(payload)


@app.command("compare-p4")
def compare_p4_command(
    data_dir: str = typer.Option("datasets/NYC", help="Directory containing NYC_train/val/test.csv"),
    train_ratio: float = typer.Option(0.8, help="Per-user chronological train ratio for long-term history"),
    min_context: int = typer.Option(1, help="Minimum visible check-ins before the session target"),
    sample_fraction: float = typer.Option(0.5, help="Fraction of all held-out sessions to evaluate"),
    seed: int = typer.Option(42, help="Stable hash seed for session sampling"),
    concurrency: int = typer.Option(4, help="Thread concurrency for DeepSeek I/O"),
    stall_timeout: int = typer.Option(600, help="Abort if no session finishes for this many seconds; 0 disables"),
    allow_fallback: bool = typer.Option(
        True,
        help="Allow heuristic fallback sessions; --no-allow-fallback marks them as strict violations without aborting",
    ),
    model: str = typer.Option("deepseek-v4-flash", help="DeepSeek model name for both variants"),
    base_url: Optional[str] = typer.Option(None, help="Optional DeepSeek-compatible base URL"),
    out: str = typer.Option("outputs/evaluation/p4_deepseek_sample50_compare.json", help="Comparison JSON output"),
) -> None:
    if not 0 < sample_fraction <= 1:
        raise typer.BadParameter("--sample-fraction must be in (0, 1]")
    if concurrency < 1:
        raise typer.BadParameter("--concurrency must be >= 1")
    if stall_timeout < 0:
        raise typer.BadParameter("--stall-timeout must be >= 0")
    os.environ["DEEPSEEK_MODEL"] = model
    if base_url:
        os.environ["DEEPSEEK_BASE_URL"] = base_url
    _preflight_deepseek()

    key_repo = NYCDataRepository(data_dir)
    key_repo.use_user_chronological_split(train_ratio)
    all_keys = key_repo.iter_session_test_keys(train_ratio=train_ratio, min_context=min_context)
    sample_keys = stable_fractional_sample(all_keys, sample_fraction, seed=seed)
    console.print(
        f"Sampled {len(sample_keys)} / {len(all_keys)} held-out sessions "
        f"({sample_fraction:.1%}, seed={seed}); concurrency={concurrency}; "
        f"stall-timeout={stall_timeout}s."
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        baseline_task = progress.add_task("mainline deepseek", total=len(sample_keys))
        baseline = evaluate_session_split_threaded(
            NYCDataRepository(data_dir),
            train_ratio=train_ratio,
            min_context=min_context,
            llm_mode="deepseek",
            run_config=RunConfig(llm_mode="deepseek"),
            session_keys=sample_keys,
            concurrency=concurrency,
            strict_llm=not allow_fallback,
            stall_timeout_seconds=stall_timeout,
            progress_callback=lambda: progress.advance(baseline_task),
        )
        _report_llm_quality("baseline", baseline.as_dict())

        p4_task = progress.add_task("p4v1 deepseek", total=len(sample_keys))
        p4 = evaluate_session_split_threaded(
            NYCDataRepository(data_dir),
            train_ratio=train_ratio,
            min_context=min_context,
            llm_mode="deepseek",
            run_config=RunConfig.p4(llm_mode="deepseek"),
            session_keys=sample_keys,
            concurrency=concurrency,
            strict_llm=not allow_fallback,
            stall_timeout_seconds=stall_timeout,
            progress_callback=lambda: progress.advance(p4_task),
        )
        _report_llm_quality("p4", p4.as_dict())

    baseline_payload = baseline.as_dict()
    p4_payload = p4.as_dict()
    payload = {
        "comparison": "mainline_vs_p4v1",
        "llm_mode": "deepseek",
        "model": model,
        "split": {
            "mode": "user_chronological_session",
            "train_ratio": train_ratio,
            "min_context": min_context,
            "total_sessions": len(all_keys),
            "sample_fraction": sample_fraction,
            "sample_seed": seed,
            "sample_sessions": len(sample_keys),
            "concurrency": concurrency,
            "stall_timeout_seconds": stall_timeout,
            "strict_llm": not allow_fallback,
            "sample_keys": [{"user_id": uid, "trajectory_id": tid} for uid, tid in sample_keys],
        },
        "variants": {
            "mainline": baseline_payload,
            "p4v1": p4_payload,
        },
        "delta_p4_minus_mainline": _metric_delta(baseline_payload, p4_payload),
    }
    write_json(out, payload)
    console.print(f"Wrote P4 comparison to {out}")
    console.print(payload["delta_p4_minus_mainline"])


def _resolve_user_target_index(
    repo: NYCDataRepository,
    user_id: str,
    train_ratio: float,
    target_index: Optional[int],
) -> int:
    if target_index is not None:
        return target_index
    info = repo.user_timeline_info(user_id, train_ratio)
    return int(info["valid_target_index_end"])


def _validate_llm_mode(llm: str) -> None:
    if llm not in {"fake", "deepseek", "openai"}:
        raise typer.BadParameter("--llm must be fake, deepseek, or openai")


def _configure_live_llm(llm: str, model: str | None, base_url: str | None) -> str:
    if llm == "deepseek":
        resolved_model = model or "deepseek-v4-flash"
        os.environ["DEEPSEEK_MODEL"] = resolved_model
        if base_url:
            os.environ["DEEPSEEK_BASE_URL"] = base_url
        return resolved_model
    resolved_model = model or "Qwen/Qwen3.8-27B-FP8"
    os.environ["OPENAI_MODEL"] = resolved_model
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    return resolved_model


def _preflight_llm(llm: str) -> None:
    if llm == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        raise typer.BadParameter("DEEPSEEK_API_KEY is required for DeepSeek evaluation")
    client = DeepSeekClient(provider=llm)
    probe = client.chat_json(
        [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return {\"ok\": true}."},
        ],
        max_tokens=20,
    )
    if probe is None or not client.last_usage:
        if probe is not None:
            console.print(
                "[yellow]LLM preflight returned valid JSON without usage; "
                "evaluation will continue and record usage_missing_count.[/yellow]"
            )
            return
        raise typer.BadParameter(
            "LLM preflight failed; check the API base URL, model name, and server logs"
        )


def _preflight_deepseek() -> None:
    _preflight_llm("deepseek")


def _report_llm_quality(label: str, payload: dict) -> None:
    usage = payload.get("llm_usage")
    usage_missing_count = int(payload.get("usage_missing_count", 0) or 0)
    fallback_count = int(payload.get("fallback_count", 0) or 0)
    if not usage or not usage.get("total_tokens"):
        console.print(
            f"[yellow]{label}: no aggregate LLM usage was returned; "
            "predictions are retained and usage is marked incomplete.[/yellow]"
        )
    if usage_missing_count:
        console.print(
            f"[yellow]{label}: {usage_missing_count} sessions returned no usage metadata; "
            "these sessions were retained.[/yellow]"
        )
    if fallback_count:
        console.print(
            f"[red]{label}: {fallback_count} sessions used heuristic intention fallback; "
            "see llm_anomalies in the output JSON.[/red]"
        )


def _metric_delta(left: dict, right: dict) -> dict:
    metrics = ("Hit@1", "Hit@5", "Hit@10", "NDCG@1", "NDCG@5", "NDCG@10", "MRR")
    return {metric: round(float(right[metric]) - float(left[metric]), 6) for metric in metrics}


def _print_stratified_report(report: dict) -> None:
    """Print overall and history-status accuracy from the same session ranks."""
    overall = report["overall"]
    console.print(
        f"\n[bold]【整体】[/bold] n={overall['n']} | "
        f"Acc@1={overall['Acc@1']:.4f} Acc@5={overall['Acc@5']:.4f} "
        f"Acc@10={overall['Acc@10']:.4f} MRR={overall['MRR']:.4f}"
    )
    dim_titles = {
        "ooh": "D1 历史状态(IH=目标 POI 出现在用户前 80% 历史中; OOH=未出现)",
        "cold_start": "D2 冷启动(用户训练段交互次数)",
        "context_len": "D3 上下文长度(当前 session 已知 check-in 数)",
    }
    for dim, groups in report["strata"].items():
        console.print(f"\n[bold cyan]=== {dim_titles.get(dim, dim)} ===[/bold cyan]")
        console.print(f"{'切片':<16}{'n':>7}{'占比':>8}{'Acc@1':>9}{'Acc@5':>9}{'Acc@10':>9}{'MRR':>9}")
        for label, m in groups.items():
            console.print(
                f"{label:<16}{m['n']:>7}{m['share']:>8.3f}{m['Acc@1']:>9.4f}"
                f"{m['Acc@5']:>9.4f}{m['Acc@10']:>9.4f}{m['MRR']:>9.4f}"
            )
    sub = report["in_history_subset"]
    console.print(
        f"\n[bold green]【IH 子集】[/bold green] "
        f"n={sub['n']}({sub['share']:.1%} 分母) | "
        f"Acc@1={sub['Acc@1']:.4f} Acc@5={sub['Acc@5']:.4f} "
        f"Acc@10={sub['Acc@10']:.4f} MRR={sub['MRR']:.4f}"
    )
