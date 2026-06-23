from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .data import NYCDataRepository
from .engine import IAAAgent, RunConfig
from .evaluation import evaluate_session_split
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
    llm: str = typer.Option("fake", help="LLM mode: fake or deepseek"),
) -> None:
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
    llm: str = typer.Option("fake", help="LLM mode: fake or deepseek"),
) -> None:
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
    llm: str = typer.Option("fake", help="LLM mode: fake or deepseek"),
) -> None:
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
    llm: str = typer.Option("fake", help="LLM mode: fake or deepseek"),
) -> None:
    repo = NYCDataRepository(data_dir)
    actual_smoke_limit = None if smoke_limit == 0 else smoke_limit
    try:
        result = evaluate_session_split(
            repo,
            train_ratio=train_ratio,
            min_context=min_context,
            smoke_limit=actual_smoke_limit,
            user_id=user_id,
            save_runs_dir=save_runs,
            llm_mode=llm,
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
    }
    write_json(out, payload)
    if save_runs is not None:
        write_json(Path(save_runs) / "summary.json", payload)
    console.print(f"Wrote evaluation results to {out}")
    if save_runs is not None:
        console.print(f"Wrote per-session traces to {save_runs}")
    console.print(payload)


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
