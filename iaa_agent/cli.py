from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .data import NYCDataRepository
from .engine import IAAAgent, RunConfig
from .evaluation import evaluate as evaluate_runs
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
    console.print({"traj_id": traj_id, "top1": payload["ranked_pois"][0]["poi_id"] if payload["ranked_pois"] else None})


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
    limit: int = typer.Option(50, help="Number of test trajectories to evaluate; use 0 for all"),
    out: str = typer.Option("outputs/evaluation/evaluation_results.json", help="Metrics JSON output"),
    llm: str = typer.Option("fake", help="LLM mode: fake or deepseek"),
) -> None:
    repo = NYCDataRepository(data_dir)
    actual_limit = None if limit == 0 else limit
    result = evaluate_runs(repo, limit=actual_limit, llm_mode=llm)
    payload = result.as_dict()
    write_json(out, payload)
    console.print(f"Wrote evaluation results to {out}")
    console.print(payload)

