"""Resumable eight-arm validation followed by NYC-first autonomous full tests."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iaa_agent.agent_runtime import atomic_json, read_json, digest, now, load_tokenizer
from iaa_agent.agent_types import AgentConfig
from iaa_agent.data import NYCDataRepository
from iaa_agent.engine import RunConfig
from iaa_agent.evidence import EvidenceStore
from scripts.autonomous_eval_support import (ARMS, AUTONOMOUS_ARMS, MODES, select_new_validation,
    run_variant, prediction_row, summarize, holm_adjust, quality_errors, metrics)
from scripts.report_autonomous import render_report


@contextmanager
def process_lock(path):
    """OS lock is released on crash; lock-file presence alone never blocks resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def source_hashes():
    paths = list((ROOT / "iaa_agent").glob("*.py"))
    paths += [ROOT / "scripts" / n for n in ("evaluate_autonomous.py", "autonomous_eval_support.py", "mm_ablation_support.py", "report_autonomous.py")]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p.read_bytes()) for p in sorted(paths)}


def rolling_results(run, cases, *, concurrency, failure_limit=4):
    """Bound admitted work, refill completed slots, and drain on failure/interrupt."""
    source = iter(enumerate(cases))
    failures = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}

        def refill():
            while len(pending) < concurrency:
                item = next(source, None)
                if item is None:
                    break
                index, case = item
                pending[pool.submit(run, case)] = index

        refill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Account for every ready result before admitting replacements. This
            # keeps a burst of provider failures from bypassing the stop gate.
            while done:
                for future in sorted(done, key=pending.get):
                    pending.pop(future)
                    row = future.result()
                    failures += row["status"] == "failed"
                    yield row
                done = {future for future in pending if future.done()}
            if failures < failure_limit:
                refill()


def baseline_audit(args, city, repo, store):
    result = {}
    source = args.source_experiment
    for filename in ("engine.py", "data.py", "models.py", "evidence.py", "utils.py"):
        frozen = source / "code" / "iaa_agent" / filename
        if digest(frozen.read_bytes()) != digest((ROOT / "iaa_agent" / filename).read_bytes()):
            raise ValueError(f"Legacy statistical implementation changed: {filename}")
    session_path = source / "results" / f"{city}_full" / "sessions.json"
    sessions = read_json(session_path)
    keys = repo.iter_session_test_keys(train_ratio=.8)
    if sessions["train_ratio"] != .8 or {tuple(k) for k in sessions["keys"]} != set(keys):
        raise ValueError("Full baseline session identities or split mismatch")
    for mode, legacy_name in (("text", "text"), ("both", "multimodal")):
        path = source / "results" / f"{city}_full" / f"{legacy_name}.json"
        payload = read_json(path)
        if payload["total"] != len(keys) or payload["session_sha256"] != sessions["session_sha256"]:
            raise ValueError("Baseline session digest/count mismatch")
        if payload["model"] != args.model or payload["thinking"] or payload["max_tokens"] != 4096:
            raise ValueError("Baseline model settings mismatch")
        if payload["csv_sources"] != store.csv_sources:
            raise ValueError("Baseline CSV fingerprint mismatch")
        if mode == "both" and payload["evidence_snapshot"]["snapshot_id"] != store.snapshot_id:
            raise ValueError("Baseline evidence fingerprint mismatch")
        cfg = dict(payload["run_config"])
        cfg["evidence_snapshot"] = None
        expected = asdict(RunConfig.p4(llm_mode="openai"))
        expected["intention_context_size"] = 5
        if cfg != expected:
            raise ValueError("Baseline RunConfig differs from the frozen reference")
        anomalies = payload.get("llm_anomalies", [])
        acceptable_repair = (city == "TKY" and mode == "both" and len(anomalies) == 1
                            and anomalies[0]["user_id"] == "1989" and anomalies[0]["trajectory_id"] == "1989_33"
                            and anomalies[0]["status"] == "invalid_intention"
                            and payload["fallback_count"] == 1 and payload["usage_missing_count"] == 0)
        if not payload["quality"]["valid"] and not acceptable_repair:
            raise ValueError("Baseline has unexpected invalid cases; refuse silent reuse")
        result[mode] = {"path": str(path), "sha256": digest(path.read_bytes()),
                        "valid": payload["quality"]["valid"], "repair_required": acceptable_repair,
                        "session_sha256": sessions["session_sha256"]}
    return result, sessions


def audit_repair_source(directory, protocol):
    """Reuse an already verified, single-request legacy repair with explicit provenance."""
    source = Path(directory)
    manifest_path = source / "manifest.json"
    manifest = read_json(manifest_path)
    previous = manifest["protocol"]
    if digest(previous) != manifest["protocol_sha256"] or previous["model"] != protocol["model"]:
        raise ValueError("Historical repair protocol/model mismatch")
    result = {}
    for city in protocol["execution"]["cities"]:
        baseline = protocol["baselines"][city]["both"]
        if not baseline["repair_required"]:
            continue
        if previous["inputs"][city] != protocol["inputs"][city]:
            raise ValueError("Historical repair data/evidence mismatch")
        for filename in ("engine.py", "data.py", "models.py", "evidence.py", "utils.py"):
            key = "iaa_agent/" + filename
            if previous["code_sha256"][key] != protocol["code_sha256"][key]:
                raise ValueError("Historical repair legacy implementation mismatch")
        report_path = source / "reused_baselines" / f"{city}_both_repaired.json"
        prediction_path = source / "baseline_repair" / city / "1989__1989_33" / "prediction.json"
        report, cached = read_json(report_path), read_json(prediction_path)
        prediction = cached["prediction"]
        if cached["identity"] != {"identity": manifest["protocol_sha256"], "engine": "fixed", "mode": "both", "query_id": "session_1989_33"}:
            raise ValueError("Historical repair request identity mismatch")
        if report["original"]["sha256"] != baseline["sha256"] or digest(prediction) != report["repair_prediction_sha256"]:
            raise ValueError("Historical repair result/source hash mismatch")
        accounting = prediction["accounting"]
        if (prediction["engine"] != "fixed" or prediction.get("heuristic_fallback") or accounting["requests"] != 1
                or accounting["invalid_attempts"] or accounting["usage_missing_count"] or not accounting["usage"].get("total_tokens")):
            raise ValueError("Historical repair was not a single valid, fully accounted request")
        result[city] = {"prediction_path": str(prediction_path), "prediction_file_sha256": digest(prediction_path.read_bytes()),
                        "source_manifest": str(manifest_path), "source_manifest_sha256": digest(manifest_path.read_bytes()),
                        "source_report": str(report_path), "source_report_sha256": digest(report_path.read_bytes()),
                        "new_model_requests": 0}
    return result


def make_manifest(args, config):
    old = read_json(args.source_ablation / "results" / "manifest.json")
    selections, inputs, baselines, full_sessions = {}, {}, {}, {}
    for city in args.cities:
        repo = NYCDataRepository(args.data_root / city)
        name = "NYC_53c3456f6380.json" if city == "NYC" else "TKY_951b9299d69c.json"
        snapshot_path = args.source_experiment / "evidence" / name
        store = EvidenceStore(snapshot_path)
        store.validate_repository(repo)
        prior_cases = old["protocol"]["selections"][city]["cases"]
        selections[city] = select_new_validation(repo, prior_cases, args.development_size, args.validation_size, args.repeat_size)
        baselines[city], sessions = baseline_audit(args, city, repo, store)
        full_sessions[city] = {"keys": sessions["keys"], "session_sha256": sessions["session_sha256"],
                               "smoke_keys": read_json(args.source_experiment / "results" / f"{city}_smoke" / "sessions.json")["keys"]}
        inputs[city] = {"snapshot": str(snapshot_path), "snapshot_id": store.snapshot_id, "csv_sources": store.csv_sources}
    protocol = {"schema_version": 1, "code_sha256": source_hashes(), "agent_config": asdict(config),
        "model": {"name": args.model, "base_url": args.base_url, "temperature": 0, "seed": 42,
                  "thinking": False, "max_model_len": 16384, "tokenizer_path": str(args.tokenizer_path)},
        "tokenizer_sha256": {p.name: digest(p.read_bytes()) for p in sorted(args.tokenizer_path.glob("*"))
                            if p.is_file() and (p.name.startswith("tokenizer") or p.name in {"chat_template.jinja", "special_tokens_map.json"})},
        "inputs": inputs, "selections": selections, "baselines": baselines, "full_sessions": full_sessions,
        "source_ablation_manifest_sha256": digest((args.source_ablation / "results" / "manifest.json").read_bytes()),
        "validation_arms": list(ARMS), "full_arms": list(AUTONOMOUS_ARMS),
        "split": {"validation_history": .7, "validation_targets": "[70%,80%)", "test_history": .8,
                  "external_evidence": "undated_static_snapshot", "catalog": "transductive_metadata_only",
                  "context_ties": "retain original session sequence; flag ties and report strict-time sensitivity",
                  "history_boundary": "strictly before target timestamp"},
        "execution": {"cities": args.cities, "concurrency": args.concurrency,
                      "scheduler": "bounded_rolling_sessions_v1",
                      "automatic_full": args.full, "development_only": args.development_only,
                      "abort_after_case_failures": 4},
        "statistics": {"bootstrap": 2000, "user_cluster_permutations": 10000, "seed": 42,
                       "primary_validation": ["autonomous__both_minus_fixed_schedule__both", "autonomous__both_minus_autonomous__text"],
                       "primary_full": ["autonomous__both_minus_fixed__both", "autonomous__both_minus_autonomous__text"],
                       "holm": "primary Hit@10 comparisons across cities, separate validation/full families"},
        "environment": {"python": platform.python_version(), "packages": {n: importlib.metadata.version(n)
                         for n in ("numpy", "pandas", "pydantic", "scikit-learn", "transformers", "vllm")}}}
    protocol["reused_repairs"] = audit_repair_source(args.baseline_repair_source, protocol) if args.baseline_repair_source else {}
    protocol = json.loads(json.dumps(protocol))
    identity = digest(protocol)
    path = args.output_dir / "manifest.json"
    if path.exists():
        previous = read_json(path)
        if previous["protocol_sha256"] != identity or previous["protocol"] != protocol:
            raise ValueError("Resume refused: code/data/config/environment changed; choose a new experiment directory")
        return previous
    manifest = {"created_at": now(), "protocol_sha256": identity, "protocol": protocol}
    atomic_json(path, manifest)
    return manifest


class Experiment:
    def __init__(self, args, manifest, tokenizer, config):
        self.args, self.manifest, self.tokenizer, self.config = args, manifest, tokenizer, config
        self.identity = manifest["protocol_sha256"]
        self.protocol = manifest["protocol"]
        self.progress_path = args.output_dir / "progress.json"
        self.lock = threading.Lock()
        self.state = {"status": "running", "started_at": now(), "pid": os.getpid(),
                      "protocol_sha256": self.identity, "completed_stages": [], "active": {}}
        self.last_progress_write = 0.
        self.local = threading.local()
        self.stores = {}
        self.baseline_rows = {}
        self.summary_sets = {"validation": {}, "full": {}}

    def update(self, **values):
        with self.lock:
            self.state.update(values, updated_at=now())
            atomic_json(self.progress_path, self.state)

    def heartbeat(self, key, event, detail):
        with self.lock:
            self.state["active"][key] = {"event": event, "detail": detail, "at": now()}
            self.state["updated_at"] = now()
            if event in {"model_started", "model_finished"} or time.monotonic() - self.last_progress_write > 5:
                atomic_json(self.progress_path, self.state)
                self.last_progress_write = time.monotonic()

    def repository(self, city, ratio):
        if getattr(self.local, "repo_key", None) != (city, ratio):
            self.local.repo = NYCDataRepository(self.args.data_root / city)
            self.stores[city].validate_repository(self.local.repo)
            self.local.repo.use_user_chronological_split(ratio)
            self.local.repo.prewarm_global_structures()
            self.local.repo_key = (city, ratio)
        return self.local.repo

    def case_path(self, phase, city, case):
        return self.args.output_dir / "cases" / phase / city / f"{case['user_id']}__{case['trajectory_id']}.json"

    def run_case(self, phase, city, case, arms, ratio, repeat=False):
        path = self.case_path(phase, city, case)
        key = f"{phase}/{city}/{case['user_id']}__{case['trajectory_id']}"
        names = list(arms) + ([a + "__repeat" for a in AUTONOMOUS_ARMS] if repeat and case.get("repeat") else [])
        if path.exists():
            saved = read_json(path)
            if saved["protocol_sha256"] != self.identity or saved["case"] != case:
                raise ValueError("Case identity mismatch")
            if saved["status"] == "completed":
                if set(saved["variants"]) != set(names):
                    raise ValueError("Incomplete cached arm matrix")
                return saved
        output = {"protocol_sha256": self.identity, "case": case, "city": city,
                  "user_id": case["user_id"], "trajectory_id": case["trajectory_id"],
                  "status": "running", "variants": {}, "started_at": now()}
        try:
            repo = self.repository(city, ratio)
            query = repo.get_session_query(case["user_id"], case["trajectory_id"], train_ratio=ratio)
            if phase != "full" and query.target_index != case["target_index"]:
                raise ValueError("Validation target index changed")
            output["ground_truth_poi_id"] = str(query.target["POI_id"])
            output["equal_time_context_count"] = int((query.context["UTC_time"] == query.target["UTC_time"]).sum())
            output["history_group"] = "IH" if str(query.target["POI_id"]) in set(query.history["POI_id"].astype(str)) else "OOH"
            baseline = {}
            # Interleave modalities in a stable label-independent way; A still precedes B.
            modes = MODES if int(digest([city, case["user_id"], case["trajectory_id"]])[:8], 16) % 2 == 0 else tuple(reversed(MODES))
            ordered = [a for engine in ("fixed", "fixed_llm_rank", "fixed_schedule", "autonomous")
                       for mode in modes if (a := f"{engine}__{mode}") in names]
            ordered += [a for a in names if a.endswith("__repeat")]
            for name in ordered:
                engine, mode, *_ = name.split("__")
                directory = self.args.output_dir / "runs" / phase / city / f"{case['user_id']}__{case['trajectory_id']}" / name
                result = run_variant(engine, mode, repo, query, self.stores[city] if mode == "both" else None,
                    directory, self.identity, self.tokenizer, self.config,
                    heartbeat=lambda event, detail: self.heartbeat(key, event, name + ":" + detail),
                    baseline_result=baseline.get(mode))
                if engine == "fixed":
                    baseline[mode] = result
                output["variants"][name] = prediction_row(result, query)
                atomic_json(path, output)
            output.update(status="completed", finished_at=now())
        except Exception as exc:
            output.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), finished_at=now())
        atomic_json(path, output)
        with self.lock:
            self.state["active"].pop(key, None)
        return output

    def stage(self, phase, city, cases, arms, ratio, repeat=False, stage_label=None):
        label = stage_label or f"{city}_{phase}"
        self.update(stage=label, completed=0, failed=0, total=len(cases))
        completed, failed = 0, 0
        rows = []
        run = lambda case: self.run_case(phase, city, case, arms, ratio, repeat)
        for row in rolling_results(run, cases, concurrency=self.args.concurrency,
                                   failure_limit=self.protocol["execution"]["abort_after_case_failures"]):
            rows.append(row)
            completed += row["status"] == "completed"
            failed += row["status"] == "failed"
            self.update(completed=completed, failed=failed)
            print(f"{label}: {completed} completed, {failed} failed / {len(cases)}", flush=True)
        rows.sort(key=lambda c: (c["user_id"], c["trajectory_id"]))
        summary = summarize(rows, cases, arms, repeats=repeat)
        if phase == "full":
            paired_rows = self.attach_baselines(city, rows)
            summary = summarize(paired_rows, cases, list(arms) + ["fixed__text", "fixed__both"])
            self.annotate_baseline_costs(city, summary, len(cases))
            strict = [c for c in paired_rows if c.get("equal_time_context_count", 0) == 0]
            summary["timestamp_ties"] = [{"user_id": c["user_id"], "trajectory_id": c["trajectory_id"],
                                          "context_ties": c["equal_time_context_count"]}
                                         for c in paired_rows if c.get("equal_time_context_count", 0)]
            if len(strict) != len(paired_rows):
                strict_ids = {(c["user_id"], c["trajectory_id"]) for c in strict}
                sensitivity = summarize(strict, [c for c in cases if (c["user_id"], c["trajectory_id"]) in strict_ids],
                                        list(arms) + ["fixed__text", "fixed__both"])
                self.annotate_baseline_costs(city, sensitivity, len(strict))
                atomic_json(self.args.output_dir / "summaries" / f"{label}_strict_time_sensitivity.json", sensitivity)
        attempts = []
        for case in cases:
            directory = self.args.output_dir / "runs" / phase / city / f"{case['user_id']}__{case['trajectory_id']}"
            for request in directory.glob("*/calls/*.json"):
                attempts.extend(read_json(request)["attempts"])
        summary["physical_request_accounting"] = {
            "requests": len(attempts), "invalid_or_interrupted": sum(not a.get("accepted") for a in attempts),
            "usage_missing": sum(not a.get("usage") for a in attempts),
            "total_tokens": sum((a.get("usage") or {}).get("total_tokens", 0) for a in attempts),
            "note": "Includes unsuccessful cases/attempts; shared A intentions counted once, old baseline requests excluded."}
        atomic_json(self.args.output_dir / "summaries" / f"{label}.json", summary)
        render_report(self.args.output_dir)
        if not summary["quality"]["valid"]:
            raise RuntimeError(f"{label}: quality gate failed: {summary['quality']['errors']}")
        self.state["completed_stages"].append(label)
        self.update()
        return summary

    def load_baselines(self, city):
        per_mode = {}
        for mode in MODES:
            identity = self.protocol["baselines"][city][mode]
            path = Path(identity["path"])
            if digest(path.read_bytes()) != identity["sha256"]:
                raise ValueError("Original baseline changed since manifest freeze")
            payload = read_json(path)
            anomalies = {(x["user_id"], x["trajectory_id"]) for x in payload.get("llm_anomalies", [])}
            rows = {}
            for old in payload["candidate_diagnostics"]["sessions"]:
                key = (old["user_id"], old["trajectory_id"])
                rows[key] = {**old, "in_observed": old["in_pool"], "raw_size": None,
                    "accounting": {"requests": 1, "logical_calls": 1, "retries": 0, "invalid_attempts": int(key in anomalies),
                                   "usage_missing_count": 0, "usage": {}, "model_seconds": 0},
                    "elapsed_seconds": 0, "tool_calls": None, "valid": key not in anomalies,
                    "observed_definition": "legacy_candidate_pool_proxy",
                    "heuristic_fallback": key in anomalies, "provenance": identity,
                    "cost_granularity": "historical_aggregate_only"}
            if len(rows) != payload["total"]:
                raise ValueError("Baseline has duplicate/missing per-session records")
            if identity["repair_required"]:
                repo = self.repository(city, .8)
                query = repo.get_session_query("1989", "1989_33", train_ratio=.8)
                directory = self.args.output_dir / "baseline_repair" / city / "1989__1989_33"
                reused = self.protocol.get("reused_repairs", {}).get(city)
                if reused:
                    source = Path(reused["prediction_path"])
                    if digest(source.read_bytes()) != reused["prediction_file_sha256"]:
                        raise ValueError("Reused repair changed since manifest freeze")
                    repaired = read_json(source)["prediction"]
                else:
                    repaired = run_variant("fixed", "both", repo, query, self.stores[city], directory,
                        self.identity, self.tokenizer, self.config,
                        heartbeat=lambda e, d: self.heartbeat("baseline_repair", e, d))
                row = prediction_row(repaired, query)
                if row["accounting"]["usage_missing_count"]:
                    raise ValueError("Baseline repair has missing request accounting")
                row.update(ground_truth_poi_id=str(query.target["POI_id"]), provenance=identity)
                rows[("1989", "1989_33")] = row
                aggregate = {k: float(np.mean([metrics(r)[k] for r in rows.values()]))
                             for k in ("Hit@1", "Hit@5", "Hit@10", "NDCG@10", "MRR@10", "CandidateRecall", "RawCandidateRecall")}
                atomic_json(self.args.output_dir / "reused_baselines" / f"{city}_{mode}_repaired.json",
                    {"original": identity, "repair_case": {"user_id": "1989", "trajectory_id": "1989_33"},
                     "repair_prediction_sha256": digest(repaired), "repair_accounting": repaired["accounting"],
                     "reused_repair": reused,
                     "metrics": aggregate, "total": len(rows), "original_preserved": True,
                     "cost_note": "Original failed attempt remains in historical usage; add repair usage, do not erase it."})
            per_mode[mode] = rows
        self.baseline_rows[city] = per_mode

    def attach_baselines(self, city, rows):
        result = []
        for case in rows:
            current = dict(case, variants=dict(case["variants"]))
            key = (case["user_id"], case["trajectory_id"])
            for mode in MODES:
                baseline = self.baseline_rows[city][mode][key]
                if baseline["ground_truth_poi_id"] != case["ground_truth_poi_id"]:
                    raise ValueError("Paired baseline ground truth mismatch")
                current["variants"]["fixed__" + mode] = baseline
            result.append(current)
        return result

    def annotate_baseline_costs(self, city, summary, count):
        for mode in MODES:
            identity = self.protocol["baselines"][city][mode]
            payload = read_json(identity["path"])
            cost = {"granularity": "historical_full_run_aggregate", "per_session_cost_available": False,
                    "full_run_total_tokens": payload["llm_usage"]["total_tokens"],
                    "full_run_elapsed_seconds": payload["elapsed_seconds"], "original": identity,
                    "note": "Historical timings were not remeasured; no fabricated per-session percentiles."}
            repair_path = self.args.output_dir / "reused_baselines" / f"{city}_{mode}_repaired.json"
            if repair_path.exists():
                cost["repair_accounting"] = read_json(repair_path)["repair_accounting"]
            summary["arms"]["fixed__" + mode]["cost"] = cost

    def execute(self):
        for city in self.args.cities:
            self.update(stage=f"{city}_load_evidence")
            self.stores[city] = EvidenceStore(self.protocol["inputs"][city]["snapshot"])
        # Development is excluded from all formal validation and full-test metrics.
        for city in self.args.cities:
            self.stage("development", city, self.protocol["selections"][city]["development"], ARMS, .7)
        if self.args.development_only:
            self.update(status="completed", finished_at=now(), scope="development_only")
            render_report(self.args.output_dir)
            return
        for city in self.args.cities:
            summary = self.stage("validation", city, self.protocol["selections"][city]["validation"], ARMS, .7, repeat=True)
            self.summary_sets["validation"][city] = summary
        adjusted = holm_adjust(self.summary_sets["validation"], self.protocol["statistics"]["primary_validation"])
        atomic_json(self.args.output_dir / "summaries" / "validation_primary_comparisons.json", adjusted)
        if self.args.full:
            for city in self.args.cities:
                self.load_baselines(city)
                definition = self.protocol["full_sessions"][city]
                make = lambda keys: [{"city": city, "user_id": u, "trajectory_id": t, "repeat": False} for u, t in keys]
                self.stage("full", city, make(definition["smoke_keys"]), AUTONOMOUS_ARMS, .8, stage_label=f"{city}_smoke50")
                self.summary_sets["full"][city] = self.stage("full", city, make(definition["keys"]), AUTONOMOUS_ARMS, .8)
            adjusted = holm_adjust(self.summary_sets["full"], self.protocol["statistics"]["primary_full"])
            atomic_json(self.args.output_dir / "summaries" / "full_primary_comparisons.json", adjusted)
        self.update(status="completed", finished_at=now())
        render_report(self.args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--source-ablation", type=Path, required=True)
    parser.add_argument("--baseline-repair-source", type=Path, help="Previous results directory with the verified TKY legacy repair")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path.home() / "Model/Qwen38-27B")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--cities", nargs="+", choices=["NYC", "TKY"], default=["NYC", "TKY"])
    parser.add_argument("--development-size", type=int, default=50)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--repeat-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.development_size, args.validation_size, args.concurrency) < 1 or not 0 <= args.repeat_size <= args.validation_size:
        parser.error("Invalid sample/concurrency settings")
    if len(args.cities) != len(set(args.cities)):
        parser.error("Duplicate cities")
    return args


def main():
    args = parse_args()
    # Background launchers can inherit SIGINT=SIG_IGN; restore Python's handler.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    os.environ.update(OPENAI_MODEL=args.model, OPENAI_BASE_URL=args.base_url,
        OPENAI_ENABLE_THINKING="0", OPENAI_TEMPERATURE="0", OPENAI_SEED="42",
        OPENAI_MAX_TOKENS="4096", OPENAI_TIMEOUT_SECONDS="180", TOKENIZERS_PARALLELISM="false")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    config = AgentConfig()
    with process_lock(args.output_dir / "runner.lock"):
        manifest = make_manifest(args, config)
        if args.prepare_only:
            print(json.dumps({"prepared": True, "protocol_sha256": manifest["protocol_sha256"]}), flush=True)
            return
        experiment = Experiment(args, manifest, load_tokenizer(args.tokenizer_path), config)
        try:
            experiment.execute()
        except BaseException as exc:
            experiment.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              error_type=type(exc).__name__, error=str(exc), finished_at=now())
            render_report(args.output_dir)
            raise


if __name__ == "__main__":
    main()
