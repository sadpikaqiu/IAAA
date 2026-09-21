"""Resumable, paired component ablations on an independent validation sample.

Run in an isolated experiment directory. Production code and evidence are hash
checked against the original frozen bundle. No original test labels/results are
used to select cases or interventions, and no heuristic fallback is evaluated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.mm_ablation_support import (
    ARMS, REPEAT_ARMS, CONTRASTS, AblationAgent, Arm, agent_config, canonical,
    compact_result, digest, masked_store, prepare_prompt, read, select_validation,
    summarize, validated_intention_attempt, write,
)
from iaa_agent.data import NYCDataRepository
from iaa_agent.engine import IAAAgent
from iaa_agent.evidence import EvidenceStore
from iaa_agent.llm import DeepSeekClient
from iaa_agent.models import Intention, ReflectionRecord


def now():
    return datetime.now(timezone.utc).isoformat()


MODEL_SETTINGS = {"OPENAI_MODEL": "Qwen/Qwen3.8-27B-FP8", "OPENAI_ENABLE_THINKING": "0",
                  "OPENAI_TEMPERATURE": "0", "OPENAI_SEED": "42", "OPENAI_MAX_TOKENS": "4096",
                  "OPENAI_TIMEOUT_SECONDS": "180", "OPENAI_REASONING_EFFORT": "medium"}


@contextmanager
def process_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_manifest(args):
    source = args.source_experiment.resolve()
    code = Path(__file__).resolve().parents[1]
    bundle = read(source / "bundle_manifest.json")
    code_hashes = {}
    for name, expected in bundle["files"].items():
        if not name.startswith("code/iaa_agent/"):
            continue
        source_hash = digest((source / name).read_bytes())
        current_hash = digest((code / Path(name).relative_to("code")).read_bytes())
        if source_hash != expected or current_hash != expected:
            raise ValueError(f"Production source differs from frozen original: {name}")
        code_hashes[name] = current_hash
    for name in ("scripts/mm_ablation_support.py", "scripts/run_multimodal_ablations.py"):
        code_hashes["code/" + name] = digest((code / name).read_bytes())
    inputs, selections = {}, {}
    for city in args.cities:
        names = [n for n in bundle["files"] if n.startswith(f"evidence/{city}_")]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one frozen snapshot for {city}")
        path = source / names[0]
        if digest(path.read_bytes()) != bundle["files"][names[0]]:
            raise ValueError("Frozen snapshot changed")
        snapshot = read(path)
        for name, expected in snapshot["csv_sources"].items():
            if digest((args.data_root / city / name).read_bytes()) != expected:
                raise ValueError(f"CSV source changed: {city}/{name}")
        inputs[city] = {"snapshot_path": str(path), "snapshot_sha256": bundle["files"][names[0]],
                        "snapshot_id": snapshot["snapshot_id"], "csv_sha256": snapshot["csv_sources"]}
        repo = NYCDataRepository(args.data_root / city)
        selections[city] = select_validation(repo, sample_size=args.sample_size, repeat_size=args.repeat_size)
        if selections[city]["selected_count"] != args.sample_size:
            raise ValueError(f"Insufficient eligible validation sessions for {city}")
        print(f"{city}: eligible={selections[city]['eligible_count']}, selected={args.sample_size}, "
              f"original_test_overlap=0, time_ties_excluded={selections[city]['excluded_non_strict_time_count']}", flush=True)
    protocol = {
        "schema_version": 1, "scope": "validation_component_attribution_not_test_confirmation",
        "data_root": str(args.data_root.resolve()), "source_experiment": str(source),
        "source_bundle_sha256": digest((source / "bundle_manifest.json").read_bytes()),
        "code_sha256": code_hashes, "inputs": inputs, "selections": selections,
        "split": {"global_and_user_history_ratio": .7, "target_window": "[70%,80%) per user",
                  "original_test": "[80%,100%) per user; no reused test sessions",
                  "chronology": "same per-user protocol as original; not a single global wall-clock split",
                  "external_evidence": "same static snapshots, unknown observation dates; no prospective freshness claim"},
        "sampling": {"seed": "mm-validation-20260917", "sample_size": args.sample_size,
                     "repeat_size": args.repeat_size, "labels_used_for_selection": False},
        "model": dict(MODEL_SETTINGS, OPENAI_BASE_URL=args.base_url),
        "arms": [asdict(a) for a in ARMS], "repeat_arms": [asdict(a) for a in REPEAT_ARMS],
        "contrasts": list(CONTRASTS), "base_config": asdict(agent_config(ARMS[0], None)),
        "modality_control": "joint TF-IDF vocabulary/IDF, masked snippets, fixed .05 coefficient per modality",
        "retry_policy": {"max_attempts_per_intention": 3, "accept": "schema_valid_and_status_success_and_usage_present_and_stop",
                         "heuristic_fallback": False, "retry_depends_on_ranking": False},
        "execution": {"concurrency": args.concurrency, "pilot_size": args.pilot_size,
                      "city_order": args.cities, "failures_before_abort": 10},
        "environment": {"python": platform.python_version(),
                        "packages": {k: importlib.metadata.version(k) for k in ("numpy", "pandas", "pydantic", "scikit-learn")}},
    }
    # JSON roundtrip normalizes tuple/list types before a resume comparison.
    import json
    protocol = json.loads(canonical(protocol))
    identity = digest(canonical(protocol))
    path = args.output_dir / "manifest.json"
    if path.exists():
        previous = read(path)
        if previous["protocol_sha256"] != identity or previous["protocol"] != protocol:
            raise ValueError("Resume refused: protocol, code, data, dependency version, or sample changed")
        return previous
    manifest = {"created_at": now(), "protocol_sha256": identity, "protocol": protocol}
    write(path, manifest)
    return manifest


def load_intention(client, messages, path, protocol_sha):
    request_hash = digest(canonical(messages))
    identity = {"protocol_sha256": protocol_sha, "messages_sha256": request_hash}
    if path.exists():
        record = read(path)
        if record["identity"] != identity or record["messages"] != messages:
            raise ValueError("Cached intention identity/prompt mismatch")
    else:
        record = {"identity": identity, "messages": messages, "attempts": [], "status": "pending"}
    for attempt in record["attempts"]:
        if attempt["valid"]:
            return Intention.model_validate(attempt["intention"]), record
    while len(record["attempts"]) < 3:
        started = time.monotonic()
        # Persist a started attempt too: an interrupted request consumes an
        # attempt rather than silently receiving unlimited retries on resume.
        record["attempts"].append({"valid": False, "status": "inflight_or_interrupted", "started_at": now()})
        write(path, record)
        result = validated_intention_attempt(client, messages)
        result.update(started_at=record["attempts"][-1]["started_at"], finished_at=now(),
                      elapsed_seconds=time.monotonic() - started)
        record["attempts"][-1] = result
        record["status"] = "success" if result["valid"] else "pending"
        write(path, record)
        if result["valid"]:
            return Intention.model_validate(result["intention"]), record
        if len(record["attempts"]) < 3:
            time.sleep(2)
    record["status"] = "failed"
    write(path, record)
    raise RuntimeError(f"No valid LLM intention after 3 attempts: {path.name}")


def case_path(output, case):
    return output / "cases" / case["city"] / f"{case['user_id']}__{case['trajectory_id']}.json"


def analyze(output, manifest, *, cities=None):
    results = {}
    for city, selection in manifest["protocol"]["selections"].items():
        if cities is not None and city not in cities:
            continue
        completed, all_rows = [], []
        for case in selection["cases"]:
            path = case_path(output, case)
            if path.exists():
                row = read(path)
                if row["protocol_sha256"] != manifest["protocol_sha256"]:
                    raise ValueError("Result protocol mismatch")
                all_rows.append(row)
                if row["status"] == "completed":
                    completed.append(row)
        summary = summarize(completed, selection["cases"])
        summary["failed_cases"] = sum(c["status"] == "failed" for c in all_rows)
        summary["llm_quality"] = {
            "all_selected_cases_completed": len(completed) == len(selection["cases"]),
            "all_evaluated_intentions_validated_llm": all(c["llm_quality"]["valid_intentions"] == len(c["intentions"]) for c in completed),
            "heuristic_fallback_count": 0,
            "accepted_usage_missing_count": 0,
            "evaluated_intentions": sum(len(c["intentions"]) for c in completed),
            "requests_for_completed_cases": sum(c["llm_quality"]["attempts"] for c in completed),
            "retry_attempts_for_completed_cases": sum(c["llm_quality"]["retries"] for c in completed),
        }
        # Include failed/in-flight requests in accounting, not just completed cases.
        caches = [read(p) for p in (output / "intentions" / city).glob("*/*.json")]
        attempts = [a for c in caches for a in c["attempts"]]
        summary["request_accounting"] = {
            "started_attempts": len(attempts), "invalid_or_interrupted_attempts": sum(not a["valid"] for a in attempts),
            "total_tokens": sum((a.get("usage") or {}).get("total_tokens", 0) for a in attempts),
            "attempts_without_usage": sum(a.get("usage") is None for a in attempts),
        }
        write(output / "summaries" / f"{city}.json", summary)
        results[city] = summary
    return results


def execute(args, manifest):
    protocol = manifest["protocol"]
    state_path = args.output_dir / "progress.json"
    state = {"status": "running", "started_at": now(), "protocol_sha256": manifest["protocol_sha256"],
             "total": sum(v["selected_count"] for v in protocol["selections"].values()),
             "cities": {}, "pid": os.getpid()}
    write(state_path, state)
    for city in args.cities:
        cases = protocol["selections"][city]["cases"]
        snapshot = Path(protocol["inputs"][city]["snapshot_path"])
        state.update(city=city, phase="building_evidence_index", updated_at=now())
        write(state_path, state)
        joint = EvidenceStore(snapshot)
        stores = {mode: masked_store(joint, mode) for mode in ("both", "images", "reviews")}
        local = threading.local()
        pilot_ids = {(c["user_id"], c["trajectory_id"]) for c in cases[:args.pilot_size]}
        city_state = {"total": len(cases), "completed": 0, "failed": 0}
        state["cities"][city] = city_state

        def run_case(case):
            path = case_path(args.output_dir, case)
            expected_arms = ARMS + REPEAT_ARMS if case["repeat"] else ARMS
            if path.exists():
                output = read(path)
                if output["protocol_sha256"] != manifest["protocol_sha256"] or output["case"] != case:
                    raise ValueError("Cached case identity changed")
                if output["status"] == "completed":
                    if set(output["variants"]) != {a.name for a in expected_arms}:
                        raise ValueError("Completed case has an incomplete arm matrix")
                    return output
            else:
                output = {"protocol_sha256": manifest["protocol_sha256"], "case": case,
                          **case, "started_at": now(), "status": "running", "variants": {}}
            try:
                if not hasattr(local, "repo"):
                    local.repo = NYCDataRepository(args.data_root / city)
                    local.repo.use_user_chronological_split(.7)
                    local.repo.prewarm_global_structures()
                repo = local.repo
                query = repo.get_session_query(case["user_id"], case["trajectory_id"], train_ratio=.7)
                if query.target_index != case["target_index"] or not case["history_cutoff"] <= query.target_index < case["test_cutoff"]:
                    raise AssertionError("Validation split changed")
                if (query.history["UTC_time"] >= query.target["UTC_time"]).any() or (query.context["UTC_time"] >= query.target["UTC_time"]).any():
                    raise AssertionError("Target is not strictly after visible events")
                intentions, records, prepared, prompts = {}, {}, None, {}
                for mode in ("text", "both", "images", "reviews"):
                    arm = Arm("prompt", mode=mode if mode != "text" else None)
                    agent = AblationAgent(repo, agent_config(arm, snapshot, live=True), arm=arm,
                                          evidence_store=stores.get(mode), prepared=prepared)
                    prompts[mode], prepared = prepare_prompt(agent, query)
                ids = ["text_0", "both_0", "images_0", "reviews_0"]
                if case["repeat"]:
                    ids += ["text_1", "both_1"]
                # Interleave prompt modes/repetitions deterministically by ID to
                # avoid always sending one condition in the same execution slot.
                ids.sort(key=lambda name: digest(canonical([case["city"], case["user_id"], case["trajectory_id"], name])))
                for name in ids:
                    cache = args.output_dir / "intentions" / city / f"{case['user_id']}__{case['trajectory_id']}" / f"{name}.json"
                    intentions[name], records[name] = load_intention(
                        DeepSeekClient(provider="openai"), prompts[name.rsplit("_", 1)[0]], cache, manifest["protocol_sha256"])
                output["intentions"] = {k: {"request_sha256": v["identity"]["messages_sha256"],
                                            "cache_sha256": digest(canonical(v)),
                                            "value": intentions[k].model_dump(mode="json")} for k, v in records.items()}
                output["llm_quality"] = {"valid_intentions": len(intentions),
                    "attempts": sum(len(r["attempts"]) for r in records.values()),
                    "retries": sum(len(r["attempts"]) - 1 for r in records.values()), "fallback_count": 0}
                # Access labels only for evaluation bookkeeping, outside agents.
                output["ground_truth_poi_id"] = str(query.target["POI_id"])
                training_pois = set(query.history["POI_id"].astype(str))
                output["history_group"] = "IH" if output["ground_truth_poi_id"] in training_pois else "OOH"
                text_reflection = None
                def forbidden(*a, **kw):
                    raise AssertionError("Cached ablation replay attempted a model request")
                for arm in expected_arms:
                    if arm.name in output["variants"]:
                        if arm.name == "text":
                            text_reflection = ReflectionRecord.model_validate(output["variants"][arm.name]["reflection"])
                        continue
                    started = time.monotonic()
                    agent = AblationAgent(repo, agent_config(arm, snapshot), arm=arm,
                                          evidence_store=stores.get(arm.mode), frozen_intention=intentions[arm.intent],
                                          prepared=prepared, text_reflection=text_reflection)
                    agent.llm.chat_json = forbidden
                    result = agent.run_query(query)
                    current = compact_result(agent, result, time.monotonic() - started)
                    if arm.name == "text":
                        text_reflection = result.reflection
                    if arm.reflection == "text_budget":
                        if current["reflection"]["triggered"] != text_reflection.triggered:
                            raise AssertionError("Text-budget reflection control diverged")
                    if arm.name == "rank_only":
                        text_rounds = output["variants"]["text"]["rounds"]
                        if [r["pool_ids"] for r in current["rounds"]] != [r["pool_ids"] for r in text_rounds]:
                            raise AssertionError("Ranking-only intervention changed candidate pools")
                    output["variants"][arm.name] = current
                    write(path, output)
                if (case["user_id"], case["trajectory_id"]) in pilot_ids:
                    # Verify the wrapper against the unchanged production engine
                    # using the actual accepted live intentions, not only fixtures.
                    for name in ("text", "full_both"):
                        arm = next(a for a in ARMS if a.name == name)
                        original = IAAAgent(repo, agent_config(arm, snapshot), evidence_store=stores.get(arm.mode))
                        original.llm.chat_json = forbidden
                        original._infer_intention = lambda *a, _intent=intentions[arm.intent], **kw: _intent.model_copy(deep=True)
                        reference = original.run_query(query)
                        current = output["variants"][name]
                        if ([p.poi_id for p in reference.ranked_pois] != current["predictions"]
                                or reference.candidate_pool_summary["candidate_poi_ids"] != current["rounds"][-1]["pool_ids"]
                                or reference.reflection.model_dump(mode="json") != current["reflection"]
                                or [p.score_decomposition for p in reference.ranked_pois] != [p["decomposition"] for p in current["top10_scores"]]):
                            raise AssertionError(f"Production equivalence failed for {name}")
                    output["production_equivalence_passed"] = True
                output.update(status="completed", finished_at=now())
                output.pop("error", None)
            except Exception as exc:
                output.update(status="failed", finished_at=now(),
                              error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            write(path, output)
            return output

        def batch(batch_cases, phase):
            state.update(phase=phase, updated_at=now())
            write(state_path, state)
            ok = True
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                jobs = {executor.submit(run_case, c): c for c in batch_cases}
                for job in as_completed(jobs):
                    case = jobs[job]
                    row = job.result()
                    city_state["completed" if row["status"] == "completed" else "failed"] += 1
                    ok = ok and row["status"] == "completed"
                    state.update(last_case=case["trajectory_id"], updated_at=now())
                    write(state_path, state)
                    print(f"{city} {phase} {city_state['completed']}/{len(cases)} failed={city_state['failed']} "
                          f"{case['trajectory_id']} {row['status']}", flush=True)
                    if (city_state["completed"] + city_state["failed"]) % 50 == 0:
                        analyze(args.output_dir, manifest, cities=[city])
                    if city_state["failed"] >= 10:
                        for pending in jobs:
                            pending.cancel()
                        raise RuntimeError("Stopping after 10 failed cases; inspect saved errors before resuming")
            return ok

        if not batch(cases[:args.pilot_size], "pilot"):
            analyze(args.output_dir, manifest, cities=[city])
            raise RuntimeError(f"{city} pilot quality gate failed; full stage not started")
        write(args.output_dir / "gates" / f"{city}_pilot.json", {
            "passed": True, "n": args.pilot_size, "checked_at": now(),
            "protocol_sha256": manifest["protocol_sha256"],
            "checks": ["all_arms_completed", "production_engine_equivalence", "no_heuristic_fallback", "accepted_usage_present", "rank_only_preserves_pools",
                       "reflection_budget_control", "strict_visible_time_order", "original_test_disjoint"]})
        batch(cases[args.pilot_size:], "validation")
        analyze(args.output_dir, manifest, cities=[city])
        city_state["status"] = "completed" if not city_state["failed"] else "failed"
        write(state_path, state)
        del stores, joint
    state.update(status="completed" if all(v["status"] == "completed" for v in state["cities"].values()) else "failed",
                 phase="finished", finished_at=now())
    write(state_path, state)
    return 0 if state["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cities", nargs="+", choices=["NYC", "TKY"], default=["NYC", "TKY"])
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--repeat-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--pilot-size", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--stage", choices=["prepare", "run", "analyze"], default="run")
    args = parser.parse_args(argv)
    if (not 1 <= args.pilot_size <= args.sample_size or not 0 <= args.repeat_size <= args.sample_size
            or args.concurrency < 1 or len(args.cities) != len(set(args.cities))):
        parser.error("Invalid sample/repetition/pilot/concurrency/city settings")
    os.environ.update(MODEL_SETTINGS, OPENAI_BASE_URL=args.base_url, OPENAI_API_KEY="EMPTY", TOKENIZERS_PARALLELISM="false")
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    with process_lock(args.output_dir / "runner.lock"):
        manifest = prepare_manifest(args)
        if args.stage == "prepare":
            print(f"Protocol frozen: {manifest['protocol_sha256']}", flush=True)
            return 0
        if args.stage == "analyze":
            analyze(args.output_dir, manifest)
            return 0
        try:
            return execute(args, manifest)
        except BaseException as exc:
            path = args.output_dir / "progress.json"
            state = read(path) if path.exists() else {}
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error_type=type(exc).__name__, error=str(exc), updated_at=now())
            write(path, state)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
