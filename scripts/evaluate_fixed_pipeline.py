"""Run paired fixed-pipeline smoke/full experiments in a persistent screen job."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iaa_agent.data import NYCDataRepository
from iaa_agent.engine import RunConfig
from iaa_agent.evaluation import evaluate_session_split_threaded
from iaa_agent.evidence import EvidenceStore
from iaa_agent.llm import DeepSeekClient
from iaa_agent.poi_image_summary import atomic_json, now


METRICS = ("Hit@1", "Hit@5", "Hit@10", "NDCG@1", "NDCG@5", "NDCG@10", "MRR")


def session_digest(keys):
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()


def sample_keys(keys, size, seed, city):
    selected = sorted(keys, key=lambda pair: hashlib.sha256(
        f"{seed}/{city}/{pair[0]}/{pair[1]}".encode()).hexdigest())[:size]
    return sorted(selected)


def quality_errors(payload, keys):
    errors = []
    if payload.get("total") != len(keys) or not keys:
        errors.append("session_count_mismatch")
    if payload.get("fallback_count") != 0:
        errors.append("heuristic_fallback")
    if payload.get("usage_missing_count") != 0:
        errors.append("missing_usage")
    if payload.get("all_sessions_used_llm") is not True:
        errors.append("not_all_sessions_used_llm")
    rows = payload.get("candidate_diagnostics", {}).get("sessions", [])
    actual = sorted((row["user_id"], row["trajectory_id"]) for row in rows)
    if actual != sorted(keys):
        errors.append("session_identity_mismatch")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--nyc-snapshot", type=Path, required=True)
    parser.add_argument("--tky-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cities", choices=["NYC", "TKY"], nargs="+", default=["NYC", "TKY"])
    parser.add_argument("--smoke-size", type=int, default=50)
    parser.add_argument("--full", action="store_true", help="After each city's valid smoke pair, run its full pair")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    args = parser.parse_args(argv)
    if args.smoke_size < 1 or args.concurrency < 1 or len(args.cities) != len(set(args.cities)):
        parser.error("Use positive sizes/concurrency and unique cities")
    if "TKY" in args.cities and args.tky_snapshot is None:
        parser.error("--tky-snapshot is required when evaluating TKY")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "progress.json"
    if status_path.exists():
        parser.error("Choose a new output directory; existing experiments are never overwritten")
    state = {"status": "preparing", "started_at": now(), "pid": os.getpid(), "completed_stages": [],
             "plan": [f"{city}_{stage}" for city in args.cities for stage in (["smoke", "full"] if args.full else ["smoke"])],
             "model": args.model, "base_url": args.base_url, "concurrency": args.concurrency,
             "python": sys.executable, "thinking": False, "max_tokens": 4096,
             "intention_context_size": 5, "sample_seed": args.seed}

    def update(**values):
        state.update(values, updated_at=now())
        atomic_json(status_path, state)

    os.environ.update(OPENAI_BASE_URL=args.base_url, OPENAI_MODEL=args.model,
                      OPENAI_ENABLE_THINKING="0", OPENAI_TEMPERATURE="0", OPENAI_MAX_TOKENS="4096",
                      OPENAI_TIMEOUT_SECONDS="180", TOKENIZERS_PARALLELISM="false")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    try:
        update()
        # Freeze/check all input identities before any model request.
        inputs = {}
        for city in args.cities:
            path = args.nyc_snapshot if city == "NYC" else args.tky_snapshot
            evidence = EvidenceStore(path)
            repo = NYCDataRepository(args.data_root / city)
            evidence.validate_repository(repo)
            keys = repo.iter_session_test_keys(train_ratio=.8, min_context=1)
            inputs[city] = (repo, evidence, keys, path)
            print(f"Input ready: {city}; sessions={len(keys)}; snapshot={evidence.snapshot_id}", flush=True)
        probe = DeepSeekClient(provider="openai")
        reply = probe.chat_json([{"role": "user", "content": 'Return only the JSON object {"ok": true}.'}])
        if not reply or reply.get("ok") is not True or not probe.last_usage:
            raise RuntimeError(f"Model preflight failed: {probe.last_call_status}")
        atomic_json(args.output_dir / "preflight.json", {"status": probe.last_call_status, "usage": probe.last_usage})
        for city in args.cities:
            repo, evidence, all_keys, snapshot_path = inputs[city]
            for stage in (["smoke", "full"] if args.full else ["smoke"]):
                keys = sample_keys(all_keys, args.smoke_size, args.seed, city) if stage == "smoke" else list(all_keys)
                stage_id = f"{city}_{stage}"
                directory = args.output_dir / stage_id
                directory.mkdir()
                identity = session_digest(keys)
                atomic_json(directory / "sessions.json", {"city": city, "stage": stage,
                            "session_sha256": identity, "keys": keys, "train_ratio": .8, "min_context": 1,
                            "all_city_sessions": len(all_keys), "sample_seed": args.seed})
                variants = {}
                for variant in ("text", "multimodal"):
                    update(status="running", stage=stage_id, variant=variant, completed_sessions=0,
                           total_sessions=len(keys), session_sha256=identity)
                    config = RunConfig.p4(llm_mode="openai")
                    config.intention_context_size = 5
                    if variant == "multimodal":
                        config.evidence_snapshot = str(snapshot_path.resolve())
                    completed = 0
                    started = time.monotonic()

                    def progress():
                        nonlocal completed
                        completed += 1
                        update(completed_sessions=completed, elapsed_seconds=round(time.monotonic() - started, 2))
                        if completed % 5 == 0 or completed == len(keys):
                            print(f"{stage_id}/{variant}: {completed}/{len(keys)}", flush=True)

                    result = evaluate_session_split_threaded(
                        repo, run_config=config, session_keys=keys, concurrency=args.concurrency,
                        strict_llm=True, stall_timeout_seconds=600, report_stratified=True,
                        report_candidates=True, evidence_store=evidence if variant == "multimodal" else None,
                        progress_callback=progress)
                    payload = result.as_dict() | {"city": city, "stage": stage, "variant": variant,
                              "run_config": asdict(config), "session_sha256": identity,
                              "model": args.model, "thinking": False, "max_tokens": 4096,
                              "elapsed_seconds": round(time.monotonic() - started, 3),
                              "csv_sources": evidence.csv_sources,
                              "evidence_snapshot": evidence.metadata() if variant == "multimodal" else None}
                    errors = quality_errors(payload, keys)
                    payload["quality"] = {"valid": not errors, "errors": errors}
                    atomic_json(directory / f"{variant}.json", payload)
                    if errors:
                        raise RuntimeError(f"{stage_id}/{variant}: quality gate failed: {', '.join(errors)}")
                    variants[variant] = payload
                paired = {"city": city, "stage": stage, "total": len(keys), "session_sha256": identity,
                          "valid": True, "scope": "sample_only" if stage == "smoke" else "full_session_split",
                          "delta_multimodal_minus_text": {metric: round(variants["multimodal"][metric] - variants["text"][metric], 6) for metric in METRICS},
                          "variants": {name: {k: p[k] for k in (*METRICS, "stratified", "llm_usage", "elapsed_seconds", "quality")}
                                       for name, p in variants.items()},
                          "candidate_recall": {name: {k: p["candidate_diagnostics"][k] for k in ("overall", "by_history")}
                                               for name, p in variants.items()}}
                atomic_json(directory / "paired.json", paired)
                state["completed_stages"].append(stage_id)
                update(status="stage_complete")
                print(f"Completed {stage_id}: {json.dumps(paired['delta_multimodal_minus_text'])}", flush=True)
        update(status="completed", finished_at=now())
        return 0
    except BaseException as exc:
        update(status="failed", error_type=type(exc).__name__, error=str(exc)[:600], finished_at=now())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
