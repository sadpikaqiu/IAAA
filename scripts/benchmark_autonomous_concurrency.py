"""Replay development requests to measure concurrency, outside formal metrics.

Run against an idle local model service after the evaluation has drained. The
same frozen request corpus is used at every level, with a final baseline repeat
to expose cache/order effects. No labels or recommendation scores select samples.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import urllib.request


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else
                          json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def corpus(source):
    """48 unique, first-attempt-valid requests, stratified by city/call kind."""
    selected = []
    for city in ("NYC", "TKY"):
        buckets = {"initial_intention": [], "decision": [], "final_ranking": []}
        for case_path in sorted((source / "results/cases/development" / city).glob("*.json")):
            # Only completion status is read; targets, ranks and scores are unused.
            if read(case_path)["status"] != "completed":
                continue
            for path in sorted((source / "results/runs/development" / city / case_path.stem).glob("*/calls/*.json")):
                record = read(path)
                if not record["attempts"] or not record["attempts"][0].get("accepted"):
                    continue
                label = path.stem
                kind = "decision" if label.startswith("decision_") else label
                if kind not in buckets:
                    continue
                relative = str(path.relative_to(source))
                buckets[kind].append({"source": relative, "source_sha256": sha(path.read_bytes()),
                    "city": city, "arm": path.parents[1].name, "label": label, "kind": kind,
                    "messages": record["messages"], "options": record["identity"]["options"],
                    "model": record["identity"]["model"]})
        for kind, count in (("initial_intention", 3), ("decision", 15), ("final_ranking", 6)):
            candidates = sorted(buckets[kind], key=lambda x: sha(["concurrency-v1", x["source"]]))
            if len(candidates) < count:
                raise ValueError(f"Not enough completed development requests: {city}/{kind}")
            selected.extend(candidates[:count])
    return sorted(selected, key=lambda x: sha(["dispatch-v1", x["source"]]))


def build_validator(item):
    """Use the actual runtime validators, including uniqueness and same-POI refs."""
    from iaa_agent.agent_runtime import strict_response_json
    from iaa_agent.agent_types import AgentConfig
    from iaa_agent.autonomous import AutonomousAgent, decode_working_selection, validate_ranking
    from iaa_agent.models import Intention
    from jsonschema import Draft202012Validator

    schema = item["options"].get("response_format", {}).get("json_schema", {}).get("schema")
    schema_validator = Draft202012Validator(schema) if schema else None
    payload = json.loads(item["messages"][1]["content"]) if schema else None
    if item["kind"] == "final_ranking":
        table = payload["candidate_facts"]
        facts = [dict(zip(table["columns"], row)) for row in table["rows"]]
        refs = {row["ref"]: row["poi_idx"] for row in facts}
        refs.update({row["ref"]: row["poi_idx"] for row in payload["external_evidence"]})
        validate = lambda raw: validate_ranking(raw, [row["poi_idx"] for row in facts], refs, payload["top_k"])
    elif item["kind"] == "decision":
        branch = schema.get("anyOf", [schema])[0]
        ids = branch["properties"]["working_poi_selection"]["properties"]
        fixed = item["arm"].startswith("fixed_schedule")
        config = AgentConfig(engine="fixed_schedule" if fixed else "autonomous")
        agent = AutonomousAgent(SimpleNamespace(registry=dict.fromkeys(ids)), None, config)
        budget = payload["budget"]
        agent.tool_count = config.max_tool_calls - budget["remaining_tool_calls"]
        required = [tuple(x) for x in payload.get("required_tool_order", [])] if fixed else None
        def validate(raw):
            decoded = decode_working_selection(raw)
            if fixed:
                return agent._validate_fixed(decoded, budget["decision"] - 1, required)
            return agent._validate_decision(decoded, budget["decision"] - 1, None)
    else:
        validate = Intention.model_validate

    def validate_text(content):
        raw = strict_response_json(content)
        if schema_validator is not None:
            schema_validator.validate(raw)
        validate(raw)
        return raw
    return validate_text


def metrics(base):
    with urllib.request.urlopen(base.removesuffix("/v1") + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    names = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
             "generation_tokens_total", "prompt_tokens_total")
    result = {name: 0. for name in names}
    for line in lines:
        for name in names:
            if line.startswith("vllm:" + name + "{") or line.startswith("vllm:" + name + " "):
                result[name] += float(line.rsplit(" ", 1)[-1])
    return result


def assert_no_evaluator():
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            words = path.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(os.fsdecode(word)).name == "evaluate_autonomous.py" for word in words if word):
            raise RuntimeError(f"Evaluation still running: {path.parent.name}")


def wait_idle(args):
    deadline = time.monotonic() + args.drain_timeout
    consecutive = 0
    while time.monotonic() < deadline:
        try:
            assert_no_evaluator()
            current = metrics(args.base_url)
            idle = current["num_requests_running"] == current["num_requests_waiting"] == 0
        except RuntimeError:
            idle = False
        consecutive = consecutive + 1 if idle else 0
        if consecutive >= 3:
            return
        time.sleep(5)
    raise TimeoutError("Evaluation/model did not drain; no benchmark requests sent")


def replay(item, validator, args, directory, index):
    payload = {"model": item["model"], "messages": item["messages"],
               "response_format": {"type": "json_object"}, **item["options"]}
    result = {"index": index, "source": item["source"], "kind": item["kind"], "city": item["city"],
              "started_at": stamp(), "transport_ok": False, "valid": False}
    started = time.monotonic()
    try:
        request = urllib.request.Request(args.base_url + "/chat/completions",
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
            raw = json.loads(response.read())
        result["response"] = raw
        result["latency_seconds"] = time.monotonic() - started
        choice = raw["choices"][0]
        result["usage"] = raw.get("usage")
        result["finish_reason"] = choice["finish_reason"]
        if choice["finish_reason"] != "stop" or not (raw.get("usage") or {}).get("total_tokens"):
            raise ValueError("Incomplete response or missing token usage")
        result["transport_ok"] = True
        validator(choice["message"]["content"])
        result["valid"] = True
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)[:2000])
    result.setdefault("latency_seconds", time.monotonic() - started)
    result["finished_at"] = stamp()
    save(directory / f"{index:03}.json", result)
    return result


def run_round(args, items, validators, concurrency, round_index):
    assert_no_evaluator()
    directory = args.output_dir / "responses" / f"{round_index:02}_c{concurrency}"
    before = metrics(args.base_url)
    if before["num_requests_running"] or before["num_requests_waiting"]:
        raise RuntimeError("Model has other active requests before benchmark round")
    rows = []
    started_at, started = stamp(), time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(replay, item, validator, args, directory, index)
                   for index, (item, validator) in enumerate(zip(items, validators))]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"c={concurrency}: {len(rows)}/{len(items)} latency={row['latency_seconds']:.1f}s "
                  f"transport={row['transport_ok']} valid={row['valid']}", flush=True)
            save(args.output_dir / "progress.json", {"status": "running", "round": round_index,
                "concurrency": concurrency, "completed": len(rows), "total": len(items), "updated_at": stamp()})
    elapsed = time.monotonic() - started
    latencies = [row["latency_seconds"] for row in rows]
    output_tokens = sum((row.get("usage") or {}).get("completion_tokens", 0) for row in rows)
    result = {"round": round_index, "concurrency": concurrency, "started_at": started_at,
        "finished_at": stamp(), "requests": len(rows), "elapsed_seconds": elapsed,
        "requests_per_hour": 3600 * len(rows) / elapsed, "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / elapsed,
        "transport_errors": sum(not row["transport_ok"] for row in rows),
        "validation_errors": sum(row["transport_ok"] and not row["valid"] for row in rows),
        "latency_mean": statistics.mean(latencies), "latency_p95": percentile(latencies, .95),
        "latency_max": max(latencies), "metrics_before": before, "metrics_after": metrics(args.base_url),
        "per_kind": {kind: {"n": len(group), "latency_mean": statistics.mean(x["latency_seconds"] for x in group),
                              "latency_max": max(x["latency_seconds"] for x in group)}
                     for kind in {x["kind"] for x in rows}
                     if (group := [x for x in rows if x["kind"] == kind])}}
    save(args.output_dir / f"round_{round_index:02}_c{concurrency}.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=[4, 8, 12, 16, 4])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--drain-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new benchmark output directory; never overwrite responses")
    if any(level < 1 or level > 16 for level in args.levels):
        raise ValueError("Concurrency must be between 1 and the current server maximum 16")
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.source / "code"))
    print("Waiting for existing evaluation to drain; no model requests yet.", flush=True)
    save(args.output_dir / "progress.json", {"status": "waiting_for_evaluation_drain", "started_at": stamp()})
    try:
        wait_idle(args)
        items = corpus(args.source)
        validators = [build_validator(item) for item in items]
        for item, validator in zip(items, validators):
            validator(read(args.source / item["source"])["attempts"][0]["raw_content"])
        manifest = {"scope": "development request replay for performance only; never formal predictions",
            "created_at": stamp(), "source": str(args.source), "source_protocol_sha256":
            read(args.source / "results/manifest.json")["protocol_sha256"],
            "script_sha256": sha(Path(__file__).read_bytes()), "levels": args.levels,
            "request_timeout": args.request_timeout, "corpus_sha256": sha(items), "corpus": items,
            "selection": "3 intentions, 15 decisions, 6 rankings per city; ID hash; first-attempt-valid completed development only",
            "dispatch": "rolling request queue; identical messages/options at each level; no retries; final c4 cache/order control",
            "limitations": "Independent cached contexts, not end-to-end agents; grammar/prefix caches warm; no cache reset on shared service."}
        save(args.output_dir / "manifest.json", manifest)
        rounds = []
        for index, level in enumerate(args.levels):
            # Do not escalate after loss of latency margin or transport failure.
            if rounds and level > rounds[-1]["concurrency"] and (
                    rounds[-1]["transport_errors"] or rounds[-1]["latency_p95"] > .8 * args.request_timeout):
                print(f"Skipping c={level}: prior level exceeded transport/latency safety margin", flush=True)
                continue
            rounds.append(run_round(args, items, validators, level, index))
            save(args.output_dir / "summary.json", {"rounds": rounds, "scope": manifest["scope"]})
        save(args.output_dir / "progress.json", {"status": "completed", "finished_at": stamp(), "rounds": len(rounds)})
    except BaseException as exc:
        save(args.output_dir / "progress.json", {"status": "failed", "error": repr(exc), "finished_at": stamp()})
        raise


if __name__ == "__main__":
    main()
