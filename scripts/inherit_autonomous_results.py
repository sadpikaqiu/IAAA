"""Copy audited predecessor results into a new evaluation protocol, without requests.

The old process must have released its OS lock. Generation code is byte-audited;
only the explicitly listed failure-observability patch is compatible. Original
request identities and budgets are retained in each inherited case.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iaa_agent.agent_runtime import read_json, atomic_json, digest, now


OBSERVABILITY_PATCH = {
    "iaa_agent/llm.py": [
        "        self.last_http_status: int | None = None\n",
        "        self.last_http_status = None\n",
        '            self.last_http_status = getattr(exc, "code", None)\n',
    ],
    "iaa_agent/agent_runtime.py": [
        '                attempt["http_status"] = getattr(self.client, "last_http_status", None)\n',
        '                attempt["exception_type"] = type(exc).__name__\n',
        '                # Unexpected program errors are not model repair opportunities.\n'
        '                if not isinstance(exc, (ValueError, ValidationError)):\n'
        '                    raise\n',
    ],
}


def audit_compatibility(source, target, source_code, target_code):
    for manifest in (source, target):
        if digest(manifest["protocol"]) != manifest["protocol_sha256"]:
            raise ValueError("Manifest digest mismatch")
    old, new = source["protocol"], target["protocol"]
    ignored = {"code_sha256", "failure_policy"}
    if {k: v for k, v in old.items() if k not in ignored} != {k: v for k, v in new.items() if k not in ignored}:
        raise ValueError("Prediction data/config/selections/environment mismatch")
    patches = []
    for code, protocol in ((source_code, old), (target_code, new)):
        for name, sha in protocol["code_sha256"].items():
            if digest((code / name).read_bytes()) != sha:
                raise ValueError(f"Frozen code hash mismatch: {name}")
    prediction_files = {k for k in old["code_sha256"] if k.startswith("iaa_agent/")}
    if prediction_files != {k for k in new["code_sha256"] if k.startswith("iaa_agent/")}:
        raise ValueError("Prediction module set changed")
    prediction_files.add("scripts/mm_ablation_support.py")
    for name in sorted(prediction_files):
        before, after = (source_code / name).read_text(encoding="utf-8"), (target_code / name).read_text(encoding="utf-8")
        if before == after:
            continue
        if name not in OBSERVABILITY_PATCH:
            raise ValueError(f"Prediction logic changed: {name}")
        for fragment in OBSERVABILITY_PATCH[name]:
            if after.count(fragment) != 1:
                raise ValueError(f"Unrecognized observability patch: {name}")
            after = after.replace(fragment, "", 1)
        if before != after:
            raise ValueError(f"More than diagnostic/error propagation changes: {name}")
        patches.append(name)
    return patches


def inherit(source_dir, target_dir, source_code, target_code):
    from scripts.evaluate_autonomous import process_lock
    from scripts.autonomous_eval_support import ARMS, AUTONOMOUS_ARMS, quality_errors
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("Inheritance requires a new directory")
    with process_lock(source_dir / "runner.lock"), process_lock(target_dir / "runner.lock"):
        source = read_json(source_dir / "manifest.json")
        target = read_json(target_dir / "manifest.json")
        patches = audit_compatibility(source, target, source_code, target_code)
        if any((target_dir / name).exists() for name in ("cases", "runs", "inheritance.json")):
            raise ValueError("Destination already has cases/runs/inheritance; refuse overwrite")
        records, files = [], []
        for path in sorted((source_dir / "cases").glob("*/*/*.json")):
            phase, city, filename = path.relative_to(source_dir / "cases").parts
            if phase not in {"development", "validation", "full"}:
                continue
            row = read_json(path)
            if row["protocol_sha256"] != source["protocol_sha256"]:
                raise ValueError("Source case protocol mismatch")
            if row.get("fatal_error"):
                raise ValueError("Source contains a fatal error; investigate before inheriting")
            case = row["case"]
            if filename != f"{case['user_id']}__{case['trajectory_id']}.json" or row["city"] != city:
                raise ValueError("Source path/case mismatch")
            if phase == "full":
                keys = {tuple(k) for k in source["protocol"]["full_sessions"][city]["keys"]}
                if (case["user_id"], case["trajectory_id"]) not in keys:
                    raise ValueError("Unexpected full case")
            elif case not in source["protocol"]["selections"][city][phase]:
                raise ValueError("Unexpected selected case")
            if row["status"] in {"completed", "completed_with_failures"}:
                errors = quality_errors([row], [case], AUTONOMOUS_ARMS if phase == "full" else ARMS,
                                        repeats=phase == "validation", allow_failures=True)
                if errors:
                    raise ValueError(f"Invalid source case: {path}: {errors}")
            original_identity = row.get("request_identity", source["protocol_sha256"])
            run_root = source_dir / "runs" / phase / city / path.stem
            for file in sorted(run_root.rglob("*.json")):
                payload = read_json(file)
                if file.name == "prediction.json":
                    if payload["identity"]["identity"] != original_identity:
                        raise ValueError("Source prediction identity mismatch")
                if file.parent.name == "calls":
                    if payload["identity"]["experiment"] != original_identity:
                        raise ValueError("Source request identity mismatch")
                    if any(a.get("status") == "inflight_or_interrupted" for a in payload["attempts"]):
                        raise ValueError("Source has undrained requests; do not silently replay them")
                destination = target_dir / file.relative_to(source_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, destination)
                sha = digest(file.read_bytes())
                if digest(destination.read_bytes()) != sha:
                    raise ValueError("Copy hash mismatch")
                files.append({"path": file.relative_to(source_dir).as_posix(), "sha256": sha})
            provenance = {"source": str(path), "sha256": digest(path.read_bytes()),
                          "protocol_sha256": source["protocol_sha256"], "original_status": row["status"]}
            row.update(protocol_sha256=target["protocol_sha256"], request_identity=original_identity,
                       inherited_from=provenance)
            atomic_json(target_dir / path.relative_to(source_dir), row)
            records.append(provenance)
        report = {"created_at": now(), "source": str(source_dir),
                  "source_manifest_sha256": digest((source_dir / "manifest.json").read_bytes()),
                  "target_protocol_sha256": target["protocol_sha256"], "diagnostic_patches": patches,
                  "cases": records, "files": files, "new_model_requests": 0,
                  "note": "Same generation inputs/budgets; original request identities retained. New aggregation protocol."}
        atomic_json(target_dir / "inheritance.json", report)
        return {"cases": len(records), "files": len(files), "new_model_requests": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--target-results", type=Path, required=True)
    args = parser.parse_args()
    print(inherit(args.source_results, args.target_results, args.source_results.parent / "code", ROOT))
