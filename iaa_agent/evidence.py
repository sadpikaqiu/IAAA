"""Frozen POI review/image evidence and deterministic content retrieval.

The index uses TF-IDF lexical relevance, not an embedding model or a verified
facility classifier. Sources are static external POI knowledge with unknown
observation times; no user check-ins or prediction labels enter this module.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .poi_image_summary import (Settings, atomic_json, discover, image_manifest, now,
                                validate_result, validate_prepared_manifest, encode_image)


POLICY = "static_external_poi_knowledge_with_unknown_observation_times"
MODES = {"both": {"review", "image"}, "reviews": {"review"}, "images": {"image"}}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_snapshot(data_root: Path, image_root: Path, city: str,
                   *, verify_image_bytes: bool = True, unavailable_manifest: Path | None = None) -> dict:
    """Build in memory; never modify the input images or the running summarizer."""
    import csv

    eligible = set()
    csv_sources = {}
    for split in ("train", "val", "test"):
        path = data_root / city / f"{city}_{split}.csv"
        raw = path.read_bytes()
        csv_sources[path.name] = _sha(raw)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            eligible.update(row["POI_id"] for row in csv.DictReader(handle))
    records: dict[str, dict] = {poi: {"review": [], "image": [], "image_uncertainties": []}
                                for poi in sorted(eligible)}
    review_path = data_root / f"{city}_WWW2024" / city / "review_summary.json"
    review_raw = review_path.read_bytes()
    seen_reviews: dict[str, set[str]] = defaultdict(set)
    for line_number, line in enumerate(review_raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid review JSONL row {line_number}")
        for group, comments in obj.items():
            poi = group.rsplit("_", 1)[-1]
            if poi not in eligible:
                continue
            if not isinstance(comments, list) or any(not isinstance(x, str) for x in comments):
                raise ValueError(f"Invalid review list on row {line_number}")
            for index, text in enumerate(comments):
                cleaned = " ".join(text.split())
                if not cleaned or cleaned in seen_reviews[poi]:
                    continue
                seen_reviews[poi].add(cleaned)
                records[poi]["review"].append({
                    "id": f"review:{poi}:{line_number}:{index}", "modality": "review",
                    "text": cleaned, "source": "review_summary.json",
                    "source_line": line_number, "source_key": group, "source_index": index,
                    "source_sha256": _sha(text.encode()), "claim_type": "visitor_report",
                    "observed_at": None,
                })
    jobs, _ = discover(data_root, city)
    expected = [job for job in jobs if job.poi_id in eligible]
    missing, invalid, valid = [], {}, []
    reviewed = {}
    unavailable_manifest_sha = None
    if unavailable_manifest is not None:
        manifest_raw = unavailable_manifest.read_bytes()
        unavailable_manifest_sha = _sha(manifest_raw)
        review = json.loads(manifest_raw)
        if review.get("schema_version") != 1 or review.get("policy") != "retain_poi_missing_visual_evidence":
            raise ValueError("Invalid unavailable-image review manifest")
        for item in review["pois"]:
            if item["city"] == city:
                if item["poi_id"] in reviewed:
                    raise ValueError("Duplicate POI in unavailable-image review")
                reviewed[item["poi_id"]] = item
        if set(reviewed) - {job.poi_id for job in expected}:
            raise ValueError("Unavailable-image review contains POIs outside this catalog's image inputs")
    unavailable, skipped_files = {}, {}
    configurations: dict[str, dict] = {}
    image_sources = {}
    for job in expected:
        path = image_root / city / "pois" / f"{job.poi_id}.json"
        if job.poi_id in reviewed:
            item = reviewed[job.poi_id]
            error_path = image_root / city / "errors" / f"{job.poi_id}.json"
            raw_error = error_path.read_bytes()
            error = json.loads(raw_error)
            if path.exists() or _sha(raw_error) != item["error_sha256"]:
                raise ValueError("Unavailable-image review is stale; re-audit the POI")
            if (error.get("status") != "error" or error.get("city") != city
                    or error.get("poi_id") != job.poi_id
                    or item["reason"] != "provider_content_rejected"
                    or not any(a.get("provider_code") in {"data_inspection_failed", "DataInspectionFailed"}
                               for a in error.get("attempts", []))):
                raise ValueError("Unavailable-image review must reference a confirmed provider content rejection")
            settings = Settings(**{f.name: error["config"][f.name] for f in fields(Settings) if f.name in error["config"]})
            if error["config"] != settings.inference_config():
                raise ValueError("Unavailable-image review uses stale generation configuration")
            if item["input_fingerprint"] != error["input_fingerprint"]:
                raise ValueError("Unavailable-image review fingerprint mismatch")
            if verify_image_bytes and image_manifest(job, settings)[1] != error["input_fingerprint"]:
                raise ValueError("Unavailable-image review no longer matches source images")
            unavailable[job.poi_id] = item
            configurations[_sha(_canonical(error["config"]))] = error["config"]
            continue
        if not path.exists():
            missing.append(job.poi_id)
            continue
        try:
            raw = path.read_bytes()
            artifact = json.loads(raw)
            if (artifact["status"] != "success" or artifact["city"] != city
                    or artifact["poi_id"] != job.poi_id):
                raise ValueError("invalid_identity_or_status")
            config = artifact["config"]
            # Enforce the current prompt/preprocessing contract when freezing a run.
            settings = Settings(**{f.name: config[f.name] for f in fields(Settings) if f.name in config})
            if config != settings.inference_config():
                raise ValueError("stale_generation_configuration")
            images = artifact["images"]
            count = sum(row["image_index"] is not None for row in images)
            result = validate_result(artifact["result"], count)
            expected_paths = {p.relative_to(job.root).as_posix() for p in job.paths}
            if len(images) != len(expected_paths) or {row["path"] for row in images} != expected_paths:
                raise ValueError("image_manifest_mismatch")
            if verify_image_bytes:
                original, fingerprint = image_manifest(job, settings)
                if fingerprint != artifact["input_fingerprint"]:
                    raise ValueError("image_content_or_configuration_changed")
                if validate_prepared_manifest(original, images) != count:
                    raise ValueError("invalid_selected_image_indices")
                for row in images:
                    if row.get("excluded_reason"):
                        try:
                            encode_image(job.root / row["path"], settings)
                        except (OSError, ValueError):
                            pass
                        else:
                            raise ValueError("excluded_image_is_decodable")
            selected = {row["image_index"]: row for row in images if row["image_index"] is not None}
            if set(selected) != set(range(1, count + 1)) or count != artifact["images_used"]:
                raise ValueError("invalid_selected_image_indices")
            for index, item in enumerate(result["visual_evidence"]):
                refs = [selected[i] for i in item["image_indices"]]
                records[job.poi_id]["image"].append({
                    "id": f"image:{job.poi_id}:{index}", "modality": "image",
                    "text": item["description"], "source": f"{city}/pois/{job.poi_id}.json",
                    "source_sha256": _sha(raw), "claim_type": "model_visual_observation",
                    "image_indices": item["image_indices"],
                    "image_paths": [r["path"] for r in refs],
                    "image_sha256": [r["sha256"] for r in refs], "observed_at": None,
                })
            # Inferred possible_activities and unsupported joint-summary sentences
            # are not promoted into factual visual evidence.
            records[job.poi_id]["image_uncertainties"] = result["uncertainties"]
            image_sources[job.poi_id] = _sha(raw)
            if any(row.get("excluded_reason") for row in images):
                skipped_files[job.poi_id] = [row["path"] for row in images if row.get("excluded_reason")]
            configurations[_sha(_canonical(config))] = config
            valid.append(job.poi_id)
        except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
            records[job.poi_id]["image"] = []
            reason = str(exc)
            invalid[job.poi_id] = reason if re.fullmatch(r"[a-z_]+", reason) else type(exc).__name__
    generation_active = (image_root / ".writer.lock").exists()
    coverage = {
        "catalog_pois": len(eligible), "expected_image_pois": len(expected),
        "valid_image_summaries": len(valid), "missing_image_pois": missing,
        "invalid_image_pois": invalid,
        "unavailable_image_pois": unavailable, "summaries_with_unreadable_images": skipped_files,
        "all_image_pois_accounted_for": not missing and not invalid,
        "pois_with_review_evidence": sum(bool(r["review"]) for r in records.values()),
        "pois_with_visual_evidence": sum(bool(r["image"]) for r in records.values()),
        "generation_active": generation_active,
        "image_bytes_verified": verify_image_bytes,
        "complete": (not missing and not invalid and not generation_active
                     and len(configurations) == 1 and verify_image_bytes),
    }
    body = {"schema_version": 1, "city": city, "knowledge_policy": POLICY,
            "csv_sources": csv_sources, "review_source_sha256": _sha(review_raw),
            "image_sources": image_sources, "image_configurations": configurations,
            "coverage": coverage, "records": records}
    if unavailable_manifest_sha:
        body["unavailable_manifest_sha256"] = unavailable_manifest_sha
    return body | {"snapshot_id": _sha(_canonical(body)), "created_at": now()}


def write_snapshot(snapshot: dict, path: Path, *, allow_partial: bool = False) -> None:
    if not snapshot["coverage"]["complete"] and not allow_partial:
        coverage = snapshot["coverage"]
        raise ValueError(
            f"{snapshot['city']} evidence is incomplete: "
            f"{coverage['valid_image_summaries']}/{coverage['expected_image_pois']} image POIs; "
            f"invalid={len(coverage['invalid_image_pois'])}; generation_active={coverage['generation_active']}. "
            "Wait for generation to finish and resolve failed/stale summaries."
        )
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("snapshot_id") != snapshot["snapshot_id"]:
            raise ValueError(f"Snapshot is immutable; choose a new output path instead of replacing {path}")
        return
    atomic_json(path, snapshot)


class EvidenceStore:
    """Immutable per-run index; search/get are reusable fixed-pipeline tools."""

    def __init__(self, path: str | Path, mode: str = "both", *, require_complete: bool = True):
        if mode not in MODES:
            raise ValueError("Evidence mode must be both, reviews, or images")
        self.path, self.mode = Path(path), mode
        snapshot = json.loads(self.path.read_text(encoding="utf-8"))
        body = {k: v for k, v in snapshot.items() if k not in {"snapshot_id", "created_at"}}
        if snapshot.get("schema_version") != 1 or _sha(_canonical(body)) != snapshot.get("snapshot_id"):
            raise ValueError("Evidence snapshot integrity/schema check failed")
        if require_complete and not snapshot["coverage"]["complete"]:
            raise ValueError("A complete frozen evidence snapshot is required before evaluation")
        self.snapshot_id, self.city = snapshot["snapshot_id"], snapshot["city"]
        self.coverage, self.csv_sources = snapshot["coverage"], snapshot["csv_sources"]
        self.records = snapshot["records"]
        self.modalities = MODES[mode]
        self.items: list[tuple[str, dict]] = []
        self.by_poi: dict[str, list[int]] = defaultdict(list)
        for poi, record in sorted(self.records.items()):
            for modality in sorted(self.modalities):
                for item in record[modality]:
                    self.by_poi[poi].append(len(self.items))
                    self.items.append((poi, item))
        self.vectorizer, self.matrix = None, None
        if self.items:
            # Character n-grams retain multilingual strings without a language
            # service. Similarities are relevance signals, never probabilities.
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         max_features=60000, sublinear_tf=True, dtype=np.float32)
            try:
                self.matrix = vectorizer.fit_transform([item["text"][:1500] for _, item in self.items])
                self.vectorizer = vectorizer
            except ValueError as exc:
                if "empty vocabulary" not in str(exc):
                    raise

    def validate_repository(self, repo) -> None:
        if self.city != repo.city:
            raise ValueError(f"Evidence city {self.city} does not match CSV city {repo.city}")
        for name, expected in self.csv_sources.items():
            if _sha((repo.data_dir / name).read_bytes()) != expected:
                raise ValueError(f"Evidence snapshot was prepared against different CSV data: {name}")

    def metadata(self) -> dict:
        return {"snapshot_id": self.snapshot_id, "city": self.city, "mode": self.mode,
                "knowledge_policy": POLICY, "retrieval": "char_ngram_tfidf_lexical",
                "complete": self.coverage["complete"],
                "unavailable_image_pois": len(self.coverage.get("unavailable_image_pois", {})),
                "summaries_with_unreadable_images": len(self.coverage.get("summaries_with_unreadable_images", {})),
                "pois_with_review_evidence": self.coverage["pois_with_review_evidence"],
                "pois_with_visual_evidence": self.coverage["pois_with_visual_evidence"]}

    def has(self, poi: str, modality: str) -> bool:
        return modality in self.modalities and bool(self.records.get(poi, {}).get(modality))

    def score(self, query: str) -> np.ndarray:
        if self.vectorizer is None or self.matrix is None:
            return np.zeros(len(self.items), dtype=np.float32)
        return (self.matrix @ self.vectorizer.transform([query[:2000]]).T).toarray().ravel()

    def get(self, poi: str, scores: np.ndarray | None = None, *, per_modality: int = 2,
            max_chars: int = 400) -> list[dict]:
        result = []
        for modality in sorted(self.modalities):
            indices = [i for i in self.by_poi.get(poi, []) if self.items[i][1]["modality"] == modality]
            if scores is not None:
                indices.sort(key=lambda i: (-float(scores[i]), self.items[i][1]["id"]))
            for i in indices[:per_modality]:
                item = dict(self.items[i][1])
                item["text"] = item["text"][:max_chars]
                item["excerpt_truncated"] = len(self.items[i][1]["text"]) > max_chars
                if scores is not None:
                    item["relevance"] = round(float(scores[i]), 6)
                result.append(item)
        return result

    def search(self, scores: np.ndarray, allowed_ids: Iterable[str], *, minimum: float = 0.03) -> list[tuple[str, float]]:
        hits = []
        for poi in allowed_ids:
            indices = self.by_poi.get(poi, [])
            score = max((float(scores[i]) for i in indices), default=0.0)
            if score >= minimum:
                hits.append((poi, score))
        return sorted(hits, key=lambda x: (-x[1], x[0]))
