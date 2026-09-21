from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from iaa_agent import cli, poi_image_summary as vision
from iaa_agent.data import NYCDataRepository
from iaa_agent.engine import IAAAgent, RunConfig
from iaa_agent.evaluation import evaluate_session_split, evaluate_session_split_threaded
from iaa_agent.evidence import EvidenceStore, build_snapshot, write_snapshot


A, B, C, D, OUTSIDE = [f"a{i:023x}" for i in range(5)]


def make_data(root: Path, city: str = "NYC") -> None:
    rows = []
    for user in ("1", "2"):
        for i, poi in enumerate([A, B, A, B, D, A, B, A, A, C]):
            stamp = pd.Timestamp("2013-01-01 10:00") + pd.Timedelta(days=i, minutes=int(user))
            rows.append(dict(user_id=user, POI_id=poi, POI_catid="cafe", POI_catid_code=1,
                             POI_catname="Coffee Shop", latitude=40.7 + int(poi[-1], 16) * .001,
                             longitude=-74., timezone=-300, UTC_time=str(stamp), local_time=str(stamp),
                             day_of_week=stamp.dayofweek, norm_in_day_time=stamp.hour / 24,
                             trajectory_id=f"{user}_{i // 2}"))
    frame = pd.DataFrame(rows)
    directory = root / city
    directory.mkdir(parents=True)
    for split, indices in (("train", list(range(6)) + list(range(10, 16))),
                           ("val", [6, 7, 16, 17]), ("test", [8, 9, 18, 19])):
        frame.iloc[indices].to_csv(directory / f"{city}_{split}.csv", index=False)
    media = root / f"{city}_WWW2024" / city
    media.mkdir(parents=True)
    comments = {f"1_{A}": ["Coffee and espresso.  Tables.", "Coffee and espresso. Tables."],
                f"2_{B}": ["Gift shop with souvenirs."],
                f"3_{C}": ["TARGET_ONLY_MARKER coffee espresso cafe tables."],
                f"5_{OUTSIDE}": ["Outside catalog coffee espresso."]}
    (media / "review_summary.json").write_text(json.dumps(comments), encoding="utf-8")
    # Snapshot construction verifies raw bytes, but never decodes or sends images.
    images = media / "image/downloaded_multimodal_data"
    images.mkdir(parents=True)
    (images / f"gmap_1_{A}_1.png").write_bytes(b"synthetic image bytes for hash validation")


def make_artifact(root: Path, image_root: Path, city: str = "NYC") -> Path:
    job = vision.discover(root, city)[0][0]
    settings = vision.Settings()
    manifest, fingerprint = vision.image_manifest(job, settings)
    result = {"summary": "UNSUPPORTED_SUMMARY", "visual_evidence": [
        {"description": "Coffee cups on tables.", "image_indices": [1]}],
        "possible_activities": [{"activity": "UNSUPPORTED_ACTIVITY", "basis": "Guess", "image_indices": [1]}],
        "uncertainties": ["Current hours unknown."],
        "image_notes": [{"image_index": 1, "description": "Coffee cups.", "usable_for_poi": True}]}
    path = image_root / city / "pois" / f"{A}.json"
    vision.atomic_json(path, dict(status="success", city=city, poi_id=A, images=manifest,
                                 images_used=1, result=result, config=settings.inference_config(),
                                 input_fingerprint=fingerprint))
    return path


@pytest.fixture
def fixture(tmp_path):
    root, images = tmp_path / "datasets", tmp_path / "summaries"
    make_data(root)
    artifact = make_artifact(root, images)
    snapshot = build_snapshot(root, images, "NYC")
    path = tmp_path / "snapshot.json"
    write_snapshot(snapshot, path)
    return root, images, artifact, path


def test_snapshot_joins_original_ids_and_retains_only_sourced_observations(fixture):
    root, images, _, path = fixture
    snapshot = json.loads(path.read_text())
    assert snapshot["coverage"]["complete"]
    assert snapshot["coverage"]["catalog_pois"] == 4
    assert OUTSIDE not in snapshot["records"]
    assert len(snapshot["records"][A]["review"]) == 1
    assert snapshot["records"][A]["review"][0]["source_line"] == 1
    assert snapshot["records"][A]["image"][0]["image_indices"] == [1]
    assert snapshot["records"][A]["image"][0]["observed_at"] is None
    assert "UNSUPPORTED" not in json.dumps(snapshot)
    assert build_snapshot(root, images, "NYC")["snapshot_id"] == snapshot["snapshot_id"]
    write_snapshot(snapshot, path)  # same-content freeze is idempotent


@pytest.mark.parametrize("fault", ["missing", "corrupt", "changed_image", "writer", "mixed_config"])
def test_snapshot_readiness_rejects_incomplete_or_changed_inputs(fixture, fault):
    root, images, artifact, path = fixture
    if fault == "missing":
        artifact.unlink()
    elif fault == "corrupt":
        artifact.write_text("{}")
    elif fault == "changed_image":
        next((root / "NYC_WWW2024/NYC/image").rglob("*.png")).write_bytes(b"changed")
    elif fault == "writer":
        (images / ".writer.lock").touch()
    else:
        data = json.loads(artifact.read_text())
        data["config"]["prompt_version"] = "old"
        vision.atomic_json(artifact, data)
    snapshot = build_snapshot(root, images, "NYC")
    assert not snapshot["coverage"]["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        write_snapshot(snapshot, path.parent / "blocked.json")
    partial = path.parent / "partial.json"
    write_snapshot(snapshot, partial, allow_partial=True)
    with pytest.raises(ValueError, match="complete"):
        EvidenceStore(partial)
    with pytest.raises(ValueError, match="immutable"):
        write_snapshot(snapshot, path, allow_partial=True)


def test_store_detects_tampering_and_csv_or_city_mismatch(fixture):
    root, _, _, path = fixture
    store = EvidenceStore(path)
    repo = NYCDataRepository(root / "NYC")
    store.validate_repository(repo)
    csv = root / "NYC/NYC_test.csv"
    csv.write_bytes(csv.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="different CSV"):
        store.validate_repository(repo)
    repo.city = "TKY"
    with pytest.raises(ValueError, match="city"):
        store.validate_repository(repo)
    snapshot = json.loads(path.read_text())
    snapshot["records"][A]["image"][0]["text"] = "Tampered"
    vision.atomic_json(path, snapshot)
    with pytest.raises(ValueError, match="integrity"):
        EvidenceStore(path)


def test_modality_filters_missing_neutral_and_allowed_catalog(fixture):
    _, _, _, path = fixture
    for mode, modalities in (("both", {"review", "image"}), ("reviews", {"review"}), ("images", {"image"})):
        store = EvidenceStore(path, mode)
        scores = store.score("coffee espresso tables")
        assert {x["modality"] for x in store.get(A, scores)} == modalities
        assert not store.get(D, scores)
        assert store.search(scores, [OUTSIDE, D]) == []
        hits = store.search(scores, [A, B, C])
        assert hits and max(value for _, value in hits) > 0
        assert store.score("").sum() == 0


def test_tky_csv_reuses_the_same_repository(tmp_path):
    make_data(tmp_path, "TKY")
    repo = NYCDataRepository(tmp_path / "TKY")
    assert repo.city == "TKY" and len(repo.all_events) == 20
    assert len(repo.iter_session_test_keys()) == 2


def test_fixed_pipeline_trace_scores_and_target_isolation(fixture):
    root, _, _, path = fixture
    repo = NYCDataRepository(root / "NYC")
    repo.use_user_chronological_split()
    query = repo.get_session_query("1", "1_4")
    config = RunConfig(evidence_snapshot=str(path), max_reflection_rounds=0)
    agent = IAAAgent(repo, config)
    result = agent.run_query(query)
    assert result.evidence_snapshot["complete"]
    assert result.dataset_capabilities.has_images and result.dataset_capabilities.has_reviews
    trace = {record.tool: record for record in result.agent_trace_summary}
    assert trace["ReadHistoricalPOIEvidence"].params["poi_ids"] == [A]
    assert not trace["ReadHistoricalPOIEvidence"].params["used_by_intention_llm"]
    assert "POIEvidenceRecall" in trace and "ReadCandidatePOIEvidence" in trace
    ranked = {poi.poi_id: poi for poi in result.ranked_pois}
    assert ranked[A].score_decomposition["image_intent_relevance"] > 0
    assert ranked[D].score_decomposition["image_intent_relevance"] == 0
    assert "images unavailable" not in ranked[A].missing_evidence
    assert "images unavailable" in ranked[D].missing_evidence
    for poi in ranked.values():
        for verdict in poi.affordance_profile.affordances:
            if verdict.name.endswith("intent_relevance"):
                assert verdict.answer in {"uncertain", "not_available"}
                assert verdict.confidence == 0
    # Swapping hidden target identity/category/coordinates at the same time must
    # change evaluation labels only, never intention, recall, ranking or reflection.
    other = query.target.copy()
    other["POI_id"], other["POI_catname"], other["latitude"] = D, "Airport", 0.
    changed = IAAAgent(repo, config).run_query(replace(query, target=other))
    assert changed.inferred_intention == result.inferred_intention
    assert changed.candidate_pool_summary == result.candidate_pool_summary
    assert changed.ranked_pois == result.ranked_pois
    assert changed.reflection == result.reflection


def test_live_intention_sees_only_prior_poi_evidence(fixture, monkeypatch):
    root, _, _, path = fixture
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    repo = NYCDataRepository(root / "NYC")
    repo.use_user_chronological_split()
    query = repo.get_session_query("1", "1_4")
    agent = IAAAgent(repo, RunConfig(llm_mode="openai", evidence_snapshot=str(path)))
    calls = []

    def chat(messages, **kwargs):
        calls.append(messages)
        return None  # tests input boundary; no paid model and no success claim

    monkeypatch.setattr(agent.llm, "chat_json", chat)
    agent.run_query(query)
    assert len(calls) == 1
    prompt = json.dumps(calls[0])
    assert "Coffee cups on tables" in prompt
    assert "TARGET_ONLY_MARKER" not in prompt and C not in prompt
    assert "untrusted data" in prompt
    assert "image_sha256" not in prompt and "image_paths" not in prompt


def test_serial_and_threaded_evaluation_share_sessions_and_candidate_metrics(fixture):
    root, _, _, path = fixture
    config = RunConfig(evidence_snapshot=str(path), max_reflection_rounds=0)
    serial = evaluate_session_split(NYCDataRepository(root / "NYC"), run_config=config,
                                    report_stratified=True).as_dict()
    threaded = evaluate_session_split_threaded(NYCDataRepository(root / "NYC"), run_config=config,
                                               concurrency=2, report_stratified=True).as_dict()
    assert serial["total"] == threaded["total"] == 2
    assert serial["candidate_diagnostics"] == threaded["candidate_diagnostics"]
    assert serial["stratified"] == threaded["stratified"]
    report = serial["candidate_diagnostics"]
    assert report["by_history"]["OOH"]["n"] == 2
    assert {row["trajectory_id"] for row in report["sessions"]} == {"1_4", "2_4"}
    assert report["overall"]["RawCandidateRecall"] >= report["overall"]["CandidateRecall"]
    baseline = evaluate_session_split(NYCDataRepository(root / "NYC"), report_candidates=True).as_dict()
    plain = evaluate_session_split(NYCDataRepository(root / "NYC")).as_dict()
    assert {k: v for k, v in baseline.items() if k != "candidate_diagnostics"} == plain


def test_cli_rejects_partial_snapshot_before_any_llm_request(fixture, monkeypatch):
    root, images, artifact, path = fixture
    artifact.unlink()
    partial = path.parent / "partial.json"
    write_snapshot(build_snapshot(root, images, "NYC"), partial, allow_partial=True)
    monkeypatch.setattr(cli, "_preflight_llm", lambda *args: pytest.fail("Premature API request"))
    result = CliRunner().invoke(cli.app, ["evaluate", "--data-dir", str(root / "NYC"),
                                         "--evidence-snapshot", str(partial), "--llm", "openai"])
    assert result.exit_code != 0
    assert "complete frozen" in result.output


def test_audited_rejection_is_missing_media_not_removed_poi(fixture):
    root, images, artifact, path = fixture
    error = json.loads(artifact.read_text())
    error.update(status="error", error="http_400", result=None,
                 attempts=[{"http_status": 400, "provider_code": "data_inspection_failed"}])
    error_path = images / "NYC/errors" / artifact.name
    vision.atomic_json(error_path, error)
    artifact.unlink()
    manifest = path.parent / "unavailable.json"
    review = {"schema_version": 1, "policy": "retain_poi_missing_visual_evidence", "pois": [
        {"city": "NYC", "poi_id": A, "reason": "provider_content_rejected",
         "error_sha256": hashlib.sha256(error_path.read_bytes()).hexdigest(),
         "input_fingerprint": error["input_fingerprint"]}]}
    vision.atomic_json(manifest, review)
    snapshot = build_snapshot(root, images, "NYC", unavailable_manifest=manifest)
    assert snapshot["coverage"]["missing_image_pois"] == []
    assert snapshot["coverage"]["complete"]
    assert A in snapshot["coverage"]["unavailable_image_pois"]
    assert snapshot["records"][A]["review"] and not snapshot["records"][A]["image"]
    # Revoking/replacing the evidence behind an exclusion invalidates the review.
    error_path.write_bytes(error_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="stale"):
        build_snapshot(root, images, "NYC", unavailable_manifest=manifest)


def test_schema_failure_cannot_be_masked_as_reviewed_content_rejection(fixture):
    root, images, artifact, path = fixture
    error = json.loads(artifact.read_text())
    error.update(status="error", error="invalid_result_fields", attempts=[{"error": "invalid_result_fields"}])
    error_path = images / "NYC/errors" / artifact.name
    vision.atomic_json(error_path, error)
    artifact.unlink()
    manifest = path.parent / "unavailable.json"
    vision.atomic_json(manifest, {"schema_version": 1, "policy": "retain_poi_missing_visual_evidence", "pois": [
        {"city": "NYC", "poi_id": A, "reason": "provider_content_rejected",
         "error_sha256": hashlib.sha256(error_path.read_bytes()).hexdigest(),
         "input_fingerprint": error["input_fingerprint"]}]})
    with pytest.raises(ValueError, match="confirmed provider"):
        build_snapshot(root, images, "NYC", unavailable_manifest=manifest)


def test_diagnostic_trace_and_frozen_intention_preserve_original_predictions(fixture):
    from iaa_agent import engine, models
    from scripts.diagnose_multimodal_pipeline import make_agent_class
    root, _, _, path = fixture
    repo = NYCDataRepository(root / "NYC")
    query = repo.get_session_query("1", "1_4")
    config = RunConfig(evidence_snapshot=str(path))
    store = EvidenceStore(path)
    original = IAAAgent(repo, config, evidence_store=store).run_query(query)
    traced_type = make_agent_class(engine, models)
    traced = traced_type(repo, config, evidence_store=store)
    observed = traced.run_query(query)
    assert observed.candidate_pool_summary == original.candidate_pool_summary
    assert observed.ranked_pois == original.ranked_pois
    assert traced.rounds and traced.rounds[-1]["profiles"]
    replay = traced_type(repo, replace(config, llm_mode="openai"), evidence_store=store,
                         frozen_intention=observed.inferred_intention,
                         prepared=(traced.profile_saved, traced.peers_saved))
    replay.llm.chat_json = lambda *a, **k: pytest.fail("Cached diagnostic replay called the model")
    replayed = replay.run_query(query)
    assert replayed.candidate_pool_summary == observed.candidate_pool_summary
    assert replayed.ranked_pois == observed.ranked_pois


def test_diagnostic_selection_labels_outcome_sampling_separately():
    from scripts.diagnose_multimodal_pipeline import select_cases
    rows = [dict(user_id=str(i), trajectory_id=f"{i}_1", in_pool=True,
                 pool_size=60, rank=1) for i in range(6)]
    altered = [dict(r) for r in rows]
    altered[0].update(in_pool=False, rank=None)
    altered[1].update(rank=None)
    altered[2].update(pool_size=30, in_pool=False, rank=None)
    rows[3].update(rank=None)
    text = {"candidate_diagnostics": {"sessions": rows}}
    mm = {"candidate_diagnostics": {"sessions": altered}, "llm_anomalies": []}
    selected = select_cases(text, mm, "NYC", per_group=1, random_size=2)
    labels = {r["trajectory_id"]: r["selection_groups"] for r in selected}
    assert "pool_loss_same_size" in labels["0_1"]
    assert "rank_loss_same_size" in labels["1_1"]
    assert "pool_loss_less_expansion" in labels["2_1"]
    assert "hit10_gain" in labels["3_1"]
    assert sum("target_independent" in groups for groups in labels.values()) == 2
    assert selected == select_cases(text, mm, "NYC", per_group=1, random_size=2)


def test_diagnostic_detects_prior_boost_displacement_without_forced_quota(fixture):
    from iaa_agent import engine, models
    from scripts.diagnose_multimodal_pipeline import make_agent_class
    from scripts.analyze_multimodal_diagnostics import selection_counterfactual
    root, _, _, path = fixture
    repo = NYCDataRepository(root / "NYC")
    config = RunConfig(evidence_snapshot=str(path), candidate_pool_size=2, evidence_quota=0)
    traced_type = make_agent_class(engine, models)
    def raw():
        return {poi: dict(poi_id=poi, poi_idx=str(i), display_name=poi, category="Coffee Shop",
                          latitude=40.7, longitude=-74., distance_km=1., source_scores=scores)
                for i, (poi, scores) in enumerate([(A, {"historical": 1.}),
                    (B, {"historical": .15}), (C, {"historical": .14, "poi_evidence": .05})])}
    agent = traced_type(repo, config)
    chosen = agent._select_candidates(raw(), expanded=False)
    assert [c.poi_id for c in chosen] == [A, C]
    diagnostic = selection_counterfactual(agent.rounds[-1], B, .1)
    assert diagnostic["boost_alone_displaces_target"]
    assert diagnostic["target_components"] == {"historical": .045}
    unboosted = traced_type(repo, replace(config, evidence_weight=0))
    assert [c.poi_id for c in unboosted._select_candidates(raw(), expanded=False)] == [A, B]
