from __future__ import annotations

import base64
from dataclasses import replace
import io
import json
from pathlib import Path
import urllib.error

import pytest

Image = pytest.importorskip("PIL.Image")

from iaa_agent import poi_image_summary as vision


POI = "49bbd6c0f964a520f4531fe3"


def write_image(path: Path, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 100), color).save(path, format="JPEG")


def make_job(tmp_path: Path, count: int = 2) -> vision.POIImages:
    root = tmp_path / "NYC_WWW2024/NYC"
    for i, color in enumerate(["red", "blue", "green"][:count], 1):
        write_image(root / "image/downloaded_multimodal_data" / f"gmap_1_{POI}_{i}.png", color)
    return vision.discover(tmp_path, "NYC")[0][0]


def valid_result(count: int = 2) -> dict:
    return {"summary": "A place with visible seating.",
            "visual_evidence": [{"description": "Tables", "image_indices": [1]}],
            "possible_activities": [{"activity": "Sitting", "basis": "Chairs", "image_indices": [1]}],
            "uncertainties": ["Opening hours are unknown."],
            "image_notes": [{"image_index": i, "description": "Tables and chairs.", "usable_for_poi": True}
                            for i in range(1, count + 1)]}


def api_response(result: dict | None = None, finish: str = "stop", usage: bool = True) -> bytes:
    data = {"id": "test-response", "model": "qwen3.8-flash",
            "choices": [{"finish_reason": finish, "message": {"content": json.dumps(result or valid_result())}}]}
    if usage:
        data["usage"] = {"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180}
    return json.dumps(data).encode()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for name in ("QWEN_API_KEY", "DASHSCOPE_API_KEY", "QWEN_BASE_URL", "QWEN_VISION_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_key_file_selects_qwen_and_preserves_dots(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text("DeepSeek：sk-wrong\nfoursquare: wrong\nqwen38-plus：sk-example.part_one.part_two\n", encoding="utf-8-sig")
    assert vision.load_api_key(path) == "sk-example.part_one.part_two"
    path.write_text("qwen-one: sk-first\nqwen-two: sk-second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected one Qwen key"):
        vision.load_api_key(path)
    assert vision.load_api_key(path, "qwen-two") == "sk-second"


def test_key_environment_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "sk-env")
    assert vision.load_api_key(tmp_path / "missing.txt") == "sk-env"


def test_grouping_uses_original_poi_and_numeric_image_order(tmp_path):
    job = make_job(tmp_path)
    write_image(job.root / "image/downloaded_multimodal_data" / f"gmap_1_{POI}_10.png", "green")
    jobs, counts = vision.discover(tmp_path, "NYC")
    assert [p.stem.rsplit("_", 1)[-1] for p in jobs[0].paths] == ["1", "2", "10"]
    assert jobs[0].poi_id == POI
    assert counts["image_files"] == 3
    write_image(job.root / "image/unknown.png")
    with pytest.raises(ValueError, match="Cannot map 1 images"):
        vision.discover(tmp_path, "NYC")


def test_mislabeled_png_is_sent_as_real_jpeg_without_upscaling(tmp_path):
    job = make_job(tmp_path, 1)
    original = job.paths[0].read_bytes()
    url, info = vision.encode_image(job.paths[0], vision.Settings())
    assert info["source_format"] == "JPEG"
    assert info["sent_size"] == [200, 100]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]).startswith(b"\xff\xd8")
    assert job.paths[0].read_bytes() == original


def test_manifest_deduplicates_and_invalidates_changed_content(tmp_path):
    job = make_job(tmp_path, 3)
    job.paths[1].write_bytes(job.paths[0].read_bytes())
    manifest, before = vision.image_manifest(job, vision.Settings())
    assert [r["image_index"] for r in manifest] == [1, None, 2]
    assert manifest[1]["duplicate_of"] == manifest[0]["path"]
    assert vision.image_manifest(job, vision.Settings(language="zh"))[1] != before
    write_image(job.paths[2], "yellow")
    assert vision.image_manifest(job, vision.Settings())[1] != before


def test_explicit_image_limit_samples_across_the_group(tmp_path):
    job = make_job(tmp_path, 3)
    manifest, _ = vision.image_manifest(job, vision.Settings(max_images_per_poi=2))
    assert [r["image_index"] for r in manifest] == [1, None, 2]
    assert manifest[1]["duplicate_of"] is None


@pytest.mark.parametrize("fault", ["missing_image", "duplicate_image", "bad_reference", "irrelevant_evidence", "truncated_schema"])
def test_bad_model_results_cannot_be_accepted(fault):
    result = valid_result()
    if fault == "missing_image":
        result["image_notes"].pop()
    elif fault == "duplicate_image":
        result["image_notes"][1]["image_index"] = 1
    elif fault == "bad_reference":
        result["visual_evidence"][0]["image_indices"] = [3]
    elif fault == "irrelevant_evidence":
        result["image_notes"][0]["usable_for_poi"] = False
    else:
        del result["summary"]
    with pytest.raises(ValueError):
        vision.validate_result(result, 2)


def test_request_contains_joint_images_and_cloud_thinking_parameter(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    captured = []

    def respond(request, timeout):
        captured.append(json.loads(request.data))
        assert timeout == 120
        assert request.get_header("Authorization") == "Bearer sk-test"
        return io.BytesIO(api_response())

    monkeypatch.setattr(vision.urllib.request, "urlopen", respond)
    settings = vision.Settings(request_interval=0)
    client = vision.QwenVisionClient(settings, "sk-test")
    result = vision.summarize_poi(job, settings, client, tmp_path / "out")
    assert result["status"] == "success"
    payload = captured[0]
    assert payload["enable_thinking"] is False
    assert "chat_template_kwargs" not in payload
    assert len([x for x in payload["messages"][1]["content"] if x["type"] == "image_url"]) == 2
    artifact = tmp_path / "out/NYC/pois" / f"{POI}.json"
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["images_used"] == 2 and saved["attempts"][0]["usage"]["total_tokens"] == 180
    assert "sk-test" not in artifact.read_text(encoding="utf-8")
    assert "base64," not in artifact.read_text(encoding="utf-8")
    assert vision.summarize_poi(job, settings, client, tmp_path / "out")["status"] == "cached"
    assert len(captured) == 1
    changed = replace(settings, language="zh")
    assert vision.summarize_poi(job, changed, vision.QwenVisionClient(changed, "sk-test"), tmp_path / "out")["status"] == "success"
    assert len(captured) == 2


def test_429_retries_but_authentication_errors_stop_without_echoing_keys(monkeypatch):
    responses = [urllib.error.HTTPError("https://example.com", 429, "busy", {"Retry-After": "0"}, io.BytesIO(b"")),
                 io.BytesIO(api_response())]

    def respond(*args, **kwargs):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(vision.urllib.request, "urlopen", respond)
    monkeypatch.setattr(vision.time, "sleep", lambda _: None)
    client = vision.QwenVisionClient(vision.Settings(request_interval=0, retries=1), "sk-secret.test")
    _, attempts = client.summarize([], 2)
    assert len(attempts) == 2 and attempts[0]["http_status"] == 429
    responses.append(urllib.error.HTTPError("https://example.com", 401, "sk-secret.test", {}, io.BytesIO(b"sk-secret.test")))
    with pytest.raises(vision.RequestFailed) as caught:
        client.summarize([], 2)
    assert caught.value.fatal
    assert "sk-secret" not in str(caught.value) + json.dumps(caught.value.attempts)
    assert len(caught.value.attempts) == 1


def test_truncated_completion_is_failure_even_if_json_parses(monkeypatch):
    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(api_response(finish="length")))
    client = vision.QwenVisionClient(vision.Settings(retries=0, request_interval=0), "sk-test")
    with pytest.raises(vision.RequestFailed) as caught:
        client.summarize([], 2)
    assert caught.value.attempts[0]["finish_reason"] == "length"
    assert caught.value.attempts[0]["usage"]["total_tokens"] == 180


def test_invalid_evidence_is_retried_with_feedback_and_usage_retained(monkeypatch):
    bad = valid_result()
    bad["visual_evidence"][0]["image_indices"] = [9]
    payloads = []

    def respond(request, **kwargs):
        payloads.append(json.loads(request.data))
        return io.BytesIO(api_response(bad if len(payloads) == 1 else valid_result()))

    monkeypatch.setattr(vision.urllib.request, "urlopen", respond)
    monkeypatch.setattr(vision.time, "sleep", lambda _: None)
    client = vision.QwenVisionClient(vision.Settings(request_interval=0, retries=1), "sk-test")
    result, attempts = client.summarize([], 2)
    assert result == valid_result()
    assert attempts[0]["error"] == "invalid_evidence_image_reference"
    assert len(attempts) == 2
    assert sum(a["usage"]["total_tokens"] for a in attempts) == 360
    assert "invalid_evidence_image_reference" in payloads[1]["messages"][-1]["content"]
    assert payloads[1]["messages"][-2]["role"] == "assistant"


def test_missing_usage_is_explicit_not_fabricated(monkeypatch):
    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(api_response(usage=False)))
    client = vision.QwenVisionClient(vision.Settings(request_interval=0), "sk-test")
    _, attempts = client.summarize([], 2)
    assert attempts[0]["usage"] is None


def test_failed_refresh_preserves_previous_success(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    settings = vision.Settings(retries=0, request_interval=0)
    client = vision.QwenVisionClient(settings, "sk-test")
    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(api_response()))
    out = tmp_path / "out"
    vision.summarize_poi(job, settings, client, out)
    good = out / "NYC/pois" / f"{POI}.json"
    before = good.read_bytes()
    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"invalid"))
    assert vision.summarize_poi(job, settings, client, out, force=True)["status"] == "error"
    assert good.read_bytes() == before
    assert (out / "NYC/errors" / f"{POI}.json").exists()


def test_corrupt_image_is_reported_before_any_api_call(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    for path in job.paths:
        path.write_bytes(b"not an image")
    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected API request"))
    settings = vision.Settings()
    result = vision.summarize_poi(job, settings, vision.QwenVisionClient(settings, "sk-test"), tmp_path / "out")
    assert result["status"] == "error" and result["attempts"] == []
    assert result["error"] == "no_decodable_images"


def test_corrupt_member_keeps_readable_images_provenance_and_resume(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    job.paths[0].write_bytes(b"<Error>download failed</Error>")
    calls = []
    def respond(request, **kwargs):
        calls.append(json.loads(request.data))
        return io.BytesIO(api_response(valid_result(1)))
    monkeypatch.setattr(vision.urllib.request, "urlopen", respond)
    settings = vision.Settings(request_interval=0)
    client = vision.QwenVisionClient(settings, "sk-test")
    out = tmp_path / "out"
    assert vision.summarize_poi(job, settings, client, out)["status"] == "success"
    artifact = json.loads((out / "NYC/pois" / f"{POI}.json").read_text())
    assert artifact["images_used"] == 1 and len(artifact["unreadable_images"]) == 1
    assert artifact["images"][0]["image_index"] is None
    assert artifact["images"][1]["image_index"] == 1
    assert len([v for v in calls[0]["messages"][1]["content"] if v["type"] == "image_url"]) == 1
    assert vision.summarize_poi(job, settings, client, out)["status"] == "cached"
    assert len(calls) == 1 and job.paths[0].read_bytes().startswith(b"<Error>")
    write_image(job.paths[0], "green")
    assert vision.image_manifest(job, settings)[1] != artifact["input_fingerprint"]


def test_cli_dry_run_does_not_read_key_or_create_outputs(tmp_path, monkeypatch):
    make_job(tmp_path)
    monkeypatch.setattr(vision, "load_api_key", lambda *a: pytest.fail("Key read during dry-run"))
    out = tmp_path / "out"
    assert vision.main(["--data-root", str(tmp_path), "--output-dir", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_cli_writes_usage_report_and_resumes(tmp_path, monkeypatch):
    make_job(tmp_path)
    monkeypatch.setenv("QWEN_API_KEY", "sk-test")
    calls = []

    def respond(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(api_response())

    monkeypatch.setattr(vision.urllib.request, "urlopen", respond)
    out = tmp_path / "out"
    args = ["--data-root", str(tmp_path), "--output-dir", str(out), "--request-interval", "0"]
    assert vision.main(args) == 0
    assert vision.main(args) == 0
    assert len(calls) == 1
    reports = [json.loads(p.read_text()) for p in (out / "runs").glob("*.json")]
    assert sum(r["api_attempts"] for r in reports) == 1
    assert sum(r["cached"] for r in reports) == 1
    assert all(r["usage_missing_attempts"] == 0 for r in reports)
    assert not (out / ".writer.lock").exists()


def test_cli_stops_dispatching_after_authentication_error(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    other_poi = "59bbd6c0f964a520f4531fe3"
    write_image(job.root / "image" / f"gmap_2_{other_poi}_1.png")
    monkeypatch.setenv("QWEN_API_KEY", "sk-test")
    calls = []

    def respond(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://example.com", 401, "unauthorized", {}, io.BytesIO(b""))

    monkeypatch.setattr(vision.urllib.request, "urlopen", respond)
    out = tmp_path / "out"
    assert vision.main(["--data-root", str(tmp_path), "--output-dir", str(out), "--workers", "1"]) == 1
    assert len(calls) == 1
    report = json.loads(next((out / "runs").glob("*.json")).read_text())
    assert report["status"] == "stopped" and report["unprocessed_pois"] == 1
    assert report["completed_pois"] == 1
    assert report["stop_reason"]["error"] == "http_401"


@pytest.mark.parametrize("code", ["data_inspection_failed", "InvalidParameter", ""])
def test_bad_image_request_does_not_cancel_unrelated_pois(tmp_path, monkeypatch, code):
    job = make_job(tmp_path)
    other_poi = "59bbd6c0f964a520f4531fe3"
    write_image(job.root / "image" / f"gmap_2_{other_poi}_1.png")
    monkeypatch.setenv("QWEN_API_KEY", "sk-test")
    calls = []

    def respond(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            body = json.dumps({"error": {"code": code, "message": "Input image rejected"}}).encode()
            raise urllib.error.HTTPError("https://example.com", 400, "bad request", {}, io.BytesIO(body))
        return io.BytesIO(api_response(valid_result(1)))

    monkeypatch.setattr(vision.urllib.request, "urlopen", respond)
    out = tmp_path / "out"
    args = ["--data-root", str(tmp_path), "--output-dir", str(out), "--workers", "1", "--request-interval", "0"]
    assert vision.main(args) == 1
    assert len(calls) == 2  # request/content 400 gets no retry
    report = json.loads(next((out / "runs").glob("*.json")).read_text())
    assert report["status"] == "completed_with_errors"
    assert report["unprocessed_pois"] == 0 and report["success"] == 1
    assert not report["stopped_on_fatal_error"]
    error = json.loads((out / "NYC/errors" / f"{POI}.json").read_text())
    assert error["attempts"][0]["provider_code"] == code


def test_billing_error_still_fatal_and_diagnostics_are_redacted(monkeypatch):
    secret = "sk-private.token.parts"
    body = {"error": {"code": "Arrearage", "message": f"Account {secret} Bearer other-secret "
                      "data:image/jpeg;base64,aabbcc account unavailable"}, "request_id": "test-id"}

    def respond(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.com", 400, "bad", {}, io.BytesIO(json.dumps(body).encode()))

    monkeypatch.setattr(vision.urllib.request, "urlopen", respond)
    client = vision.QwenVisionClient(vision.Settings(request_interval=0), secret)
    with pytest.raises(vision.RequestFailed) as caught:
        client.summarize([], 1)
    assert caught.value.fatal and len(caught.value.attempts) == 1
    saved = json.dumps(caught.value.attempts)
    assert secret not in saved and "other-secret" not in saved and "base64" not in saved
    assert "Arrearage" in saved and "test-id" in saved


def test_empty_json_repair_resends_images_with_required_field_diagnostics(monkeypatch):
    payloads = []
    empty = json.loads(api_response())
    empty["choices"][0]["message"]["content"] = "{}"

    def respond(request, **kwargs):
        payloads.append(json.loads(request.data))
        return io.BytesIO(json.dumps(empty).encode() if len(payloads) == 1 else api_response())

    monkeypatch.setattr(vision.urllib.request, "urlopen", respond)
    monkeypatch.setattr(vision.time, "sleep", lambda _: None)
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,test"}}]
    client = vision.QwenVisionClient(vision.Settings(request_interval=0, retries=1), "sk-test")
    _, attempts = client.summarize(content, 2)
    assert attempts[0]["validation_details"]["missing_fields"] == sorted(valid_result())
    assert attempts[0]["next_retry"] == "fresh_multimodal_schema_request"
    assert len(payloads[1]["messages"]) == 2
    assert payloads[1]["messages"][-1]["content"][0] == content[0]
    assert "base64" not in json.dumps(attempts)


def test_interrupt_drains_and_accounts_for_inflight_results(tmp_path, monkeypatch):
    make_job(tmp_path)
    monkeypatch.setenv("QWEN_API_KEY", "sk-test")
    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(api_response()))

    def interrupt_after_completion(futures, **kwargs):
        for future in futures:
            future.result()
        raise KeyboardInterrupt

    monkeypatch.setattr(vision, "wait", interrupt_after_completion)
    out = tmp_path / "out"
    assert vision.main(["--data-root", str(tmp_path), "--output-dir", str(out)]) == 130
    report = json.loads(next((out / "runs").glob("*.json")).read_text())
    assert report["status"] == "interrupted" and report["success"] == 1
    assert report["api_attempts"] == 1 and report["usage_totals"]["total_tokens"] == 180
    assert not (out / ".writer.lock").exists()
