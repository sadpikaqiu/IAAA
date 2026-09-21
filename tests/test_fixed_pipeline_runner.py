from scripts.evaluate_fixed_pipeline import quality_errors, sample_keys, session_digest


def test_paired_keys_are_reproducible_without_target_labels():
    keys = [("1", str(i)) for i in range(100)]
    sample = sample_keys(keys, 50, 20260914, "NYC")
    assert len(sample) == 50
    assert sample == sample_keys(list(reversed(keys)), 50, 20260914, "NYC")
    assert session_digest(sample) == session_digest(list(sample))
    assert sample != sample_keys(keys, 50, 20260915, "NYC")


def test_quality_gate_rejects_fallback_usage_and_mismatched_sessions():
    keys = [("1", "1_4")]
    payload = {"total": 1, "fallback_count": 0, "usage_missing_count": 0,
               "all_sessions_used_llm": True, "candidate_diagnostics": {"sessions": [
                   {"user_id": "1", "trajectory_id": "1_4"}]}}
    assert quality_errors(payload, keys) == []
    assert "heuristic_fallback" in quality_errors(payload | {"fallback_count": 1}, keys)
    assert "missing_usage" in quality_errors(payload | {"usage_missing_count": 1}, keys)
    assert "session_identity_mismatch" in quality_errors(payload, [("2", "2_4")])
