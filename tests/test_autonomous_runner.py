from copy import deepcopy

from test_poi_evidence import fixture
from iaa_agent.data import NYCDataRepository
from scripts.autonomous_eval_support import select_new_validation, quality_errors, paired_contrast, holm_adjust, metrics


def example(rank=1):
    return {"rank": rank, "in_pool": True, "in_raw": True, "in_observed": True,
            "pool_size": 30, "raw_size": 80, "predictions": [str(i) for i in range(10)],
            "valid": True, "heuristic_fallback": False,
            "accounting": {"requests": 1, "usage_missing_count": 0, "usage": {"total_tokens": 100}}}


def test_new_split_excludes_old_cases_and_does_not_use_labels(fixture):
    root, _, _, _ = fixture
    repo = NYCDataRepository(root / "NYC")
    first = select_new_validation(repo, [], development_size=1, validation_size=1, repeat_size=1)
    assert first["development"][0] != first["validation"][0]
    repo.all_events["POI_id"] = "changed hidden labels"
    repo.all_events["POI_catname"] = "Airport"
    assert select_new_validation(repo, [], 1, 1, 1) == first
    import pytest
    with pytest.raises(ValueError, match="unused"):
        select_new_validation(repo, first["development"], 1, 1, 1)


def test_quality_gate_rejects_missing_failed_duplicate_and_fallback():
    expected = [{"user_id": "1", "trajectory_id": "t", "repeat": False}]
    case = {**expected[0], "status": "completed", "variants": {"a": example()}}
    assert quality_errors([case], expected, ["a"]) == []
    assert quality_errors([], expected, ["a"])
    assert quality_errors([case, case], expected, ["a"])
    bad = deepcopy(case)
    bad["variants"]["a"]["heuristic_fallback"] = True
    assert "invalid_or_fallback_prediction" in quality_errors([bad], expected, ["a"])
    bad = deepcopy(case)
    bad["variants"]["a"]["accounting"]["usage_missing_count"] = 1
    assert "usage_missing" in quality_errors([bad], expected, ["a"])
    assert "incomplete_case" in quality_errors([dict(case, status="failed")], expected, ["a"])


def test_missing_historical_raw_size_is_not_fabricated():
    a, b = example(None), example(2)
    a["raw_size"] = None
    assert metrics(a)["mean_raw_size"] is None
    cases = [{"user_id": str(i), "variants": {"a": a, "b": b}} for i in range(3)]
    result = paired_contrast(cases, "a", "b", draws=20, permutations=100)
    assert result["delta"]["Hit@10"] == 1
    assert "mean_raw_size" not in result["delta"]
    assert result["hit10_gains"] == 3


def test_holm_family_is_explicit_and_monotonic():
    summaries = {"NYC": {"contrasts": {"one": {"hit10_cluster_permutation_p": .01}, "two": {"hit10_cluster_permutation_p": .04}}},
                 "TKY": {"contrasts": {"one": {"hit10_cluster_permutation_p": .02}, "two": {"hit10_cluster_permutation_p": .9}}}}
    adjusted = holm_adjust(summaries, ["one", "two"])
    assert adjusted["NYC"]["contrasts"]["one"]["hit10_holm_p"] == .04
    assert adjusted["TKY"]["contrasts"]["one"]["hit10_holm_p"] == .06
    assert adjusted["NYC"]["contrasts"]["two"]["hit10_holm_p"] == .08
    assert adjusted["TKY"]["contrasts"]["two"]["holm_family_size"] == 4
