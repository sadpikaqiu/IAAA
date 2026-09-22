from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from iaa_agent.agent_runtime import PromptBudget, strict_response_json
from iaa_agent.autonomous import decode_ranked_selection, ranking_selection_schema, validate_ranking


@pytest.mark.parametrize("wrapped", [False, True])
def test_prompt_count_uses_ids_not_batch_encoding_keys(wrapped):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            assert kwargs["preserve_thinking"] is False
            ids = list(range(12409))
            return {"input_ids": [ids] if wrapped else ids, "attention_mask": []}
    assert PromptBudget(Tokenizer()).count([]) == 12409


def test_prompt_count_rejects_ambiguous_multi_conversation_batch():
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [[1, 2], [3, 4]], "attention_mask": []}
    with pytest.raises(ValueError, match="exactly one conversation"):
        PromptBudget(Tokenizer()).count([])


def ranking():
    def entry(rank, ref):
        return {"rank": rank, "reason": "Recorded evidence", "evidence_refs": [ref],
                "affordances": dict.fromkeys(("category", "spatial", "temporal", "revisit", "transition"), "uncertain")}
    return {"ranked_pois_by_id": {"P1": entry(2, "F1"), "P2": entry(1, "F2")}}


def test_model_assigned_ranks_not_key_order_determine_output():
    decoded = decode_ranked_selection(ranking(), 2)
    validated = validate_ranking(decoded, ["P1", "P2", "P3"], {"F1": "P1", "F2": "P2", "F3": "P3"}, 2)
    assert [row.poi_idx for row in validated.ranked_pois] == ["P2", "P1"]


@pytest.mark.parametrize("value", [1, 0, 3, True, 1.0])
def test_rank_decoder_rejects_duplicate_out_of_range_or_non_integer_ranks(value):
    raw = ranking()
    raw["ranked_pois_by_id"]["P1"]["rank"] = value
    with pytest.raises(ValueError):
        decode_ranked_selection(raw, 2)


def test_rank_schema_binds_refs_to_poi_and_enforces_exact_selection_size():
    schema = ranking_selection_schema(["P1", "P2", "P3"], {"F1": "P1", "F2": "P2", "F3": "P3"}, 2)
    validator = Draft202012Validator(schema)
    good = ranking()
    assert validator.is_valid(good)
    wrong_ref = deepcopy(good)
    wrong_ref["ranked_pois_by_id"]["P1"]["evidence_refs"] = ["F2"]
    assert not validator.is_valid(wrong_ref)
    fewer = deepcopy(good)
    fewer["ranked_pois_by_id"].pop("P1")
    assert not validator.is_valid(fewer)
    more = deepcopy(good)
    more["ranked_pois_by_id"]["P3"] = deepcopy(good["ranked_pois_by_id"]["P1"])
    more["ranked_pois_by_id"]["P3"]["evidence_refs"] = ["F3"]
    assert not validator.is_valid(more)
    unknown = deepcopy(good)
    unknown["ranked_pois_by_id"]["P4"] = unknown["ranked_pois_by_id"].pop("P1")
    assert not validator.is_valid(unknown)
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        strict_response_json('{"ranked_pois_by_id":{"P1":{"rank":1},"P1":{"rank":2}}}')
