from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from iaa_agent.agent_runtime import JournaledModel, PromptBudget, read_json, atomic_json
from iaa_agent.agent_types import AgentConfig
from iaa_agent.autonomous import RankingTopUpRepair, ranking_array_schema, validate_ranking
from test_autonomous import CharacterTokenizer


IDS = ['P1', 'P2', 'P3', 'P4', 'P5']
REFS = {'F'+idx[1:]: idx for idx in IDS}
MESSAGES = [{'role': 'system', 'content': 'Rank allowed POIs.'},
            {'role': 'user', 'content': 'Original visible facts, intention and evidence.'}]


def answer(*ids):
    return {'ranked_pois': [{'poi_idx': idx, 'reason': 'Model chose '+idx,
        'affordances': dict.fromkeys(('category','spatial','temporal','revisit','transition'), 'uncertain'),
        'evidence_refs': ['F'+idx[1:]], 'missing_evidence': [], 'conflicts': []} for idx in ids]}


class Client:
    model, base_url = 'test', 'local-test'
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
    def chat_json(self, messages, **kwargs):
        self.calls.append((deepcopy(messages),deepcopy(kwargs)))
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        self.last_raw_content = json.dumps(reply)
        self.last_call_status, self.last_finish_reason, self.last_error_type = 'success','stop',None
        self.last_usage = {'prompt_tokens': 100,'completion_tokens': 30,'total_tokens': 130}
        return reply


def run(path, replies, top_k=3, retry_budget=2):
    client = Client(replies)
    model = JournaledModel(path, 'topup-test', PromptBudget(CharacterTokenizer()),
                           AgentConfig(top_k=top_k,retry_budget=retry_budget),client=client)
    validator = lambda raw: validate_ranking(raw,IDS,REFS,top_k)
    return model, client, lambda: model.call('final_ranking',MESSAGES,4096,validator,
        schema=ranking_array_schema(IDS,REFS,top_k),
        repair_builder=RankingTopUpRepair(MESSAGES,IDS,REFS,top_k))


def test_duplicate_only_topup_preserves_model_order_and_excludes_kept_ids(tmp_path):
    initial=answer('P3','P1','P3')
    model,client,execute=run(tmp_path,[initial,answer('P5')])
    result=execute()
    assert [p.poi_idx for p in result.ranked_pois]==['P3','P1','P5']
    assert [p.model_dump() for p in result.ranked_pois[:2]]==initial['ranked_pois'][:2]
    request,kwargs=client.calls[1]
    assert request[:2]==MESSAGES
    contract=kwargs['request_options']['response_format']['json_schema']['schema']
    assert contract['properties']['ranked_pois']['minItems']==1
    check=Draft202012Validator(contract)
    assert check.is_valid(answer('P5'))
    assert not check.is_valid(answer('P3'))
    assert not check.is_valid(answer('P1'))
    journal=read_json(tmp_path/'final_ranking.json')
    assert [a['accepted'] for a in journal['attempts']]==[False,True]
    assert journal['attempts'][1]['repair_metadata']['excluded_ids']==['P3','P1']
    assert model.accounting()['retries']==1
    assert model.accounting()['invalid_attempts']==1
    assert model.accounting()['usage']['total_tokens']==260
    resumed,_,resume=run(tmp_path,[])
    assert resume()==result
    assert resumed.accounting()==model.accounting()


def test_second_topup_can_finish_a_duplicate_topup_without_rewriting_prefix(tmp_path):
    model,client,execute=run(tmp_path,[answer('P3','P3','P3','P3'),answer('P1','P1','P5'),answer('P2')],top_k=4)
    result=execute()
    assert [p.poi_idx for p in result.ranked_pois]==['P3','P1','P5','P2']
    assert model.accounting()['retries']==2
    assert 'Return ONLY 1 additional' in client.calls[2][0][-1]['content']


def test_shared_retry_budget_cannot_be_reset_by_topups(tmp_path):
    model,client,execute=run(tmp_path,[answer('P3','P3','P3')])
    atomic_json(tmp_path/'decision_01.json',{'attempts':[{'accepted':False,'usage':{'total_tokens':10}},
        {'accepted':False,'usage':{'total_tokens':10}},{'accepted':True,'usage':{'total_tokens':10}}]})
    with pytest.raises(RuntimeError,match='Repair budget exhausted'):
        execute()
    assert len(client.calls)==1
    assert model.accounting()['retries']==2


def test_inflight_topup_is_charged_and_can_resume_with_remaining_budget(tmp_path):
    model,client,execute=run(tmp_path,[answer('P3','P1','P3'),KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):execute()
    assert model.accounting()['retries']==1
    resumed,client2,resume=run(tmp_path,[answer('P5')])
    assert [p.poi_idx for p in resume().ranked_pois]==['P3','P1','P5']
    assert len(client2.calls)==1
    assert resumed.accounting()['retries']==2
    assert resumed.accounting()['usage_missing_count']==1


@pytest.mark.parametrize('kind',['wrong_ref','unknown_id','too_short','invalid_field'])
def test_other_errors_cannot_seed_partial_topup(tmp_path,kind):
    bad=answer('P3','P1','P3')
    if kind=='wrong_ref':bad['ranked_pois'][-1]['evidence_refs']=['F1']
    if kind=='unknown_id':bad['ranked_pois'][-1]['poi_idx']='P999'
    if kind=='too_short':bad['ranked_pois'].pop()
    if kind=='invalid_field':bad['ranked_pois'][-1]['reason']=''
    _,client,execute=run(tmp_path,[bad,answer('P2','P1','P5')])
    assert [p.poi_idx for p in execute().ranked_pois]==['P2','P1','P5']
    assert 'Return ONLY' not in client.calls[1][0][-1]['content']
    assert read_json(tmp_path/'final_ranking.json')['attempts'][1]['repair_context']=='full_previous_response'


def test_invalid_topup_cannot_replace_or_corrupt_retained_entries(tmp_path):
    model,client,execute=run(tmp_path,[answer('P3','P1','P3'),answer('P1'),answer('P5')])
    assert [p.poi_idx for p in execute().ranked_pois]==['P3','P1','P5']
    assert model.accounting()['retries']==2
    assert read_json(tmp_path/'final_ranking.json')['attempts'][2]['repair_metadata']['excluded_ids']==['P3','P1']


def test_tampered_cached_retained_prefix_is_rejected_without_request(tmp_path):
    _,_,execute=run(tmp_path,[answer('P3','P1','P3'),answer('P5')]);execute()
    p=tmp_path/'final_ranking.json';j=read_json(p)
    j['attempts'][0]['parsed']=answer('P3','P2','P3')
    j['attempts'][0]['raw_content']=json.dumps(j['attempts'][0]['parsed'])
    atomic_json(p,j)
    _,_,resume=run(tmp_path,[])
    with pytest.raises(ValueError,match='Cached repair request mismatch'):resume()


def test_valid_original_ranking_does_not_request_topup(tmp_path):
    model,client,execute=run(tmp_path,[answer('P5','P3','P1')])
    assert [p.poi_idx for p in execute().ranked_pois]==['P5','P3','P1']
    assert len(client.calls)==1
    assert model.accounting()['retries']==0
