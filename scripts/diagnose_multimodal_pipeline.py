"""Trace frozen pipeline cases and isolate evidence interventions for diagnosis.

This is an explanatory case study, not a new benchmark. Outcome-selected cases
are explicitly labeled. Original experiment code, evidence and results stay frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def row_key(row):
    return str(row['user_id']), str(row['trajectory_id'])


def select_cases(text, multimodal, city, per_group, random_size):
    base = {row_key(r): r for r in text['candidate_diagnostics']['sessions']}
    mm = {row_key(r): r for r in multimodal['candidate_diagnostics']['sessions']}
    assert set(base) == set(mm)
    groups = {name: [] for name in ('pool_loss_same_size', 'rank_loss_same_size',
                                   'pool_loss_less_expansion', 'hit10_gain')}
    for key, a in base.items():
        b = mm[key]
        if a['in_pool'] and not b['in_pool'] and a['pool_size'] == b['pool_size']:
            groups['pool_loss_same_size'].append(key)
        if a['rank'] and not b['rank'] and b['in_pool'] and a['pool_size'] == b['pool_size']:
            groups['rank_loss_same_size'].append(key)
        if a['in_pool'] and not b['in_pool'] and a['pool_size'] > b['pool_size']:
            groups['pool_loss_less_expansion'].append(key)
        if not a['rank'] and b['rank']:
            groups['hit10_gain'].append(key)
    def ordered(keys, group):
        return sorted(keys, key=lambda k: sha(f'20260917/{city}/{group}/{k[0]}/{k[1]}'.encode()))
    selected = {}
    for group, keys in groups.items():
        for key in ordered(keys, group)[:per_group]:
            selected.setdefault(key, []).append(group)
    for key in ordered(base, 'target_independent')[:random_size]:
        selected.setdefault(key, []).append('target_independent')
    for anomaly in multimodal.get('llm_anomalies', []):
        selected.setdefault(row_key(anomaly), []).append('llm_anomaly')
    return [{'city': city, 'user_id': k[0], 'trajectory_id': k[1], 'selection_groups': groups,
             'original': {'text': base[k], 'multimodal': mm[k]}}
            for k, groups in sorted(selected.items(), key=lambda item: ('llm_anomaly' not in item[1], item[0]))]


def make_agent_class(engine, models):
    class TracedAgent(engine.IAAAgent):
        def __init__(self, *args, frozen_intention=None, prepared=None, ranking_evidence=True,
                     force_reflection=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.frozen_intention = frozen_intention
            self.prepared = prepared
            self.ranking_evidence = ranking_evidence
            self.force_reflection = force_reflection
            self.rounds = []
            self.llm_record = None
            self.profile_saved = None
            self.peers_saved = None
            chat = self.llm.chat_json
            def recorded_chat(messages, **kwargs):
                parsed = chat(messages, **kwargs)
                errors = []
                try:
                    models.Intention.model_validate(parsed)
                except Exception as exc:
                    errors = (exc.errors(include_input=False, include_url=False)
                              if hasattr(exc, 'errors') else [{'type': type(exc).__name__, 'message': str(exc)}])
                self.llm_record = {'messages': messages, 'parsed': parsed,
                                   'raw_content': self.llm.last_raw_content,
                                   'finish_reason': self.llm.last_finish_reason,
                                   'call_status': self.llm.last_call_status,
                                   'error_type': self.llm.last_error_type,
                                   'usage': self.llm.last_usage, 'validation_errors': errors}
                return parsed
            self.llm.chat_json = recorded_chat

        def _build_user_profile(self, query):
            self.profile_saved = (self.prepared[0].model_copy(deep=True) if self.prepared
                                  else super()._build_user_profile(query))
            return self.profile_saved

        def _find_peer_users(self, query, profile):
            self.peers_saved = copy.deepcopy(self.prepared[1]) if self.prepared else super()._find_peer_users(query, profile)
            return self.peers_saved

        def _infer_intention(self, context, profile, peers, query):
            if self.frozen_intention is None:
                return super()._infer_intention(context, profile, peers, query)
            self.last_intention_source = 'frozen_diagnostic_intention'
            self.last_llm_status = 'not_called_diagnostic_replay'
            return self.frozen_intention.model_copy(deep=True)

        def _reserve_evidence_candidates(self, chosen, all_candidates, size):
            result = super()._reserve_evidence_candidates(chosen, all_candidates, size)
            old_ids, new_ids = {c.poi_id for c in chosen}, {c.poi_id for c in result}
            self.rounds[-1]['quota'] = {'before': [c.poi_id for c in chosen],
                                        'added': sorted(new_ids - old_ids), 'evicted': sorted(old_ids - new_ids)}
            return result

        def _select_candidates(self, raw, expanded):
            self.rounds.append({'expanded': expanded})
            selected = super()._select_candidates(raw, expanded)
            self.rounds[-1].update(raw=[{'poi_id': k, 'category': v['category'],
                                        'distance_km': v['distance_km'], 'prior_score': v['prior_score'],
                                        'source_scores': dict(v['source_scores'])} for k, v in raw.items()],
                                   chosen=[c.poi_id for c in selected])
            return selected

        def _build_affordances(self, *args, **kwargs):
            profiles = super()._build_affordances(*args, **kwargs)
            if not self.ranking_evidence:
                for profile in profiles:
                    profile.score_decomposition = {k: v for k, v in profile.score_decomposition.items()
                                                   if k not in {'image_intent_relevance', 'review_intent_relevance'}}
                    profile.alignment_score = round(sum(profile.score_decomposition.values()), 6)
            self.rounds[-1]['profiles'] = [
                {'poi_id': p.poi_id, 'category': p.category, 'score': p.alignment_score,
                 'confidence': p.confidence, 'distance_km': p.distance_km,
                 'score_decomposition': p.score_decomposition,
                 'evidence': [ref for verdict in p.affordances for ref in verdict.evidence_refs]}
                for p in self._rank_profiles(profiles)]
            return profiles

        def _maybe_reflect(self, *args, **kwargs):
            natural = super()._maybe_reflect(*args, **kwargs)
            self.rounds[-1]['natural_reflection'] = natural.model_dump(mode='json')
            if self.force_reflection is not None:
                self.rounds[-1]['diagnostic_forced_reflection'] = self.force_reflection.model_dump(mode='json')
                return self.force_reflection.model_copy(deep=True)
            return natural
    return TracedAgent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cities', nargs='+', choices=['NYC', 'TKY'], default=['NYC', 'TKY'])
    parser.add_argument('--per-group', type=int, default=2)
    parser.add_argument('--random-size', type=int, default=4)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--anomaly-only', action='store_true')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    args = parser.parse_args(argv)
    if min(args.per_group, args.random_size) < 0 or args.concurrency < 1:
        parser.error('Invalid case counts or concurrency')
    root = args.experiment_root.resolve()
    if (args.output_dir / 'manifest.json').exists():
        parser.error('Use a fresh output directory; no results are overwritten')
    # Import and check the original frozen implementation, not the working tree.
    manifest = read(root / 'bundle_manifest.json')
    for relative, expected in manifest['files'].items():
        if relative.startswith('code/iaa_agent/'):
            assert sha((root / relative).read_bytes()) == expected, relative
    sys.path.insert(0, str(root / 'code'))
    engine = importlib.import_module('iaa_agent.engine')
    models = importlib.import_module('iaa_agent.models')
    data = importlib.import_module('iaa_agent.data')
    evidence_module = importlib.import_module('iaa_agent.evidence')
    assert Path(engine.__file__).resolve().is_relative_to(root / 'code')
    os.environ.update(OPENAI_BASE_URL=args.base_url, OPENAI_MODEL='Qwen/Qwen3.8-27B-FP8',
                      OPENAI_ENABLE_THINKING='0', OPENAI_TEMPERATURE='0', OPENAI_SEED='42',
                      OPENAI_MAX_TOKENS='4096', OPENAI_TIMEOUT_SECONDS='180', TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')
    TracedAgent = make_agent_class(engine, models)
    selected = []
    inputs = {}
    for city in args.cities:
        folder = root / 'results' / (city + '_full')
        originals = {v: read(folder / (v + '.json')) for v in ('text', 'multimodal')}
        cases = select_cases(originals['text'], originals['multimodal'], city, args.per_group, args.random_size)
        if args.anomaly_only:
            cases = [c for c in cases if 'llm_anomaly' in c['selection_groups']]
        selected.extend(cases)
        snapshot = Path(originals['multimodal']['run_config']['evidence_snapshot'])
        # A locally copied experiment can use the same frozen snapshot basename.
        if not snapshot.exists():
            snapshot = root / 'evidence' / snapshot.name
        store = evidence_module.EvidenceStore(snapshot)
        repo = data.NYCDataRepository(args.data_root / city)
        store.validate_repository(repo)
        inputs[city] = (store, snapshot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / 'manifest.json', {'created_at': now(), 'scope': 'diagnostic_selected_cases_only',
          'selection_policy': 'hash_order_within_labeled_outcome_strata_plus_target_independent_controls',
          'script_sha256': sha(Path(__file__).read_bytes()), 'frozen_bundle_sha256': sha((root / 'bundle_manifest.json').read_bytes()),
          'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'cases': selected})
    state = {'started_at': now(), 'status': 'running', 'total': len(selected), 'completed': 0, 'failed': 0}
    write(args.output_dir / 'progress.json', state)
    local = threading.local()

    def run_case(case):
        city = case['city']
        if not hasattr(local, 'repos'):
            local.repos = {}
        if city not in local.repos:
            repo = data.NYCDataRepository(args.data_root / city)
            repo.use_user_chronological_split(.8)
            repo.prewarm_global_structures()
            local.repos[city] = repo
        repo = local.repos[city]
        store, snapshot = inputs[city]
        query = repo.get_session_query(case['user_id'], case['trajectory_id'], train_ratio=.8, min_context=1)
        output = dict(case, started_at=now(), variants={})
        saved = {}
        text_config = engine.RunConfig.p4(llm_mode='openai')
        text_config.intention_context_size = 5
        mm_config = replace(text_config, evidence_snapshot=str(snapshot))
        prepared = None

        def run(name, config, *, frozen=None, rank_evidence=True, forced=None):
            nonlocal prepared
            started = time.monotonic()
            agent = TracedAgent(repo, config, evidence_store=store if config.evidence_snapshot else None,
                                frozen_intention=frozen, prepared=prepared, ranking_evidence=rank_evidence,
                                force_reflection=forced)
            result = agent.run_query(query)
            prepared = (agent.profile_saved, agent.peers_saved)
            predictions = [p.poi_id for p in result.ranked_pois]
            # Ground truth enters diagnostics only after inference and ranking.
            gt = result.ground_truth_poi_id
            observation = {'config': asdict(config), 'elapsed_seconds': round(time.monotonic()-started, 3),
                           'mode': 'live_model' if frozen is None else 'frozen_intention_replay',
                           'predictions': predictions, 'rank': predictions.index(gt)+1 if gt in predictions else None,
                           'in_pool': gt in result.candidate_pool_summary['candidate_poi_ids'],
                           'in_raw': gt in result.candidate_pool_summary['raw_retrieved_poi_ids'],
                           'intention': result.inferred_intention.model_dump(mode='json'),
                           'reflection': result.reflection.model_dump(mode='json'),
                           'llm_source': agent.last_intention_source, 'llm_status': agent.last_llm_status,
                           'llm': agent.llm_record, 'rounds': agent.rounds}
            if name.endswith('_live'):
                original = case['original']['text' if name == 'text_live' else 'multimodal']
                observation['original_predictions_match'] = predictions == original['predictions']
                observation['original_pool_hit_match'] = observation['in_pool'] == original['in_pool']
            output['variants'][name] = observation
            saved[name] = result
            return result

        text_result = run('text_live', text_config)
        mm_result = run('multimodal_live', mm_config)
        if not args.anomaly_only:
            # Control: instrumentation and cached replay must preserve predictions.
            replay = run('multimodal_replay_control', mm_config, frozen=mm_result.inferred_intention)
            assert [p.poi_id for p in replay.ranked_pois] == [p.poi_id for p in mm_result.ranked_pois]
            assert replay.candidate_pool_summary == mm_result.candidate_pool_summary
            run('intention_only', text_config, frozen=mm_result.inferred_intention)
            run('text_intention_mm_downstream', mm_config, frozen=text_result.inferred_intention)
            run('mm_no_quota', replace(mm_config, evidence_quota=0), frozen=mm_result.inferred_intention)
            run('mm_no_ranking_evidence', mm_config, frozen=mm_result.inferred_intention, rank_evidence=False)
            run('mm_text_reflection', mm_config, frozen=mm_result.inferred_intention, forced=text_result.reflection)
            run('text_intention_mm_no_quota', replace(mm_config, evidence_quota=0), frozen=text_result.inferred_intention)
        output['ground_truth_poi_id'] = str(query.target['POI_id'])
        output['finished_at'] = now()
        output['status'] = 'completed'
        write(args.output_dir / 'cases' / city / f"{case['user_id']}_{case['trajectory_id']}.json", output)
        return {'city': city, 'user_id': case['user_id'], 'trajectory_id': case['trajectory_id'],
                'live_statuses': {v: output['variants'][v]['llm_status'] for v in ('text_live','multimodal_live')}}

    completed = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        jobs = {executor.submit(run_case, c): c for c in selected}
        for job in as_completed(jobs):
            case = jobs[job]
            try:
                item = job.result()
                state['completed'] += 1
                completed.append(item)
                print(f"{state['completed']}/{len(selected)} {item}", flush=True)
            except Exception as exc:
                state['failed'] += 1
                write(args.output_dir / 'errors' / f"{case['city']}_{case['trajectory_id']}.json",
                      {'case': case, 'error_type': type(exc).__name__, 'error': str(exc)})
                print(f"ERROR {case['city']} {case['trajectory_id']}: {type(exc).__name__}: {exc}", flush=True)
            state['updated_at'] = now()
            write(args.output_dir / 'progress.json', state)
    state.update(status='completed' if not state['failed'] else 'failed', finished_at=now())
    write(args.output_dir / 'progress.json', state)
    write(args.output_dir / 'live_calls.json', completed)
    return int(state['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
