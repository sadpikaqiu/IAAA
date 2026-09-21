"""Offline controls using previously captured intentions; never calls an API."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import importlib
from pathlib import Path
import sys
import threading

from diagnose_multimodal_pipeline import read, write, now, sha, make_agent_class


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--source-results', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args(argv)
    if (args.output_dir / 'manifest.json').exists():
        parser.error('Use a fresh output directory')
    if read(args.source_results / 'progress.json')['status'] != 'completed':
        parser.error('The source diagnostic run must finish first')
    root = args.experiment_root.resolve()
    bundle = read(root / 'bundle_manifest.json')
    for name, expected in bundle['files'].items():
        if name.startswith('code/iaa_agent/'):
            assert sha((root / name).read_bytes()) == expected
    sys.path.insert(0, str(root / 'code'))
    engine = importlib.import_module('iaa_agent.engine')
    models = importlib.import_module('iaa_agent.models')
    data = importlib.import_module('iaa_agent.data')
    evidence = importlib.import_module('iaa_agent.evidence')
    TracedAgent = make_agent_class(engine, models)

    class RecallPriorControl(TracedAgent):
        def __init__(self, *a, suppress_prior=False, **kw):
            super().__init__(*a, **kw)
            self.suppress_prior = suppress_prior
        def _select_candidates(self, raw, expanded):
            original = self.config
            if self.suppress_prior:
                self.config = replace(original, evidence_weight=0)
            try:
                return super()._select_candidates(raw, expanded)
            finally:
                self.config = original

    paths = sorted((args.source_results / 'cases').glob('*/*.json'))
    cases = [read(path) for path in paths]
    inputs = {}
    for city in sorted({c['city'] for c in cases}):
        path = root / 'evidence' / Path(next(c for c in cases if c['city']==city)['variants']['multimodal_live']['config']['evidence_snapshot']).name
        store = evidence.EvidenceStore(path)
        store.validate_repository(data.NYCDataRepository(args.data_root / city))
        inputs[city] = store
    write(args.output_dir / 'manifest.json', {'created_at': now(), 'scope': 'offline_same_intention_controls',
          'source_cases': {str(p.relative_to(args.source_results)): sha(p.read_bytes()) for p in paths},
          'script_sha256': sha(Path(__file__).read_bytes()), 'api_calls': 0,
          'control': 'Set evidence_weight=0 only inside candidate selection; ranking keeps the original weight.'})
    state = {'status': 'running', 'started_at': now(), 'total': len(cases), 'completed': 0, 'failed': 0}
    write(args.output_dir / 'progress.json', state)
    local = threading.local()

    def run_case(case):
        if not hasattr(local, 'repos'):
            local.repos = {}
        city = case['city']
        if city not in local.repos:
            repo = data.NYCDataRepository(args.data_root / city)
            repo.use_user_chronological_split(.8)
            repo.prewarm_global_structures()
            local.repos[city] = repo
        repo = local.repos[city]
        query = repo.get_session_query(case['user_id'], case['trajectory_id'])
        source = case['variants']
        intention = models.Intention.model_validate(source['multimodal_live']['intention'])
        reflection = models.ReflectionRecord.model_validate(source['text_live']['reflection'])
        config = engine.RunConfig(**source['multimodal_live']['config'])
        output = {k: case[k] for k in ('city', 'user_id', 'trajectory_id', 'selection_groups', 'ground_truth_poi_id')}
        output['variants'] = {}
        prepared = None
        def run(name, current_config, suppress_prior, rank_evidence=True, forced=None):
            nonlocal prepared
            agent = RecallPriorControl(repo, current_config, evidence_store=inputs[city],
                                       frozen_intention=intention, prepared=prepared,
                                       ranking_evidence=rank_evidence, force_reflection=forced,
                                       suppress_prior=suppress_prior)
            def forbidden(*args, **kwargs):
                raise AssertionError('Offline replay attempted a model request')
            agent.llm.chat_json = forbidden
            result = agent.run_query(query)
            prepared = (agent.profile_saved, agent.peers_saved)
            predictions = [r.poi_id for r in result.ranked_pois]
            gt = result.ground_truth_poi_id
            output['variants'][name] = {'predictions': predictions,
                'rank': predictions.index(gt)+1 if gt in predictions else None,
                'in_pool': gt in result.candidate_pool_summary['candidate_poi_ids'],
                'pool_size': result.candidate_pool_summary['candidate_count'],
                'reflection': result.reflection.model_dump(mode='json'), 'rounds': agent.rounds,
                'suppressed_recall_prior': suppress_prior, 'ranking_evidence': rank_evidence,
                'evidence_quota': current_config.evidence_quota, 'api_calls': 0}
            return result
        control = run('cached_original_control', config, False)
        assert output['variants']['cached_original_control']['predictions'] == source['multimodal_live']['predictions']
        assert control.candidate_pool_summary['candidate_poi_ids'] == source['multimodal_live']['rounds'][-1]['chosen']
        run('no_recall_prior_boost', config, True)
        run('no_prior_boost_no_quota', replace(config, evidence_quota=0), True)
        run('no_prior_boost_no_quota_text_reflection', replace(config, evidence_quota=0), True, forced=reflection)
        run('no_prior_boost_no_quota_no_ranking', replace(config, evidence_quota=0), True, rank_evidence=False)
        output['finished_at'] = now()
        write(args.output_dir / 'cases' / city / f"{case['user_id']}_{case['trajectory_id']}.json", output)
        return case['trajectory_id']

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        jobs = {executor.submit(run_case, c): c for c in cases}
        for job in as_completed(jobs):
            case = jobs[job]
            try:
                result = job.result()
                state['completed'] += 1
                print(f"{state['completed']}/{len(cases)} {case['city']} {result}", flush=True)
            except Exception as exc:
                state['failed'] += 1
                write(args.output_dir / 'errors' / f"{case['city']}_{case['trajectory_id']}.json",
                      {'type': type(exc).__name__, 'message': str(exc)})
                print(f'ERROR {case["trajectory_id"]}: {exc}', flush=True)
            state['updated_at'] = now()
            write(args.output_dir / 'progress.json', state)
    state.update(status='completed' if not state['failed'] else 'failed', finished_at=now())
    write(args.output_dir / 'progress.json', state)
    return int(state['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
