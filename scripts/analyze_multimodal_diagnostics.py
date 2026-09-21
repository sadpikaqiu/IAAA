"""Summarize original full runs and labeled counterfactual diagnostic cases."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def key(row):
    return row['user_id'], row['trajectory_id']


def source_summary(folder):
    a, b = [read(folder / f'{v}.json') for v in ('text', 'multimodal')]
    a = {key(r): r for r in a['candidate_diagnostics']['sessions']}
    b = {key(r): r for r in b['candidate_diagnostics']['sessions']}
    out = {name: Counter() for name in ('hit10_losses', 'hit10_gains', 'pool_transitions', 'pool_losses')}
    for k, x in a.items():
        y = b[k]
        shape = f"{x['pool_size']}->{y['pool_size']}"
        out['pool_transitions'][shape] += 1
        if x['rank'] and not y['rank']:
            reason = 'absent_from_raw' if not y['in_raw'] else 'lost_from_final_pool' if not y['in_pool'] else 'below_top10_in_pool'
            out['hit10_losses'][reason] += 1
        if not x['rank'] and y['rank']:
            reason = 'new_raw_recall' if not x['in_raw'] else 'new_pool_retention' if not x['in_pool'] else 'reranked_into_top10'
            out['hit10_gains'][reason] += 1
        if x['in_pool'] and not y['in_pool']:
            out['pool_losses'][shape] += 1
    return {k: dict(v) for k, v in out.items()}


def rank_in_profiles(profiles, target, remove_evidence=False):
    def sortkey(profile):
        score = profile['score']
        if remove_evidence:
            score = round(sum(v for k, v in profile['score_decomposition'].items()
                              if k not in {'image_intent_relevance', 'review_intent_relevance'}), 6)
        return score, profile['confidence'], -profile['distance_km']
    ids = [p['poi_id'] for p in sorted(profiles, key=sortkey, reverse=True)]
    return ids.index(target) + 1 if target in ids else None


def selection_counterfactual(round_data, target, evidence_weight):
    """Hold the raw candidates and pool size fixed; remove only recall-prior boost."""
    weights = {'historical': .30, 'spatial': .20, 'category_intent': .20,
               'transition': .15, 'temporal_popularity': .10, 'peer': .05,
               'poi_evidence': evidence_weight}
    raw = round_data['raw']
    maxima = {}
    for row in raw:
        for source, score in row['source_scores'].items():
            maxima[source] = max(maxima.get(source, 0), score)
    components = {}
    for row in raw:
        parts = {s: weights.get(s, .05) * v / maxima[s] if maxima[s] else 0
                 for s, v in row['source_scores'].items()}
        assert abs(sum(parts.values()) - row['prior_score']) < 1e-9
        components[row['poi_id']] = parts
    current = sorted(raw, key=lambda r: (r['prior_score'], -r['distance_km']), reverse=True)
    without = sorted(raw, key=lambda r: (sum(v for k, v in components[r['poi_id']].items()
                                            if k != 'poi_evidence'), -r['distance_km']), reverse=True)
    size = len(round_data['chosen'])
    ids = [r['poi_id'] for r in current]
    clean_ids = [r['poi_id'] for r in without]
    return {'raw_count': len(raw), 'pool_size': size,
            'prior_rank': ids.index(target) + 1 if target in ids else None,
            'prior_rank_without_evidence_boost': clean_ids.index(target) + 1 if target in clean_ids else None,
            'target_components': components.get(target),
            'would_enter_without_boost_and_quota': target in clean_ids[:size],
            'boost_alone_displaces_target': target in clean_ids[:size] and target not in ids[:size],
            'boost_alone_promotes_target': target not in clean_ids[:size] and target in ids[:size],
            'top_evidence_source_score': maxima.get('poi_evidence', 0)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-results', type=Path, required=True)
    parser.add_argument('--diagnostic-results', type=Path, required=True)
    parser.add_argument('--replay-results', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    output = {'scope': 'diagnostic_outcome_selected_cases_not_benchmark',
              'original_full': {c: source_summary(args.original_results / (c + '_full')) for c in ('NYC', 'TKY')},
              'diagnostic_progress': read(args.diagnostic_results / 'progress.json'), 'by_city': {}, 'cases': []}
    for path in sorted((args.diagnostic_results / 'cases').glob('*/*.json')):
        case = read(path)
        city = case['city']
        variants = case['variants']
        target = case['ground_truth_poi_id']
        summary = output['by_city'].setdefault(city, {'n': 0, 'live_reproduction': Counter(),
                                                    'live_statuses': Counter(), 'interventions': {},
                                                    'quota_target_evictions': 0,
                                                    'recall_prior_boost': Counter(),
                                                    'same_pool_remove_ranking': Counter()})
        summary['n'] += 1
        row = {'city': city, 'user_id': case['user_id'], 'trajectory_id': case['trajectory_id'],
               'selection_groups': case['selection_groups'], 'target': target, 'variants': {}}
        for v in ('text_live', 'multimodal_live'):
            run = variants[v]
            summary['live_reproduction'][v + '_exact' if run['original_predictions_match'] else v + '_changed'] += 1
            summary['live_statuses'][v + '/' + run['llm_status']] += 1
            row[v + '_validation_errors'] = run['llm']['validation_errors'] if run['llm'] else None
        for name, run in variants.items():
            row['variants'][name] = {'rank': run['rank'], 'in_pool': run['in_pool'],
                                    'pool_size': len(run['rounds'][-1]['chosen']),
                                    'reflected': run['reflection']['triggered']}
        mm = variants['multimodal_live']
        last = mm['rounds'][-1]
        evicted = target in last.get('quota', {}).get('evicted', [])
        summary['quota_target_evictions'] += evicted
        row['target_evicted_by_forced_quota'] = evicted
        selection = selection_counterfactual(last, target, mm['config']['evidence_weight'])
        row['fixed_round_selection_counterfactual'] = selection
        summary['recall_prior_boost']['target_displaced'] += selection['boost_alone_displaces_target']
        summary['recall_prior_boost']['target_promoted'] += selection['boost_alone_promotes_target']
        if evicted:
            raw = {r['poi_id']: r for r in last['raw']}
            row['quota_evict_target_details'] = raw[target]
            row['quota_added'] = [raw[p] for p in last['quota']['added']]
        before = rank_in_profiles(last['profiles'], target)
        removed = rank_in_profiles(last['profiles'], target, remove_evidence=True)
        row['same_pool_target_rank_with_without_evidence_score'] = [before, removed]
        if before is not None:
            summary['same_pool_remove_ranking']['eligible'] += 1
            summary['same_pool_remove_ranking']['improved' if removed < before else 'worsened' if removed > before else 'same'] += 1
            if before > 10 and removed <= 10:
                summary['same_pool_remove_ranking']['hit10_recovered'] += 1
            if before <= 10 and removed > 10:
                summary['same_pool_remove_ranking']['hit10_lost'] += 1
        for name in ('intention_only', 'text_intention_mm_downstream', 'mm_no_quota',
                     'mm_no_ranking_evidence', 'mm_text_reflection', 'text_intention_mm_no_quota'):
            if name not in variants:
                continue
            # Intent-only and text-intent downstream compare against the text run;
            # remaining changes compare against the same cached multimodal run.
            baseline = variants['text_live'] if name in {'intention_only', 'text_intention_mm_downstream', 'text_intention_mm_no_quota'} else mm
            alt = variants[name]
            counts = summary['interventions'].setdefault(name, Counter())
            counts['n'] += 1
            counts['pool_gained'] += not baseline['in_pool'] and alt['in_pool']
            counts['pool_lost'] += baseline['in_pool'] and not alt['in_pool']
            counts['hit10_gained'] += not baseline['rank'] and bool(alt['rank'])
            counts['hit10_lost'] += bool(baseline['rank']) and not alt['rank']
        output['cases'].append(row)
    if args.replay_results:
        offline = {'progress': read(args.replay_results / 'progress.json'), 'by_city': {}, 'cases': []}
        inputs = read(args.replay_results / 'manifest.json')['source_cases']
        import hashlib
        for relative, expected in inputs.items():
            assert hashlib.sha256((args.diagnostic_results / relative).read_bytes()).hexdigest() == expected
        for path in sorted((args.replay_results / 'cases').glob('*/*.json')):
            case = read(path)
            source = read(args.diagnostic_results / 'cases' / case['city'] / path.name)
            runs = case['variants']
            assert runs['cached_original_control']['predictions'] == source['variants']['multimodal_live']['predictions']
            if 'intention_only' in source['variants']:
                assert runs['no_prior_boost_no_quota_no_ranking']['predictions'] == source['variants']['intention_only']['predictions']
            city = offline['by_city'].setdefault(case['city'], {'n': 0, 'interventions': {}})
            city['n'] += 1
            for name, run in runs.items():
                if name == 'cached_original_control':
                    continue
                base = runs['cached_original_control']
                count = city['interventions'].setdefault(name, Counter())
                count['pool_gained'] += not base['in_pool'] and run['in_pool']
                count['pool_lost'] += base['in_pool'] and not run['in_pool']
                count['hit10_gained'] += not base['rank'] and bool(run['rank'])
                count['hit10_lost'] += bool(base['rank']) and not run['rank']
            offline['cases'].append({k: case[k] for k in ('city', 'trajectory_id', 'selection_groups')} |
                {'variants': {name: {k: v[k] for k in ('rank', 'in_pool', 'pool_size')} for name, v in runs.items()}})
        offline['cached_controls_verified'] = True
        offline['all_downstream_disabled_matches_intention_only'] = True
        output['offline_prior_controls'] = offline
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    compact = {k: v for k, v in output.items() if k not in {'cases', 'offline_prior_controls'}}
    if args.replay_results:
        compact['offline_prior_controls'] = {k: v for k, v in offline.items() if k != 'cases'}
    print(json.dumps(compact, ensure_ascii=True))


if __name__ == '__main__':
    main()
