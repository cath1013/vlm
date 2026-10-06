#!/usr/bin/env python3
"""Final frozen 0–2 s paired API evaluation. No fitting or calibration.

Example: python examples/evaluate_frozen_models_api_val104.py --model MODEL
--out out/frozen_api --dry-run. English only. Replay is intentionally required
on resume as well: saved classifier decisions are never a preflight substitute.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_llm import providers, i18n

RUN = ROOT / 'out/collision_risk_hazard/run_20261006_window_balanced'
COHORT = RUN / 'final_analysis/v4_vs_crn4_2s_all_windows.csv'
EXPECTED = {'v4': dict(TP=46, FP=55, TN=111, FN=7),
            'crn4': dict(TP=41, FP=40, TN=126, FN=12)}
PROMPT_KIND = 'accident_frozen_network'
PROMPT = """You are an expert in cooperative-driving (V2X) accident prediction.
Using the observed vehicle motion history and supplied network evidence, judge
whether an actual physical collision will occur in each requested future 1-second interval.
Positions use local ENU coordinates in metres; heading is clockwise from north.
Only information up to the observation endpoint is observed evidence.
The network_evidence block is produced by a learned model and is fallible evidence,
not a ground-truth label. A high risk score, accepted interaction, small predicted
clearance, or predicted contact does not by itself prove collision. Conversely,
absence of predicted trajectory contact does not by itself prove safety when
observed motion and other evidence support imminent collision. Risk/verifier
scores are model-specific outputs, not calibrated probabilities of ground-truth
collision. Do not mechanically threshold or copy them.
Traffic-signal state, stop lines, and lane-marking types are unavailable.
Set accident_expected=true only when the total supplied evidence supports physical
vehicle contact/body overlap in that interval. Mere convergence, small separation,
shared road/junction use, opposing travel, or possible crossing is not itself collision.
Consider recent speed and signed acceleration trends and whether conflict will
persist or be avoided. Use only supplied actor ids. When accident_expected=false,
involved_actor_ids must be empty. Do not invent missing sensor information.
Give one concise sentence describing the physical evidence supporting or arguing
against collision in this interval. Do not use the network score or network decision
itself as the reason. Return exactly k=1 for (0,1] and k=2 for (1,2], relative to t_end.
"""
QUESTION = 'Will physical vehicle contact occur in each interval: k=1 (0,1], k=2 (1,2] after t_end? horizon_s=2. Return both buckets.'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def schema():
    result = i18n.prediction_schema('en', n_buckets=2)
    item = result['properties']['predictions']['items']['properties']
    item['k']['enum'] = [1, 2]
    item['reason']['description'] = ('Give one concise sentence describing the physical evidence supporting or '
        'arguing against collision in this interval. Do not use the network score or network decision itself as the reason.')
    return result


def load_cohort(path=COHORT):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 219 and len({r['window_id'] for r in rows}) == 219,
            'Frozen cohort must contain exactly 219 unique window_ids')
    return {r['window_id']: r for r in rows}


def confusion(rows):
    return {label: sum(('TP' if r['decision'] else 'FN') == label if r['actual'] else
                       ('FP' if r['decision'] else 'TN') == label for r in rows)
            for label in ('TP', 'FP', 'TN', 'FN')}


def assert_networks(cases, ledger):
    require(set(cases) == set(ledger), 'Replay/cohort window join mismatch')
    for condition in EXPECTED:
        cm = confusion([dict(actual=c['gt_bucket'] in (1, 2), decision=c[condition]['decision'])
                        for c in cases.values()])
        require(cm == EXPECTED[condition], f'{condition} frozen confusion mismatch: {cm}')
        for wid, c in cases.items():
            require(c[condition]['decision'] == (ledger[wid][condition + '_predicted'] == 'True'),
                    f'{condition} frozen per-window mismatch: {wid}')
            require(c['gt_bucket'] == int(ledger[wid]['gt_bucket']), f'GT mismatch: {wid}')
    return EXPECTED


def observation(window):
    from traffic_llm.swept_path import actor_footprint_m
    actors = []
    fields = ('actor_id kind cls world_xy heading_deg speed_mps accel_mps2 maneuver observed_by '
              'confidence position_quality track_age_s heading_source observed_range_m source_track_ids').split()
    for a in window.last.actors:
        row = {key: getattr(a, key) for key in fields}
        row['placement'] = asdict(a.placement) if a.placement else None
        row['footprint_length_width_m'] = actor_footprint_m(a)
        points = {float(t): [x, y] for t, x, y in a.track_history
                  if window.t_end - 5 <= t <= window.t_end}
        for snap in window.snapshots:
            if window.t_end - 5 <= snap.t <= window.t_end:
                for past in snap.actors:
                    if past.actor_id == a.actor_id:
                        points[float(snap.t)] = list(past.world_xy)
        row['observed_history'] = [[t-window.t_end, *xy] for t, xy in sorted(points.items())]
        actors.append(row)
    ctx = getattr(window.last, 'map_context', {}) or {}
    return dict(observation_window=dict(t_start_s=window.t_start, t_end_s=window.t_end,
                    history_s=5, time_reference='history offsets relative to t_end'), actors=actors,
                map_conventions=dict(coordinates='local ENU metres', heading='clockwise from north degrees',
                    drive_side=ctx.get('drive_side', 'right'), lane_numbering=ctx.get('lane_numbering', 'from_median'),
                    lane_width_m=ctx.get('lane_width_m')))


def paths(actor):
    result = []
    for p in actor.predictions:
        times = p.waypoint_times_s or [i * p.horizon_s / (len(p.waypoints)-1)
                                      for i in range(len(p.waypoints))]
        result.append(dict(coordinates_enu_m=[[t, *xy] for t, xy in zip(times, p.waypoints) if 0 <= t <= 2]))
    return result


def v4_evidence(window, models, wcfg, gt):
    from examples import evaluate_pair_reranker_v4_0_2 as frozen
    actors, _ = frozen.rank_actors(window.last, wcfg.actor_cap(len(window.last.actors)))
    by_id = {a.actor_id: a for a in actors}
    pairs = frozen.swept_pair_clearances(actors, horizon_s=wcfg.horizon_s,
        sample_dt_s=wcfg.swept_sample_dt_s, contact_margin_m=wcfg.swept_contact_margin_m,
        exclude_touching_now=wcfg.swept_exclude_touching_now,
        exclude_static_pairs=wcfg.swept_exclude_static_pairs)
    candidates = {1: [], 2: []}
    accepted = []
    for k, pool in frozen.bucket_pools(pairs).items():
        for p in pool:
            features = frozen.pair_features_v4(by_id[p.actor_a], by_id[p.actor_b], p,
                                               wcfg.horizon_s, window.last.interactions)
            candidates[k].append(dict(actor_a=p.actor_a, actor_b=p.actor_b,
                                      predicted_contact=bool(p.predicted_contact), features=features))
            score = models[k].predict_features(features)
            if score >= models[k].threshold:
                # Candidate selection/features remain the exact frozen 5-s path.
                # Visible geometry is restricted to the requested 0–2 s horizon.
                visible_pairs = frozen.swept_pair_clearances(
                    [by_id[p.actor_a], by_id[p.actor_b]], horizon_s=2.,
                    sample_dt_s=wcfg.swept_sample_dt_s,
                    contact_margin_m=wcfg.swept_contact_margin_m,
                    exclude_touching_now=wcfg.swept_exclude_touching_now,
                    exclude_static_pairs=False)
                require(len(visible_pairs) == 1, 'Missing accepted-pair 2-s geometry')
                visible = visible_pairs[0]
                accepted.append(dict(actor_a=p.actor_a, actor_b=p.actor_b, k=k, verifier_score=score,
                    future_coordinates={aid: paths(by_id[aid]) for aid in (p.actor_a, p.actor_b)},
                    predicted_minimum_footprint_clearance_m=visible.minimum_clearance_m,
                    time_of_predicted_minimum_clearance_s=visible.time_after_observation_s,
                    predicted_contact=bool(visible.predicted_contact), first_predicted_contact_s=visible.first_contact_s,
                    current_clearance_m=visible.clearance_at_observation_m, contact_duration_s=visible.contact_duration_s,
                    contact_event=visible.contact_event, contact_at_observation=visible.contact_at_observation,
                    contact_before_observation=visible.contact_before_observation))
    record = frozen.evaluate_window(gt, candidates, models)
    require(record is not None, 'Frozen cohort became censored')
    require(record['predicted'] == bool(accepted), 'Accepted interaction mismatch')
    return dict(decision=record['predicted'], pairs=record['selected_pairs'],
                evidence=dict(source_type='trajectory_pair_verifier', accepted_interactions=accepted))


def reproduce(args, ledger):
    import torch
    from examples import evaluate_pair_reranker_v4_0_2 as v4
    from examples import diagnose_crn4_false_positives as crn
    from traffic_llm.collision_risk_net import CollisionRiskNet, load_v2
    checkpoint = RUN / 'crn4/collision_risk_final.pt'
    require(hashlib.sha256(checkpoint.read_bytes()).hexdigest() ==
        '311bac77b615d7c71bc05567f053cdeb139e7018e21de26c3d97cc9f410f454f', 'CRN checkpoint changed')
    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    require(saved['max_horizon'] == 4, 'Expected frozen CRN-4 horizon')
    model = CollisionRiskNet(load_v2(saved['v2_checkpoint'], args.device), saved['hidden_dim'],
        saved['dropout'], max_horizon=4).to(args.device)
    model.load_state_dict(saved['state_dict'])
    model.frozen_thresholds = saved['threshold_by_horizon']
    crn.evaluator.verify_manifest(RUN / 'val104.jsonl', 'val', smoke=False)
    rows = crn.evaluator.read_rows(RUN / 'val104.jsonl', 'val')
    require(not set(saved['train_scenario_ids']) & {r['scenario_id'] for r in rows}, 'Train/val overlap')
    risks, _, _ = crn.infer(model, rows, args.device, args.batch_size,
                           saved['threshold_by_horizon']['2'], ledger)
    risk_by_id = {r['window_id']: r for r in risks}
    models = v4.load_frozen_models(v4.DEFAULT_BUCKET1_MODEL, v4.DEFAULT_BUCKET2_MODEL)
    predictor = ROOT / 'out/predict_model_joint_scene_v2/joint_scene_motionnet_v2_best.pt'
    cfg = v4.PipelineConfig()
    cfg.deepaccident.observation_mode = 'sensor3d'
    cfg.predictor = v4.JointSceneTorchPredictor(str(predictor), device=args.device)
    runner = v4.DeepAccidentRunner(args.root, cfg)
    scenarios = sorted((s for s in runner.list_scenarios() if s.split == 'val'), key=lambda s:s.scenario_id)
    require(len(scenarios) == 104, 'Expected official val104 scenario resolution')
    wcfg = v4.WindowConfig(window_s=5., stride_s=1., horizon_s=5., snapshot_rate_hz=2.,
                          warmup=False, full_window=False)
    cases = {}
    for s in scenarios:
        if not any(r['scenario_id'] == s.scenario_id for r in ledger.values()):
            continue
        xodr = v4.find_xodr(s.town, [args.carla_maps])
        require(bool(xodr), f'Missing map: {s.town}')
        result = runner.build(s.scenario, s.scenario_type, opendrive_path=xodr)
        snapshots = list(result.snapshots(rate_hz=2.))
        collision = v4.estimate_collision(s, cfg.deepaccident)
        windows, _ = v4.build_windows(snapshots, wcfg,
            collision_time_s=collision.time_s if collision and collision.occurred else None)
        ids = {agent:s.meta.agent_id_of(agent) for agent in s.agents}
        for w in windows:
            wid = f'{s.scenario_id}:{w.index}:{w.t_end:.3f}'
            if wid not in ledger:
                continue
            gt = v4.window_ground_truth(w, wcfg, collision=collision, agent_carla_ids=ids,
                scenario_id=s.scenario_id, scenario_split=s.scenario_type, dataset_split=s.split,
                town=s.town, data_end_s=snapshots[-1].t)
            r = risk_by_id[wid]
            pair = json.loads(r['top1_pair'])
            evidence = dict(actor_a=pair[0], actor_b=pair[1], **{key:r[key] for key in
                ('hazard_1', 'hazard_2', 'event_probability_1', 'event_probability_2',
                 'cumulative_risk_1', 'cumulative_risk_2')})
            truth = v4.evaluation_truth(gt)
            require(truth is not None, f'Censored window: {wid}')
            require(truth[0] == (r['gt_bucket'] if r['gt_bucket'] in (1,2) else None),
                    f'Independent V4/CRN truth mismatch: {wid}')
            cases[wid] = dict(gt_bucket=r['gt_bucket'], gt=gt,
                gt_groups=[sorted(g) for g in truth[1]], observation=observation(w),
                v4=v4_evidence(w, models, wcfg, gt),
                crn4=dict(decision=r['status'] in ('TP', 'FP'), pairs=[pair],
                    evidence=dict(source_type='observation_pair_risk', selected_interaction=evidence)))
        print(f'Replayed {s.scenario_id}: {len(cases)}/219', flush=True)
    assert_networks(cases, ledger)
    files = [checkpoint, predictor, Path(saved['v2_checkpoint']), v4.DEFAULT_BUCKET1_MODEL,
             v4.DEFAULT_BUCKET2_MODEL, RUN/'val104.jsonl', COHORT]
    provenance = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return cases, dict(network_checkpoint_SHA256=provenance,
        frozen_thresholds=dict(v4={str(k):m.threshold for k,m in models.items()},
                               crn4=saved['threshold_by_horizon']))


COMMON_KEYS = {'observation_window', 'actors', 'map_conventions'}
FORBIDDEN = ('gt_', 'ground_truth', 'collision_bucket', 'comparison_group', 'checkpoint',
             'threshold', 'training', 'validation_performance', 'later_collision')


def audit_payload(payload, condition, network):
    def walk(value):
        if isinstance(value, dict):
            for key, v in value.items():
                require(key.lower() != 'gt' and not any(word in key.lower() for word in FORBIDDEN), f'Forbidden key: {key}')
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, str):
            require(not any(word in value.lower() for word in ('checkpoint', '.pt', '.pth', 'threshold',
                'comparison_group', 'ground_truth', 'crn4', 'jointscene', 'v4')), 'Forbidden visible value')
            require(value not in ('TP', 'FP', 'TN', 'FN', 'positive', 'negative'), 'Visible grading label')
    walk(payload)
    require(set(payload['observation']) == COMMON_KEYS, 'Unexpected common evidence')
    ids = {a['actor_id'] for a in payload['observation']['actors']}
    evidence = payload['network_evidence']
    if condition == 'crn4':
        require(set(evidence) == {'source_type', 'selected_interaction'}, 'Unexpected CRN evidence')
        interactions = [evidence['selected_interaction']]
        require(set(interactions[0]) == {'actor_a', 'actor_b', 'hazard_1', 'hazard_2',
            'event_probability_1', 'event_probability_2', 'cumulative_risk_1', 'cumulative_risk_2'},
            'CRN contains trajectory or unexpected evidence')
    else:
        interactions = evidence['accepted_interactions']
        require([[p['actor_a'], p['actor_b']] for p in interactions] == network['pairs'],
                'V4 future geometry must be restricted to accepted pairs')
        for p in interactions:
            require(set(p['future_coordinates']) == {p['actor_a'], p['actor_b']}, 'Unrelated future geometry')
            require(p['k'] in (1,2), 'Invalid V4 interval')
    for p in interactions:
        require({p['actor_a'], p['actor_b']} <= ids, 'Evidence actor absent from observed actor list')
    for a in payload['observation']['actors']:
        require(set(a) <= set(('actor_id kind cls world_xy heading_deg speed_mps accel_mps2 maneuver observed_by '
            'confidence position_quality track_age_s heading_source observed_range_m source_track_ids '
            'placement footprint_length_width_m observed_history').split()), 'Unexpected actor evidence')
        require(all(-5 <= point[0] <= 0 for point in a['observed_history']), 'Future history leakage')


def prepare(cases, args):
    requests = []
    conditions = ('v4', 'crn4') if args.condition == 'both' else (args.condition,)
    for wid, c in sorted(cases.items()):
        common = canonical(c['observation'])
        for condition in conditions:
            payload = dict(observation=json.loads(common), network_evidence=c[condition]['evidence'], horizon_s=2)
            audit_payload(payload, condition, c[condition])
            require(canonical(payload['observation']) == common, 'Common observation differs')
            extra = {'generationConfig': {'temperature':args.temperature}} if args.provider == 'gemini' else {'temperature':args.temperature}
            if args.thinking_config is not None:
                require(args.provider == 'gemini', '--thinking-config is Gemini-only')
                extra['generationConfig']['thinkingConfig'] = json.loads(args.thinking_config)
            body = providers.build_request(args.provider, system=PROMPT, blocks=[canonical(payload), QUESTION],
                schema=schema(), model=args.model, max_tokens=args.max_output_tokens, effort='', extra=extra)
            requests.append(dict(request_id=f'{wid}__{condition}', window_id=wid, condition=condition,
                                 body=body, observation_hash=digest(common)))
    random.Random(args.seed).shuffle(requests)
    require(len(requests) == 219 * len(conditions), 'Wrong request count')
    if args.condition == 'both':
        by_window = {}
        for r in requests:
            by_window.setdefault(r['window_id'], []).append(r['observation_hash'])
        require(all(len(h)==2 and h[0]==h[1] for h in by_window.values()), 'Paired observation mismatch')
    return requests


def validate_response(response, actor_ids):
    def check(value, spec):
        typ = spec['type']
        valid = {'object': isinstance(value, dict), 'array': isinstance(value, list),
                 'string': isinstance(value, str), 'integer': type(value) is int,
                 'boolean': type(value) is bool}.get(typ, False)
        require(valid, f'Response schema type violation: {typ}')
        if 'enum' in spec:
            require(value in spec['enum'], 'Response enum violation')
        if typ == 'object':
            require(set(spec.get('required', [])) <= set(value), 'Missing response field')
            require(set(value) <= set(spec['properties']), 'Unexpected response field')
            for key, child in value.items():
                check(child, spec['properties'][key])
        elif typ == 'array':
            for child in value:
                check(child, spec['items'])
    check(response, schema())
    predictions = response['predictions']
    require(len(predictions) == 2 and {p['k'] for p in predictions} == {1,2}, 'Require exactly k=1 and k=2')
    for p in predictions:
        require(p['interval_s'].replace(' ', '') == {1:'(0,1]', 2:'(1,2]'}[p['k']], 'Wrong interval_s')
        ids = p['involved_actor_ids']
        require(len(ids)==len(set(ids)) and set(ids) <= set(actor_ids), 'Unknown/duplicate actor ids')
        require(not ids if not p['accident_expected'] else len(ids)>=2, 'Invalid collision actor ids')
    return response


def grade(case, response):
    if response is None:
        return dict(status='FAILED', binary_2s_window=None, strict_bucket=None, actor_pair=None)
    positives = [p for p in response['predictions'] if p['accident_expected']]
    ks = sorted(p['k'] for p in positives)
    ids = sorted({aid for p in positives for aid in p['involved_actor_ids']})
    actual = case['gt_bucket'] in (1,2)
    expected = [case['gt_bucket']] if actual else []
    groups = [set(g) for g in case['gt_groups']]
    hit = any(any((a in g and b in h) or (b in g and a in h)
          for i,g in enumerate(groups) for h in groups[i+1:])
          for p in positives if len(p['involved_actor_ids']) == 2
          for a,b in itertools.combinations(p['involved_actor_ids'],2)) if actual else False
    recovered = sum(bool(set(ids) & g) for g in groups)
    return dict(status='SCORED', decision=bool(ks), bucket=ks, actor_ids=ids,
        binary_2s_window=('TP' if ks else 'FN') if actual else ('FP' if ks else 'TN'),
        strict_bucket=dict(correct=ks==expected, bucket_correct={str(k):(k in ks)==(k in expected) for k in (1,2)}),
        actor_pair=dict(exact_gt_pair_hit=hit, vehicle_recall=recovered/len(groups) if groups else None,
            actor_precision=sum(any(a in g for g in groups) for a in ids)/len(ids) if ids else None))


def transitions(cases, results, condition):
    counts = Counter()
    for wid, c in cases.items():
        g = results.get((wid, condition))
        if not g or g['status'] != 'SCORED':
            continue
        actual = c['gt_bucket'] in (1,2)
        n = ('TP' if c[condition]['decision'] else 'FN') if actual else ('FP' if c[condition]['decision'] else 'TN')
        counts[f'network_{n} -> LLM_{g["binary_2s_window"]}'] += 1
    keys = [('TP','TP'), ('TP','FN'), ('FN','TP'), ('FN','FN'), ('FP','TN'), ('FP','FP'), ('TN','FP'), ('TN','TN')]
    out = {f'network_{a} -> LLM_{b}':counts[f'network_{a} -> LLM_{b}'] for a,b in keys}
    ratio = lambda a,b: a/b if b else None
    out.update(fraction_of_network_FPs_filtered_by_LLM=ratio(counts['network_FP -> LLM_TN'],
        sum(counts[f'network_FP -> LLM_{b}'] for b in ('TN','FP'))),
        fraction_of_network_TPs_destroyed_by_LLM=ratio(counts['network_TP -> LLM_FN'],
        sum(counts[f'network_TP -> LLM_{b}'] for b in ('TP','FN'))),
        new_FPs_introduced_by_LLM=counts['network_TN -> LLM_FP'],
        network_FNs_recovered_by_LLM=counts['network_FN -> LLM_TP'],
        unscored=219-sum(counts.values()), fraction_denominator='scored network strata only')
    return out


def metrics(grades):
    scored = [g for g in grades if g['status']=='SCORED']
    cm = Counter(g['binary_2s_window'] for g in scored)
    tp,fp,tn,fn = (cm[k] for k in ('TP','FP','TN','FN'))
    div = lambda a,b: a/b if b else None
    p,r,s = div(tp,tp+fp),div(tp,tp+fn),div(tn,tn+fp)
    mean = lambda vals: sum(vals)/len(vals) if vals else None
    return dict(**{k:cm[k] for k in ('TP','FP','TN','FN')}, scored=len(scored), unscored=len(grades)-len(scored),
        precision=p, recall=r, specificity=s, false_positive_rate=div(fp,fp+tn), F1=div(2*tp,2*tp+fp+fn),
        balanced_accuracy=(r+s)/2 if r is not None and s is not None else None,
        strict_bucket_accuracy=mean([g['strict_bucket']['correct'] for g in scored]),
        per_bucket_accuracy={str(k):mean([g['strict_bucket']['bucket_correct'][str(k)] for g in scored]) for k in (1,2)},
        exact_GT_pair_hits=sum(g['actor_pair']['exact_gt_pair_hit'] for g in scored),
        vehicle_recall=mean([g['actor_pair']['vehicle_recall'] for g in scored if g['actor_pair']['vehicle_recall'] is not None]),
        actor_precision=mean([g['actor_pair']['actor_precision'] for g in scored if g['actor_pair']['actor_precision'] is not None]))


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def write_jsonl(path, rows):
    """Publish a complete JSONL snapshot atomically."""
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(canonical(row)+'\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def request_definitions(out, requests, resume):
    """Save once; a resume must match every immutable definition exactly."""
    path = out/'requests.jsonl'
    require(len({r['request_id'] for r in requests}) == len(requests),
            'Duplicate request definitions')
    if path.exists():
        require(resume, 'Request definitions exist; use --resume')
        with path.open(encoding='utf-8') as f:
            saved = [json.loads(line) for line in f]
        require(canonical(saved) == canonical(requests), 'Resume request definitions changed')
    else:
        write_jsonl(path, requests)


def response_path(out, request):
    # Stable reversible request ids live inside the files; digest avoids slash/path traversal.
    return out/'parsed_responses'/f'{digest(request["request_id"])}.json'


def resume_response(out, request, ids):
    path = response_path(out, request)
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    require(saved['request_id'] == request['request_id'] and saved['body_hash'] == digest(canonical(request['body'])),
            'Resume request changed')
    try:
        return validate_response(saved['parsed_response'], ids)
    except Exception:
        return None


def call_api(args, body):
    key = os.environ.get(args.api_key_env)
    require(bool(key), f'Missing credential environment variable: {args.api_key_env}')
    headers = {'Content-Type':'application/json'}
    if args.provider == 'gemini':
        headers['x-goog-api-key'] = key
    elif args.provider == 'openai':
        headers['Authorization'] = f'Bearer {key}'
    else:
        headers.update({'x-api-key':key, 'anthropic-version':'2023-06-01'})
    req = urllib.request.Request(providers.endpoint_for(args.provider, args.model),
                                 data=canonical(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        return response.status, response.read().decode()


def execute(args, request, ids, append):
    path = response_path(args.out, request)
    existing = resume_response(args.out, request, ids) if args.resume else None
    if existing is not None:
        saved = json.loads(path.read_text())
        return existing, saved['usage'], saved['latency_s']
    require(not path.exists() or args.resume, 'Response exists; use --resume')
    for attempt in range(args.retries+1):
        start = time.monotonic()
        raw, parsed, error, http_status, usage = None, None, None, None, {}
        retry = False
        try:
            http_status, raw = call_api(args, request['body'])
            api = json.loads(raw)
            usage = providers.usage_of(args.provider, api)
            parsed = validate_response(json.loads(''.join(providers.response_texts(args.provider, api))), ids)
        except urllib.error.HTTPError as exc:
            http_status, raw = exc.code, exc.read().decode(errors='replace')
            error = f'HTTP {exc.code}'
            retry = exc.code == 429 or 500 <= exc.code < 600
        except (urllib.error.URLError, TimeoutError) as exc:
            error, retry = str(exc), True
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        latency = time.monotonic()-start
        saved = dict(request_id=request['request_id'], window_id=request['window_id'], condition=request['condition'],
            body_hash=digest(canonical(request['body'])), attempt=attempt, http_status=http_status,
            raw_response=raw, parsed_response=parsed, usage=usage, latency_s=latency, error=error)
        write_json(args.out/'raw_responses'/f'{digest(request["request_id"])}__{time.time_ns()}.json', saved)
        append('api_attempts.jsonl', {k:v for k,v in saved.items() if k not in ('raw_response','parsed_response')})
        if parsed is not None:
            require(not path.exists() or resume_response(args.out, request, ids) is None,
                    'Refusing to overwrite valid parsed response')
            write_json(path, saved)
            return parsed, usage, latency
        append('failures.jsonl', saved)
        if not retry or attempt == args.retries:
            return None, usage, latency
        time.sleep(min(30., args.backoff * 2**attempt))


def outputs(args, cases, records):
    results = {(r['window_id'],r['condition']):r['grading'] for r in records}
    conditions = ('v4','crn4') if args.condition=='both' else (args.condition,)
    scores = {c:metrics([results.get((wid,c), grade(case,None)) for wid,case in cases.items()]) for c in conditions}
    scores['primary_metric'] = 'binary_2s_window'
    paired, summary = [], Counter()
    for wid, c in sorted(cases.items()):
        a,b = (results.get((wid,condition), grade(c,None)) for condition in ('v4','crn4'))
        row = dict(window_id=wid, gt_binary_2s=c['gt_bucket'] in (1,2), gt_bucket=c['gt_bucket'],
                   v4_network_decision=c['v4']['decision'], crn_network_decision=c['crn4']['decision'])
        for condition,g in (('v4',a),('crn',b)):
            row.update({condition+'_llm_decision':g.get('decision'), condition+'_llm_bucket':canonical(g.get('bucket')),
                        condition+'_llm_actor_ids':canonical(g.get('actor_ids'))})
        if a['status']==b['status']=='SCORED':
            ca,cb = (g['binary_2s_window'] in ('TP','TN') for g in (a,b))
            summary[['both_wrong','CRN+LLM_only_correct','V4+LLM_only_correct','both_correct'][int(ca)*2+int(cb)]] += 1
            if not row['gt_binary_2s']:
                summary[['both_FP','V4+LLM_FP_only','CRN+LLM_FP_only','both_TN'][int(ca)*2+int(cb)]] += 1
        else:
            summary['unscored_pairs'] += 1
        paired.append(row)
    scores['paired'] = {key:summary[key] for key in ('both_correct','V4+LLM_only_correct',
        'CRN+LLM_only_correct','both_wrong','both_TN','V4+LLM_FP_only','CRN+LLM_FP_only','both_FP','unscored_pairs')}
    write_json(args.out/'scores.json', scores)
    write_json(args.out/'network_to_llm_transitions.json', {c:transitions(cases,results,c) for c in conditions})
    with (args.out/'paired_v4_crn_results.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['v4','crn4','both'], default='both')
    parser.add_argument('--provider', choices=providers.PROVIDERS, default='gemini')
    parser.add_argument('--model', required=True)
    parser.add_argument('--api-key-env', default='GEMINI_API_KEY')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-output-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--thinking-config', help='Explicit Gemini thinkingConfig JSON; omitted means provider default')
    parser.add_argument('--seed', type=int, default=20261007, help='Request shuffle seed; not an API sampling seed')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--root', default=str(Path.home()/'inclab-nas/DeepAccident'))
    parser.add_argument('--carla-maps', default=str(ROOT/'carla_map'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--backoff', type=float, default=2.)
    parser.add_argument('--timeout', type=float, default=180.)
    args = parser.parse_args(argv)
    require(args.workers>0 and args.batch_size>0 and args.max_output_tokens>0 and args.retries>=0,
            'Invalid counts')
    require(args.temperature>=0 and args.backoff>=0 and args.timeout>0, 'Invalid generation/retry setting')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest_path = args.out/'experiment_manifest.json'
    require(args.resume or not manifest_path.exists(), 'Output exists; use --resume or a new --out')
    try:
        ledger = load_cohort()
        cases, provenance = reproduce(args, ledger)
        requests = prepare(cases, args)
        audit = dict(passed=True, windows=len(cases), requests=len(requests), confusion=EXPECTED,
            invariants=['strict cohort join','independent network reproduction','paired byte-identical observations',
                'no GT/model metadata leakage','CRN risk only','V4 accepted geometry only',
                'actor membership','exact two response buckets validated'], api_calls=0)
    except Exception as exc:
        write_json(args.out/'dry_run_audit.json', dict(passed=False,error=f'{type(exc).__name__}: {exc}',api_calls=0))
        raise
    write_json(args.out/'dry_run_audit.json', audit)
    manifest = dict(provider=args.provider, model_id=args.model, temperature=args.temperature,
        max_output_tokens=args.max_output_tokens, thinking_configuration=json.loads(args.thinking_config) if args.thinking_config else 'provider default',
        worker_count=args.workers, seed=args.seed, api_seed_support='not enabled by this provider adapter', api_seed_sent=False,
        seed_scope='request ordering only; provider sampling determinism not asserted', prompt_kind=PROMPT_KIND,
        prompt_hash=digest(PROMPT), schema_hash=digest(canonical(schema())), question_hash=digest(QUESTION),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        evaluator_source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        condition=args.condition, request_hashes={r['request_id']:digest(canonical(r['body'])) for r in requests},
        retries=args.retries, backoff=args.backoff, timeout=args.timeout, **provenance)
    if manifest_path.exists():
        require(json.loads(manifest_path.read_text()) == manifest, 'Resume manifest/settings changed')
    else:
        write_json(manifest_path, manifest)
    request_definitions(args.out, requests, args.resume)
    for directory in ('raw_responses','parsed_responses'):
        (args.out/directory).mkdir(exist_ok=True)
    for name in ('api_attempts.jsonl','failures.jsonl'):
        (args.out/name).touch(exist_ok=True)
    lock = threading.Lock()
    def append(name, value):
        with lock, (args.out/name).open('a') as f:
            f.write(canonical(value)+'\n')
            f.flush()
            os.fsync(f.fileno())
    records = []
    if not args.dry_run:
        require(bool(os.environ.get(args.api_key_env)), f'Missing {args.api_key_env}')
    def work(r):
        c = cases[r['window_id']]
        ids = [a['actor_id'] for a in c['observation']['actors']]
        if args.dry_run:
            parsed, usage, latency = None, {}, None
        else:
            append('api_attempts.jsonl', dict(request_id=r['request_id'],
                window_id=r['window_id'], condition=r['condition'],
                body_hash=digest(canonical(r['body'])), status='QUEUED'))
            parsed, usage, latency = execute(args,r,ids,append)
        record = dict(request_id=r['request_id'], window_id=r['window_id'], condition=r['condition'],
            record_type='DRY_RUN' if args.dry_run else 'FINAL',
            frozen_network_decision=c[r['condition']]['decision'], frozen_network_pairs=c[r['condition']]['pairs'],
            llm_parsed_response=parsed, grading=grade(c,parsed), token_usage=usage, latency_s=latency)
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work,r) for r in requests]
        for future in as_completed(futures):
            records.append(future.result())
    by_id = {r['request_id']:r for r in records}
    require(len(by_id) == len(records) == len(requests), 'Incomplete or duplicate final records')
    records = [by_id[r['request_id']] for r in requests]
    # Dry runs use the same snapshot file; live resume replaces the entire file.
    write_jsonl(args.out/'records.jsonl', records)
    outputs(args,cases,records)
    print(f'Wrote {len(records)} records to {args.out}', flush=True)
    if not args.dry_run:
        failures = sum(r['grading']['status'] != 'SCORED' for r in records)
        if failures:
            raise SystemExit(f'{failures} API requests FAILED/UNSCORED; use --resume to retry')


if __name__ == '__main__':
    main()
