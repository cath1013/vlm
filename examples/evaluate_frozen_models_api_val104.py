#!/usr/bin/env python3
"""Final frozen 0–2 s paired API evaluation. No fitting or calibration.

Example: python examples/evaluate_frozen_models_api_val104.py --model MODEL
--out out/frozen_api --dry-run. English only. Replay is intentionally required
on resume as well unless a source-validated replay cache is explicitly supplied.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_llm import providers, i18n

RUN = ROOT / 'out/collision_risk_hazard/run_20261006_window_balanced'
COHORT = RUN / 'final_analysis/v4_vs_crn4_2s_all_windows.csv'
FROZEN_CRN4_SHA256 = '311bac77b615d7c71bc05567f053cdeb139e7018e21de26c3d97cc9f410f454f'
EXPECTED = {'v4': dict(TP=46, FP=55, TN=111, FN=7),
            'crn4': dict(TP=41, FP=40, TN=126, FN=12)}
PROMPT_KIND = 'accident_frozen_network'
PROMPT = """You are an expert in cooperative-driving (V2X) accident prediction.
Using the observed vehicle motion history and supplied network evidence, identify
the interval containing the first new physical vehicle collision onset after t_end.
accident_expected refers to the ONSET of a new physical vehicle collision,
not continued contact after a collision that began in an earlier interval.
At most one of k=1 or k=2 should have accident_expected=true.
If accident_expected=true, involved_actor_ids must contain exactly the two actors
involved in that collision. If no new collision begins during (0,2], both buckets
must be false.
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
Set accident_expected=true only when the total supplied evidence supportsthe onset of physical vehicle contact/body overlap in that interval. Mere convergence, small separation,
shared road/junction use, opposing travel, or possible crossing is not itself collision.
Consider recent speed and signed acceleration trends and whether conflict will
persist or be avoided. Use only supplied actor ids. When accident_expected=false,
involved_actor_ids must be empty. Do not invent missing sensor information.
Give one concise sentence describing the physical evidence supporting or arguing
against collision in this interval. Do not use the network score or network decision
itself as the reason. Return exactly k=1 for (0,1] and k=2 for (1,2], relative to t_end.
"""
QUESTION = ('In which interval, if any, does the first new physical vehicle collision begin after t_end? '
            'Use k=1 for onset in (0,1] and k=2 for onset in (1,2]. '
            'If no new collision begins within 2 s, return false for both buckets. '
            'Return both bucket entries.')
CRN4_PROMPT_V2 = """You are a CRN4 selected-pair collision verifier, NOT a scene-wide collision detector.
Evaluate ONLY network_evidence.selected_interaction.actor_a and actor_b.
Other observed actors may be used only as context and must never become the predicted collision pair.
If accident_expected=true, involved_actor_ids must contain exactly the two selected-interaction actor IDs.
If accident_expected=false, involved_actor_ids must be empty.
CRN hazards, risk, and event probabilities are fallible evidence, not ground-truth labels;
do not mechanically threshold or copy them.
Predict true only when evidence strongly supports simultaneous physical footprint overlap.
High risk, small gap, rapid closing, convergence, crossing paths, opposing travel,
a shared junction, or a projected conflict point are not sufficient by themselves.
Treat observed speed, acceleration, and heading as local estimates, not guarantees of unchanged future motion.
For k=2 specifically, constant-velocity/constant-acceleration extrapolation or projected
path intersection alone is insufficient. Require persistent evidence in the observed
history plus strong support for physical overlap within (1,2].
If evidence cannot distinguish collision from a plausible near-miss/safe passage, return false.
For one projected collision, mark only the bucket containing first physical contact as true;
do not mark both k=1 and k=2 true. Continued contact is not a new collision onset.
Return exactly k=1 for (0,1] and k=2 for (1,2], relative to t_end.
Positions use local ENU coordinates in metres; heading is clockwise from north.
Only information up to the observation endpoint is observed evidence.
Traffic-signal state, stop lines, and lane-marking types are unavailable. Do not invent missing sensor information.
Keep each reason to one concise sentence describing physical evidence and do not use
the CRN score or decision itself as the reason.
"""


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
    item['k']['description'] = 'Integer bucket index. Must be 1 or 2.'
    item['interval_s']['enum'] = ['(0,1]', '(1,2]']
    item['interval_s']['description'] = 'Exact interval label corresponding to k: "(0,1]" for k=1 and "(1,2]" for k=2.'
    item['reason']['description'] = ('Give one concise sentence describing the physical evidence supporting or '
        'arguing against collision in this interval. Do not use the network score or network decision itself as the reason.')
    return result


def load_cohort(path=COHORT):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 219 and len({r['window_id'] for r in rows}) == 219,
            'Frozen cohort must contain exactly 219 unique window_ids')
    return {r['window_id']: r for r in rows}


def load_window_ids(path, cohort):
    """Read one window ID per nonblank line after frozen-cohort validation."""
    ids = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(bool(ids), 'Window IDs file is empty')
    require(len(ids) == len(set(ids)), 'Duplicate window IDs in subset file')
    unknown = sorted(set(ids) - set(cohort))
    require(not unknown, f'Unknown window IDs in subset file: {unknown}')
    return ids


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
        FROZEN_CRN4_SHA256, 'CRN checkpoint changed')
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


REPLAY_CACHE_VERSION = 1
REPLAY_CASE_KEYS = {'observation', 'gt_bucket', 'gt_groups', 'v4', 'crn4'}


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def replay_code_sha256():
    """Fingerprint replay/cache semantics without prompt or API configuration."""
    names = {'canonical', 'digest', 'require', 'load_cohort', 'assert_networks',
             'observation', 'paths', 'v4_evidence', 'reproduce', 'audit_payload',
             'file_sha256', 'replay_code_sha256', 'replay_source_fingerprints',
             'validate_replay_cases', 'cached_replay', 'save_replay_cache'}
    constants = {'EXPECTED', 'COMMON_KEYS', 'FORBIDDEN', 'FROZEN_CRN4_SHA256',
                 'REPLAY_CACHE_VERSION', 'REPLAY_CASE_KEYS'}
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if
             isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants
                                                  for t in node.targets)]
    require({node.name for node in nodes if isinstance(node, ast.FunctionDef)} == names,
            'Replay fingerprint function inventory mismatch')
    return digest(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False))


def replay_source_fingerprints(args, ledger, v2_checkpoint=None):
    """Hash replay inputs without network execution or scenario reconstruction.

    Sensor3d replay reads metadata, calibration pickles and label text, not
    camera pixels. Fingerprint all val metadata (scenario resolution) and all
    calibration/label inputs for cohort scenarios, including file inventories.
    """
    from importlib.metadata import version
    from traffic_llm.carla_map import find_xodr
    checkpoint = RUN/'crn4/collision_risk_final.pt'
    checkpoint_hash = file_sha256(checkpoint)
    require(checkpoint_hash == FROZEN_CRN4_SHA256, 'CRN checkpoint changed')
    import torch
    # Loading checkpoint metadata does not construct or execute either network.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(saved['max_horizon'] == 4, 'Expected frozen CRN-4 horizon')
    saved_v2 = str(Path(saved['v2_checkpoint']).resolve())
    if v2_checkpoint is not None:
        require(str(Path(v2_checkpoint).resolve()) == saved_v2, 'Cached V2 checkpoint identity mismatch')
    v2_checkpoint = saved_v2
    artifacts = [checkpoint, ROOT/'out/predict_model_joint_scene_v2/joint_scene_motionnet_v2_best.pt',
                 Path(v2_checkpoint), ROOT/'out/pair_reranker_v4/model_bucket1/pair_reranker_v4.json',
                 ROOT/'out/pair_reranker_v4/model_bucket2/pair_reranker_v4.json',
                 RUN/'val104.jsonl', COHORT, RUN/'val104.manifest.json']
    artifact_hashes = {str(p.resolve()):file_sha256(p) for p in artifacts}
    code_paths = list((ROOT/'traffic_llm').rglob('*.py'))
    # Provider request conversion and language/prompt schemas do not produce replay evidence.
    code_paths = [p for p in code_paths if p.name not in ('providers.py', 'i18n.py')]
    code_paths.extend(ROOT/'examples'/name for name in (
        'evaluate_pair_reranker_v4_0_2.py', 'diagnose_crn4_false_positives.py',
        'train_collision_risk_net.py', 'audit_gt_carla_boxes.py', 'plot_v4_crn4_error_cases.py',
        'audit_gt_xy_source_ablation.py', 'audit_gt_heading_source_ablation.py',
        'audit_predictor_trajectory_contacts.py'))
    code = {str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(code_paths)}
    code['evaluator_replay_semantics'] = replay_code_sha256()
    source_root = Path(args.root).resolve(strict=True)
    val_root = source_root if source_root.name == 'val' else source_root/'val'
    require(val_root.is_dir(), f'Missing official val source directory: {val_root}')
    data_files = set(val_root.glob('*/meta/*.txt'))
    require(bool(data_files), 'Missing val scenario metadata')
    # Calibration/label inventories for non-cohort scenarios affect whether
    # the source resolves to the official 104 scenarios, even though those
    # scenarios are never reconstructed by the frozen cohort replay.
    resolution_inventory = {}
    for meta in sorted(data_files):
        base, scenario = meta.parent.parent, meta.stem
        inputs = list(base.glob(f'*/calib/{scenario}/*.pkl')) + list(base.glob(f'*/label/{scenario}/*.txt'))
        resolution_inventory[str(meta.relative_to(val_root))] = sorted(str(p.relative_to(val_root)) for p in inputs)
    for sid in sorted({r['scenario_id'] for r in ledger.values()}):
        scenario_type, scenario = sid.split('/')
        base = val_root/scenario_type
        require((base/'meta'/f'{scenario}.txt').is_file(), f'Missing cohort scenario source: {sid}')
        calibrations = set(base.glob(f'*/calib/{scenario}/*.pkl'))
        labels = set(base.glob(f'*/label/{scenario}/*.txt'))
        require(bool(calibrations) and bool(labels), f'Missing cohort sensor3d inputs: {sid}')
        data_files.update(calibrations | labels)
    dataset_inventory = {str(p.relative_to(val_root)):file_sha256(p) for p in sorted(data_files)}
    maps_root = Path(args.carla_maps).resolve(strict=True)
    maps = {}
    for town in sorted({r['scenario_id'].rsplit('/', 1)[-1].split('_')[0] for r in ledger.values()}):
        path = find_xodr(town, [str(maps_root)])
        require(bool(path), f'Missing map: {town}')
        maps[str(Path(path).resolve())] = file_sha256(path)
    identity = lambda p: dict(path=str(p), device=p.stat().st_dev, inode=p.stat().st_ino)
    return dict(v2_checkpoint=v2_checkpoint, frozen_crn_thresholds=saved['threshold_by_horizon'],
                artifacts=artifact_hashes, code=code,
                dataset_source=identity(source_root), val_source=identity(val_root),
                dataset_inventory_sha256=digest(canonical(dataset_inventory)),
                scenario_resolution_inventory_sha256=digest(canonical(resolution_inventory)),
                dataset_input_files=len(dataset_inventory), map_source=identity(maps_root), maps=maps,
                runtime=dict(python=sys.version, torch=version('torch'), numpy=version('numpy')),
                replay_settings=dict(device=args.device, batch_size=args.batch_size))


def validate_replay_cases(cases, ledger, provenance, sources):
    """Recheck compact evidence against both immutable ledger and collector GT."""
    from examples import train_collision_risk_net as collector
    require(len(cases) == 219 and len(ledger) == 219, 'Replay cache requires exactly 219 unique windows')
    assert_networks(cases, ledger)
    collector.verify_manifest(RUN/'val104.jsonl', 'val', smoke=False)
    collector_rows = {r['window_id']:r for r in collector.read_rows(RUN/'val104.jsonl', 'val')}
    require(set(cases) <= set(collector_rows), 'Cache window missing from frozen collector')
    expected_artifacts = set(sources['artifacts']) - {str((RUN/'val104.manifest.json').resolve())}
    hashes = {str(Path(p).resolve()):h for p,h in provenance['network_checkpoint_SHA256'].items()}
    require(set(hashes) == expected_artifacts and all(hashes[p] == sources['artifacts'][p] for p in hashes),
            'Replay provenance artifact mismatch')
    thresholds = provenance['frozen_thresholds']
    require(thresholds['crn4'] == sources['frozen_crn_thresholds'], 'Frozen CRN threshold metadata mismatch')
    for k in (1,2):
        model_path = ROOT/f'out/pair_reranker_v4/model_bucket{k}/pair_reranker_v4.json'
        require(thresholds['v4'][str(k)] == json.loads(model_path.read_text())['decision_threshold'],
                'Frozen V4 threshold mismatch')
    for wid,c in cases.items():
        require(set(c) == REPLAY_CASE_KEYS, f'Unexpected cached case fields: {wid}')
        require(type(c['gt_bucket']) is int and c['gt_bucket'] == collector_rows[wid]['gt_bucket'],
                f'Collector GT mismatch: {wid}')
        actors = c['observation']['actors']
        require(isinstance(actors,list), 'Invalid cached observation actor schema')
        actor_keys = set(('actor_id kind cls world_xy heading_deg speed_mps accel_mps2 maneuver observed_by '
                         'confidence position_quality track_age_s heading_source observed_range_m source_track_ids '
                         'placement footprint_length_width_m observed_history').split())
        require(all(isinstance(a,dict) and set(a) == actor_keys for a in actors),
                'Invalid cached observation actor schema')
        window = c['observation']['observation_window']
        require(set(window) == {'t_start_s','t_end_s','history_s','time_reference'} and window['history_s'] == 5,
                'Invalid cached observation window schema')
        require(set(c['observation']['map_conventions']) ==
                {'coordinates','heading','drive_side','lane_numbering','lane_width_m'}, 'Invalid cached map schema')
        ids = [a['actor_id'] for a in actors]
        require(all(isinstance(a,str) for a in ids) and len(ids) == len(set(ids)), 'Invalid cached actor IDs')
        groups = c['gt_groups']
        require(isinstance(groups,list) and all(isinstance(g,list) and all(isinstance(a,str) for a in g)
                                                and len(g) == len(set(g)) for g in groups), 'Invalid cached GT groups')
        # Keep the original grading representation, including empty groups
        # for unobserved vehicles and aliases shared by identity groups.
        require(c['gt_bucket'] in (1,2) or not groups, 'Invalid cached GT pair groups')
        match = lambda pair: any((pair[0] in g and pair[1] in h) or (pair[1] in g and pair[0] in h)
                                 for i,g in enumerate(groups) for h in groups[i+1:])
        row = collector_rows[wid]
        collector_ids = [a['actor_id'] for a in row['actors']]
        pair_labels = {frozenset(pair):label for pair,label in zip(itertools.combinations(collector_ids,2), row['pair_labels'])}
        for pair,label in zip(itertools.combinations(collector_ids,2), row['pair_labels']):
            if c['gt_bucket'] in (1,2):
                require(match(pair) == (label == c['gt_bucket']), f'Collector GT-pair membership mismatch: {wid}')
        for condition in ('v4','crn4'):
            network = c[condition]
            require(set(network) == {'decision','pairs','evidence'} and type(network['decision']) is bool,
                    'Invalid cached network fields')
            require(isinstance(network['pairs'],list) and all(isinstance(p,list) and len(p) == 2 and
                    p[0] != p[1] and set(p) <= set(ids) for p in network['pairs']), 'Invalid cached selected pair')
            audit_payload(dict(observation=c['observation'], network_evidence=network['evidence'], horizon_s=2),
                          condition, network)
            hit = c['gt_bucket'] in (1,2) and network['decision'] and any(match(p) for p in network['pairs'])
            require(hit == (ledger[wid][condition+'_correct_pair_hit'] == 'True'), f'Frozen GT-pair hit mismatch: {wid}')
        require(c['v4']['pairs'] == json.loads(ledger[wid]['v4_selected_pairs']), f'Frozen V4 pair mismatch: {wid}')
        selected = list(ast.literal_eval(ledger[wid]['crn4_selected_pair']))
        require(c['crn4']['pairs'] == [selected], f'Frozen CRN4 pair mismatch: {wid}')
        require(pair_labels.get(frozenset(selected)) == int(ledger[wid]['crn4_selected_pair_gt_label']),
                f'Frozen selected-pair GT membership mismatch: {wid}')
        evidence = c['crn4']['evidence']['selected_interaction']
        require([evidence['actor_a'],evidence['actor_b']] == selected, 'CRN evidence selected-pair mismatch')
        require(c['crn4']['evidence']['source_type'] == 'observation_pair_risk', 'Invalid CRN evidence source')
        for key in ('hazard_1','hazard_2','event_probability_1','event_probability_2','cumulative_risk_1','cumulative_risk_2'):
            require(type(evidence[key]) in (int,float) and 0 <= evidence[key] <= 1, 'Invalid CRN probability')
        h1,h2 = evidence['hazard_1'],evidence['hazard_2']
        require(all(math.isclose(evidence[key],value,abs_tol=1e-12) for key,value in (
            ('event_probability_1',h1), ('event_probability_2',(1-h1)*h2),
            ('cumulative_risk_1',h1), ('cumulative_risk_2',1-(1-h1)*(1-h2)))), 'Inconsistent CRN evidence probabilities')
        require(math.isclose(evidence['cumulative_risk_2'], float(ledger[wid]['crn4_risk_2s']), abs_tol=2e-6),
                'Frozen CRN risk mismatch')
        threshold = thresholds['crn4']['2']
        require(threshold == float(ledger[wid]['crn4_threshold_2s']) and
                c['crn4']['decision'] == (evidence['cumulative_risk_2'] >= threshold), 'Frozen CRN threshold/decision mismatch')
        require(c['v4']['evidence']['source_type'] == 'trajectory_pair_verifier', 'Invalid V4 evidence source')
        require(set(c['v4']['evidence']) == {'source_type','accepted_interactions'}, 'Invalid V4 evidence schema')
        accepted = c['v4']['evidence']['accepted_interactions']
        require(c['v4']['decision'] == bool(accepted), 'Cached V4 acceptance mismatch')
        for k in (1,2):
            require(any(p['k'] == k for p in accepted) == (ledger[wid][f'v4_bucket{k}_prediction'] == 'True'),
                    f'Frozen V4 bucket decision mismatch: {wid}')
        for p in accepted:
            require(set(p) == {'actor_a','actor_b','k','verifier_score','future_coordinates',
                'predicted_minimum_footprint_clearance_m','time_of_predicted_minimum_clearance_s',
                'predicted_contact','first_predicted_contact_s','current_clearance_m','contact_duration_s',
                'contact_event','contact_at_observation','contact_before_observation'}, 'Invalid V4 interaction schema')
            require(type(p['k']) is int and type(p['verifier_score']) in (int,float) and
                    math.isfinite(p['verifier_score']) and p['verifier_score'] >= thresholds['v4'][str(p['k'])],
                    'Invalid accepted V4 score')
            for paths_for_actor in p['future_coordinates'].values():
                require(isinstance(paths_for_actor,list) and all(isinstance(path,dict) and
                    set(path) == {'coordinates_enu_m'} and isinstance(path['coordinates_enu_m'],list) and
                    all(isinstance(point,list) and len(point) == 3 and all(type(v) in (int,float)
                        and math.isfinite(v) for v in point) and 0 <= point[0] <= 2
                        for point in path['coordinates_enu_m']) for path in paths_for_actor),
                    'Invalid V4 future-coordinate schema')


def save_replay_cache(path, document):
    """Atomically publish a new cache; never replace an existing cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as f:
            temp_path = Path(f.name)
            f.write(canonical(document)+'\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink()


def cached_replay(args, ledger):
    path = args.replay_cache
    if path.exists() or path.is_symlink():
        cache_bytes = path.read_bytes()
        cache_hash = hashlib.sha256(cache_bytes).hexdigest()
        document = json.loads(cache_bytes)
        require(set(document) == {'version','cases','provenance','sources','content_sha256'}, 'Invalid replay cache envelope')
        require(type(document['version']) is int and document['version'] == REPLAY_CACHE_VERSION,
                'Incompatible replay cache version')
        content = {k:v for k,v in document.items() if k != 'content_sha256'}
        require(digest(canonical(content)) == document['content_sha256'], 'Replay cache content integrity mismatch')
        sources = replay_source_fingerprints(args, ledger, document['sources']['v2_checkpoint'])
        require(sources == document['sources'], 'Stale replay cache source-artifact fingerprints')
        rows = document['cases']
        require(isinstance(rows,list) and len(rows) == 219 and
                len({r['window_id'] for r in rows}) == 219, 'Replay cache requires exactly 219 unique windows')
        require(all(set(r) == REPLAY_CASE_KEYS | {'window_id'} for r in rows), 'Unexpected cached window schema')
        cases = {r['window_id']:{k:v for k,v in r.items() if k != 'window_id'} for r in rows}
        provenance = document['provenance']
        validate_replay_cases(cases, ledger, provenance, sources)
        origin = 'validated_cache'
    else:
        sources = replay_source_fingerprints(args, ledger)
        cases, provenance = reproduce(args, ledger)
        cases = {wid:json.loads(canonical({k:c[k] for k in REPLAY_CASE_KEYS})) for wid,c in cases.items()}
        require(replay_source_fingerprints(args, ledger, sources['v2_checkpoint']) == sources,
                'Replay sources changed during cache creation')
        validate_replay_cases(cases, ledger, provenance, sources)
        content = dict(version=REPLAY_CACHE_VERSION,
                       cases=[dict(window_id=wid, **c) for wid,c in cases.items()],
                       provenance=provenance, sources=sources)
        document = dict(content, content_sha256=digest(canonical(content)))
        save_replay_cache(path, document)
        cache_hash = hashlib.sha256((canonical(document)+'\n').encode('utf-8')).hexdigest()
        origin = 'fresh_replay'
    require(file_sha256(path) == cache_hash, 'Replay cache changed during validation')
    return cases, provenance, dict(evidence_source=origin, replay_cache_path=str(path.resolve()),
                                  replay_cache_sha256=cache_hash,
                                  replay_cache_content_sha256=document['content_sha256'],
                                  replay_cache_provenance=document['sources'])


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
            prompt = CRN4_PROMPT_V2 if condition == 'crn4' else PROMPT
            body = providers.build_request(args.provider, system=prompt, blocks=[canonical(payload), QUESTION],
                schema=schema(), model=args.model, max_tokens=args.max_output_tokens, effort='', extra=extra)
            requests.append(dict(request_id=f'{wid}__{condition}', window_id=wid, condition=condition,
                                 body=body, observation_hash=digest(common)))
            if condition == 'crn4':
                selected = c[condition]['evidence']['selected_interaction']
                requests[-1]['selected_pair'] = [selected['actor_a'], selected['actor_b']]
    random.Random(args.seed).shuffle(requests)
    require(len(requests) == 219 * len(conditions), 'Wrong request count')
    if args.condition == 'both':
        by_window = {}
        for r in requests:
            by_window.setdefault(r['window_id'], []).append(r['observation_hash'])
        require(all(len(h)==2 and h[0]==h[1] for h in by_window.values()), 'Paired observation mismatch')
    return requests


def validate_response(response, actor_ids, selected_pair=None):
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
        if selected_pair is not None and p['accident_expected']:
            require(len(ids) == 2 and set(ids) == set(selected_pair), 'CRN4 prediction must use exactly the selected pair')
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
        unscored=len(cases)-sum(counts.values()), fraction_denominator='scored network strata only')
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
        return validate_response(saved['parsed_response'], ids, request.get('selected_pair'))
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
            parsed = validate_response(json.loads(''.join(providers.response_texts(args.provider, api))), ids,
                                       request.get('selected_pair'))
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
    transition_results = {c:transitions(cases,results,c) for c in conditions}
    prefix = ''
    if getattr(args, 'window_ids_file', None) is not None:
        prefix = 'diagnostic_subset_'
        scope = dict(evaluation_scope='diagnostic_subset', full_cohort_metrics=False,
                     window_count=len(cases), selected_window_ids=list(cases))
        scores.update(scope)
        transition_results.update(scope)
    write_json(args.out/(prefix+'scores.json'), scores)
    write_json(args.out/(prefix+'network_to_llm_transitions.json'), transition_results)
    with (args.out/(prefix+'paired_v4_crn_results.csv')).open('w',newline='') as f:
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
    parser.add_argument('--max-output-tokens', type=int, default=32768)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--thinking-config', help='Explicit Gemini thinkingConfig JSON; omitted means provider default')
    parser.add_argument('--seed', type=int, default=20261007, help='Request shuffle seed; not an API sampling seed')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--window-ids-file', type=Path,
                        help='Diagnostic subset: one frozen window ID per nonblank line; full replay still required')
    parser.add_argument('--replay-cache', type=Path,
                        help='Reuse source-validated frozen 219-window evidence; create atomically if absent')
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
    if not args.dry_run:
        require(bool(os.environ.get(args.api_key_env)), f'Missing {args.api_key_env}')
    require(args.workers>0 and args.batch_size>0 and args.max_output_tokens>0 and args.retries>=0,
            'Invalid counts')
    require(args.temperature>=0 and args.backoff>=0 and args.timeout>0, 'Invalid generation/retry setting')
    if args.replay_cache is not None:
        output_root, cache_path = args.out.resolve(), args.replay_cache.resolve()
        reserved = {'experiment_manifest.json','dry_run_audit.json','requests.jsonl','records.jsonl',
                    'api_attempts.jsonl','failures.jsonl','scores.json','network_to_llm_transitions.json',
                    'paired_v4_crn_results.csv','diagnostic_subset_scores.json',
                    'diagnostic_subset_network_to_llm_transitions.json','diagnostic_subset_paired_v4_crn_results.csv'}
        reserved |= {name+'.tmp' for name in reserved}
        require(cache_path not in {output_root/name for name in reserved} and cache_path != output_root and
                not any(cache_path.is_relative_to(output_root/name) for name in ('raw_responses','parsed_responses')),
                'Replay cache path conflicts with experiment outputs')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest_path = args.out/'experiment_manifest.json'
    require(args.resume or not manifest_path.exists(), 'Output exists; use --resume or a new --out')
    try:
        ledger = load_cohort()
        cache_info = None
        if args.replay_cache is None:
            cases, provenance = reproduce(args, ledger)
        else:
            cases, provenance, cache_info = cached_replay(args, ledger)
        requests = prepare(cases, args)
        selected_window_ids = None
        if args.window_ids_file is not None:
            selected_window_ids = load_window_ids(args.window_ids_file, ledger)
            require(set(selected_window_ids) <= set(cases), 'Subset window absent from reproduced cohort')
            selected = set(selected_window_ids)
            requests = [r for r in requests if r['window_id'] in selected]
        audit = dict(passed=True, windows=len(cases), requests=len(requests), confusion=EXPECTED,
            invariants=['strict cohort join','independent network reproduction','paired byte-identical observations',
                'no GT/model metadata leakage','CRN risk only','V4 accepted geometry only',
                'actor membership','exact two response buckets validated'], api_calls=0)
        if cache_info is not None:
            audit.update(cache_info)
        if selected_window_ids is not None:
            audit.update(evaluation_scope='diagnostic_subset', full_frozen_cohort_validated=True,
                         selected_window_ids=selected_window_ids, selected_windows=len(selected_window_ids))
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
    if cache_info is not None:
        manifest.update(cache_info)
        manifest['prompt_hashes'] = {condition:digest(CRN4_PROMPT_V2 if condition == 'crn4' else PROMPT)
                                     for condition in sorted({r['condition'] for r in requests})}
    if selected_window_ids is not None:
        manifest.update(evaluation_scope='diagnostic_subset', full_cohort_metrics=False,
                        window_ids_file=str(args.window_ids_file.resolve()),
                        selected_window_ids=selected_window_ids, full_frozen_cohort_windows=len(cases))
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if cache_info is not None:
            # A cache created by this experiment is loaded on resume. Preserve
            # its creation-time origin in the manifest; the audit records the
            # current invocation's fresh/cache source.
            manifest['evidence_source'] = previous.get('evidence_source')
        require(previous == manifest, 'Resume manifest/settings changed')
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
    output_cases = cases if selected_window_ids is None else {wid:cases[wid] for wid in selected_window_ids}
    outputs(args,output_cases,records)
    print(f'Wrote {len(records)} records to {args.out}', flush=True)
    if not args.dry_run:
        failures = sum(r['grading']['status'] != 'SCORED' for r in records)
        if failures:
            raise SystemExit(f'{failures} API requests FAILED/UNSCORED; use --resume to retry')


if __name__ == '__main__':
    main()
