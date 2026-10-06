#!/usr/bin/env python3
"""Descriptive-only audit of the frozen V4 0–2 s verifier.

Inference calls the frozen evaluator's predictor, ranking, geometry, pools,
features, models and OR decision. No training, threshold search or relabeling.
The frozen 219-window join and its per-window evaluator ledger are mandatory.
All decisions and counts must match before diagnostics are written.

CSV feature_<name> and NPZ raw_features preserve FEATURE_NAMES_V4 verbatim,
including frozen clipping and unavailable-value sentinels. Unprefixed swept
measurements preserve the unrounded SweptPair evidence. Observed-state aliases
use the frozen feature values (including their clipping). Closing speed is
positive for approach. ETA slots are sorted, NOT assigned to actor_a/actor_b.
Oracle geometry uses existing GT replay, exact raw yaw/size and 0.1-s samples
on (0,2], (2,5]; minima with incomplete coverage describe available samples.
A geometry gap cannot establish no contact. No raw yaw/size interpolation or
GT path extrapolation is allowed. Contact flags denote sampled footprint
contact; first-contact time uses the audit's new/boundary/recontact semantics.
V2 predicted contact and oracle contact are distinct. Oracle/GT labels are
attached only after feature extraction and score computation.

Effects are descriptive, with dependent windows/pairs within scenarios.
Feature highlights rank relative effect sizes; they impose no cutoff and do
not constitute feature selection or evidence of a causal mechanism.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from examples import evaluate_pair_reranker_v4_0_2 as frozen
from examples import audit_gt_xy_source_ablation as gt_replay
from examples import plot_v4_crn4_error_cases as geometry_audit
from examples.diagnose_crn4_false_positives import selected_geometry as _shared_geometry
from traffic_llm.pair_reranker_v2 import (FEATURE_NAMES_V4,
    RISK_FEATURE_NAMES_V3, ROUTE_ETA_FEATURE_NAMES_V4, ACTOR_STAT_NAMES)
from traffic_llm.collision_risk_net import ground_truth_bucket

ANALYSIS = ROOT/'out/collision_risk_hazard/run_20261006_window_balanced/final_analysis'
COHORT = ANALYSIS/'v4_vs_crn4_2s_all_windows.csv'
FROZEN_ROWS = ROOT/'out/pair_reranker_v4/eval_0_2_val104_with_rows_windows.jsonl'
EXPECTED = dict(TP=46, FP=55, TN=111, FN=7)
EXPECTED_THRESHOLDS = {1:0.08578953170775334,2:5.17511995858515e-05}
OUTPUTS = ('v4_all_windows_diagnostics.csv','v4_negative_window_diagnostics.csv',
    'v4_fp_diagnostics.csv','v4_candidate_diagnostics.csv',
    'v4_accepted_fp_pair_diagnostics.csv','v4_candidate_features.npz',
    'v4_fp_vs_tn_summary.json','v4_fp_vs_tp_pair_summary.json','v4_fp_mechanism_summary.json')
BINARY = set(('predicted_contact same_road same_lane both_have_road any_ego both_ego '
    'both_speed_known both_accel_known both_range_known contact_within_1s both_braking '
    'either_braking both_stopped mutually_observed known_nonmutual_observation '
    'same_next_junction route_eta_available has_crossing_interaction '
    'crossing_conflict_orthogonal crossing_conflict_oncoming_turn').split())
BINARY.update(n for n in FEATURE_NAMES_V4 if n.startswith('pair_type_'))
ALIASES = dict(center_distance_m='center_distance_now_m',relative_speed_mps='relative_speed_mps',
    closing_speed_mps='closing_speed_mps',cv_time_to_closest_s='cv_time_to_closest_s',
    heading_cos='heading_cosine',same_road='same_road',same_lane='same_lane')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def classify(actual, predicted):
    return ('TP' if predicted else 'FN') if actual else ('FP' if predicted else 'TN')


def load_cohort():
    rows = geometry_audit.read_csv(COHORT)
    require(len(rows)==219 and len({r['window_id'] for r in rows})==219,'Expected exact 219-window frozen join')
    ledger = {r['window_id']:r for r in rows}
    with FROZEN_ROWS.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    require(len(records)==219 and len({r['window_id'] for r in records})==219,'Invalid frozen evaluator ledger')
    require({r['window_id'] for r in records}==set(ledger),'Frozen ledgers have different cohorts')
    by_id = {r['window_id']:r for r in records}
    for wid,r in ledger.items():
        old = by_id[wid]
        require(old['scenario_id']==r['scenario_id'] and
                old['predicted']==(r['v4_predicted']=='True') and
                old['actual']==(r['truth_positive']=='True'),f'Frozen ledger disagreement: {wid}')
    validate_records(records,ledger,by_id)
    return ledger,by_id


def validate_records(records, ledger, reference):
    require(len(records)==219 and len({r['window_id'] for r in records})==219 and
            {r['window_id'] for r in records}==set(ledger),'Replay cohort differs from frozen 219 windows')
    counts = Counter(classify(r['actual'],r['predicted']) for r in records)
    require(dict(counts)==EXPECTED,f'Frozen confusion mismatch: {dict(counts)} != {EXPECTED}')
    for row in records:
        wid = row['window_id']
        saved,old = ledger[wid],reference[wid]
        require(row['predicted']==(saved['v4_predicted']=='True') and
                row['actual']==(saved['truth_positive']=='True'),f'Frozen decision mismatch: {wid}')
        # Includes exact selected pair order, per-bucket firing, identity coverage.
        for key in ('actual','positive_bucket','bucket1_prediction','bucket2_prediction',
                    'baseline_predicted','gt_pair_covered','correct_pair_hit','selected_pairs',
                    'bucket1_selected_pairs','bucket2_selected_pairs'):
            require(row[key]==old[key],f'Frozen evaluator field mismatch {key}: {wid}')
    return dict(counts)


def build_candidates(window, wcfg):
    """Observation/predictor only: no GT or oracle argument is accepted."""
    actors,_ = frozen.rank_actors(window.last,wcfg.actor_cap(len(window.last.actors)))
    by_id = {a.actor_id:a for a in actors}
    pairs = frozen.swept_pair_clearances(actors,horizon_s=wcfg.horizon_s,
        sample_dt_s=wcfg.swept_sample_dt_s,contact_margin_m=wcfg.swept_contact_margin_m,
        exclude_touching_now=wcfg.swept_exclude_touching_now,
        exclude_static_pairs=wcfg.swept_exclude_static_pairs)
    candidates,evidence = {1:[],2:[]},{1:[],2:[]}
    for bucket,pool in frozen.bucket_pools(pairs).items():
        for pair in pool:
            features = frozen.pair_features_v4(by_id[pair.actor_a],by_id[pair.actor_b],
                                               pair,wcfg.horizon_s,window.last.interactions)
            require(len(features)==len(FEATURE_NAMES_V4),'Unexpected V4 feature width')
            candidates[bucket].append(dict(actor_a=pair.actor_a,actor_b=pair.actor_b,
                predicted_contact=bool(pair.predicted_contact),features=features))
            evidence[bucket].append(pair)
    return candidates,evidence


def score_features(model, raw):
    """Export the exact Python-double normalization used by the frozen MLP."""
    require(len(raw)==len(model.means)==len(model.scales)==len(FEATURE_NAMES_V4),'Feature schema mismatch')
    require(all(math.isfinite(s) and s>0 for s in model.scales),'Invalid frozen scales')
    normalized = [(x-m)/s for x,m,s in zip(raw,model.means,model.scales)]
    np.testing.assert_array_equal(np.asarray(normalized),
        (np.asarray(raw,dtype=np.float64)-np.asarray(model.means))/np.asarray(model.scales))
    score = model.predict_features(raw)
    # Same arithmetic order as predict_features: confirms the captured MLP input.
    hidden = [max(0.,b+sum(w*x for w,x in zip(weights,normalized)))
              for weights,b in zip(model.hidden_weights,model.hidden_bias)]
    logit = model.output_bias+sum(w*x for w,x in zip(model.output_weights,hidden))
    z = math.exp(-logit) if logit>=0 else math.exp(logit)
    reconstructed = 1/(1+z) if logit>=0 else z/(1+z)
    require(score==reconstructed,'Normalized export disagrees with frozen MLP')
    return score,normalized


def truth_groups(gt,bucket):
    return [set(v.get('actor_ids') or []) for row in gt['expected']
            if row['k']==bucket and row['accident_expected'] for v in row.get('involved_vehicles') or []]


def diagnostic_window(window, scenario_id, gt, candidates, evidence, models):
    thresholds = {k:models[k].threshold for k in (1,2)}
    # Scores and features exist before any oracle/later labels are attached.
    scored = {k:[score_features(models[k],c['features']) for c in candidates[k]] for k in (1,2)}
    decision = frozen.evaluate_window(gt,candidates,models)
    require(decision is not None,'Unidentifiable window entered diagnostics')
    wid = f'{scenario_id}:{window.index}:{window.t_end:.3f}'
    decision.update(window_id=wid,scenario_id=scenario_id,t_end_s=window.t_end)
    bucket = ground_truth_bucket(gt)
    s = classify(decision['actual'],decision['predicted'])
    expected = {r['k']:r for r in gt['expected']}
    later_scorable = all(k in expected and bool(expected[k].get('scorable')) for k in (3,4,5))
    require(decision['actual']==(0<bucket<=2),'GT bucket disagrees with frozen evaluator')
    early = truth_groups(gt,bucket) if 0<bucket<=2 else []
    later = truth_groups(gt,bucket) if bucket in (3,4,5) else []
    later_pairs = sorted({tuple(sorted((a,b))) for i,x in enumerate(later) for y in later[i+1:]
                          for a,b in itertools.product(x,y) if a!=b})
    row = dict(window_id=wid,scenario_id=scenario_id,gt_bucket=bucket,status=s,
        bucket1_prediction=decision['bucket1_prediction'],bucket2_prediction=decision['bucket2_prediction'],
        baseline_predicted_contact=decision['baseline_predicted'],gt_pair_candidate_covered=decision['gt_pair_covered'],
        correct_gt_pair_hit=decision['correct_pair_hit'],later_collision=bucket in (3,4,5),
        later_2_5_fully_scorable=later_scorable,
        no_collision_through_5s_observed=bucket==0 and later_scorable,
        later_gt_bucket=bucket if bucket in (3,4,5) else None,
        later_gt_pair=json.dumps(later_pairs) if later_pairs else None)
    rows,raw_vectors,normalized_vectors = [],[],[]
    for k in (1,2):
        scores = [score for score,_ in scored[k]]
        order = sorted(range(len(scores)),key=lambda p:-scores[p])
        top1 = scores[order[0]] if order else None
        top2 = scores[order[1]] if len(order)>1 else None
        n_above = sum(score>=thresholds[k] for score in scores)
        row.update({f'n_bucket{k}_candidates':len(scores),f'n_bucket{k}_accepted':n_above,
            f'bucket{k}_threshold':thresholds[k],f'bucket{k}_top1_score':top1,f'bucket{k}_top2_score':top2,
            f'bucket{k}_top1_minus_top2':top1-top2 if top2 is not None else None,
            f'bucket{k}_top1_margin_to_threshold':top1-thresholds[k] if top1 is not None else None,
            f'bucket{k}_top2_margin_to_threshold':top2-thresholds[k] if top2 is not None else None,
            f'bucket{k}_n_candidates_above_threshold':n_above})
        accepted = []
        for index,(c,pair,(score,normalized)) in enumerate(zip(candidates[k],evidence[k],scored[k])):
            raw = c['features']
            values = dict(zip(FEATURE_NAMES_V4,raw))
            accepted_flag = score>=thresholds[k]
            r = dict(window_id=wid,scenario_id=scenario_id,status=s,bucket=k,
                actor_a=c['actor_a'],actor_b=c['actor_b'],candidate_rank_by_clearance=index+1,
                candidate_rank_by_score=order.index(index)+1,verifier_score=score,frozen_threshold=thresholds[k],
                margin_to_threshold=score-thresholds[k],accepted=accepted_flag,
                predicted_contact=c['predicted_contact'],predicted_min_clearance_m=pair.minimum_clearance_m,
                predicted_min_clearance_time_s=pair.time_after_observation_s,
                clearance_at_observation_m=pair.clearance_at_observation_m,
                clearance_at_1s_m=pair.clearance_at_1s_m,clearance_at_horizon_m=pair.clearance_at_horizon_m,
                first_predicted_contact_s=pair.first_contact_s,contact_duration_s=pair.contact_duration_s,
                contact_event=pair.contact_event,joint_path_probability=pair.joint_path_probability,
                **{alias:values[name] for alias,name in ALIASES.items()},
                **{name:values[name] for name in (*RISK_FEATURE_NAMES_V3,*ROUTE_ETA_FEATURE_NAMES_V4)},
                gt_pair_label_0_2=bool(early and frozen.matches_gt_pair(c,early)),
                matches_later_gt_pair=bool(frozen.matches_gt_pair(c,later)) if later else None)
            r.update({'feature_'+name:value for name,value in values.items()})
            rows.append(r); raw_vectors.append(list(raw)); normalized_vectors.append(normalized)
            if accepted_flag:
                accepted.append([c['actor_a'],c['actor_b']])
        require(accepted==decision[f'bucket{k}_selected_pairs'],'Diagnostic accepted pairs differ from frozen evaluator')
        require(models[k].threshold==thresholds[k],'Frozen threshold changed')
    return decision,row,rows,raw_vectors,normalized_vectors


def replay_scenario(scenario,args,models,runner=None,cfg=None):
    # Only the serial main path passes a shared runner/predictor. Concurrent
    # scenarios retain their own model and runner, including on CUDA.
    if runner is None:
        cfg = frozen.PipelineConfig(); cfg.deepaccident.observation_mode='sensor3d'
        cfg.predictor = frozen.JointSceneTorchPredictor(args.predictor,device=args.device)
        runner = frozen.DeepAccidentRunner(args.root,cfg)
    result = runner.build(scenario.scenario,scenario.scenario_type,
                         opendrive_path=frozen.find_xodr(scenario.town,[args.carla_maps]))
    snapshots = list(result.snapshots(rate_hz=2.0))
    collision = frozen.estimate_collision(scenario,cfg.deepaccident)
    wcfg = frozen.WindowConfig(window_s=5.,stride_s=1.,horizon_s=5.,snapshot_rate_hz=2.,warmup=False,full_window=False)
    windows,_ = frozen.build_windows(snapshots,wcfg,collision_time_s=collision.time_s if collision and collision.occurred else None)
    ids = {a:scenario.meta.agent_id_of(a) for a in scenario.agents}
    records,rows,candidates,raw,norm = [],[],[],[],[]
    for window in windows:
        gt = frozen.window_ground_truth(window,wcfg,collision=collision,agent_carla_ids=ids,
            scenario_id=scenario.scenario_id,scenario_split=scenario.scenario_type,
            dataset_split=scenario.split,town=scenario.town,data_end_s=snapshots[-1].t)
        if frozen.evaluation_truth(gt) is None:
            continue
        pools,evidence = build_candidates(window,wcfg)
        rec,row,cs,rs,ns = diagnostic_window(window,scenario.scenario_id,gt,pools,evidence,models)
        records.append(rec); rows.append(row); candidates.extend(cs); raw.extend(rs); norm.extend(ns)
    print(f'Inference {scenario.scenario_id}: {len(rows)} identifiable windows',flush=True)
    return records,rows,candidates,raw,norm


def selected_geometry(actors,pair,identities,frames,cutoff,horizon):
    """Require actual timestamped GT support before invoking the audit helper.

    Production motion helpers may extend a stopped one-point predictor path.
    A t0-only GT path has no oracle future support and must stay unknown.
    """
    supported = {}
    for aid in pair:
        if aid not in actors:
            continue
        a = copy.deepcopy(actors[aid])
        a.predictions = [p for p in a.predictions if p.waypoint_times_s is not None
            and len(p.waypoint_times_s)==len(p.waypoints) and len(p.waypoints)>=2
            and p.waypoint_times_s[0]==0 and p.waypoint_times_s[-1]>0
            and all(y>x for x,y in zip(p.waypoint_times_s,p.waypoint_times_s[1:]))]
        supported[aid] = a
    return _shared_geometry(supported,pair,identities,frames,cutoff,horizon)


def oracle_geometry(scenario,accepted,args):
    """A separate GT replay after all frozen decisions have been validated."""
    windows,ids,_,_ = gt_replay.replay_windows_profiled(scenario.root,
        dict(town=scenario.town,scenario=scenario.scenario,scenario_type=scenario.scenario_type),
        dict(mode='sensor3d',window_s=5.,stride_s=1.,horizon_s=5.,history_stride_s=1.),args.carla_maps)
    frames = gt_replay._FrameBoxCache(scenario)
    by_id = {f'{scenario.scenario_id}:{w.index}:{w.t_end:.3f}':w for w in windows.values()}
    cache = {}
    for row in accepted:
        require(row['window_id'] in by_id,f'GT replay missing window: {row["window_id"]}')
        w = by_id[row['window_id']]
        pair = (row['actor_a'],row['actor_b'])
        key = (row['window_id'],tuple(sorted(pair)))
        if key not in cache:
            identities = geometry_audit.geometry.identities(w,ids)
            cache[key] = selected_geometry({a.actor_id:a for a in w.last.actors},pair,identities,frames,w.t_end,5.)
        g = cache[key]
        row.update({(key if key.startswith('geometry_complete_') else 'gt_'+key):value for key,value in g.items()})
        # Incomplete earlier geometry cannot prove an exact first contact time.
        t = row['gt_first_contact_time_s']
        if t is not None and (not row['geometry_complete_0_2'] or (t>2 and not row['geometry_complete_2_5'])):
            row['gt_first_contact_time_s'] = None
        row['accepted_pair_matches_later_gt_pair'] = row['matches_later_gt_pair']
    print(f'Oracle {scenario.scenario_id}: {len(accepted)} accepted FP candidates',flush=True)


def describe(values):
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return dict(n=0,missing=len(values),median=None,q1=None,q3=None,iqr=None,min=None,max=None)
    q1,median,q3 = np.quantile(valid,[.25,.5,.75]).tolist()
    return dict(n=len(valid),missing=len(values)-len(valid),median=median,q1=q1,q3=q3,iqr=q3-q1,min=min(valid),max=max(valid))


def compare(a,b,binary=False):
    x,y = ([float(v) for v in values if v is not None] for values in (a,b))
    if binary:
        require(all(v in (0,1) for v in x+y),'Nonbinary feature in binary summary')
        px,py = (sum(v)/len(v) if v else None for v in (x,y))
        diff = px-py if px is not None and py is not None else None
        return dict(kind='binary',a_n=len(x),b_n=len(y),a_missing=len(a)-len(x),b_missing=len(b)-len(y),
                    a_proportion=px,b_proportion=py,percentage_point_difference=100*diff if diff is not None else None,
                    effect=diff)
    # Sorted-search Cliff's delta avoids quadratic work for candidate summaries.
    yy = np.sort(y)
    delta = float(sum(int(np.searchsorted(yy,v,'left'))-(len(y)-int(np.searchsorted(yy,v,'right')))
                      for v in x)/(len(x)*len(y))) if x and y else None
    return dict(kind='continuous',a=describe(a),b=describe(b),cliffs_delta=delta,effect=delta)


def feature_comparison(a,b):
    return {name:compare([r['feature_'+name] for r in a],[r['feature_'+name] for r in b],name in BINARY)
            for name in FEATURE_NAMES_V4}


def concentration(windows,candidates):
    accepted = defaultdict(set)
    for r in candidates:
        if r['accepted']:
            accepted[r['window_id']].add(tuple(sorted((r['actor_a'],r['actor_b']))))
    counts = Counter(r['scenario_id'] for r in windows if r['status']=='FP')
    previous,links,repeated = {},[],[]
    for r in sorted(windows,key=lambda r:(r['scenario_id'],float(r['window_id'].rsplit(':',1)[1]),r['window_id'])):
        old = previous.get(r['scenario_id'])
        if old and old['status']==r['status']=='FP' and math.isclose(
                float(r['window_id'].rsplit(':',1)[1])-float(old['window_id'].rsplit(':',1)[1]),1.,abs_tol=1e-8):
            links.append([old['window_id'],r['window_id']])
            common = sorted(accepted[old['window_id']] & accepted[r['window_id']])
            if common:
                repeated.append(dict(windows=links[-1],pairs=common))
        previous[r['scenario_id']] = r
    return dict(unique_fp_scenarios=len(counts),fp_windows_per_scenario=dict(sorted(counts.items())),
        consecutive_fp_window_links=links,repeated_accepted_pairs_across_consecutive_fp_windows=repeated)


def summaries(windows,candidates,accepted):
    groups = {s:[r for r in windows if r['status']==s] for s in EXPECTED}
    tops = {(s,k):[r for r in candidates if r['status']==s and r['bucket']==k and r['candidate_rank_by_score']==1]
            for s in ('FP','TN') for k in (1,2)}
    window_stats = {}
    for k in (1,2):
        for field in (f'n_bucket{k}_candidates',f'bucket{k}_top1_score',f'bucket{k}_top2_score',
            f'bucket{k}_top1_minus_top2',f'bucket{k}_top1_margin_to_threshold',
            f'bucket{k}_top2_margin_to_threshold',f'bucket{k}_n_candidates_above_threshold'):
            window_stats[field] = compare([r[field] for r in groups['FP']],[r[field] for r in groups['TN']])
    tn_features = {str(k):feature_comparison(tops['FP',k],tops['TN',k]) for k in (1,2)}
    fp_pairs = [r for r in candidates if r['status']=='FP' and r['accepted']]
    tp_pairs = [r for r in candidates if r['status']=='TP' and r['accepted'] and r['gt_pair_label_0_2']]
    tp_features = feature_comparison(fp_pairs,tp_pairs)
    tp_by_bucket = {str(k):feature_comparison([r for r in fp_pairs if r['bucket']==k],
                        [r for r in tp_pairs if r['bucket']==k]) for k in (1,2)}
    highlights = []
    for name in FEATURE_NAMES_V4:
        effects = [tn_features[str(k)][name]['effect'] for k in (1,2)]
        effects = [abs(v) for v in effects if v is not None]
        v = tp_features[name]['effect']
        if effects and v is not None:
            tn_effect,tp_effect = max(effects),abs(v)
            highlights.append(dict(feature=name,fp_tn_max_bucket_absolute_effect=tn_effect,
                fp_correct_tp_absolute_effect=tp_effect,recall_overlap_rank_signal=tn_effect-tp_effect,
                failure_specific_rank_signal=min(tn_effect,tp_effect)))
    pair_summary = dict(populations={'a':'accepted FP window × bucket × pair rows','b':'accepted TP rows matching GT actor groups'},
        fp_pair_rows=len(fp_pairs),correct_tp_pair_rows=len(tp_pairs),features=tp_features,by_bucket=tp_by_bucket,
        recall_overlap_ranking=sorted(highlights,key=lambda r:(-r['recall_overlap_rank_signal'],r['feature'])),
        failure_specific_ranking=sorted(highlights,key=lambda r:(-r['failure_specific_rank_signal'],r['feature'])),
        interpretation='Relative rankings only: FP-vs-TN uses top candidates; FP-vs-TP uses accepted candidates. '
                       'Different conditioning can confound these comparisons; no cutoff or feature selection.')
    feature_groups = dict(trajectory_evidence=list(FEATURE_NAMES_V4[:12]),
        observed_pair_state=list(FEATURE_NAMES_V4[12:25]),
        actor_state_and_path=[f'{name}_{suffix}' for name in ACTOR_STAT_NAMES for suffix in ('min','max','absdiff')],
        pair_type=[n for n in FEATURE_NAMES_V4 if n.startswith('pair_type_')],
        v3_risk_gating=list(RISK_FEATURE_NAMES_V3),v4_route_eta=list(ROUTE_ETA_FEATURE_NAMES_V4))
    effects_by_group = {}
    for group,names in feature_groups.items():
        values = [abs(tn_features[str(k)][n]['effect']) for k in (1,2) for n in names
                  if tn_features[str(k)][n]['effect'] is not None]
        tp = [abs(tp_features[n]['effect']) for n in names if tp_features[n]['effect'] is not None]
        effects_by_group[group] = dict(fp_tn_mean_absolute_effect=float(np.mean(values)) if values else None,
            fp_correct_tp_mean_absolute_effect=float(np.mean(tp)) if tp else None)
    fp = groups['FP']
    by_window = defaultdict(list)
    for r in accepted:
        by_window[r['window_id']].append(r)
    fired = dict(bucket1_only=sum(r['bucket1_prediction'] and not r['bucket2_prediction'] for r in fp),
        bucket2_only=sum(r['bucket2_prediction'] and not r['bucket1_prediction'] for r in fp),
        both=sum(r['bucket1_prediction'] and r['bucket2_prediction'] for r in fp))
    structure = dict(exactly_one_accepted_pair=0,multiple_accepted_pairs=0,
                     exactly_one_candidate_above_threshold=0,multiple_candidates_above_threshold=0)
    geom = Counter(); future = Counter({
        'no_collision_through_5s_observed':0,'later_2_5_censored_or_unknown':0,
        'collision_after_2s':0,'later_collision_with_same_accepted_pair':0,
        'later_collision_with_different_pair':0})
    contact = Counter(); erroneous_v2_windows = 0
    for w in fp:
        rows = by_window[w['window_id']]
        require(rows,'FP window has no accepted pair')
        unique = {tuple(sorted((r['actor_a'],r['actor_b']))) for r in rows}
        structure['exactly_one_accepted_pair' if len(unique)==1 else 'multiple_accepted_pairs'] += 1
        structure['exactly_one_candidate_above_threshold' if len(rows)==1 else 'multiple_candidates_above_threshold'] += 1
        for flag,label in ((True,'with_predicted_contact'),(False,'without_predicted_contact')):
            if any(r['predicted_contact']==flag for r in rows):
                contact[label+'_windows'] += 1
            contact[label+'_pair_rows'] += sum(r['predicted_contact']==flag for r in rows)
        erroneous_v2_windows += any(r['predicted_contact'] and r['gt_contact_0_2'] is False for r in rows)
        early = any(r['gt_contact_0_2'] is True for r in rows)
        late = any(r['gt_contact_2_5'] is True for r in rows)
        complete = all(r['geometry_complete_0_2'] and r['geometry_complete_2_5'] for r in rows)
        geom['actual_gt_contact_0_2_windows'] += early
        geom['actual_gt_contact_after_2_windows'] += late
        geom['incomplete_geometry_windows'] += not complete
        geom['no_observed_contact_complete_geometry_windows'] += complete and not early and not late
        geom['accepted_pair_rows_incomplete_geometry'] += sum(not(r['geometry_complete_0_2'] and r['geometry_complete_2_5']) for r in rows)
        if not w['later_collision']:
            outcome = ('no_collision_through_5s_observed' if w['no_collision_through_5s_observed']
                       else 'later_2_5_censored_or_unknown')
            future[outcome] += 1
        else:
            future['collision_after_2s'] += 1
            matches = [r['matches_later_gt_pair'] for r in rows]
            if any(v is True for v in matches):
                future['later_collision_with_same_accepted_pair'] += 1
            elif all(v is False for v in matches):
                future['later_collision_with_different_pair'] += 1
            else:
                future['later_pair_unknown'] += 1
    signals = {
        'a) trajectory/candidate problem':erroneous_v2_windows,
        'b) pair-verifier representation problem':contact['without_predicted_contact_windows'],
        'c) pairwise-loss vs window-OR mismatch':structure['exactly_one_candidate_above_threshold'],
        'd) horizon/timing problem':future['later_collision_with_same_accepted_pair'],
    }
    supported = [name for name,count in signals.items() if count]
    hypothesis = supported[0] if len(supported)==1 else 'e) mixture'
    negative_summary = dict(populations={'a':'FP','b':'TN'},counts={s:len(groups[s]) for s in ('FP','TN')},
        window_statistics=window_stats,top_candidate_features_by_bucket=tn_features,feature_groups=effects_by_group,
        top_candidate_missing_windows={str(k):{s:len(groups[s])-len(tops[s,k]) for s in ('FP','TN')} for k in (1,2)})
    mechanisms = dict(sanity_counts={s:len(groups[s]) for s in EXPECTED},which_verifier_fired=fired,
        v2_contact_evidence={**dict(contact),'predicted_contact_but_complete_gt_no_contact_0_2_windows':erroneous_v2_windows},gt_physical_outcome=dict(future),oracle_geometry=dict(geom),
        pair_multiplicity_and_score_structure=structure,
        oracle_clearance_distributions={f'{a}_{b}':dict(
            all_available=describe([r[f'gt_min_clearance_{a}_{b}_m'] for r in accepted]),
            complete_only=describe([r[f'gt_min_clearance_{a}_{b}_m'] for r in accepted if r[f'geometry_complete_{a}_{b}']]),
            complete_pair_rows=sum(r[f'geometry_complete_{a}_{b}'] for r in accepted),
            unknown_contact_pair_rows=sum(r[f'gt_contact_{a}_{b}'] is None for r in accepted)) for a,b in ((0,2),(2,5))},
        hard_negative_hypothesis=dict(
            explanation='Frozen pair thresholds act through a window OR. Counts describe accepted candidates, '
                        'not an untested alternative window-aware model or an invented elevated-score cutoff.',
            single_candidate_windows=structure['exactly_one_candidate_above_threshold'],
            multiple_candidate_windows=structure['multiple_candidates_above_threshold'],
            predominance='one isolated candidate' if structure['exactly_one_candidate_above_threshold']>len(fp)/2 else 'several candidates',
            negative_window_score_distributions=window_stats),
        recall_context=dict(gt_positive_windows=53,fn_windows=len(groups['FN']),
            uncovered_positive_windows=sum(not r['gt_pair_candidate_covered'] for r in windows if r['status'] in ('TP','FN')),
            uncovered_fn_windows=sum(not r['gt_pair_candidate_covered'] for r in groups['FN']),
            tp_with_wrong_pair_only=sum(not r['correct_gt_pair_hit'] for r in groups['TP']),
            fn_window_ids=[r['window_id'] for r in groups['FN']]),
        scenario_concentration=concentration(windows,candidates),conventions=__doc__,
        descriptive_hypothesis=dict(label=hypothesis,nonexclusive_evidence_window_counts=signals,
            explanation='The label reflects the observed diagnostic signals, not causal attribution. A single accepted '
                        'candidate is consistent with OR exposure but does not prove a loss mismatch; acceptance without '
                        'V2 contact is consistent with representation effects but does not prove them. Multiple signals '
                        'are reported as a mixture. This audit cannot validate a training change.'))
    return negative_summary,pair_summary,mechanisms


def compact_answers(negative,pair,mechanisms):
    fired = mechanisms['which_verifier_fired']
    contact = mechanisms['v2_contact_evidence']
    structure = mechanisms['pair_multiplicity_and_score_structure']
    future = mechanisms['gt_physical_outcome']
    geom = mechanisms['oracle_geometry']
    ranked = sorted(negative['feature_groups'].items(),key=lambda kv:-(kv[1]['fp_tn_mean_absolute_effect'] or 0))
    answers = [
        f"1. Verifier firing (55 FP windows): {fired}.",
        f"2. Accepted V2 contact evidence: {contact}. Oracle 0–2 contact windows: {geom.get('actual_gt_contact_0_2_windows',0)}; "
        'predicted contact alone does not establish trajectory error without complete oracle coverage.',
        f"3. Candidate structure: {structure}; predominance: {mechanisms['hard_negative_hypothesis']['predominance']}.",
        f"4. Dataset outcomes: {future}; TP wrong-pair-only windows: {mechanisms['recall_context']['tp_with_wrong_pair_only']}. "
        'Bucket 0 is observed through 5 s only when k3–k5 are all scorable; otherwise the later outcome is unknown.',
        '5. Largest FP–TN feature-group effects: '+json.dumps(ranked[:3])+'.',
        '6. FP similarity to correct TP: see group FP–TP effects above and recall-overlap ranking: '+
            ', '.join(r['feature'] for r in pair['recall_overlap_ranking'][:5])+'.',
        '7. Descriptive hypothesis: '+mechanisms['descriptive_hypothesis']['label']+'. '+mechanisms['descriptive_hypothesis']['explanation'],
    ]
    return answers


def write_outputs(out,windows,candidates,accepted,raw,norm,negative,pair,mechanisms):
    # All validations and all oracle work complete before the first output write.
    require(len(windows)==219 and Counter(r['status'] for r in windows)==Counter(EXPECTED),'Output population mismatch')
    require(len(raw)==len(norm)==len(candidates),'Feature mapping length mismatch')
    require(not any((out/name).exists() for name in OUTPUTS),'Outputs exist; choose a new --out')
    out.mkdir(parents=True,exist_ok=True)
    def csv_file(name,rows,fields):
        with (out/name).open('x',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    for name,rows in ((OUTPUTS[0],windows),(OUTPUTS[1],[r for r in windows if r['status'] in ('FP','TN')]),
                      (OUTPUTS[2],[r for r in windows if r['status']=='FP'])):
        csv_file(name,rows,list(windows[0]))
    # Oracle fields are confined to accepted-FP export; candidate rows stay inference-only plus analysis labels.
    base_fields = list(candidates[0]) if candidates else []
    csv_file(OUTPUTS[3],candidates,base_fields)
    oracle_fields = ['gt_min_clearance_0_2_m','gt_min_clearance_0_2_time_s','gt_contact_0_2',
        'gt_min_clearance_2_5_m','gt_min_clearance_2_5_time_s','gt_contact_2_5','gt_first_contact_time_s',
        'geometry_complete_0_2','geometry_complete_2_5','gt_contact_event','accepted_pair_matches_later_gt_pair']
    csv_file(OUTPUTS[4],accepted,base_fields+oracle_fields)
    np.savez_compressed(out/OUTPUTS[5],raw_features=np.asarray(raw,dtype=np.float64).reshape(-1,len(FEATURE_NAMES_V4)),
        normalized_features=np.asarray(norm,dtype=np.float64).reshape(-1,len(FEATURE_NAMES_V4)),
        feature_names=np.asarray(FEATURE_NAMES_V4),
        **{name:np.asarray([r[name] for r in candidates]) for name in
            ('window_id','bucket','actor_a','actor_b','status','accepted','verifier_score','gt_pair_label_0_2')})
    for name,value in zip(OUTPUTS[6:],(negative,pair,mechanisms)):
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # These three are required by the frozen evaluator too.
    parser.add_argument('--root',required=True)
    parser.add_argument('--carla-maps',required=True)
    parser.add_argument('--predictor',required=True)
    parser.add_argument('--bucket1-model',default=str(frozen.DEFAULT_BUCKET1_MODEL))
    parser.add_argument('--bucket2-model',default=str(frozen.DEFAULT_BUCKET2_MODEL))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.workers is None:
        args.workers = 1 if 'cuda' in args.device.lower() else 4
    require(args.workers>0,'Workers must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    require(not any((args.out/n).exists() for n in OUTPUTS),'Outputs exist; choose a new --out')
    ledger,reference = load_cohort()
    models = frozen.load_frozen_models(args.bucket1_model,args.bucket2_model)
    original_thresholds = {k:m.threshold for k,m in models.items()}
    require(original_thresholds==EXPECTED_THRESHOLDS,'Thresholds differ from the frozen V4 evaluation')
    cfg = frozen.PipelineConfig(); cfg.deepaccident.observation_mode='sensor3d'
    if args.workers==1:
        cfg.predictor = frozen.JointSceneTorchPredictor(args.predictor,device=args.device)
    runner = frozen.DeepAccidentRunner(args.root,cfg)
    scenarios = sorted((s for s in runner.list_scenarios() if s.split=='val'),key=lambda s:s.scenario_id)
    require(len(scenarios)==104,'Expected official val104')
    records,windows,candidates,raw,norm = [],[],[],[],[]
    if args.workers==1:
        for scenario in scenarios:
            recs,ws,cs,rs,ns = replay_scenario(scenario,args,models,runner=runner,cfg=cfg)
            records.extend(recs); windows.extend(ws); candidates.extend(cs); raw.extend(rs); norm.extend(ns)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for recs,ws,cs,rs,ns in pool.map(lambda s:replay_scenario(s,args,models),scenarios):
                records.extend(recs); windows.extend(ws); candidates.extend(cs); raw.extend(rs); norm.extend(ns)
    validate_records(records,ledger,reference)
    require(original_thresholds=={k:m.threshold for k,m in models.items()},'Frozen thresholds changed')
    require(all(w['gt_bucket']==int(ledger[w['window_id']]['gt_bucket']) for w in windows),'Frozen GT bucket mismatch')
    print(f'Exact frozen cohort and decisions reproduced: {EXPECTED}',flush=True)
    # Copy to prevent oracle fields from leaking into inference candidate exports.
    accepted = [dict(r) for r in candidates if r['status']=='FP' and r['accepted']]
    by_scenario = defaultdict(list)
    for r in accepted:
        by_scenario[r['scenario_id']].append(r)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda s:oracle_geometry(s,by_scenario[s.scenario_id],args),
                      [s for s in scenarios if s.scenario_id in by_scenario]))
    negative,pair,mechanisms = summaries(windows,candidates,accepted)
    paths = [COHORT,FROZEN_ROWS,Path(args.predictor),Path(args.bucket1_model),Path(args.bucket2_model),
             Path(frozen.__file__),Path(__file__),Path(gt_replay.__file__),
             Path(geometry_audit.__file__),Path(inspect.getsourcefile(_shared_geometry))]
    mechanisms['provenance'] = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    mechanisms['frozen_thresholds'] = original_thresholds
    answers = compact_answers(negative,pair,mechanisms)
    mechanisms['compact_answers'] = answers
    write_outputs(args.out,windows,candidates,accepted,raw,norm,negative,pair,mechanisms)
    print('\n'.join(answers),flush=True)
    print(f'Diagnostics: {args.out}',flush=True)


if __name__=='__main__':
    main()
