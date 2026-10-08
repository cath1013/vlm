#!/usr/bin/env python3
"""Diagnostic-only frozen CRN-4 audit; no fitting, threshold search or relabeling.

Run with --root ~/inclab-nas/DeepAccident --carla-maps carla_map.
CRN inputs are exclusively the original collector JSONL via the evaluator's
collate/forward/aggregate/report path. Oracle geometry is computed afterwards.
Footprint clearance is continuous in metres (not a binary distance), evaluated
at the existing audit's 0.1-s samples on (0,2] and (2,5]. Raw yaw and dimensions
are exact lookups, never interpolated. Incomplete-coverage minima describe only
available samples; absence of contact is null unless coverage is complete.
Kinematics use observed snapshot/history XY only, bounded by window start and
cutoff. Velocity is the latest backward difference; the -1s quantities require
an exact observed sample and its preceding sample, without extrapolation.
GT contact flags denote sampled overlap; first contact uses existing audit
new/boundary/recontact onset semantics. V2 agreement uses those onset semantics.
V2 context is a separate frozen rollout from the same collector inputs.
Closing speed is positive toward the other actor; distance/speed/closing changes
are current minus one second earlier. Heading difference is unsigned [0,180]
from the observed ActorState headings. CV closest time is the unconstrained future
minimizer (clamped below at zero); CV minimum distance clamps time to [0,2].
Features are captured verbatim by a hazard-head pre-hook, before time append.
All summaries are descriptive; windows from the same scenario are dependent.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
from examples import train_collision_risk_net as evaluator
from examples import plot_v4_crn4_error_cases as audit
from examples import audit_gt_xy_source_ablation as replay
from traffic_llm.schemas import PredictedPath
from traffic_llm.collision_risk_net import CollisionRiskNet, cumulative_risk, hazard_to_event_probabilities, load_v2

RUN = ROOT / 'out/collision_risk_hazard/run_20261006_window_balanced'
EXPECTED = dict(TP=41, FP=40, TN=126, FN=12)
KINEMATICS = ('center_distance_t0_m relative_speed_t0_mps closing_speed_t0_mps '
              'heading_difference_deg cv_time_to_closest_s cv_min_center_distance_0_2_m '
              'observed_distance_change_last_1s_m actor_a_speed_change_last_1s_mps '
              'actor_b_speed_change_last_1s_mps relative_closing_change_last_1s_mps').split()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def status(bucket, predicted):
    return ('TP' if predicted else 'FN') if 0 < bucket <= 2 else ('FP' if predicted else 'TN')


@torch.no_grad()
def infer(model, rows, device, batch_size, threshold, ledger):
    """Same batch padding and float64 aggregation as the frozen evaluator."""
    records, windows, pairs, vectors = [], [], [], []
    captured = []
    def capture(_module, args):
        head_input = args[0].detach().cpu()
        features = head_input[..., 0, :-1].clone()
        require(torch.equal(head_input[..., :-1], features.unsqueeze(-2).expand_as(head_input[..., :-1])),
                'Head features differ across time')
        require(torch.equal(head_input[..., -1], model.time_encoding().expand_as(head_input[..., -1])),
                'Unexpected head time input')
        captured.append(features)
    model.eval()
    handle = model.head.register_forward_pre_hook(capture)
    try:
        for batch in DataLoader(evaluator.eligible_rows(rows, 4), batch_size=batch_size,
                                collate_fn=partial(evaluator.collate, max_horizon=4)):
            captured.clear()
            logits, mask = evaluator.forward(model, batch, device)
            probabilities = logits.sigmoid().cpu()
            require(len(captured) == 1, 'Expected one hazard head invocation')
            for b, row in enumerate(batch['rows']):
                valid = mask[b].cpu()
                h = probabilities[b, valid].double()
                rec = evaluator.aggregate_window(row, h.tolist(), 4)
                records.append(rec)
                if not evaluator.identifiable(row, 2):
                    continue
                wid = row['window_id']
                require(wid in ledger, f'Window absent from frozen join: {wid}')
                saved = ledger[wid]
                risk = cumulative_risk(h)
                events, _ = hazard_to_event_probabilities(h)
                ids = [a['actor_id'] for a in row['actors']]
                local_pairs = list(itertools.combinations(ids, 2))
                require(len(local_pairs) > 0, f'No pairs: {wid}')
                order = sorted(range(len(local_pairs)), key=lambda p: -float(risk[p, 1]))
                top = order[0]
                predicted = evaluator.decide_window(rec, threshold, 2)
                label = status(row['gt_bucket'], predicted)
                require(rec['selected_pairs'][1] == local_pairs[top], f'Ranking mismatch: {wid}')
                require(audit.parse_pairs(saved['crn4_selected_pair'], single=True)[0] == local_pairs[top],
                        f'Frozen selected pair mismatch: {wid}')
                require(int(saved['gt_bucket']) == row['gt_bucket'] and
                        (saved['crn4_predicted'] == 'True') == predicted and
                        float(saved['crn4_threshold_2s']) == threshold and
                        math.isclose(float(saved['crn4_risk_2s']), rec['horizon_risks'][1], abs_tol=2e-6),
                        f'Frozen join disagreement: {wid}')
                require(torch.equal(risk[:, 1], 1 - (1-h[:, 0])*(1-h[:, 1])), 'Risk reconstruction failed')
                require(torch.allclose(events[:, :2].sum(-1), risk[:, 1], atol=1e-15, rtol=0),
                        'Event reconstruction failed')
                top_risk = float(risk[top, 1])
                logit = lambda p: math.log(p / (1-p)) if 0 < p < 1 else None
                lt, lr = logit(threshold), logit(top_risk)
                top2 = float(risk[order[1], 1]) if len(order) > 1 else None
                w = dict(window_id=wid, scenario_id=row['scenario_id'], status=label,
                         comparison_group=saved.get('comparison_group', saved['transition']),
                         gt_bucket=row['gt_bucket'], later_collision=row['gt_bucket'] in (3,4,5),
                         n_actors=len(ids), n_pairs=len(local_pairs), threshold_2s=threshold,
                         top1_pair=json.dumps(local_pairs[top]), top1_risk_2s=top_risk,
                         top2_risk_2s=top2, top1_minus_top2=top_risk-top2 if top2 is not None else None,
                         n_pairs_above_threshold=int((risk[:, 1] >= threshold).sum()),
                         raw_margin=top_risk-threshold, top1_risk_logit=lr, threshold_logit=lt,
                         logit_margin=lr-lt if lr is not None and lt is not None else None)
                for k in range(2):
                    w.update({f'hazard_{k+1}': float(h[top,k]),
                              f'event_probability_{k+1}': float(events[top,k]),
                              f'cumulative_risk_{k+1}': float(risk[top,k])})
                features = captured[0][b, valid].numpy()
                ranks = {p: rank+1 for rank,p in enumerate(order)}
                for p, (a,c) in enumerate(local_pairs):
                    pairs.append(dict(window_id=wid, status=label, actor_a=a, actor_b=c,
                                      pair_rank_at_2s=ranks[p], risk_1s=float(risk[p,0]), risk_2s=float(risk[p,1]),
                                      hazard_1=float(h[p,0]), hazard_2=float(h[p,1]),
                                      event_probability_1=float(events[p,0]), event_probability_2=float(events[p,1]),
                                      above_2s_threshold=bool(risk[p,1]>=threshold), gt_pair_label=row['pair_labels'][p]))
                    vectors.append(features[p].copy())
                require(w['n_pairs_above_threshold'] == sum(p['above_2s_threshold'] for p in pairs[-len(local_pairs):]),
                        'Threshold count mismatch')
                windows.append(w)
    finally:
        handle.remove()
    metrics = evaluator.report(records, rows, model.frozen_thresholds, 4)['per_horizon']['2']
    require({k: metrics[k.lower()] for k in EXPECTED} == EXPECTED, 'Frozen confusion matrix mismatch')
    require(Counter(w['status'] for w in windows) == Counter(EXPECTED), 'Diagnostic confusion matrix mismatch')
    require({w['window_id'] for w in windows} == set(ledger) and len(windows) == 219, 'Population mismatch')
    return windows, pairs, np.stack(vectors)


def observed_kinematics(window, pair):
    histories = []
    headings = []
    for aid in pair:
        actor = next(a for a in window.last.actors if a.actor_id == aid)
        headings.append(actor.heading_deg)
        points = {float(t): np.array([x,y], float) for t,x,y in actor.track_history
                  if window.t_start <= t <= window.t_end}
        for snap in window.snapshots:
            if window.t_start <= snap.t <= window.t_end:
                for a in snap.actors:
                    if a.actor_id == aid:
                        points[float(snap.t)] = np.array(a.world_xy, float)
        points[window.t_end] = np.array(actor.world_xy, float)
        histories.append(points)
    def at(points, t):
        exact = next((s for s in points if abs(s-t)<1e-8), None)
        if exact is None:
            return None, None
        earlier = [s for s in points if s < exact-1e-8]
        prev = max(earlier) if earlier else None
        return points[exact], (points[exact]-points[prev])/(exact-prev) if prev is not None else None
    result = dict.fromkeys(KINEMATICS)
    pa, va = at(histories[0], window.t_end)
    pb, vb = at(histories[1], window.t_end)
    r = pb-pa
    distance = float(np.linalg.norm(r))
    result['center_distance_t0_m'] = distance
    if all(h is not None for h in headings):
        result['heading_difference_deg'] = abs((headings[0]-headings[1]+180)%360-180)
    def closing(r,v):
        d = np.linalg.norm(r)
        return -float(np.dot(r,v))/d if d>0 else None
    if va is not None and vb is not None:
        v = vb-va
        speed = float(np.linalg.norm(v))
        tc = max(0., -float(np.dot(r,v))/speed**2) if speed>0 else None
        result.update(relative_speed_t0_mps=speed, closing_speed_t0_mps=closing(r,v),
                      cv_time_to_closest_s=tc,
                      cv_min_center_distance_0_2_m=float(np.linalg.norm(r+v*min(2.,tc))) if tc is not None else distance)
    qa, wa = at(histories[0], window.t_end-1)
    qb, wb = at(histories[1], window.t_end-1)
    if qa is not None and qb is not None:
        result['observed_distance_change_last_1s_m'] = distance-float(np.linalg.norm(qb-qa))
    for name, now, before in [('actor_a',va,wa),('actor_b',vb,wb)]:
        if now is not None and before is not None:
            result[name+'_speed_change_last_1s_mps'] = float(np.linalg.norm(now)-np.linalg.norm(before))
    if all(x is not None for x in (va,vb,qa,qb,wa,wb)):
        old = closing(qb-qa, wb-wa)
        now = closing(r,vb-va)
        if old is not None and now is not None:
            result['relative_closing_change_last_1s_mps'] = now-old
    return result


def selected_geometry(actors, pair, identities, frames, cutoff, horizon):
    """Reuse exact audit poses/footprints; unknown remains unknown at gaps."""
    heading = audit.geometry._HeadingLookup(frames, identities, cutoff)
    size = audit.geometry._ExactFootprintLookup(frames, identities, cutoff)
    a,b = (actors.get(aid) for aid in pair)
    values = []
    if a is not None and b is not None:
        options = []
        for aa,bb in itertools.product(a.predictions,b.predictions):
            ma = audit.geometry._future_pose(a,aa,horizon,heading)
            mb = audit.geometry._future_pose(b,bb,horizon,heading)
            if ma is None or mb is None:
                continue
            samples = []
            for step in range(int(horizon*10)+1):
                t = step/10
                valid = t<=min(ma[1],mb[1])+1e-8 and not any(lo<t<hi for lo,hi in (*ma[2],*mb[2]))
                pa,pb = (ma[0](t),mb[0](t)) if valid else (None,None)
                ea,eb = size(a,t),size(b,t)
                gap = audit._footprint_clearance(pa,ea,pb,eb) if all(x is not None for x in (pa,pb,ea,eb)) else None
                samples.append((t,gap))
            options.append(samples)
        if options:
            values = min(options, key=lambda ss:min((g for _,g in ss if g is not None),default=math.inf))
    output = {}
    for start,end in ((0,2),(2,5)):
        if end>horizon:
            continue
        samples = [(t,g) for t,g in values if start<t<=end or (start==0 and t==0)]
        full = len(samples)==(21 if start==0 else 30) and all(g is not None for _,g in samples)
        available = [(g,t) for t,g in samples if g is not None and t>start]
        minimum = min(available) if available else (None,None)
        contact = True if any(g<=0 for g,t in available) else (False if full else None)
        output.update({f'min_clearance_{start}_{end}_m':minimum[0], f'min_clearance_{start}_{end}_time_s':minimum[1],
                       f'contact_{start}_{end}':contact, f'geometry_complete_{start}_{end}':full})
    # Existing audit event semantics distinguish preexisting contact and onset.
    first = None
    event = 'GEOMETRY_UNCERTAIN'
    if values and values[0][1] is not None:
        before,gapped = audit.geometry._raw_history_before(a,b,heading,size,0.)
        missing = [(max(0,t-.05),min(horizon,t+.05)) for t,g in values if g is None]
        event,at = audit.geometry._contact_event(values[0][1]<=0,before,gapped,
            [(t,g<=0) for t,g in values if t>0 and g is not None],
            not missing,missing)
        if event in {'NEW_CONTACT','BOUNDARY_ONSET','RECONTACT'}:
            first = at
    output['first_contact_time_s'] = first
    output['contact_event'] = event
    return output


@torch.no_grad()
def v2_paths(model, row, actors, device):
    """Frozen V2 rollout from collector inputs; runtime adapter's XY conversion.

    Oracle ActorState copies supply only observed t0 XY/history for plotting
    geometry. They are never encoded: all V2 inputs come from the collector.
    """
    batch = evaluator.collate([row],4)
    offsets, _ = evaluator.forward(model,batch,device)
    result = {}
    for i,encoded in enumerate(row['actors']):
        aid = encoded['actor_id']
        if aid not in actors:
            continue
        actor = copy.deepcopy(actors[aid])
        origin = encoded['origin_enu']
        require(np.allclose(origin,actor.world_xy,atol=1e-8,rtol=0),f'Replay t0 origin mismatch: {aid}')
        arr = offsets[0,i].reshape(-1,2).float().cpu().tolist()
        actor.predictions = [PredictedPath('Frozen JointScene V2 context',1.,
            [tuple(origin)]+[(origin[0]+de,origin[1]+dn) for de,dn in arr],5.,truncated=False)]
        result[aid] = actor
    return result


def enrich(windows, args, predictor, collector_rows):
    cfg = audit.PipelineConfig()
    cfg.deepaccident.observation_mode = 'sensor3d'
    runner = audit.DeepAccidentRunner(args.root,cfg)
    scenarios = [s for s in runner.list_scenarios() if s.split=='val']
    require(len(scenarios)==104, 'Expected official val104')
    by_id = {r['window_id']:r for r in collector_rows}
    v2_model = load_v2(predictor,args.device).to(args.device).eval()
    by_scenario = defaultdict(list)
    for row in windows:
        by_scenario[row['scenario_id']].append(row)
    require(set(by_scenario) <= {s.scenario_id for s in scenarios},'Missing val104 scenario')
    def process(scenario):
        print(f'Diagnostic replay: {scenario.scenario_id}',flush=True)
        gt,ids,_,_ = replay.replay_windows_profiled(scenario.root,
            dict(town=scenario.town,scenario=scenario.scenario,scenario_type=scenario.scenario_type),
            dict(mode='sensor3d',window_s=5.,stride_s=1.,horizon_s=5.,history_stride_s=1.),
            args.carla_maps)
        frames = replay._FrameBoxCache(scenario)
        for row in by_scenario[scenario.scenario_id]:
            matches = [w for w in gt.values() if f'{scenario.scenario_id}:{w.index}:{w.t_end:.3f}'==row['window_id']]
            require(len(matches)==1, f'Missing replay window: {row["window_id"]}')
            window = matches[0]
            pair = json.loads(row['top1_pair'])
            row.update(observed_kinematics(window,pair))
            identities = audit.geometry.identities(window,ids)
            truth = gt.get(window.label)
            result = selected_geometry({a.actor_id:a for a in truth.last.actors} if truth else {},
                                       pair,identities,frames,window.t_end,5)
            for key,value in result.items():
                row[('gt_'+key if key.startswith('geometry_complete') else 'selected_pair_gt_'+key)] = value
            later = []
            if row['later_collision']:
                collision = audit.estimate_collision(scenario,cfg.deepaccident)
                target = list(collision.carla_ids or [])
                if len(target)==2:
                    aliases = [[aid for aid,cids in identities.items() if cid in cids] for cid in target]
                    later = [list(p) for p in itertools.product(*aliases) if p[0]!=p[1]]
                row['later_gt_collision_pair'] = json.dumps(later) if later else None
                row['crn_pair_matches_later_gt_pair'] = any(set(pair)==set(p) for p in later) if later else None
            else:
                row['later_gt_collision_pair'] = None
                row['crn_pair_matches_later_gt_pair'] = None
            predicted_actors = v2_paths(v2_model,by_id[row['window_id']],
                                       {a.actor_id:a for a in window.last.actors},args.device)
            v2 = selected_geometry(predicted_actors,pair,identities,frames,window.t_end,2)
            row['v2_predicted_contact_0_2'] = (True if v2['first_contact_time_s'] is not None else
                False if v2['geometry_complete_0_2'] and v2['contact_event'] in {'NONE','PREEXISTING_PERSISTENT'} else None)
            row['v2_predicted_min_clearance_0_2_m'] = v2['min_clearance_0_2_m']
    requested = sorted((s for s in scenarios if s.scenario_id in by_scenario),key=lambda s:s.scenario_id)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(process,requested))


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(n=0,median=None,q1=None,q3=None,iqr=None,min=None,max=None)
    q1,median,q3 = np.quantile(values,[.25,.5,.75]).tolist()
    return dict(n=len(values),median=median,q1=q1,q3=q3,iqr=q3-q1,min=min(values),max=max(values))


def summary(windows):
    groups = {s:[r for r in windows if r['status']==s] for s in ('FP','TN')}
    fp = groups['FP']
    fields = ['n_pairs','top1_risk_2s','top2_risk_2s','top1_minus_top2','logit_margin',
              'hazard_1','hazard_2','event_probability_1','event_probability_2',*KINEMATICS]
    stats = {}
    for field in fields:
        stats[field] = {s:{**describe([r[field] for r in rows]),
                          'missing':sum(r[field] is None for r in rows)} for s,rows in groups.items()}
        if field in KINEMATICS:
            a,b = ([r[field] for r in groups[s] if r[field] is not None] for s in ('FP','TN'))
            stats[field]['cliffs_delta_fp_minus_tn'] = (sum(int(x>y)-int(x<y) for x in a for y in b)/(len(a)*len(b)) if a and b else None)
    concentration = Counter(r['scenario_id'] for r in fp)
    repeated = []
    previous = {}
    for r in sorted(windows,key=lambda r:(r['scenario_id'],float(r['window_id'].rsplit(':',1)[1]))):
        old = previous.get(r['scenario_id'])
        if old and old['status']==r['status']=='FP' and set(json.loads(old['top1_pair']))==set(json.loads(r['top1_pair'])):
            dt = float(r['window_id'].rsplit(':',1)[1])-float(old['window_id'].rsplit(':',1)[1])
            if abs(dt-1)<1e-8:
                repeated.append([old['window_id'],r['window_id'],r['top1_pair']])
        previous[r['scenario_id']] = r
    return dict(sanity_counts=dict(Counter(r['status'] for r in windows)),
        negative_scenario_counts={s:len({r['scenario_id'] for r in rows}) for s,rows in groups.items()},
        fp_future_outcome={**{f'bucket{k}':sum(r['gt_bucket']==k for r in fp) for k in (0,3,4,5)},
            'bucket0_meaning':'no recorded collision within the window 5-s horizon; not proof of complete 5-s coverage',
            'later_same_pair':sum(r['later_collision'] and r['crn_pair_matches_later_gt_pair'] is True for r in fp),
            'later_different_pair':sum(r['later_collision'] and r['crn_pair_matches_later_gt_pair'] is False for r in fp),
            'later_pair_unknown':sum(r['later_collision'] and r['crn_pair_matches_later_gt_pair'] is None for r in fp)},
        descriptive_statistics=stats,
        hard_pair_structure={s:dict(n_pairs_distribution=dict(Counter(r['n_pairs'] for r in rows)),
            exactly_one_above_threshold=sum(r['n_pairs_above_threshold']==1 for r in rows),
            multiple_above_threshold=sum(r['n_pairs_above_threshold']>1 for r in rows)) for s,rows in groups.items()},
        hazard_contributions={s:dict(event_p1_greater=sum(r['event_probability_1']>r['event_probability_2'] for r in rows),
            event_p2_greater=sum(r['event_probability_2']>r['event_probability_1'] for r in rows),
            equal=sum(r['event_probability_1']==r['event_probability_2'] for r in rows)) for s,rows in groups.items()},
        selected_pair_gt_geometry={f'{start}_{end}':dict(
            fp_min_clearance=describe([r[f'selected_pair_gt_min_clearance_{start}_{end}_m'] for r in fp]),
            fp_min_clearance_complete_only=describe([r[f'selected_pair_gt_min_clearance_{start}_{end}_m'] for r in fp
                if r[f'gt_geometry_complete_{start}_{end}']]),
            fp_contacts=sum(r[f'selected_pair_gt_contact_{start}_{end}'] is True for r in fp),
            coverage={s:dict(complete=sum(r[f'gt_geometry_complete_{start}_{end}'] for r in rows),
                            total=len(rows),contact_unknown=sum(r[f'selected_pair_gt_contact_{start}_{end}'] is None for r in rows))
                      for s,rows in groups.items()}) for start,end in ((0,2),(2,5))},
        fp_windows_per_scenario=dict(concentration),consecutive_fp_same_pair_links=repeated,
        v2_agreement=dict(fp_contact=sum(r['v2_predicted_contact_0_2'] is True for r in fp),
                          fp_no_contact=sum(r['v2_predicted_contact_0_2'] is False for r in fp),
                          fp_unknown=sum(r['v2_predicted_contact_0_2'] is None for r in fp)),
        conventions=__doc__,
        hazard_comparison='event p1=h1 versus event p2=(1-h1)*h2; ties reported separately',
        feature_layout=['state_i + state_j','abs(state_i - state_j)',
                        'normalized edge_ij + normalized edge_ji','abs(normalized edge_ij - normalized edge_ji)'])


def write_csv(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path.home()/'inclab-nas/DeepAccident'))
    parser.add_argument('--carla-maps',default=str(ROOT/'carla_map'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--workers',type=int,default=4,help='Independent scenario geometry replay threads')
    parser.add_argument('--out',type=Path,default=RUN/'final_analysis/crn4_diagnostics')
    args = parser.parse_args(argv)
    require(args.batch_size>0 and args.workers>0,'Batch size and workers must be positive')
    names = ['crn4_negative_window_diagnostics.csv','crn4_fp_diagnostics.csv','crn4_all_pair_risks.csv',
             'crn4_pair_head_features.npz','crn4_fp_vs_tn_summary.json','crn4_all_window_diagnostics.csv']
    require(not any((args.out/name).exists() for name in names),'Outputs already exist; choose a new --out')
    checkpoint = RUN/'crn4/collision_risk_final.pt'
    require(hashlib.sha256(checkpoint.read_bytes()).hexdigest() ==
            '311bac77b615d7c71bc05567f053cdeb139e7018e21de26c3d97cc9f410f454f',
            'Frozen checkpoint SHA256 mismatch')
    saved = torch.load(checkpoint,map_location=args.device,weights_only=False)
    require(saved['max_horizon']==4,'Expected frozen CRN-4')
    model = CollisionRiskNet(load_v2(saved['v2_checkpoint'],args.device),saved['hidden_dim'],saved['dropout'],max_horizon=4).to(args.device)
    model.load_state_dict(saved['state_dict'])
    model.frozen_thresholds = saved['threshold_by_horizon']
    evaluator.verify_manifest(RUN/'val104.jsonl','val',smoke=False)
    rows = evaluator.read_rows(RUN/'val104.jsonl','val')
    require(not set(saved['train_scenario_ids']) & {r['scenario_id'] for r in rows},'Train/val overlap')
    ledger = audit.read_csv(RUN/'final_analysis/v4_vs_crn4_2s_all_windows.csv')
    require(len(ledger)==219 and len({r['window_id'] for r in ledger})==219,'Invalid frozen ledger population')
    windows,pairs,features = infer(model,rows,args.device,args.batch_size,saved['threshold_by_horizon']['2'],{r['window_id']:r for r in ledger})
    print(f'Frozen confusion matrix reproduced: {EXPECTED}; {len(pairs)} pair vectors',flush=True)
    enrich(windows,args,saved['v2_checkpoint'],rows)
    result = summary(windows)
    result['provenance'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (checkpoint,RUN/'val104.jsonl',RUN/'final_analysis/v4_vs_crn4_2s_all_windows.csv')}
    negative = [r for r in windows if r['status'] in ('FP','TN')]
    fp = [r for r in windows if r['status']=='FP']
    require(len(negative)==166 and len(fp)==40,'Negative population mismatch')
    args.out.mkdir(parents=True,exist_ok=True)
    for name,data in ((names[0],negative),(names[1],fp),(names[2],pairs),(names[5],windows)):
        write_csv(args.out/name,data)
    np.savez_compressed(args.out/names[3],features=features,
        **{k:np.asarray([r[k] for r in pairs]) for k in
           ('window_id','actor_a','actor_b','status','gt_pair_label','pair_rank_at_2s')})
    (args.out/names[4]).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Wrote diagnostics to {args.out}',flush=True)


if __name__=='__main__':
    main()
