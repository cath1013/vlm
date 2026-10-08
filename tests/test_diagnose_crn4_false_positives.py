"""Frozen audit regressions and data-boundary tests; no training."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from examples import diagnose_crn4_false_positives as d
from test_plot_v4_crn4_error_cases import fixture


@pytest.mark.parametrize('scalar_type', [np.float32, np.float64, np.int64])
@pytest.mark.parametrize('fp_values,tn_values,expected', [
    ([1, 2], [2, 3], -0.75),
    ([2, 3], [1, 2], 0.75),
    ([2], [2], 0.0),
])
def test_summary_cliffs_delta_with_numpy_scalars(scalar_type, fp_values, tn_values, expected):
    windows = []
    for status, values in [('FP', fp_values), ('TN', tn_values)]:
        for value in values:
            row = d.defaultdict(lambda: 0)
            row.update(status=status, scenario_id=f'scenario_{len(windows)}',
                       window_id=f'window_{len(windows)}:0')
            row.update({field: scalar_type(value) for field in d.KINEMATICS})
            windows.append(row)
    stats = d.summary(windows)['descriptive_statistics']
    for field in d.KINEMATICS:
        assert stats[field]['cliffs_delta_fp_minus_tn'] == expected


@pytest.fixture(scope='module')
def frozen():
    torch.set_num_threads(2)
    saved = torch.load(d.RUN/'crn4/collision_risk_final.pt',map_location='cpu',weights_only=False)
    model = d.CollisionRiskNet(d.load_v2(saved['v2_checkpoint']),saved['hidden_dim'],saved['dropout'],max_horizon=4)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    model.frozen_thresholds = saved['threshold_by_horizon']
    rows = d.evaluator.read_rows(d.RUN/'val104.jsonl','val')
    ledger = {r['window_id']:r for r in d.audit.read_csv(d.RUN/'final_analysis/v4_vs_crn4_2s_all_windows.csv')}
    result = d.infer(model,rows,'cpu',16,saved['threshold_by_horizon']['2'],ledger)
    return model,rows,ledger,result


def test_exact_frozen_population_ranking_threshold_and_hazards(frozen):
    _,rows,_,(windows,pairs,features) = frozen
    assert dict(d.Counter(w['status'] for w in windows)) == d.EXPECTED
    assert len(windows)==219
    assert sum(w['status'] in ('FP','TN') for w in windows)==166
    assert features.shape[0]==len(pairs)
    by_window = d.defaultdict(list)
    for p in pairs:
        by_window[p['window_id']].append(p)
        assert p['risk_1s']==1-(1-p['hazard_1'])
        assert p['risk_2s']==1-(1-p['hazard_1'])*(1-p['hazard_2'])
        assert p['event_probability_2']==(1-p['hazard_1'])*p['hazard_2']
        assert abs(p['event_probability_1']+p['event_probability_2']-p['risk_2s'])<1e-15
    for w in windows:
        ps = by_window[w['window_id']]
        top = next(p for p in ps if p['pair_rank_at_2s']==1)
        assert [top['actor_a'],top['actor_b']]==json.loads(w['top1_pair'])
        assert top['risk_2s']==max(p['risk_2s'] for p in ps)
        assert w['n_pairs_above_threshold']==sum(p['risk_2s']>=w['threshold_2s'] for p in ps)


def test_export_matches_independent_pair_tensor_and_head_input(frozen):
    model,rows,_,(_,pairs,features) = frozen
    row = next(r for r in rows if r['window_id']==pairs[0]['window_id'])
    # Match original evaluation batch padding, which affects tiny float errors.
    batch = d.evaluator.collate(d.evaluator.eligible_rows(rows,4)[:16],4)
    b = model.backbone
    with torch.no_grad():
        state,route = b._encode_actors(*(batch[k] for k in d.evaluator.INPUT_KEYS[:7]))
        velocity = b.initial_velocity(batch['global_features'],batch['actor_mask'])
        edge = b.normalize_edges(b.edge_features(batch['origin'],velocity,route,batch['global_features']))
        for block in b.blocks:
            state = block(state,edge,batch['actor_mask'])
        i,j = torch.triu_indices(batch['actor_mask'].shape[1],batch['actor_mask'].shape[1],1)
        expected = torch.cat((state[:,i]+state[:,j],(state[:,i]-state[:,j]).abs(),
                              edge[:,i,j]+edge[:,j,i],(edge[:,i,j]-edge[:,j,i]).abs()),-1)
        captured = []
        handle = model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].clone()))
        try:
            _,mask = d.evaluator.forward(model,batch,'cpu')
        finally:
            handle.remove()
    index = next(k for k,r in enumerate(batch['rows']) if r is row)
    assert torch.equal(expected,captured[0][...,0,:-1])
    np.testing.assert_array_equal(features[:len(row['pair_labels'])],expected[index,mask[index]].numpy())


def test_future_oracle_values_cannot_enter_crn_inputs(frozen):
    model,rows,_,_ = frozen
    original = d.evaluator.collate(rows[:2],4)
    changed = copy.deepcopy(rows[:2])
    for r in changed:
        r['selected_pair_gt_min_clearance_0_2_m'] = -999999
        r['oracle_future'] = 'must never enter inputs'
        r['gt_bucket'] = 0
        r['pair_labels'] = [0]*len(r['pair_labels'])
    poisoned = d.evaluator.collate(changed,4)
    for key in d.evaluator.INPUT_KEYS:
        assert torch.equal(original[key],poisoned[key])
    with torch.no_grad():
        assert torch.equal(d.evaluator.forward(model,original,'cpu')[0],d.evaluator.forward(model,poisoned,'cpu')[0])


def test_observed_kinematics_ignore_future_and_do_not_fill_missing():
    case,window,_,_,_ = fixture()
    a,b = window.last.actors
    a.track_history = [(3.5,-3.,0.),(4.,-2.,0.),(4.5,-1.,0.),(5.,0.,0.),(5.5,999.,999.)]
    b.track_history = [(3.5,13.,0.),(4.,12.,0.),(4.5,11.,0.),(5.,10.,0.),(5.5,-999.,999.)]
    b.heading_deg = 270.
    pair = ['A','B']
    result = d.observed_kinematics(window,pair)
    assert result['center_distance_t0_m']==10
    assert result['relative_speed_t0_mps']==4
    assert result['closing_speed_t0_mps']==4
    assert result['cv_time_to_closest_s']==2.5
    assert result['cv_min_center_distance_0_2_m']==2
    assert result['heading_difference_deg']==180
    assert result['observed_distance_change_last_1s_m']==-4
    assert result['actor_a_speed_change_last_1s_mps']==0
    a.track_history[-1] = (5.5,-1e8,1e8)
    window.snapshots.append(SimpleNamespace(t=6.,actors=[a,b]))
    # Keep the observation endpoint fixed; a future snapshot cannot be the last.
    window.snapshots.insert(0,window.snapshots.pop())
    assert d.observed_kinematics(window,pair)==result
    a.track_history = [(4.5,-1.,0.),(5.,0.,0.)]
    assert d.observed_kinematics(window,pair)['observed_distance_change_last_1s_m'] is None


def test_missing_geometry_remains_unknown_and_coverage_is_interval_specific():
    _,window,truth,frames,_ = fixture()
    actors = {a.actor_id:a for a in truth.last.actors}
    identities = {'A':{1},'B':{2}}
    full = d.selected_geometry(actors,['A','B'],identities,frames,5.,5.)
    assert full['contact_0_2'] is False
    assert full['geometry_complete_0_2'] is True
    assert full['geometry_complete_2_5'] is True
    # Frame lookup is floor(t/.1+epsilon)+1; remove all to avoid assumptions.
    missing = d.selected_geometry(actors,['A','B'],identities,{},5.,5.)
    assert missing['contact_0_2'] is None
    assert missing['contact_2_5'] is None
    assert missing['min_clearance_0_2_m'] is None
    assert missing['geometry_complete_0_2'] is False
    absent = d.selected_geometry({},['A','B'],identities,frames,5.,5.)
    assert absent['contact_0_2'] is None
    # Truncated raw GT can establish only the early interval.
    for a in actors.values():
        a.predictions[0].waypoint_times_s = [0.,2.]
    partial = d.selected_geometry(actors,['A','B'],identities,frames,5.,5.)
    assert partial['geometry_complete_0_2'] is True
    assert partial['geometry_complete_2_5'] is False
    assert partial['contact_2_5'] is None


def test_wrong_aggregate_fails(frozen):
    model,rows,ledger,_ = frozen
    wrong = {**model.frozen_thresholds,'2':0.}
    with patch.object(model,'frozen_thresholds',wrong), pytest.raises(ValueError):
        d.infer(model,rows,'cpu',16,0.,ledger)


def test_isolated_raw_geometry_hole_is_not_interpolated():
    _,_,truth,frames,_ = fixture()
    actors = {a.actor_id:a for a in truth.last.actors}
    del frames[52]  # Exact +0.1s sample; adjacent raw frames remain available.
    result = d.selected_geometry(actors,['A','B'],{'A':{1},'B':{2}},frames,5.,5.)
    assert result['geometry_complete_0_2'] is False
    assert result['contact_0_2'] is None
    assert result['min_clearance_0_2_m'] is not None
    assert result['geometry_complete_2_5'] is True


def test_v2_context_rollout_uses_collector_only(frozen):
    from traffic_llm.schemas import ActorState, PredictedPath
    model,rows,_,_ = frozen
    row = rows[0]
    actors = {a['actor_id']:ActorState(a['actor_id'],'observed','car',tuple(a['origin_enu']),90.,0.,0.)
              for a in row['actors']}
    batch = d.evaluator.collate([row],4)
    with torch.no_grad():
        offsets,_ = d.evaluator.forward(model.backbone,batch,'cpu')
    predicted = d.v2_paths(model.backbone,row,actors,'cpu')
    for i,a in enumerate(row['actors']):
        xy = np.asarray(predicted[a['actor_id']].predictions[0].waypoints)
        np.testing.assert_array_equal(xy[0],a['origin_enu'])
        np.testing.assert_allclose(xy[1:]-a['origin_enu'],offsets[0,i].numpy(),atol=1e-10,rtol=0)
        actors[a['actor_id']].predictions = [PredictedPath('oracle poison',1.,[(999.,999.)],5.)]
    again = d.v2_paths(model.backbone,row,actors,'cpu')
    for aid in predicted:
        assert again[aid].predictions[0].waypoints==predicted[aid].predictions[0].waypoints
