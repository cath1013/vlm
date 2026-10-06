"""Focused frozen-data regressions; expensive raw replay is opt-in.

RUN_V4_DIAGNOSTIC_REPLAY=1 enables complete fresh val104 replay. Ordinary
regressions recompute all 166 negative decisions from saved exact bucket pools,
check the complete frozen ledger, and test candidate extraction with fixtures.
No training, threshold search, labels on disk or frozen files are modified.
"""
import copy
import inspect
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from examples import diagnose_v4_false_positives as d
from traffic_llm.schemas import ActorState, PredictedPath
from traffic_llm.swept_path import SweptPair


@pytest.fixture(scope='module')
def frozen_data():
    ledger,reference = d.load_cohort()
    models = d.frozen.load_frozen_models(d.frozen.DEFAULT_BUCKET1_MODEL,d.frozen.DEFAULT_BUCKET2_MODEL)
    pools = {}
    for k in (1,2):
        with (d.ROOT/f'out/pair_reranker_v4/val_bucket{k}.jsonl').open() as f:
            pools[k] = {(r['scenario'],float(r['window'].split('-')[-1])):r['candidates']
                        for r in map(json.loads,f)}
    return ledger,reference,models,pools


def actor(aid,x):
    return ActorState(aid,'observed','car',(x,0.),90.,2.,-1.,
        track_history=[(3.5,x-3.,0.),(4.,x-2.,0.),(4.5,x-1.,0.),(5.,x,0.)],
        predictions=[PredictedPath('straight',1.,[(x,0.),(x+2.,0.),(x+10.,0.)],5.)])


def pair(a='V001',b='V002',bucket=1,gap=1.):
    return SweptPair(a,b,gap,.5 if bucket==1 else 1.5,bucket,False,1.,'straight','straight',
                     gap+2.,None,0.,gap+1.,gap+3.,'NONE',False,False)


def truth(bucket=0):
    return {'expected':[dict(k=k,scorable=True,accident_expected=k==bucket,
        involved_vehicles=[dict(actor_ids=['V001']),dict(actor_ids=['V002'])] if k==bucket else [])
        for k in range(1,6)]}


def fixture_window():
    return SimpleNamespace(index=0,t_end=5.,last=SimpleNamespace(actors=[actor('V001',0.),actor('V002',10.)],interactions=[]))


def test_exact_cohort_and_confusion(frozen_data):
    ledger,reference,_,_ = frozen_data
    assert len(ledger)==219
    assert d.validate_records(list(reference.values()),ledger,reference)==d.EXPECTED
    assert sum(not r['actual'] for r in reference.values())==166
    assert d.Counter(d.classify(r['actual'],r['predicted']) for r in reference.values())==d.Counter(d.EXPECTED)


def test_individual_decision_mismatch_aborts_even_when_aggregates_match(frozen_data):
    ledger,reference,_,_ = frozen_data
    rows = copy.deepcopy(list(reference.values()))
    fp = next(r for r in rows if not r['actual'] and r['predicted'])
    tn = next(r for r in rows if not r['actual'] and not r['predicted'])
    fp['predicted'],tn['predicted'] = False,True
    with pytest.raises(ValueError,match='decision mismatch'):
        d.validate_records(rows,ledger,reference)
    with pytest.raises(ValueError,match='cohort'):
        d.validate_records(rows[:-1],ledger,reference)
    fp['predicted'] = True
    with pytest.raises(ValueError,match='confusion mismatch'):
        d.validate_records(rows,ledger,reference)


def test_saved_exact_negative_pools_reproduce_all_fp_tn_and_accepted_pairs(frozen_data):
    _,reference,models,pools = frozen_data
    counts = d.Counter()
    thresholds = {k:m.threshold for k,m in models.items()}
    complete_windows = 0
    for old in reference.values():
        key = old['scenario_id'],old['t_end_s']
        if key not in pools[2]:
            assert old['actual'] and old['positive_bucket']==1
            continue  # 33 bucket-1 event windows: bucket-2 collector was censored.
        cs = {k:pools[k][key] for k in (1,2)}
        # Positive identity assertions are in the full replay test. Here check
        # the exact scores, selected pairs and OR on complete saved pools.
        rec = d.frozen.evaluate_window(truth(),cs,models)
        for k in (1,2):
            assert rec[f'bucket{k}_selected_pairs']==old[f'bucket{k}_selected_pairs']
            assert rec[f'bucket{k}_prediction']==old[f'bucket{k}_prediction']
            for c in cs[k]:
                score,normalized = d.score_features(models[k],c['features'])
                assert score==models[k].predict_features(c['features'])
                np.testing.assert_array_equal(normalized,
                    (np.asarray(c['features'])-models[k].means)/models[k].scales)
        assert rec['predicted']==old['predicted']
        if not old['actual']:
            counts['FP' if rec['predicted'] else 'TN'] += 1
        complete_windows += 1
    assert complete_windows==186
    assert dict(counts)==dict(FP=55,TN=111)
    assert thresholds=={k:m.threshold for k,m in models.items()}


def test_filtering_before_cap_and_exact_feature_mapping():
    window = fixture_window()
    config = d.frozen.WindowConfig(window_s=5.,stride_s=1.,horizon_s=5.)
    pairs = [pair(bucket=2,gap=float(i)) for i in range(55)]+[pair(bucket=1,gap=99.)]
    expected = d.frozen.pair_features_v4(*window.last.actors,pairs[-1],5.,window.last.interactions)
    with patch.object(d.frozen,'rank_actors',return_value=(window.last.actors,None)) as rank, \
         patch.object(d.frozen,'swept_pair_clearances',return_value=pairs) as sweep:
        pools,evidence = d.build_candidates(window,config)
    assert len(pools[1])==1 and len(pools[2])==50
    assert evidence[1]==pairs[-1:] and evidence[2]==pairs[:50]
    assert pools[1][0]['features']==expected
    assert dict(zip(d.FEATURE_NAMES_V4,pools[1][0]['features']))==dict(zip(d.FEATURE_NAMES_V4,expected))
    rank.assert_called_once_with(window.last,config.actor_cap(2))
    assert sweep.call_args.kwargs==dict(horizon_s=config.horizon_s,sample_dt_s=config.swept_sample_dt_s,
        contact_margin_m=config.swept_contact_margin_m,exclude_touching_now=config.swept_exclude_touching_now,
        exclude_static_pairs=config.swept_exclude_static_pairs)


def test_diagnostic_scores_features_acceptance_and_thresholds(frozen_data):
    _,_,models,_ = frozen_data
    window = fixture_window()
    raw = d.frozen.pair_features_v4(*window.last.actors,pair(),5.,[])
    cs = {1:[dict(actor_a='V001',actor_b='V002',features=raw,predicted_contact=False)],2:[]}
    evidence = {1:[pair()],2:[]}
    before = [m.threshold for m in models.values()]
    rec,w,rows,features,normalized = d.diagnostic_window(window,'scene',truth(),cs,evidence,models)
    assert rec['selected_pairs']==[[r['actor_a'],r['actor_b']] for r in rows if r['accepted']]
    assert w['n_bucket1_accepted']==sum(r['accepted'] for r in rows)
    assert w['bucket2_top1_score'] is None and w['bucket1_top2_score'] is None
    assert rows[0]['verifier_score']==models[1].predict_features(raw)
    for name,value in zip(d.FEATURE_NAMES_V4,raw):
        assert rows[0]['feature_'+name]==value
    assert features==[raw]
    np.testing.assert_array_equal(normalized[0],(np.asarray(raw)-models[1].means)/models[1].scales)
    assert before==[m.threshold for m in models.values()]
    for k,path in ((1,d.frozen.DEFAULT_BUCKET1_MODEL),(2,d.frozen.DEFAULT_BUCKET2_MODEL)):
        assert models[k].threshold==json.loads(Path(path).read_text())['decision_threshold']
    assert models[1].threshold==0.08578953170775334
    assert models[2].threshold==5.17511995858515e-05


def test_oracle_and_later_labels_cannot_enter_feature_extraction_or_scores(frozen_data):
    _,_,models,_ = frozen_data
    window = fixture_window()
    p = pair()
    with patch.object(d.frozen,'rank_actors',return_value=(window.last.actors,None)), \
         patch.object(d.frozen,'swept_pair_clearances',return_value=[p]):
        cs,evidence = d.build_candidates(window,d.frozen.WindowConfig())
        poisoned = copy.deepcopy(window)
        poisoned.gt_future = 'forbidden oracle'
        for a in poisoned.last.actors:
            a.gt_min_clearance_0_2_m = -999.
            a.oracle_future = [(1e9,1e9)]
        with patch.object(d.frozen,'rank_actors',return_value=(poisoned.last.actors,None)):
            other,_ = d.build_candidates(poisoned,d.frozen.WindowConfig())
    assert other==cs
    assert list(inspect.signature(d.build_candidates).parameters)==['window','wcfg']
    first = d.diagnostic_window(window,'scene',truth(3),cs,evidence,models)
    second = d.diagnostic_window(window,'scene',truth(5),cs,evidence,models)
    assert first[0]['selected_pairs']==second[0]['selected_pairs']
    assert first[0]['predicted']==second[0]['predicted']
    assert first[3:]==second[3:]
    assert first[1]['gt_bucket']==3 and second[1]['gt_bucket']==5


def test_missing_oracle_geometry_remains_unknown():
    actors = {a.actor_id:a for a in fixture_window().last.actors}
    result = d.selected_geometry(actors,('V001','V002'),{'V001':{1},'V002':{2}},{},5.,5.)
    assert result['contact_0_2'] is None and result['contact_2_5'] is None
    assert result['min_clearance_0_2_m'] is None
    assert not result['geometry_complete_0_2'] and not result['geometry_complete_2_5']


def test_scenario_and_consecutive_counting_deterministic():
    windows = [dict(window_id=f's:{i}:{t:.3f}',scenario_id='s',status=s)
               for i,t,s in [(0,5.,'FP'),(1,6.,'FP'),(2,7.,'TN'),(3,8.,'FP'),(4,10.,'FP'),(5,11.,'FP')]]
    candidates = [dict(window_id=w['window_id'],accepted=True,actor_a='a',actor_b='b') for w in windows]
    candidates.append(dict(window_id=windows[1]['window_id'],accepted=True,actor_a='b',actor_b='a'))
    result = d.concentration(windows,candidates)
    assert result==d.concentration(list(reversed(windows)),list(reversed(candidates)))
    assert result['unique_fp_scenarios']==1
    assert result['fp_windows_per_scenario']=={'s':5}
    assert len(result['consecutive_fp_window_links'])==2
    assert len(result['repeated_accepted_pairs_across_consecutive_fp_windows'])==2
    assert result['repeated_accepted_pairs_across_consecutive_fp_windows'][0]['pairs']==[('a','b')]


def test_effects_and_cuda_worker_default():
    assert d.compare([1.,2.],[3.,4.])['cliffs_delta']==-1.
    assert d.compare([0.,1.],[1.,0.])['cliffs_delta']==0.
    assert d.compare([1,1],[0,1],True)['percentage_point_difference']==50.
    assert d.compare([],[])['effect'] is None
    argv = ['--root','data','--carla-maps','maps','--predictor','v2.pt','--out','diag']
    assert d.parse_args(argv+['--device','cuda:0']).workers==1
    assert d.parse_args(argv).workers==4


@pytest.mark.skipif(os.environ.get('RUN_V4_DIAGNOSTIC_REPLAY')!='1',reason='Opt-in complete raw val104/V2 replay')
def test_fresh_full_val104_replay_and_exports(tmp_path):
    d.main(['--root',str(Path.home()/'inclab-nas/DeepAccident'),'--carla-maps',str(d.ROOT/'carla_map'),
            '--predictor',str(d.ROOT/'out/predict_model_joint_scene_v2/joint_scene_motionnet_v2_best.pt'),
            '--device',os.environ.get('V4_DIAGNOSTIC_DEVICE','cuda'),'--workers','1','--out',str(tmp_path)])
    assert all((tmp_path/n).is_file() for n in d.OUTPUTS)
    assert len(d.geometry_audit.read_csv(tmp_path/d.OUTPUTS[0]))==219
    assert len(d.geometry_audit.read_csv(tmp_path/d.OUTPUTS[1]))==166
    assert len(d.geometry_audit.read_csv(tmp_path/d.OUTPUTS[2]))==55


def test_archived_scores_reproduce_all_219_window_or_decisions(frozen_data):
    """OR-only regression, not a substitute for exact fresh candidate replay.

    For the 33 censored bucket-2 collections, the global-cap archive retains
    enough accepted candidates to check the OR decision, but can omit pairs.
    The production diagnostic never uses this archive fallback.
    """
    ledger,reference,models,pools = frozen_data
    with (d.ROOT/'out/pair_reranker_v4/val_full5s.jsonl').open() as f:
        full = {(r['scenario'],float(r['window'].split('-')[-1])):r['candidates'] for r in map(json.loads,f)}
    counts = d.Counter()
    for wid,old in reference.items():
        key = old['scenario_id'],old['t_end_s']
        cs = {1:pools[1][key],2:pools[2].get(key,[r for r in full[key] if r['interval_index']==2])}
        gt = truth(int(ledger[wid]['gt_bucket']))
        record = d.frozen.evaluate_window(gt,cs,models)
        assert record['predicted']==old['predicted']
        assert record['actual']==old['actual']
        counts[d.classify(record['actual'],record['predicted'])] += 1
    assert dict(counts)==dict(TP=46,FP=55,TN=111,FN=7)


def test_isolated_raw_geometry_hole_and_truncation_are_unknown():
    from examples.audit_gt_carla_boxes import WorldBox
    actors = {a.actor_id:a for a in fixture_window().last.actors}
    for a in actors.values():
        a.predictions = [PredictedPath('GT fixture',1.,[a.world_xy,a.world_xy],5.,waypoint_times_s=[0.,5.])]
    frames = {f:{cid:WorldBox(cid,(0.,0.),0.,(1.,0.),2.,1.,1.) for cid in (1,2)} for f in range(1,102)}
    identities = {'V001':{1},'V002':{2}}
    full = d.selected_geometry(actors,('V001','V002'),identities,frames,5.,5.)
    assert full['contact_0_2'] is False and full['contact_2_5'] is False
    del frames[52]
    missing = d.selected_geometry(actors,('V001','V002'),identities,frames,5.,5.)
    assert missing['contact_0_2'] is None and not missing['geometry_complete_0_2']
    assert missing['min_clearance_0_2_m'] is not None
    assert missing['geometry_complete_2_5']
    for a in actors.values():
        a.predictions[0].waypoint_times_s = [0.,2.]
    truncated = d.selected_geometry(actors,('V001','V002'),identities,frames,5.,5.)
    assert truncated['contact_2_5'] is None and not truncated['geometry_complete_2_5']


def test_summary_exports_counts_mapping_and_unknown_geometry(tmp_path,frozen_data):
    """Synthetic one-candidate fixture tests summary/export plumbing only."""
    ledger,reference,models,pools = frozen_data
    with (d.ROOT/'out/pair_reranker_v4/val_full5s.jsonl').open() as f:
        full = {(r['scenario'],float(r['window'].split('-')[-1])):r['candidates'] for r in map(json.loads,f)}
    windows,candidates,raw,norm,accepted = [],[],[],[],[]
    censored_window_id = None
    for wid,old in reference.items():
        key = old['scenario_id'],old['t_end_s']
        all_cs = {1:pools[1][key],2:pools[2].get(key,[r for r in full[key] if r['interval_index']==2])}
        # Keep a single genuine scored candidate where there is one; exact
        # full-pool extraction is covered by separate tests and raw replay.
        single = {1:[],2:[]}
        chosen_bucket = None
        for k in (1,2):
            found = next((r for r in all_cs[k] if models[k].predict_features(r['features'])>=models[k].threshold),None)
            if found:
                single[k] = [found]; chosen_bucket=k; break
        window = fixture_window()
        window.index = int(wid.rsplit(':',2)[1]); window.t_end=old['t_end_s']
        evidence = {k:[pair(c['actor_a'],c['actor_b'],k) for c in single[k]] for k in (1,2)}
        gt = truth(int(ledger[wid]['gt_bucket']))
        if not int(ledger[wid]['gt_bucket']) and old['predicted'] and censored_window_id is None:
            gt['expected'][2]['scorable'] = False
            censored_window_id = wid
        _,w,cs,rs,ns = d.diagnostic_window(window,old['scenario_id'],gt,single,evidence,models)
        assert w['status']==d.classify(old['actual'],old['predicted'])
        # For this plumbing fixture, use the frozen reference identity flags.
        w['correct_gt_pair_hit']=old['correct_pair_hit']; w['gt_pair_candidate_covered']=old['gt_pair_covered']
        windows.append(w); candidates.extend(cs); raw.extend(rs); norm.extend(ns)
        for c in cs:
            if c['status']=='FP' and c['accepted']:
                a = dict(c)
                a.update(gt_min_clearance_0_2_m=None,gt_min_clearance_0_2_time_s=None,gt_contact_0_2=None,
                         gt_min_clearance_2_5_m=None,gt_min_clearance_2_5_time_s=None,gt_contact_2_5=None,
                         gt_first_contact_time_s=None,geometry_complete_0_2=False,geometry_complete_2_5=False,
                         gt_contact_event='GEOMETRY_UNCERTAIN',accepted_pair_matches_later_gt_pair=c['matches_later_gt_pair'])
                accepted.append(a)
    negative,tp,mechanisms = d.summaries(windows,candidates,accepted)
    assert negative['counts']==dict(FP=55,TN=111)
    assert mechanisms['sanity_counts']==d.EXPECTED
    outcome = mechanisms['gt_physical_outcome']
    fp_windows = [w for w in windows if w['status']=='FP']
    assert 'no_recorded_collision_within_5s' not in outcome
    assert outcome['later_2_5_censored_or_unknown']==1
    assert outcome['no_collision_through_5s_observed']==sum(w['gt_bucket']==0 for w in fp_windows)-1
    assert outcome['collision_after_2s']==sum(w['gt_bucket'] in (3,4,5) for w in fp_windows)
    assert sum(outcome[k] for k in ('no_collision_through_5s_observed','later_2_5_censored_or_unknown','collision_after_2s'))==55
    assert 'later_collision_with_same_accepted_pair' in outcome
    assert 'later_collision_with_different_pair' in outcome
    assert mechanisms['oracle_geometry']['incomplete_geometry_windows']==55
    assert mechanisms['oracle_geometry']['no_observed_contact_complete_geometry_windows']==0
    assert len(negative['top_candidate_features_by_bucket']['1'])==len(d.FEATURE_NAMES_V4)
    assert len(tp['features'])==len(d.FEATURE_NAMES_V4)
    assert len(d.compact_answers(negative,tp,mechanisms))==7
    d.write_outputs(tmp_path,windows,candidates,accepted,raw,norm,negative,tp,mechanisms)
    assert len(d.geometry_audit.read_csv(tmp_path/d.OUTPUTS[0]))==219
    assert len(d.geometry_audit.read_csv(tmp_path/d.OUTPUTS[1]))==166
    assert len(d.geometry_audit.read_csv(tmp_path/d.OUTPUTS[2]))==55
    with np.load(tmp_path/d.OUTPUTS[5],allow_pickle=False) as data:
        np.testing.assert_array_equal(data['feature_names'],d.FEATURE_NAMES_V4)
        np.testing.assert_array_equal(data['raw_features'],raw)
        np.testing.assert_array_equal(data['normalized_features'],norm)
        for i,c in enumerate(candidates):
            for field in ('window_id','bucket','actor_a','actor_b','status','accepted','verifier_score','gt_pair_label_0_2'):
                assert data[field][i]==c[field]
            m = models[c['bucket']]
            np.testing.assert_array_equal(data['normalized_features'][i],(data['raw_features'][i]-m.means)/m.scales)
    with pytest.raises(ValueError,match='Outputs exist'):
        d.write_outputs(tmp_path,windows,candidates,accepted,raw,norm,negative,tp,mechanisms)


def test_stationary_t0_only_gt_path_is_not_extrapolated():
    from examples.audit_gt_carla_boxes import WorldBox
    a,b = actor('V001',0.),actor('V002',10.)
    for value in (a,b):
        value.speed_mps = 0.
        value.predictions = [PredictedPath('GT with no future',1.,[value.world_xy],5.,
                                         waypoint_times_s=[0.],truncated=True)]
    frames = {f:{cid:WorldBox(cid,(0.,0.),0.,(1.,0.),2.,1.,1.) for cid in (1,2)} for f in range(1,102)}
    g = d.selected_geometry({'V001':a,'V002':b},('V001','V002'),{'V001':{1},'V002':{2}},frames,5.,5.)
    assert g['min_clearance_0_2_m'] is None and g['contact_0_2'] is None
    assert g['contact_2_5'] is None and not g['geometry_complete_2_5']
    assert len(a.predictions[0].waypoints)==1  # No mutation of the GT reference.


def test_wrong_threshold_rejected_before_replay(tmp_path):
    models = d.frozen.load_frozen_models(d.frozen.DEFAULT_BUCKET1_MODEL,d.frozen.DEFAULT_BUCKET2_MODEL)
    models = copy.deepcopy(models)
    models[1].threshold += .001  # Test-only copy; frozen JSON stays untouched.
    with patch.object(d.frozen,'load_frozen_models',return_value=models), \
         patch.object(d.frozen,'DeepAccidentRunner') as runner, \
         pytest.raises(ValueError,match='Thresholds differ'):
        d.main(['--root','unused','--carla-maps','unused','--predictor','unused','--out',str(tmp_path)])
    runner.assert_not_called()
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize('bucket,unscorable,missing,fully_scorable,safe',[
    (0,None,None,True,True),
    (0,3,None,False,False),(0,4,None,False,False),(0,5,None,False,False),
    (0,None,3,False,False),(0,None,4,False,False),(0,None,5,False,False),
    (3,None,None,True,False),(4,None,None,True,False),(5,None,None,True,False),
    (3,4,None,False,False),
])
def test_later_observability_does_not_change_inference(frozen_data,bucket,unscorable,missing,fully_scorable,safe):
    _,_,models,_ = frozen_data
    window = fixture_window()
    raw = d.frozen.pair_features_v4(*window.last.actors,pair(),5.,[])
    candidates = {1:[dict(actor_a='V001',actor_b='V002',features=raw,predicted_contact=False)],2:[]}
    evidence = {1:[pair()],2:[]}
    baseline = d.diagnostic_window(window,'scene',truth(bucket),candidates,evidence,models)
    gt = truth(bucket)
    for r in gt['expected']:
        if r['k']==unscorable:
            r['scorable']=False
    gt['expected'] = [r for r in gt['expected'] if r['k']!=missing]
    actual = d.diagnostic_window(window,'scene',gt,candidates,evidence,models)
    decision,w,rows,features,normalized = actual
    assert w['later_2_5_fully_scorable'] is fully_scorable
    assert w['no_collision_through_5s_observed'] is safe
    assert w['later_collision'] is (bucket in (3,4,5))
    # These are diagnostic fields only. Decisions, candidates, scores, labels
    # and raw/normalized feature vectors remain byte/numerically identical.
    assert json.dumps(decision)==json.dumps(baseline[0])
    assert json.dumps(rows)==json.dumps(baseline[2])
    np.testing.assert_array_equal(features,baseline[3])
    np.testing.assert_array_equal(normalized,baseline[4])


def test_serial_cuda_main_constructs_one_predictor_and_hashes_oracle_sources(tmp_path,frozen_data):
    from contextlib import ExitStack
    ledger,reference,models,_ = frozen_data
    scenarios = [SimpleNamespace(scenario_id=f's{i:03d}',split='val') for i in range(104)]
    runner = SimpleNamespace(list_scenarios=lambda:list(reversed(scenarios)))
    predictor_path = tmp_path/'predictor.pt'
    predictor_path.write_bytes(b'frozen predictor fixture')
    windows = [dict(window_id=wid,gt_bucket=int(ledger[wid]['gt_bucket'])) for wid in reference]
    def replay(scenario,args,passed_models,**kwargs):
        assert args.workers==1 and args.device=='cuda:0'
        assert kwargs['runner'] is runner
        assert kwargs['cfg'].predictor is predictor
        assert passed_models is models
        return (list(reference.values()),windows,[],[],[]) if scenario.scenario_id=='s000' else ([],[],[],[],[])
    predictor = object()
    with ExitStack() as stack:
        stack.enter_context(patch.object(d.frozen,'load_frozen_models',return_value=models))
        construct_predictor = stack.enter_context(patch.object(d.frozen,'JointSceneTorchPredictor',return_value=predictor))
        construct_runner = stack.enter_context(patch.object(d.frozen,'DeepAccidentRunner',return_value=runner))
        replay_mock = stack.enter_context(patch.object(d,'replay_scenario',side_effect=replay))
        stack.enter_context(patch.object(d,'summaries',return_value=({}, {}, {})))
        stack.enter_context(patch.object(d,'compact_answers',return_value=[]))
        write = stack.enter_context(patch.object(d,'write_outputs'))
        stack.enter_context(patch('builtins.print'))
        d.main(['--root','data','--carla-maps','maps','--predictor',str(predictor_path),
                '--device','cuda:0','--out',str(tmp_path/'out')])
    construct_predictor.assert_called_once_with(str(predictor_path),device='cuda:0')
    assert construct_runner.call_count==1 and replay_mock.call_count==104
    assert [call.args[0].scenario_id for call in replay_mock.call_args_list]==[s.scenario_id for s in scenarios]
    mechanisms = write.call_args.args[-1]
    paths = [d.COHORT,d.FROZEN_ROWS,predictor_path,d.frozen.DEFAULT_BUCKET1_MODEL,d.frozen.DEFAULT_BUCKET2_MODEL,
             Path(d.frozen.__file__),Path(d.__file__),Path(d.gt_replay.__file__),Path(d.geometry_audit.__file__),
             Path(inspect.getsourcefile(d._shared_geometry))]
    assert mechanisms['provenance']=={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert mechanisms['frozen_thresholds']==d.EXPECTED_THRESHOLDS


def test_shared_and_local_runner_replay_are_identical(frozen_data):
    _,_,models,_ = frozen_data
    window = fixture_window()
    snapshot = window.last
    snapshot.t=window.t_end
    config = d.frozen.PipelineConfig()
    config.deepaccident.observation_mode='sensor3d'
    config.predictor=object()
    runner = SimpleNamespace(build=lambda *a,**kw:SimpleNamespace(snapshots=lambda **kw:[snapshot]))
    scenario = SimpleNamespace(scenario_id='scene',scenario='scene',scenario_type='type',town='Town01',split='val',agents=[])
    args = SimpleNamespace(root='data',carla_maps='maps',predictor='v2.pt',device='cuda')
    with patch.object(d.frozen,'JointSceneTorchPredictor',return_value=object()) as predictor, \
         patch.object(d.frozen,'DeepAccidentRunner',return_value=runner), \
         patch.object(d.frozen,'find_xodr',return_value=None), \
         patch.object(d.frozen,'estimate_collision',return_value=None), \
         patch.object(d.frozen,'build_windows',return_value=([window],None)), \
         patch.object(d.frozen,'window_ground_truth',return_value=truth()), \
         patch.object(d.frozen,'rank_actors',return_value=(snapshot.actors,None)), \
         patch.object(d.frozen,'swept_pair_clearances',return_value=[pair()]), \
         patch('builtins.print'):
        shared = d.replay_scenario(scenario,args,models,runner=runner,cfg=config)
        predictor.assert_not_called()
        local = d.replay_scenario(scenario,args,models)
        predictor.assert_called_once_with('v2.pt',device='cuda')
    assert json.dumps(shared)==json.dumps(local)
    np.testing.assert_array_equal(shared[3],local[3])
    np.testing.assert_array_equal(shared[4],local[4])
