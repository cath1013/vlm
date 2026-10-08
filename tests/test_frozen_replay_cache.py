"""Validated replay cache tests; no inference, reconstruction, training or HTTP."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from examples import evaluate_frozen_models_api_val104 as e
from examples import train_collision_risk_net as collector


@pytest.fixture
def replay_env(tmp_path, monkeypatch):
    project = tmp_path/'project'
    run = project/'run'
    run.mkdir(parents=True)
    cohort_path = run/'cohort.csv'
    monkeypatch.setattr(e, 'ROOT', project)
    monkeypatch.setattr(e, 'RUN', run)
    monkeypatch.setattr(e, 'COHORT', cohort_path)
    monkeypatch.setattr(e.subprocess, 'check_output', lambda *a,**k:'test-commit\n')
    for name in ('evaluate_pair_reranker_v4_0_2.py', 'diagnose_crn4_false_positives.py',
                 'train_collision_risk_net.py', 'audit_gt_carla_boxes.py', 'plot_v4_crn4_error_cases.py',
                 'audit_gt_xy_source_ablation.py', 'audit_gt_heading_source_ablation.py',
                 'audit_predictor_trajectory_contacts.py'):
        path = project/'examples'/name
        path.parent.mkdir(exist_ok=True)
        path.write_text('# replay dependency\n')
    code = project/'traffic_llm'/'collision_risk_net.py'
    code.parent.mkdir()
    code.write_text('# network dependency\n')
    checkpoint = run/'crn4'/'collision_risk_final.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'frozen checkpoint')
    monkeypatch.setattr(e, 'FROZEN_CRN4_SHA256', e.file_sha256(checkpoint))
    predictor = project/'out/predict_model_joint_scene_v2/joint_scene_motionnet_v2_best.pt'
    predictor.parent.mkdir(parents=True)
    predictor.write_bytes(b'frozen encoder')
    for k in (1,2):
        path = project/f'out/pair_reranker_v4/model_bucket{k}/pair_reranker_v4.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'decision_threshold':.5}))
    data = tmp_path/'dataset'
    scenario_type = data/'val'/'type1_subtype1_accident'
    scenario = 'Town01_type001_subtype0001_scenario00001'
    for directory, filename in (('meta', scenario+'.txt'),
                                ('ego_vehicle/calib/'+scenario, 'frame.pkl'),
                                ('ego_vehicle/label/'+scenario, 'frame.txt')):
        path = scenario_type/directory/filename
        path.parent.mkdir(parents=True)
        path.write_bytes(b'original dataset input')
    maps = tmp_path/'maps'
    maps.mkdir()
    (maps/'Town01.xodr').write_text('<OpenDRIVE/>')
    ledger, cases, rows = {}, {}, []
    sid = 'type1_subtype1_accident/'+scenario
    for i in range(219):
        wid = f'{sid}:{i}:5.000'
        positive = i < 53
        bucket = 1 if positive else 0
        v4_decision = i < 46 if positive else i < 108
        crn_decision = i < 41 if positive else i < 93
        hazard = .2 if crn_decision else .01
        risk = 1-(1-hazard)**2
        pairs = [['a','b']] if v4_decision else []
        accepted = [dict(actor_a='a',actor_b='b',k=1,verifier_score=.8,
                         future_coordinates={'a':[], 'b':[]},predicted_minimum_footprint_clearance_m=-.1,
                         time_of_predicted_minimum_clearance_s=1.,predicted_contact=True,
                         first_predicted_contact_s=1.,current_clearance_m=2.,contact_duration_s=.1,
                         contact_event='NEW',contact_at_observation=False,
                         contact_before_observation=False)] if v4_decision else []
        ledger[wid] = dict(window_id=wid,scenario_id=sid,gt_bucket=str(bucket),
            v4_predicted=str(v4_decision),crn4_predicted=str(crn_decision),
            v4_correct_pair_hit=str(positive and v4_decision),
            crn4_correct_pair_hit=str(positive and crn_decision),
            v4_bucket1_prediction=str(v4_decision),v4_bucket2_prediction='False',
            v4_selected_pairs=json.dumps(pairs),crn4_selected_pair="('a', 'b')",
            crn4_selected_pair_gt_label=str(bucket), crn4_risk_2s=str(risk),crn4_threshold_2s='.3')
        actors = []
        for a in ('a','b'):
            actor = {key:None for key in ('actor_id kind cls world_xy heading_deg speed_mps accel_mps2 '
                'maneuver observed_by confidence position_quality track_age_s heading_source observed_range_m '
                'source_track_ids placement footprint_length_width_m observed_history').split()}
            actor.update(actor_id=a,observed_history=[[-1,0,0],[0,1,1]])
            actors.append(actor)
        cases[wid] = dict(observation=dict(observation_window=dict(t_start_s=0.,t_end_s=5.,history_s=5,
            time_reference='history offsets relative to t_end'),actors=actors,
            map_conventions=dict(coordinates='local ENU metres',heading='clockwise from north degrees',
                                 drive_side='right',lane_numbering='from_median',lane_width_m=3.5)),
            gt_bucket=bucket,gt_groups=[['a'],['b']] if positive else [],
            v4=dict(decision=v4_decision,pairs=pairs,
                    evidence=dict(source_type='trajectory_pair_verifier',accepted_interactions=accepted)),
            crn4=dict(decision=crn_decision,pairs=[['a','b']],
                evidence=dict(source_type='observation_pair_risk',selected_interaction=dict(
                    actor_a='a',actor_b='b',hazard_1=hazard,hazard_2=hazard,
                    event_probability_1=hazard,event_probability_2=(1-hazard)*hazard,
                    cumulative_risk_1=hazard,cumulative_risk_2=risk))))
        rows.append(dict(window_id=wid,gt_bucket=bucket,actors=[{'actor_id':'a'},{'actor_id':'b'}],
                         pair_labels=[bucket]))
    cohort_path.write_text(e.canonical(ledger))
    (run/'val104.jsonl').write_text('\n'.join(e.canonical(r) for r in rows)+'\n')
    (run/'val104.manifest.json').write_text('{}')
    monkeypatch.setattr(collector, 'verify_manifest', lambda *a,**k:None)
    monkeypatch.setattr(collector, 'read_rows', lambda *a:copy.deepcopy(rows))
    monkeypatch.setattr(e, 'load_cohort', lambda:copy.deepcopy(ledger))
    import torch
    monkeypatch.setattr(torch, 'load', lambda *a,**k:{'v2_checkpoint':str(predictor),
                                                   'max_horizon':4,'threshold_by_horizon':{'2':.3}})
    args = SimpleNamespace(root=str(data),carla_maps=str(maps),device='cpu',batch_size=16,
                           replay_cache=tmp_path/'cache.json')
    sources = e.replay_source_fingerprints(args,ledger)
    hashes = {p:h for p,h in sources['artifacts'].items() if p != str(run/'val104.manifest.json')}
    provenance = dict(network_checkpoint_SHA256=hashes,
                      frozen_thresholds=dict(v4={'1':.5,'2':.5},crn4={'2':.3}))
    e.assert_networks(cases,ledger)
    return SimpleNamespace(args=args,ledger=ledger,cases=cases,provenance=provenance,rows=rows,
                           project=project,run=run,data=data,maps=maps)


def create_cache(env):
    with patch.object(e,'reproduce',return_value=(copy.deepcopy(env.cases),copy.deepcopy(env.provenance))) as replay:
        result = e.cached_replay(env.args,env.ledger)
    replay.assert_called_once()
    return result


def rewrite_cache(env, mutation, rehash=False):
    document = json.loads(env.args.replay_cache.read_text())
    mutation(document)
    if rehash:
        content = {k:v for k,v in document.items() if k != 'content_sha256'}
        document['content_sha256'] = e.digest(e.canonical(content))
    env.args.replay_cache.write_text(e.canonical(document))


def test_cache_creation_and_reuse_without_replay(replay_env):
    env = replay_env
    env.cases[next(iter(env.cases))]['gt'] = object()
    cases, provenance, info = create_cache(env)
    assert len(cases) == 219 and all(set(c) == e.REPLAY_CASE_KEYS for c in cases.values())
    assert provenance == env.provenance
    assert info['evidence_source'] == 'fresh_replay'
    original = env.args.replay_cache.read_bytes()
    with patch.object(e,'reproduce',side_effect=AssertionError('must reuse cache')) as replay:
        loaded, loaded_provenance, loaded_info = e.cached_replay(env.args,env.ledger)
    replay.assert_not_called()
    assert loaded == cases and loaded_provenance == provenance
    assert loaded_info['evidence_source'] == 'validated_cache'
    assert loaded_info['replay_cache_sha256'] == info['replay_cache_sha256']
    assert env.args.replay_cache.read_bytes() == original
    assert not list(env.args.replay_cache.parent.glob('cache.json.*.tmp'))


@pytest.mark.parametrize('artifact', ['checkpoint','collector','cohort','code','dataset','map','manifest'])
def test_stale_artifacts_rejected_without_regeneration(replay_env, artifact):
    env = replay_env
    create_cache(env)
    original = env.args.replay_cache.read_bytes()
    paths = dict(checkpoint=env.run/'crn4/collision_risk_final.pt',collector=env.run/'val104.jsonl',
                 cohort=e.COHORT,code=env.project/'traffic_llm/collision_risk_net.py',
                 dataset=next(env.data.rglob('*.pkl')),map=env.maps/'Town01.xodr',
                 manifest=env.run/'val104.manifest.json')
    with paths[artifact].open('ab') as f:
        f.write(b'changed')
    with patch.object(e,'reproduce') as replay:
        with pytest.raises(ValueError,match='checkpoint changed|Stale replay cache'):
            e.cached_replay(env.args,env.ledger)
    replay.assert_not_called()
    assert env.args.replay_cache.read_bytes() == original


def test_content_tampering_rejected(replay_env):
    env = replay_env
    create_cache(env)
    rewrite_cache(env,lambda doc:doc['cases'][0].update(gt_bucket=5))
    corrupted = env.args.replay_cache.read_bytes()
    with patch.object(e,'reproduce') as replay:
        with pytest.raises(ValueError,match='content integrity'):
            e.cached_replay(env.args,env.ledger)
    replay.assert_not_called()
    assert env.args.replay_cache.read_bytes() == corrupted


def test_changed_noncohort_scenario_resolution_rejected(replay_env):
    env = replay_env
    base = env.data/'val'/'other_type'
    (base/'meta').mkdir(parents=True)
    (base/'meta'/'Town01_unused.txt').write_text('unchanged metadata')
    create_cache(env)
    for kind,filename in [('calib','frame.pkl'),('label','frame.txt')]:
        directory = base/'ego_vehicle'/kind/'Town01_unused'
        directory.mkdir(parents=True)
        (directory/filename).write_text('newly resolvable scenario')
    with patch.object(e,'reproduce') as replay:
        with pytest.raises(ValueError,match='Stale replay cache'):
            e.cached_replay(env.args,env.ledger)
    replay.assert_not_called()


@pytest.mark.parametrize('mutation,error', [
    (lambda doc:doc['cases'].pop(), '219 unique'),
    (lambda doc:doc['cases'].__setitem__(1,doc['cases'][0]), '219 unique'),
    (lambda doc:doc['cases'][0]['crn4'].update(decision=False), 'confusion mismatch'),
    (lambda doc:doc['cases'][0].update(gt_bucket=5), 'confusion mismatch|GT mismatch'),
    (lambda doc:doc['cases'][0]['observation'].update(gt_bucket=1), 'Forbidden key'),
    (lambda doc:doc['cases'][0]['crn4'].update(pairs=[['a','unknown']]), 'Invalid cached selected pair'),
    (lambda doc:doc['cases'][0].update(gt_groups=[['a','b'],[]]), 'GT-pair membership mismatch'),
    (lambda doc:doc.update(version=999), 'Incompatible'),
    (lambda doc:doc.update(version=True), 'Incompatible'),
])
def test_rehashed_invalid_cache_still_fails_semantic_audits(replay_env, mutation, error):
    env = replay_env
    create_cache(env)
    rewrite_cache(env,mutation,rehash=True)
    with patch.object(e,'reproduce') as replay:
        with pytest.raises(ValueError,match=error):
            e.cached_replay(env.args,env.ledger)
    replay.assert_not_called()


def test_cache_reusable_across_prompt_and_api_settings(replay_env, monkeypatch):
    env = replay_env
    create_cache(env)
    monkeypatch.setattr(e,'PROMPT','New generic prompt')
    monkeypatch.setattr(e,'CRN4_PROMPT_V2','New CRN prompt')
    env.args.model = 'another-model'
    env.args.temperature = 2.
    env.args.max_output_tokens = 123
    env.args.provider = 'gemini'
    with patch.object(e,'reproduce',side_effect=AssertionError('must reuse')):
        assert e.cached_replay(env.args,env.ledger)[2]['evidence_source'] == 'validated_cache'


@pytest.mark.parametrize('change,valid', [('prompt',True), ('replay',False)])
def test_source_fingerprint_distinguishes_prompt_from_replay_edits(replay_env, monkeypatch, change, valid):
    env = replay_env
    create_cache(env)
    source = Path(e.__file__).read_text()
    if change == 'prompt':
        source = source.replace('You are an expert in cooperative-driving (V2X) accident prediction.',
                                'A different prompt for another experiment.')
    else:
        source = source.replace("history offsets relative to t_end", "changed replay time reference")
    revised = env.project/'revised_evaluator.py'
    revised.write_text(source)
    monkeypatch.setattr(e,'__file__',str(revised))
    with patch.object(e,'reproduce') as replay:
        if valid:
            assert e.cached_replay(env.args,env.ledger)[2]['evidence_source'] == 'validated_cache'
        else:
            with pytest.raises(ValueError,match='Stale replay cache'):
                e.cached_replay(env.args,env.ledger)
    replay.assert_not_called()


def test_truncated_cache_rejected_without_replacement(replay_env):
    env = replay_env
    env.args.replay_cache.write_text('{"version":')
    with patch.object(e,'reproduce') as replay:
        with pytest.raises(json.JSONDecodeError):
            e.cached_replay(env.args,env.ledger)
    replay.assert_not_called()
    assert env.args.replay_cache.read_text() == '{"version":'


def test_subset_and_resume_use_full_validated_cache(replay_env, tmp_path):
    env = replay_env
    create_cache(env)
    ids = list(env.ledger)[:2]
    subset = tmp_path/'ids.txt'
    subset.write_text('\n'.join(ids)+'\n')
    out = tmp_path/'experiment'
    argv = ['--model','test','--condition','crn4','--out',str(out),'--dry-run',
            '--root',env.args.root,'--carla-maps',env.args.carla_maps,
            '--replay-cache',str(env.args.replay_cache),'--window-ids-file',str(subset)]
    with patch.object(e,'reproduce',side_effect=AssertionError('no replay')) as replay, patch.object(e,'call_api') as api:
        e.main(argv)
        definitions = (out/'requests.jsonl').read_bytes()
        e.main(argv+['--resume'])
    replay.assert_not_called()
    api.assert_not_called()
    assert (out/'requests.jsonl').read_bytes() == definitions
    assert len(definitions.splitlines()) == 2
    manifest = json.loads((out/'experiment_manifest.json').read_text())
    assert manifest['evidence_source'] == 'validated_cache'
    assert manifest['selected_window_ids'] == ids
    assert manifest['replay_cache_sha256'] == e.file_sha256(env.args.replay_cache)
    assert manifest['prompt_hashes'] == {'crn4':e.digest(e.CRN4_PROMPT_V2)}
    audit = json.loads((out/'dry_run_audit.json').read_text())
    assert audit['windows'] == 219 and audit['evidence_source'] == 'validated_cache'
    assert (out/'diagnostic_subset_scores.json').exists()


def test_cache_creation_then_resume_records_original_and_current_source(replay_env, tmp_path):
    env = replay_env
    out = tmp_path/'experiment'
    argv = ['--model','test','--out',str(out),'--dry-run', '--root',env.args.root,
            '--carla-maps',env.args.carla_maps,'--replay-cache',str(env.args.replay_cache)]
    with patch.object(e,'reproduce',return_value=(env.cases,env.provenance)) as replay, patch.object(e,'call_api') as api:
        e.main(argv)
        original = (out/'experiment_manifest.json').read_bytes()
        e.main(argv+['--resume'])
    replay.assert_called_once()
    api.assert_not_called()
    assert (out/'experiment_manifest.json').read_bytes() == original
    assert json.loads(original)['evidence_source'] == 'fresh_replay'
    assert json.loads((out/'dry_run_audit.json').read_text())['evidence_source'] == 'validated_cache'


def test_atomic_cache_publication_never_overwrites_existing_file(tmp_path):
    path = tmp_path/'cache.json'
    path.write_text('existing invalid cache')
    with pytest.raises(FileExistsError):
        e.save_replay_cache(path,{'replacement':True})
    assert path.read_text() == 'existing invalid cache'


@pytest.mark.parametrize('filename', ['scores.json','diagnostic_subset_scores.json',
                                      'dry_run_audit.json','records.jsonl.tmp','raw_responses/cache.json'])
def test_cache_path_cannot_collide_with_experiment_outputs(tmp_path, filename):
    out = tmp_path/'experiment'
    cache = out/filename
    with patch.object(e,'reproduce') as replay, patch.object(e,'call_api') as api:
        with pytest.raises(ValueError,match='cache path conflicts'):
            e.main(['--model','test','--out',str(out),'--replay-cache',str(cache),'--dry-run'])
    replay.assert_not_called()
    api.assert_not_called()
    assert not out.exists()
