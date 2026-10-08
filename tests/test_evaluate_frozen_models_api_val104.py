import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from examples import evaluate_frozen_models_api_val104 as e


def response(ks=(), ids=('a','b')):
    return dict(predictions=[dict(k=k, interval_s={1:'(0,1]',2:'(1,2]'}[k],
        accident_expected=k in ks, involved_actor_ids=list(ids) if k in ks else [],
        reason='The vehicles overlap.' if k in ks else 'The vehicles remain separated.', confidence='medium')
        for k in (1,2)], overall_assessment='Assessment', data_limitations='Limited')


def case(bucket=1, network=True):
    return dict(gt_bucket=bucket, gt_groups=[['a','alias'],['b']] if bucket in (1,2) else [],
        v4=dict(decision=network, pairs=[], evidence=dict(source_type='trajectory_pair_verifier', accepted_interactions=[])),
        crn4=dict(decision=network, pairs=[['a','b']], evidence=dict(source_type='observation_pair_risk',
            selected_interaction=dict(actor_a='a',actor_b='b', **{k:.2 for k in ('hazard_1','hazard_2',
            'event_probability_1','event_probability_2','cumulative_risk_1','cumulative_risk_2')}))),
        observation=dict(observation_window={}, actors=[dict(actor_id=a,observed_history=[[-5,0,0],[0,1,1]])
            for a in ('a','b')],map_conventions={}))


def test_cohort_exact_and_duplicate_rejected(tmp_path):
    ledger=e.load_cohort()
    assert len(ledger)==219
    path=tmp_path/'bad.csv'
    path.write_text('window_id\nx\nx\n')
    with pytest.raises(ValueError): e.load_cohort(path)


@pytest.mark.parametrize('condition', ['v4','crn4'])
def test_frozen_aggregate_gate(condition):
    ledger=e.load_cohort()
    cases={wid:dict(gt_bucket=int(r['gt_bucket']), **{c:dict(decision=r[c+'_predicted']=='True')
        for c in ('v4','crn4')}) for wid,r in ledger.items()}
    assert e.assert_networks(cases,ledger)==e.EXPECTED
    wid=next(iter(cases))
    cases[wid][condition]['decision']=not cases[wid][condition]['decision']
    with pytest.raises(ValueError,match='confusion mismatch'): e.assert_networks(cases,ledger)


def test_common_prompt_schema_and_payloads():
    cases={f'w{i}':case() for i in range(219)}
    args=SimpleNamespace(condition='both',provider='gemini',model='test',temperature=0.,
        thinking_config=None,max_output_tokens=100,seed=42)
    requests=e.prepare(cases,args)
    assert len(requests)==438
    for wid in cases:
        pair=[r for r in requests if r['window_id']==wid]
        assert pair[0]['observation_hash']==pair[1]['observation_hash']
        for r in pair:
            expected = e.CRN4_PROMPT_V2 if r['condition'] == 'crn4' else e.PROMPT
            assert r['body']['systemInstruction']['parts'][0]['text'] == expected
            if r['condition'] == 'crn4':
                assert r['selected_pair'] == ['a','b']
        assert pair[0]['body']['generationConfig']==pair[1]['body']['generationConfig']
        common=[json.loads(r['body']['contents'][0]['parts'][0]['text'])['observation'] for r in pair]
        assert e.canonical(common[0])==e.canonical(common[1])
    assert e.prepare(cases,args)==requests


@pytest.mark.parametrize('key', ['gt_bucket','threshold','checkpoint','comparison_group','later_collision'])
def test_no_leakage(key):
    c=case()
    payload=dict(observation=c['observation'], network_evidence=c['v4']['evidence'])
    payload[key]='secret'
    with pytest.raises(ValueError): e.audit_payload(payload,'v4',c['v4'])


def test_crn_no_trajectory_and_actor_membership():
    c=case()
    p=dict(observation=c['observation'],network_evidence=c['crn4']['evidence'])
    e.audit_payload(p,'crn4',c['crn4'])
    p['network_evidence']['selected_interaction']['future_path']=[]
    with pytest.raises(ValueError): e.audit_payload(p,'crn4',c['crn4'])
    del p['network_evidence']['selected_interaction']['future_path']
    p['network_evidence']['selected_interaction']['actor_a']='missing'
    with pytest.raises(ValueError): e.audit_payload(p,'crn4',c['crn4'])


def test_v4_only_accepted_geometry():
    c=case()
    p=dict(observation=c['observation'],network_evidence=c['v4']['evidence'])
    p['network_evidence']['accepted_interactions']=[dict(actor_a='a',actor_b='b',k=1,future_coordinates={'a':[], 'b':[]})]
    with pytest.raises(ValueError): e.audit_payload(p,'v4',c['v4'])
    c['v4']['pairs']=[['a','b']]
    e.audit_payload(p,'v4',c['v4'])
    p['network_evidence']['accepted_interactions'][0]['future_coordinates']['unaccepted']=[]
    with pytest.raises(ValueError): e.audit_payload(p,'v4',c['v4'])


def test_binary_timing_actor_grading_and_failure():
    c=case()
    g=e.grade(c,response((2,),('alias','b')))
    assert g['binary_2s_window']=='TP'
    assert not g['strict_bucket']['correct']
    assert g['actor_pair']==dict(exact_gt_pair_hit=True, vehicle_recall=1.,actor_precision=1.)
    assert e.grade(c,response((1,)))['strict_bucket']['correct']
    assert not e.grade(c,response((1,2)))['strict_bucket']['correct']
    assert e.grade(case(0),response())['binary_2s_window']=='TN'
    assert e.grade(case(0),response((1,)))['binary_2s_window']=='FP'
    assert e.grade(c,None)['binary_2s_window'] is None


def test_schema_semantic_buckets():
    e.validate_response(response(),['a','b'])
    r=response(); r['predictions'][1]['k']=1
    with pytest.raises(ValueError): e.validate_response(r,['a','b'])
    r=response(); r['predictions'][1]['interval_s']='(0,1]'
    with pytest.raises(ValueError): e.validate_response(r,['a','b'])


def test_gemini_bucket_schema_keeps_integer_without_numeric_enum():
    body = e.providers.build_request('gemini', system=e.PROMPT, blocks=[e.QUESTION],
        schema=e.schema(), model='test', max_tokens=100, effort='')
    k = body['generationConfig']['responseSchema']['properties']['predictions']['items']['properties']['k']
    assert k['type'] == 'INTEGER'
    assert 'enum' not in k
    assert k['description'] == 'Integer bucket index. Must be 1 or 2.'
    interval_s = body['generationConfig']['responseSchema']['properties']['predictions']['items']['properties']['interval_s']
    assert interval_s['type'] == 'STRING'
    assert interval_s['enum'] == ['(0,1]', '(1,2]']
    assert interval_s['description'] == 'Exact interval label corresponding to k: "(0,1]" for k=1 and "(1,2]" for k=2.'
    e.validate_response(response(), ['a','b'])


@pytest.mark.parametrize('ks', [(), (1,), (2,), (1,1), (2,2), (0,2), (1,3),
                                    (1,2,1), ('1',2), (1,2.0), (True,2)])
def test_response_still_requires_exactly_integer_buckets_one_and_two(ks):
    r = response()
    template = r['predictions'][0]
    r['predictions'] = [dict(template, k=k, interval_s='(1,2]' if k == 2 else '(0,1]')
                        for k in ks]
    with pytest.raises(ValueError):
        e.validate_response(r, ['a','b'])


def test_transitions_all_eight():
    cases={}; results={}
    for bucket in (0,1):
        for n in (False,True):
            for llm in (False,True):
                wid=str((bucket,n,llm)); c=case(bucket,n); cases[wid]=c
                results[wid,'v4']=e.grade(c,response((1,) if llm else ()))
    t=e.transitions(cases,results,'v4')
    assert all(v==1 for k,v in t.items() if ' -> ' in k)
    assert t['fraction_of_network_FPs_filtered_by_LLM']==.5
    assert t['fraction_of_network_TPs_destroyed_by_LLM']==.5
    assert t['network_FNs_recovered_by_LLM']==1


def test_resume_skips_valid_and_retries_failed(tmp_path):
    (tmp_path/'parsed_responses').mkdir(); (tmp_path/'raw_responses').mkdir()
    req=dict(request_id='window__v4',window_id='window',condition='v4',body={})
    saved=dict(request_id=req['request_id'],body_hash=e.digest(e.canonical(req['body'])),
        parsed_response=response(),usage={},latency_s=.1)
    e.write_json(e.response_path(tmp_path,req),saved)
    args=SimpleNamespace(out=tmp_path,resume=True,retries=0,provider='gemini')
    with patch.object(e,'call_api',side_effect=AssertionError('must skip')):
        assert e.execute(args,req,['a','b'],lambda *a:None)[0]==response()
    saved['parsed_response']=None; e.write_json(e.response_path(tmp_path,req),saved)
    with patch.object(e,'call_api',side_effect=ValueError('schema error')) as api:
        assert e.execute(args,req,['a','b'],lambda *a:None)[0] is None
        assert api.call_count==1


def test_dry_run_zero_api_calls(tmp_path, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    ledger=e.load_cohort()
    cases={wid:case(int(r['gt_bucket'])) for wid,r in ledger.items()}
    with patch.object(e,'reproduce',return_value=(cases,{})), patch.object(e,'call_api',side_effect=AssertionError('API forbidden')) as api:
        e.main(['--model','test','--out',str(tmp_path),'--dry-run'])
    assert api.call_count==0
    assert json.loads((tmp_path/'dry_run_audit.json').read_text())['passed']
    assert len((tmp_path/'requests.jsonl').read_text().splitlines())==438
    assert json.loads((tmp_path/'scores.json').read_text())['v4']['scored']==0


def test_real_frozen_reproduction_opt_in(tmp_path):
    """Expensive dataset/checkpoint integration check, explicitly opt in."""
    import os
    if os.environ.get('RUN_FROZEN_VAL104_REPLAY') != '1':
        pytest.skip('Set RUN_FROZEN_VAL104_REPLAY=1 to replay the real val104 networks')
    e.main(['--model','preflight-only','--out',str(tmp_path),'--dry-run'])
    audit=json.loads((tmp_path/'dry_run_audit.json').read_text())
    assert audit['windows']==219 and audit['requests']==438
    assert audit['confusion']==e.EXPECTED
    assert audit['api_calls']==0


def test_preflight_failure_aborts_before_api(tmp_path, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-credential')
    with patch.object(e,'reproduce',side_effect=ValueError('frozen confusion mismatch')), patch.object(e,'call_api') as api:
        with pytest.raises(ValueError,match='confusion mismatch'):
            e.main(['--model','test','--out',str(tmp_path)])
    assert api.call_count==0
    assert json.loads((tmp_path/'dry_run_audit.json').read_text())['passed'] is False


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('key_env', ['GEMINI_API_KEY', 'DIAGNOSTIC_GEMINI_KEY'])
def test_missing_credential_fails_before_frozen_replay(tmp_path, monkeypatch, resume, key_env):
    monkeypatch.delenv(key_env, raising=False)
    out = tmp_path/'run'
    argv = ['--model','test','--out',str(out),'--api-key-env',key_env]
    if resume:
        argv.append('--resume')
    with patch.object(e,'load_cohort') as cohort, patch.object(e,'reproduce') as replay, \
         patch.object(e,'call_api') as api:
        with pytest.raises(ValueError,match=f'Missing {key_env}'):
            e.main(argv)
    cohort.assert_not_called()
    replay.assert_not_called()
    api.assert_not_called()
    assert not out.exists()


def test_observation_excludes_predictions_and_future_history():
    from traffic_llm.schemas import ActorState, PredictedPath
    a=ActorState('a','ego','car',(0.,0.),90.,2.,-.5,
        track_history=[(-6,0,0),(-5,1,0),(0,2,0),(1,3,0)],
        predictions=[PredictedPath('secret',1.,[(0,0),(1,1)],1.)])
    w=SimpleNamespace(last=SimpleNamespace(actors=[a],map_context={}), t_start=-5.,t_end=0.,snapshots=[])
    o=e.observation(w)
    assert o['actors'][0]['observed_history']==[[-5.,1,0],[0.,2,0]]
    assert 'predictions' not in o['actors'][0]
    assert o['actors'][0]['accel_mps2']==-.5


def test_bounded_transient_retry_persists_each_attempt(tmp_path):
    import io
    import urllib.error
    (tmp_path/'parsed_responses').mkdir(); (tmp_path/'raw_responses').mkdir()
    args=SimpleNamespace(out=tmp_path,resume=False,retries=2,provider='gemini',backoff=0.)
    req=dict(request_id='w__crn4',window_id='w',condition='crn4',body={})
    entries=[]
    errors=[urllib.error.HTTPError('url',429,'rate limit',{},io.BytesIO(b'error')) for _ in range(3)]
    with patch.object(e,'call_api',side_effect=errors) as api:
        parsed,_,_=e.execute(args,req,['a','b'],lambda name,row:entries.append((name,row)))
    assert parsed is None and api.call_count==3
    assert len(list((tmp_path/'raw_responses').iterdir()))==3
    assert len([r for name,r in entries if name=='failures.jsonl'])==3
    assert not e.response_path(tmp_path,req).exists()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_unique_log(path, count=438):
    rows = read_jsonl(path)
    assert len(rows) == len({r['request_id'] for r in rows}) == count
    return rows


@pytest.mark.parametrize('first_run', ['live', 'dry_run', 'partial_failure'])
def test_idempotent_request_and_final_record_logs(tmp_path, monkeypatch, first_run):
    """Exercise real persistence/resume with only inference and HTTP mocked."""
    monkeypatch.setenv('GEMINI_API_KEY', 'test-credential')
    cases = {f'w{i}':case() for i in range(219)}
    raw = json.dumps({'candidates':[{'content':{'parts':[{'text':json.dumps(response())}]}}],
                      'usageMetadata':{'promptTokenCount':10,'candidatesTokenCount':20}})
    argv = ['--model','test','--out',str(tmp_path),'--workers','1']
    calls = []
    def api(args, body):
        calls.append(e.digest(e.canonical(body)))
        if first_run == 'partial_failure' and len(calls) == 1:
            raise ValueError('Permanent schema error')
        # Definition snapshots must exist before the first HTTP call.
        if len(calls) == 1:
            assert len(read_jsonl(tmp_path/'requests.jsonl')) == 438
        return 200, raw
    with patch.object(e,'reproduce',return_value=(cases,{})), patch.object(e,'call_api',side_effect=api):
        if first_run == 'partial_failure':
            with pytest.raises(SystemExit,match='1 API requests FAILED/UNSCORED'):
                e.main(argv)
        else:
            e.main(argv + (['--dry-run'] if first_run == 'dry_run' else []))
        definitions = assert_unique_log(tmp_path/'requests.jsonl')
        assert all(set(r) == {'request_id','window_id','condition','body','observation_hash'} |
                   ({'selected_pair'} if r['condition'] == 'crn4' else set()) for r in definitions)
        original_bytes = (tmp_path/'requests.jsonl').read_bytes()
        hashes = {r['request_id']:e.digest(e.canonical(r['body'])) for r in definitions}
        first_records = assert_unique_log(tmp_path/'records.jsonl')
        if first_run == 'dry_run':
            assert not calls
            assert all(r['record_type']=='DRY_RUN' and r['grading']['binary_2s_window'] is None
                       for r in first_records)
            assert (tmp_path/'api_attempts.jsonl').read_text() == ''
        elif first_run == 'partial_failure':
            assert sum(r['grading']['status']=='FAILED' for r in first_records)==1
        else:
            assert all(r['record_type']=='FINAL' and r['grading']['status']=='SCORED' for r in first_records)
        before = len(calls)
        saved_responses = {p:p.read_bytes() for p in (tmp_path/'parsed_responses').iterdir()}
        e.main(argv + ['--resume'])
        assert len(calls)-before == {'live':0,'dry_run':438,'partial_failure':1}[first_run]
        assert (tmp_path/'requests.jsonl').read_bytes() == original_bytes
        resumed = assert_unique_log(tmp_path/'requests.jsonl')
        assert {r['request_id']:e.digest(e.canonical(r['body'])) for r in resumed} == hashes
        final = assert_unique_log(tmp_path/'records.jsonl')
        assert all(r['record_type']=='FINAL' and r['grading']['status']=='SCORED' for r in final)
        assert all(p.read_bytes()==value for p,value in saved_responses.items())
        for p in (tmp_path/'parsed_responses').iterdir():
            saved = json.loads(p.read_text())
            assert saved['body_hash'] == hashes[saved['request_id']]
        assert all('body' not in r for r in read_jsonl(tmp_path/'api_attempts.jsonl'))


@pytest.mark.parametrize('mutation', ['body', 'duplicate', 'missing'])
def test_resume_rejects_changed_request_definitions_before_http(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv('GEMINI_API_KEY','test-credential')
    cases = {f'w{i}':case() for i in range(219)}
    argv = ['--model','test','--out',str(tmp_path)]
    with patch.object(e,'reproduce',return_value=(cases,{})), patch.object(e,'call_api') as api:
        e.main(argv + ['--dry-run'])
        path = tmp_path/'requests.jsonl'
        rows = read_jsonl(path)
        if mutation=='body':
            rows[0]['body']['generationConfig']['temperature'] = 2.
        elif mutation=='duplicate':
            rows.append(rows[0])
        else:
            rows.pop()
        e.write_jsonl(path,rows)
        previous_records = (tmp_path/'records.jsonl').read_bytes()
        with pytest.raises(ValueError,match='Resume request definitions changed'):
            e.main(argv + ['--resume'])
        assert api.call_count==0
        assert (tmp_path/'records.jsonl').read_bytes()==previous_records


def test_prompt_and_question_request_first_collision_onset():
    prompt = ' '.join(e.PROMPT.split())
    assert 'interval containing the first new physical vehicle collision onset after t_end' in prompt
    assert 'ONSET of a new physical vehicle collision' in prompt
    assert 'not continued contact after a collision that began in an earlier interval' in prompt
    assert 'At most one of k=1 or k=2 should have accident_expected=true' in prompt
    assert 'involved_actor_ids must contain exactly the two actors involved in that collision' in prompt
    assert 'If no new collision begins during (0,2], both buckets must be false' in prompt
    assert 'first new physical vehicle collision begin after t_end' in e.QUESTION
    assert 'k=1 for onset in (0,1] and k=2 for onset in (1,2]' in e.QUESTION
    assert 'If no new collision begins within 2 s, return false for both buckets' in e.QUESTION
    assert 'Return both bucket entries' in e.QUESTION
    assert 'Do not use the network score or network decision itself as the reason' in prompt


@pytest.mark.parametrize('ids', [('a','c'), ('a','b','c')])
def test_crn4_rejects_other_observed_collision_pairs(ids):
    r = response((1,), ids)
    # V4 continues to accept these observed actor IDs.
    e.validate_response(r, ['a','b','c'])
    with pytest.raises(ValueError, match='exactly the selected pair'):
        e.validate_response(r, ['a','b','c'], selected_pair=['a','b'])


def test_crn4_accepts_selected_pair_in_either_order_and_empty_negatives():
    e.validate_response(response((2,), ('b','a')), ['a','b','c'], selected_pair=['a','b'])
    e.validate_response(response(), ['a','b','c'], selected_pair=['a','b'])
    r = response()
    r['predictions'][0]['involved_actor_ids'] = ['a','b']
    with pytest.raises(ValueError, match='Invalid collision actor ids'):
        e.validate_response(r, ['a','b','c'], selected_pair=['a','b'])


def test_crn4_wrong_pair_rejected_on_api_and_resume(tmp_path):
    (tmp_path/'parsed_responses').mkdir()
    (tmp_path/'raw_responses').mkdir()
    req = dict(request_id='w__crn4', window_id='w', condition='crn4', body={}, selected_pair=['a','b'])
    r = response((1,), ('a','c'))
    saved = dict(request_id=req['request_id'], body_hash=e.digest(e.canonical(req['body'])),
                 parsed_response=r, usage={}, latency_s=.1)
    e.write_json(e.response_path(tmp_path,req), saved)
    assert e.resume_response(tmp_path,req,['a','b','c']) is None
    args = SimpleNamespace(out=tmp_path, resume=True, retries=0, provider='gemini')
    raw = json.dumps({'candidates':[{'content':{'parts':[{'text':json.dumps(r)}]}}]})
    entries = []
    with patch.object(e,'call_api',return_value=(200,raw)):
        assert e.execute(args,req,['a','b','c'],lambda name,row:entries.append((name,row)))[0] is None
    assert any('exactly the selected pair' in row['error'] for name,row in entries if name == 'failures.jsonl')


def test_crn4_only_requests_use_verifier_v2():
    cases = {f'w{i}':case() for i in range(219)}
    args = SimpleNamespace(condition='crn4', provider='gemini', model='test', temperature=1.,
                           thinking_config=None, max_output_tokens=32768, seed=42)
    requests = e.prepare(cases,args)
    assert len(requests) == 219
    for r in requests:
        assert r['body']['systemInstruction']['parts'][0]['text'] == e.CRN4_PROMPT_V2
        payload = json.loads(r['body']['contents'][0]['parts'][0]['text'])
        assert payload['network_evidence'] == cases[r['window_id']]['crn4']['evidence']
        assert r['selected_pair'] == ['a','b']
        assert r['body']['contents'][0]['parts'][1]['text'] == e.QUESTION


@pytest.mark.parametrize('condition', ['crn4', 'both'])
def test_diagnostic_subset_filters_after_full_replay_and_labels_outputs(tmp_path, monkeypatch, condition):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-credential')
    cases = {f'w{i}':case() for i in range(219)}
    subset = tmp_path/'windows.txt'
    subset.write_text('w7\nw2\n')
    out = tmp_path/'run'
    args = SimpleNamespace(condition=condition, provider='gemini', model='test', temperature=1.,
                           thinking_config=None, max_output_tokens=32768, seed=20261007)
    full_requests = e.prepare(cases,args)
    def reproduce(cli, ledger):
        assert len(ledger) == 219
        assert not (out/'requests.jsonl').exists()
        # Reading/filtering the IDs happens only after this full replay returns.
        assert not subset.exists()
        subset.write_text('w7\nw2\n')
        return cases, {}
    raw = json.dumps({'candidates':[{'content':{'parts':[{'text':json.dumps(response())}]}}]})
    subset.unlink()
    with patch.object(e,'load_cohort',return_value=cases), patch.object(e,'reproduce',side_effect=reproduce) as replay, \
         patch.object(e,'call_api',return_value=(200,raw)) as api:
        e.main(['--model','test','--condition',condition,'--out',str(out),
                '--window-ids-file',str(subset),'--workers','1'])
    replay.assert_called_once()
    requests = read_jsonl(out/'requests.jsonl')
    assert requests == [r for r in full_requests if r['window_id'] in {'w7','w2'}]
    assert api.call_count == (4 if condition == 'both' else 2)
    manifest = json.loads((out/'experiment_manifest.json').read_text())
    assert manifest['window_ids_file'] == str(subset.resolve())
    assert manifest['selected_window_ids'] == ['w7','w2']
    assert manifest['full_frozen_cohort_windows'] == 219
    audit = json.loads((out/'dry_run_audit.json').read_text())
    assert audit['windows'] == 219 and audit['full_frozen_cohort_validated']
    scores = json.loads((out/'diagnostic_subset_scores.json').read_text())
    transitions = json.loads((out/'diagnostic_subset_network_to_llm_transitions.json').read_text())
    for report in (scores, transitions):
        assert report['evaluation_scope'] == 'diagnostic_subset'
        assert report['full_cohort_metrics'] is False
        assert report['selected_window_ids'] == ['w7','w2']
        assert report['crn4']['unscored'] == 0
    assert scores['crn4']['scored'] == 2
    assert not (out/'scores.json').exists()
    assert not (out/'network_to_llm_transitions.json').exists()
    assert len((out/'diagnostic_subset_paired_v4_crn_results.csv').read_text().splitlines()) == 3


@pytest.mark.parametrize('contents,error', [('w0\nw0\n','Duplicate'), ('unknown\n','Unknown'), ('\n','empty')])
def test_invalid_subset_rejected_after_full_replay_before_api(tmp_path, contents, error, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-credential')
    subset = tmp_path/'windows.txt'
    subset.write_text(contents)
    cases = {f'w{i}':case() for i in range(219)}
    with patch.object(e,'load_cohort',return_value=cases), patch.object(e,'reproduce',return_value=(cases,{})) as replay, \
         patch.object(e,'call_api') as api:
        with pytest.raises(ValueError,match=error):
            e.main(['--model','test','--out',str(tmp_path/'run'),'--window-ids-file',str(subset)])
    replay.assert_called_once()
    api.assert_not_called()


def test_subset_cannot_bypass_full_frozen_validation(tmp_path, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-credential')
    subset = tmp_path/'windows.txt'
    subset.write_text('w0\n')
    with patch.object(e,'reproduce',side_effect=ValueError('CRN checkpoint changed')) as replay, \
         patch.object(e,'load_window_ids') as load_ids, patch.object(e,'call_api') as api:
        with pytest.raises(ValueError,match='CRN checkpoint changed'):
            e.main(['--model','test','--out',str(tmp_path/'run'),'--window-ids-file',str(subset)])
    replay.assert_called_once()
    load_ids.assert_not_called()
    api.assert_not_called()


@pytest.mark.parametrize('mutation', ['decision', 'gt'])
def test_subset_rejects_frozen_mismatch_in_unselected_window(tmp_path, mutation, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-credential')
    ledger = e.load_cohort()
    cases = {wid:case(int(r['gt_bucket'])) for wid,r in ledger.items()}
    for wid,r in ledger.items():
        for condition in ('v4','crn4'):
            cases[wid][condition]['decision'] = r[condition+'_predicted'] == 'True'
    selected, excluded = list(cases)[:2]
    subset = tmp_path/'windows.txt'
    subset.write_text(selected+'\n')
    if mutation == 'decision':
        cases[excluded]['crn4']['decision'] = not cases[excluded]['crn4']['decision']
    else:
        # Preserve binary GT strata to reach the per-window GT check.
        cases[excluded]['gt_bucket'] = 2 if cases[excluded]['gt_bucket'] == 1 else (
            1 if cases[excluded]['gt_bucket'] == 2 else 5 if cases[excluded]['gt_bucket'] == 0 else 0)
    def replay(args, full_ledger):
        e.assert_networks(cases, full_ledger)
        return cases, {}
    with patch.object(e,'reproduce',side_effect=replay), patch.object(e,'call_api') as api:
        with pytest.raises(ValueError,match='frozen confusion mismatch|GT mismatch'):
            e.main(['--model','test','--out',str(tmp_path/'run'),'--window-ids-file',str(subset)])
    api.assert_not_called()


def test_subset_resume_preserves_ids_and_rejects_changed_selection(tmp_path):
    subset = tmp_path/'windows.txt'
    subset.write_text('w0\nw1\n')
    cases = {f'w{i}':case() for i in range(219)}
    out = tmp_path/'run'
    argv = ['--model','test','--condition','crn4','--out',str(out),
            '--window-ids-file',str(subset),'--dry-run']
    with patch.object(e,'load_cohort',return_value=cases), patch.object(e,'reproduce',return_value=(cases,{})) as replay, \
         patch.object(e,'call_api') as api:
        e.main(argv)
        definitions = (out/'requests.jsonl').read_bytes()
        e.main(argv+['--resume'])
        assert (out/'requests.jsonl').read_bytes() == definitions
        subset.write_text('w0\nw2\n')
        with pytest.raises(ValueError,match='Resume manifest/settings changed'):
            e.main(argv+['--resume'])
        assert (out/'requests.jsonl').read_bytes() == definitions
        assert replay.call_count == 3
        api.assert_not_called()


@pytest.mark.parametrize('thinking', [None, {'thinkingLevel':'HIGH'}])
def test_frozen_gemini_cli_defaults_and_configurable_thinking(tmp_path, thinking):
    cases = {f'w{i}':case() for i in range(219)}
    argv = ['--model','test','--out',str(tmp_path),'--dry-run']
    if thinking is not None:
        argv += ['--thinking-config',json.dumps(thinking)]
    with patch.object(e,'reproduce',return_value=(cases,{})), patch.object(e,'call_api') as api:
        e.main(argv)
    assert api.call_count == 0
    manifest = json.loads((tmp_path/'experiment_manifest.json').read_text())
    assert manifest['temperature'] == 1.0
    assert manifest['max_output_tokens'] == 32768
    assert manifest['thinking_configuration'] == (thinking if thinking is not None else 'provider default')
    for r in assert_unique_log(tmp_path/'requests.jsonl'):
        gen = r['body']['generationConfig']
        assert gen['temperature'] == 1.0
        assert gen['maxOutputTokens'] == 32768
        assert gen.get('thinkingConfig') == thinking
        payload = json.loads(r['body']['contents'][0]['parts'][0]['text'])
        c = cases[r['window_id']]
        assert payload == json.loads(e.canonical(dict(observation=c['observation'],
            network_evidence=c[r['condition']]['evidence'], horizon_s=2)))
        assert gen['responseSchema'] == e.providers.gemini_schema(e.schema())
    # The revised request is an instruction, not a change to response validation
    # or grading: a two-positive response still receives the original strict grade.
    double = response((1,2))
    e.validate_response(double, ['a','b'])
    assert e.grade(case(1),double)['binary_2s_window'] == 'TP'
    assert e.grade(case(1),double)['strict_bucket']['correct'] is False
