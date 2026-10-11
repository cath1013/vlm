"""Focused CRN5-R1 tests: no official data, checkpoints, or expensive CRN fits."""
from types import SimpleNamespace

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

from examples import train_crn5_r1 as r1


def actor(actor_id, x=0., speed=2.):
    return {'actor_id': actor_id, 'origin_enu': [x, 0.],
            'global': [speed, 1., 0., 1., 1., 0., 1.],
            'history': [[-speed, 0., 0., 0., 1.]] * 5}


def row(sid='s', bucket=1, labels=None, observed=(True, True)):
    actors = [actor('a'), actor('b', 3.), actor('c', 10.)]
    labels = [bucket, 0, 0] if labels is None else labels
    masks = [[k < bucket if label else obs and (not bucket or k < bucket)
              for k, obs in enumerate(list(observed) + [True] * 3)] for label in labels]
    return dict(window_id=sid, scenario_id=sid, actors=actors, gt_bucket=bucket,
                gt_positive=bool(bucket), gt_pair_present=any(labels),
                pair_labels=labels, observed_negative_buckets=list(observed) + [True] * 3,
                hazard_target=[[int(label == k + 1) for k in range(5)] for label in labels],
                hazard_mask=masks)


HAZARDS = [[.5, .2, .1, .1], [.5, .2, .1, .1], [.1, .1, .1, .1]]


def record(wid, truth, score, crn_risk=.8):
    return dict(window_id=wid, truth=truth, crn_risk=crn_risk,
                filter_only_score=score, rerank_plus_filter_score=score,
                baseline_hit=truth, reranked_hit=truth, gt_pair_present=truth,
                candidate_hit=truth)


class TestCRN5R1(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def patch(self, target, name, value):
        patcher = patch.object(target, name, value)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_features_have_no_label_dependency_and_stable_risk_ties(self):
        a = row()
        b = row(bucket=0, labels=[0, 0, 0])
        x = r1.examples([a], {'s': HAZARDS}, 2)[0]
        y = r1.examples([b], {'s': HAZARDS}, 2)[0]
        assert x['features'] == y['features']
        assert x['indices'] == [0, 1]
        assert len(x['features'][0]) == len(r1.FEATURE_NAMES)
        assert x['hits'] == [True, False]
        assert y['hits'] == [False, False]
        assert r1.motion_features(a['actors'][0], a['actors'][1]) == r1.motion_features(a['actors'][1], a['actors'][0])


    def test_missing_gt_pair_and_topk_miss_never_create_pair_positives(self):
        missing = r1.examples([row(labels=[0, 0, 0])], {'s': HAZARDS}, 2)[0]
        omitted = r1.examples([row(labels=[0, 0, 1])], {'s': HAZARDS}, 2)[0]
        for e in (missing, omitted):
            assert not any(e['hits'])
            assert not any(any(t) for t in e['targets'])
            assert not e['window_supervised']


    def test_unidentifiable_windows_excluded(self):
        assert r1.examples([row(bucket=0, labels=[0, 0, 0], observed=(True, False))], {'s': HAZARDS}, 5) == []


    def test_censored_logits_receive_no_pair_loss_gradient(self):
        e = r1.examples([row(labels=[0, 0, 0], observed=(True, False))], {'s': HAZARDS}, 2)[0]
        assert e['masks'] == [[True, False], [True, False]]
        logits = torch.tensor([[0., 100.], [0., -100.]], requires_grad=True)
        loss = r1.reranker_loss(logits, e)
        loss.backward()
        assert torch.isfinite(loss)
        assert (logits.grad[:, 0] > 0).all()
        assert (logits.grad[:, 1] == 0).all()


    def test_correct_candidate_and_window_losses_are_finite(self):
        e = r1.examples([row()], {'s': HAZARDS}, 3)[0]
        logits = torch.tensor([[-1000., -1000.], [1000., 1000.], [0., 0.]], requires_grad=True)
        loss = r1.reranker_loss(logits, e)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()
        assert logits.grad[0, 0] < 0


    def test_positive_window_loss_rewards_only_best_gt_correct_candidate(self):
        e = r1.examples([row(labels=[1, 0, 1])], {'s': HAZARDS}, 3)[0]
        logits = torch.tensor([[-2., -2.], [3., 3.], [-4., -4.]],
                              dtype=torch.float64, requires_grad=True)
        risks = 1 - (1 - logits.sigmoid()).prod(-1)
        assert risks[1] > risks[0] > risks[2]
        assert e['hits'] == [True, False, True]
        # Subtract the identical pair BCE and disable ranking to isolate the
        # positive-window term without changing censor masks or pair targets.
        with patch.dict(r1.R1_CONFIG, window_loss_weight=0., ranking_loss_weight=0.):
            pair_bce = r1.reranker_loss(logits, e)
        with patch.dict(r1.R1_CONFIG, window_loss_weight=1., ranking_loss_weight=0.):
            window_loss = r1.reranker_loss(logits, e) - pair_bce
        torch.testing.assert_close(window_loss, -risks[0].log())
        window_loss.backward()
        assert (logits.grad[0] < 0).all()
        torch.testing.assert_close(logits.grad[1:], torch.zeros_like(logits.grad[1:]))


    def test_threshold_minimizes_fp_and_rounds_retention_up(self):
        records = [record(str(i), True, .8 if i < 97 else .2) for i in range(100)]
        records += [record('negative', False, .3), record('gated', False, .99, .1)]
        assert r1.select_threshold(records, 'filter_only', .5) == .8
        small = [record('p', True, .2), record('n', False, .3)]
        assert r1.select_threshold(small, 'filter_only', .5) == .2
        with self.assertRaisesRegex(ValueError, 'positive'):
            r1.select_threshold([record('n', False, .2)], 'filter_only', .5)


    def test_fold_thresholds_and_final_selection_are_scored_separately(self):
        records = [record('p', True, .3), record('n', False, .6)]
        for r in records:
            for rule in r1.RULES:
                r[rule + '_threshold'] = .5
        nested = r1.evaluate(records, .5)
        selected = r1.evaluate(records, .5, {rule: .2 for rule in r1.RULES})
        assert nested['filter_only']['tp'] == 0
        assert selected['filter_only']['tp'] == 1
        assert nested['filter_only']['transitions'] == {'TP->FN': 1, 'FP->FP': 1}


    def test_nested_models_and_calibration_never_train_on_outer_scenarios(self):
        tmp_path = self.tmp_path
        rows = [row(f's{i}') for i in range(30)]
        assignments = r1.crn.folds_by_scenario(rows, 0)
        original = {r['window_id']: HAZARDS for r in rows}
        fits = []

        # Track each CRN crossfit fitting population and all nested exclusions.
        class CheckedCache:
            def crossfit(self, source, seed):
                scenarios = {r['scenario_id'] for r in source}
                outer = (seed - 100) // 100
                excluded = {r['scenario_id'] for r in rows if assignments[r['scenario_id']] == outer}
                assert not scenarios & excluded
                for fitting, held in r1.splits(source, seed):
                    r1.disjoint(fitting, held)
                return {r['window_id']: HAZARDS for r in source}

            def rerank(self, fitting_data, held_data, seed):
                return predict(fit(fitting_data, 'cpu', 2, seed), held_data, 'cpu')

        def fit(data, device, batch_size, seed):
            scenarios = {e['scenario_id'] for e in data}
            fits.append(scenarios)
            return scenarios

        def predict(model, data, device):
            assert not model & {e['scenario_id'] for e in data}
            result = []
            for e in data:
                result.append({**record(e['window_id'], e['truth'], .7, e['crn_risk']),
                               'scenario_id': e['scenario_id']})
            return result

        self.patch(r1, 'fit_reranker', fit)
        self.patch(r1, 'predict_reranker', predict)
        args = SimpleNamespace(seed=0, device='cpu', batch_size=2, top_k=5, out=tmp_path)
        predictions, audit = r1.run_nested(rows, original, assignments, CheckedCache(), args, .5)
        assert len(predictions) == len(rows)
        assert len(fits) == 30  # Five calibration fits plus one outer fit, per fold.
        for fold in audit:
            excluded = set(fold['held_scenarios'])
            for inner in fold['inner_folds']:
                assert not excluded & set(inner['fitting_scenarios'])
                assert not excluded & set(inner['calibration_scenarios'])
                assert not set(inner['fitting_scenarios']) & set(inner['calibration_scenarios'])


    def test_prediction_cache_reuses_fit_and_rejects_tampering(self):
        tmp_path = self.tmp_path
        rows = [row(f's{i}') for i in range(5)]
        saved = dict(v2_checkpoint='unused', hidden_dim=128, dropout=.1)
        args = SimpleNamespace(out=tmp_path / 'experiment', device='cpu', batch_size=2)
        cache = r1.CRNCache(rows, saved, {'batch_size': 16}, args, {'source': 'abc'})
        calls = []
        class Model:
            def eval(self):
                return self
            def requires_grad_(self, value):
                assert value is False
                return self
        self.patch(r1.crn, 'fit', lambda *a, **k: (calls.append(k) or Model()))
        self.patch(r1, 'pair_predict', lambda model, rows, *a: {r['window_id']: HAZARDS for r in rows})
        fitting, held = rows[:4], rows[4:]
        assert cache.infer(fitting, held, 0) == cache.infer(fitting, held, 0)
        assert len(calls) == 1
        assert calls[0]['epochs'] == 10 and calls[0]['batch_size'] == 16
        assert calls[0]['window_loss_weight'] == .1
        path = next(cache.root.glob('*.json'))
        doc = r1.json.loads(path.read_text())
        doc['predictions']['s4'][0][0] = .99
        r1.write_json(path, doc)
        with self.assertRaisesRegex(ValueError, 'corruption'):
            cache.infer(fitting, held, 0)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            cache.infer(rows, held, 0)


    def test_existing_experiment_refused_before_reading_inputs(self):
        tmp_path = self.tmp_path
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            r1.main(['--out', str(tmp_path), '--train', '/missing'])

    def test_small_fit_freezes_model_and_checkpoint_roundtrips(self):
        rows = [row('positive'), row('negative', bucket=0, labels=[0, 0, 0])]
        data = r1.examples(rows, {r['window_id']: HAZARDS for r in rows}, 3)
        with patch.dict(r1.R1_CONFIG, epochs=2):
            model = r1.fit_reranker(data, 'cpu', 2, 0)
        assert all(not p.requires_grad for p in model.parameters())
        assert not model.training
        expected = r1.predict_reranker(model, data, 'cpu')
        path = self.tmp_path / 'checkpoint.pt'
        torch.save(model.state_dict(), path)
        state = torch.load(path, weights_only=True)
        restored = r1.PairReranker(state['mean'], state['scale'])
        restored.load_state_dict(state)
        assert r1.predict_reranker(restored, data, 'cpu') == expected
        features = torch.tensor([f for e in data for f in e['features']])
        torch.testing.assert_close(model.mean, features.mean(0))

    def test_reranker_prediction_cache_reuses_fits(self):
        rows = [row('fitting'), row('held')]
        args = SimpleNamespace(out=self.tmp_path / 'experiment', device='cpu', batch_size=2)
        cache = r1.CRNCache(rows, {}, {'batch_size': 16}, args, {'source': 'abc'})
        data = r1.examples(rows, {r['window_id']: HAZARDS for r in rows}, 3)
        calls = []
        self.patch(r1, 'fit_reranker', lambda *a: calls.append(a) or object())
        self.patch(r1, 'predict_reranker', lambda model, data, device: [{'window_id': data[0]['window_id']}])
        assert cache.rerank(data[:1], data[1:], 0) == cache.rerank(data[:1], data[1:], 0)
        assert len(calls) == 1
        with self.assertRaisesRegex(ValueError, 'leakage'):
            cache.rerank(data, data[1:], 0)

    def test_original_oof_baseline_assertions(self):
        rows = [row('p'), row('n', bucket=0, labels=[0, 0, 0])]
        predictions = {r['window_id']: HAZARDS for r in rows}
        records = r1.baseline_records(rows, predictions)
        thresholds = r1.crn.select_threshold(records, rows, 4)
        saved = {'threshold_by_horizon': thresholds}
        expected = r1.crn.report(records, rows, thresholds, 4)['per_horizon']['2']
        report = {'train_oof_2s': {**expected, **{k.upper(): expected[k] for k in ('tp', 'fp', 'tn', 'fn')}}}
        assert r1.verify_original(rows, predictions, saved, report) == expected
        report['train_oof_2s']['correct_gt_actor_pair_hits'] += 1
        with self.assertRaisesRegex(ValueError, 'hits mismatch'):
            r1.verify_original(rows, predictions, saved, report)
