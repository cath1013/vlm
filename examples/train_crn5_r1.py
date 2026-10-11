#!/usr/bin/env python3
"""Train CRN5-R1 using collector observations only; no validation or raw replay.

Original CRN5-A head fits reproduce its saved five-fold OOF run; the supplied
checkpoint and V2 backbone are never updated. Nested evaluation uses five outer
folds, five calibration folds, and five CRN cross-fitting folds inside each
calibration fitting split. Prediction caches are content-addressed in a sibling
.crn5_r1_cache directory and reusable across NEW experiment directories.

Both rules gate on the fixed CRN5-A 2s decision. Filter-only retains its rank-1
pair; rerank-plus-filter selects the highest R1 risk among the top-k CRN pairs.
Nested fold thresholds are fitted without outer labels. Final thresholds are
selected on pooled outer OOF predictions; their selection metrics are explicitly
not the unbiased nested estimate. No hyperparameter search is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.utils.data import DataLoader
from examples import train_collision_risk_net as crn

FROZEN_SHA = '9946c3480885dd3e7260c835c069553cd3f7cb5b382cb35e743919f795b2fddd'
RULES = ('filter_only', 'rerank_plus_filter')
R1_CONFIG = dict(epochs=20, lr=0.001, hidden_dim=32, weight_decay=1e-4,
                 window_loss_weight=0.1, ranking_loss_weight=0.1)
FEATURE_NAMES = ([f'hazard_{i}' for i in range(1, 5)] +
                 [f'risk_{i}' for i in range(1, 5)] +
                 [f'event_{i}' for i in range(1, 5)] +
                 ['rank_fraction', 'margin_to_top', 'margin_to_next', 'mean_risk', 'log_pair_count',
                  'distance_m', 'relative_speed_mps', 'closing_speed_mps', 'cross_speed_mps',
                  'tca_2s', 'dca_2s_m', 'heading_cos', 'speed_min', 'speed_max',
                  'accel_min', 'accel_max', 'motion_known', 'accel_known',
                  'history_relative_distance_change', 'history_known'])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def disjoint(fitting, held):
    a = {r['scenario_id'] for r in fitting}
    b = {r['scenario_id'] for r in held}
    require(not a & b, 'Scenario leakage between fitting and held splits')


def splits(rows, seed):
    assignments = crn.folds_by_scenario(rows, seed)
    return [([r for r in rows if assignments[r['scenario_id']] != fold],
             [r for r in rows if assignments[r['scenario_id']] == fold]) for fold in range(5)]


@torch.inference_mode()
def pair_predict(model, rows, device, batch_size):
    """No labels are passed to the model; store every unordered pair's hazards."""
    model.eval()
    loader = DataLoader(crn.eligible_rows(rows, model.max_horizon), batch_size=batch_size,
                        collate_fn=partial(crn.collate, max_horizon=model.max_horizon))
    result = {}
    for batch in loader:
        logits, mask = crn.forward(model, batch, device)
        probabilities = logits.sigmoid().cpu()
        for i, row in enumerate(batch['rows']):
            result[row['window_id']] = probabilities[i, mask[i].cpu()].tolist()
    return result


class CRNCache:
    def __init__(self, rows, saved, config, args, sources):
        self.rows, self.saved, self.config, self.args = rows, saved, config, args
        self.identity = dict(sources=sources, crn_config=config, device=args.device,
                             inference_batch_size=args.batch_size, torch_version=torch.__version__)
        self.root = args.out.parent / '.crn5_r1_cache' / digest(self.identity)
        self.root.mkdir(parents=True, exist_ok=True)
        self.audit = []

    def infer(self, fitting, held, seed):
        disjoint(fitting, held)
        fit_ids = sorted({r['scenario_id'] for r in fitting})
        held_ids = sorted({r['scenario_id'] for r in held})
        key = dict(fit_scenarios=fit_ids, seed=seed)
        path = self.root / (digest(key) + '.json')
        # Predict the full complement so future consumers can reuse this fit.
        fit_set = set(fit_ids)
        complement = [r for r in self.rows if r['scenario_id'] not in fit_set]
        expected_ids = {r['window_id'] for r in crn.eligible_rows(complement, 4)}
        if path.exists():
            cached = json.loads(path.read_text())
            require(cached['key'] == key and cached['identity'] == self.identity,
                    'CRN cache provenance mismatch')
            require(cached['prediction_digest'] == digest(cached['predictions']), 'CRN cache corruption')
            predictions = cached['predictions']
            origin = 'cache'
        else:
            print(f'CRN fit: {len(fit_ids)} scenarios; seed={seed}', flush=True)
            model = crn.fit(fitting, self.saved['v2_checkpoint'], self.args.device,
                            epochs=10, batch_size=self.config['batch_size'], lr=0.001,
                            hidden_dim=self.saved['hidden_dim'], dropout=self.saved['dropout'],
                            seed=seed, max_horizon=4, window_loss_weight=0.1)
            # Every inference model is frozen, including the already frozen V2.
            model.eval().requires_grad_(False)
            predictions = pair_predict(model, complement, self.args.device, self.args.batch_size)
            del model
            payload = dict(key=key, identity=self.identity, predictions=predictions,
                           prediction_digest=digest(predictions))
            temporary = path.with_suffix('.tmp')
            write_json(temporary, payload)
            temporary.replace(path)
            origin = 'fit'
        require(set(predictions) == expected_ids, 'Incomplete CRN prediction cache')
        self.audit.append(dict(cache=str(path), fitting_scenarios=fit_ids,
                               held_scenarios=held_ids, seed=seed, origin=origin))
        return {r['window_id']: predictions[r['window_id']] for r in crn.eligible_rows(held, 4)}

    def crossfit(self, rows, seed):
        predictions = {}
        for fold, (fitting, held) in enumerate(splits(rows, seed)):
            predictions.update(self.infer(fitting, held, seed + fold))
        return predictions

    def rerank(self, fitting_data, held_data, seed):
        """Cache calibration/outer scores too; their training features are hashed."""
        disjoint(fitting_data, held_data)
        key = dict(kind='reranker', fitting_digest=digest(fitting_data),
                   held_digest=digest(held_data), seed=seed,
                   config=R1_CONFIG, batch_size=self.args.batch_size)
        path = self.root / ('r1_' + digest(key) + '.json')
        if path.exists():
            cached = json.loads(path.read_text())
            require(cached['key'] == key and cached['prediction_digest'] == digest(cached['predictions']),
                    'Reranker prediction cache corruption')
            return cached['predictions']
        model = fit_reranker(fitting_data, self.args.device, self.args.batch_size, seed)
        predictions = predict_reranker(model, held_data, self.args.device)
        del model
        temporary = path.with_suffix('.tmp')
        write_json(temporary, dict(key=key, predictions=predictions, prediction_digest=digest(predictions)))
        temporary.replace(path)
        return predictions


def pair_statistics(hazards):
    survival, risks, events = 1., [], []
    for h in hazards:
        events.append(survival * h)
        survival *= 1 - h
        risks.append(1 - survival)
    return risks, events


def motion_features(a, b):
    """Symmetric ENU features from current origin, telemetry and past positions."""
    ga, gb = a['global'], b['global']
    delta = [b['origin_enu'][i] - a['origin_enu'][i] for i in (0, 1)]
    distance = math.hypot(*delta)
    known = bool(ga[1] and ga[6] and gb[1] and gb[6])
    velocity = [gb[0] * gb[4 + i] - ga[0] * ga[4 + i] for i in (0, 1)] if known else [0., 0.]
    speed2 = sum(x * x for x in velocity)
    dot = sum(x * y for x, y in zip(delta, velocity))
    tca = max(0., min(2., -dot / speed2)) if speed2 > 1e-12 else 0.
    dca = math.hypot(*(delta[i] + tca * velocity[i] for i in (0, 1)))
    history_delta, history_known = 0., False
    for ha, hb in zip(reversed(a['history']), reversed(b['history'])):
        if ha[4] and hb[4]:
            past = math.hypot(*(delta[i] + hb[i] - ha[i] for i in (0, 1)))
            history_delta, history_known = distance - past, True
            break
    return [distance, math.sqrt(speed2), -dot / max(distance, 1e-6),
            abs(delta[0] * velocity[1] - delta[1] * velocity[0]) / max(distance, 1e-6),
            tca, dca, ga[4] * gb[4] + ga[5] * gb[5] if known else 0.,
            min(ga[0], gb[0]), max(ga[0], gb[0]), min(ga[2], gb[2]), max(ga[2], gb[2]),
            float(known), float(bool(ga[3] and gb[3])), history_delta, float(history_known)]


def candidate_features(actors, hazards, top_k):
    """Feature allowlist: this function cannot access any collector GT fields."""
    pairs = list(itertools.combinations(range(len(actors)), 2))
    require(len(pairs) == len(hazards), 'Pair hazards do not match collector actor order')
    require(all(len(h) == 4 and all(math.isfinite(x) and 0 <= x <= 1 for x in h)
                for h in hazards), 'Invalid four-bucket CRN hazards')
    statistics = [pair_statistics(h) for h in hazards]
    order = sorted(range(len(pairs)), key=lambda i: (-statistics[i][0][1], i))
    scores = [statistics[i][0][1] for i in order]
    features = []
    for rank, p in enumerate(order[:top_k]):
        risks, events = statistics[p]
        next_score = scores[rank + 1] if rank + 1 < len(scores) else 0.
        a, b = pairs[p]
        features.append(hazards[p] + risks + events +
                        [rank / max(1, len(pairs) - 1), scores[0] - scores[rank],
                         scores[rank] - next_score, sum(scores) / len(scores), math.log1p(len(scores))] +
                        motion_features(actors[a], actors[b]))
    require(all(math.isfinite(x) for f in features for x in f), 'Nonfinite observation features')
    return order[:top_k], features


def examples(rows, predictions, top_k):
    eligible_ids = {r['window_id'] for r in crn.eligible_rows(rows, 4)}
    require({r['window_id'] for r in rows if r['window_id'] in eligible_ids and crn.identifiable(r, 2)}
            <= set(predictions), 'Missing identifiable CRN cross-fit predictions')
    result = []
    for row in rows:
        if not crn.identifiable(row, 2):
            continue
        hazards = predictions.get(row['window_id'], [])
        # The official CRN evaluator excludes windows with no supervised pairs.
        if row['window_id'] not in predictions:
            continue
        indices, features = candidate_features(row['actors'], hazards, top_k)
        pairs = list(itertools.combinations([a['actor_id'] for a in row['actors']], 2))
        masks = [row['hazard_mask'][p][:2] for p in indices]
        targets = [row['hazard_target'][p][:2] for p in indices]
        hits = [row['gt_bucket'] in (1, 2) and row['pair_labels'][p] == row['gt_bucket'] for p in indices]
        truth = row['gt_bucket'] in (1, 2)
        result.append(dict(window_id=row['window_id'], scenario_id=row['scenario_id'],
                           gt_bucket=row['gt_bucket'], truth=truth, features=features,
                           indices=indices, pairs=[pairs[p] for p in indices], masks=masks,
                           targets=targets, hits=hits, gt_pair_present=row['gt_pair_present'],
                           window_supervised=not truth or any(hits),
                           crn_risk=pair_statistics(hazards[indices[0]])[0][1] if indices else 0.))
    return result


class PairReranker(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES), R1_CONFIG['hidden_dim']), nn.ReLU(),
                                 nn.Linear(R1_CONFIG['hidden_dim'], 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        # Residual calibration of two frozen CRN hazards; no gradients to CRN.
        base = torch.logit(features[..., :2].clamp(1e-6, 1 - 1e-6))
        return base + self.net((features - self.mean) / self.scale)


def reranker_loss(logits, example):
    targets = logits.new_tensor(example['targets'])
    masks = torch.tensor(example['masks'], device=logits.device, dtype=torch.bool)
    positive, negative = masks & targets.bool(), masks & ~targets.bool()
    terms = [nn.functional.binary_cross_entropy_with_logits(logits[m], targets[m])
             for m in (positive, negative) if m.any()]
    loss = torch.stack(terms).mean() if terms else logits.sum() * 0
    # Log-space cumulative risk avoids vanished gradients near probability 0/1.
    log_risk = torch.logaddexp(nn.functional.logsigmoid(logits[:, 0]),
                             nn.functional.logsigmoid(-logits[:, 0]) + nn.functional.logsigmoid(logits[:, 1]))
    hits = torch.tensor(example['hits'], dtype=torch.bool, device=logits.device)
    if example['window_supervised']:
        if example['truth']:
            window_loss = -log_risk[hits].max()
        else:
            best = log_risk.argmax()
            window_loss = -nn.functional.logsigmoid(-logits[best]).sum()
        loss = loss + R1_CONFIG['window_loss_weight'] * window_loss
    if hits.any():
        ranking = torch.logsumexp(log_risk, 0) - torch.logsumexp(log_risk[hits], 0)
        loss = loss + R1_CONFIG['ranking_loss_weight'] * ranking
    return loss


def fit_reranker(data, device, batch_size, seed):
    usable = [e for e in data if e['features'] and any(any(m) for m in e['masks'])]
    require(bool(usable), 'No identifiable reranker training windows')
    x = torch.tensor([f for e in usable for f in e['features']], dtype=torch.float32)
    torch.manual_seed(seed)
    model = PairReranker(x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=R1_CONFIG['lr'], weight_decay=R1_CONFIG['weight_decay'])
    loader = DataLoader(usable, batch_size=batch_size, shuffle=True, collate_fn=list,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(R1_CONFIG['epochs']):
        for batch in loader:
            losses = [reranker_loss(model(torch.tensor(e['features'], device=device)), e) for e in batch]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def predict_reranker(model, data, device):
    result = []
    for e in data:
        if e['features']:
            logits = model(torch.tensor(e['features'], device=device))
            risks = (1 - (1 - logits.sigmoid()).prod(-1)).cpu().tolist()
            selected = max(range(len(risks)), key=lambda i: risks[i])
        else:
            risks, selected = [], None
        result.append(dict(window_id=e['window_id'], scenario_id=e['scenario_id'], truth=e['truth'],
                           crn_risk=e['crn_risk'], baseline_hit=bool(e['hits'] and e['hits'][0]),
                           gt_pair_present=e['gt_pair_present'], candidate_hit=any(e['hits']),
                           crn_pair=e['pairs'][0] if e['pairs'] else None,
                           reranked_pair=e['pairs'][selected] if selected is not None else None,
                           reranked_hit=e['hits'][selected] if selected is not None else False,
                           filter_only_score=risks[0] if risks else 0.,
                           rerank_plus_filter_score=risks[selected] if selected is not None else 0.))
    return result


def select_threshold(records, rule, crn_threshold, retention=0.97):
    baseline_tp = sum(r['truth'] and r['crn_risk'] >= crn_threshold for r in records)
    require(any(r['truth'] for r in records), 'Threshold calibration needs positive windows')
    target = math.ceil(retention * baseline_tp)
    candidates = crn.threshold_candidates([r[rule + '_score'] for r in records])
    choices = []
    for threshold in candidates:
        decisions = [r['crn_risk'] >= crn_threshold and r[rule + '_score'] >= threshold for r in records]
        tp = sum(r['truth'] and p for r, p in zip(records, decisions))
        fp = sum(not r['truth'] and p for r, p in zip(records, decisions))
        if tp >= target:
            choices.append(((fp, -tp, -threshold), threshold))
    require(bool(choices), 'Recall retention constraint is infeasible')
    return min(choices)[1]


def outcome(truth, prediction):
    return ('TP' if prediction else 'FN') if truth else ('FP' if prediction else 'TN')


def evaluate(records, crn_threshold, thresholds=None):
    truth = [r['truth'] for r in records]
    baseline = [r['crn_risk'] >= crn_threshold for r in records]
    metrics = dict(baseline=crn.binary_metrics(truth, baseline))
    positives = sum(truth)
    safe = lambda a, b: a / b if b else None
    metrics['baseline'].update(correct_gt_pair_hits=sum(p and r['baseline_hit'] for p, r in zip(baseline, records)))
    metrics['baseline']['correct_gt_pair_recall'] = safe(metrics['baseline']['correct_gt_pair_hits'], positives)
    for rule in RULES:
        decisions = [b and r[rule + '_score'] >= (thresholds[rule] if thresholds is not None else r[rule + '_threshold'])
                     for b, r in zip(baseline, records)]
        m = crn.binary_metrics(truth, decisions)
        hit = 'baseline_hit' if rule == 'filter_only' else 'reranked_hit'
        hits = sum(p and r[hit] for p, r in zip(decisions, records))
        transitions = Counter(outcome(t, b) + '->' + outcome(t, p) for t, b, p in zip(truth, baseline, decisions))
        m.update(correct_gt_pair_hits=hits, correct_gt_pair_recall=safe(hits, positives),
                 FPs_eliminated=metrics['baseline']['fp'] - m['fp'],
                 TPs_lost=metrics['baseline']['tp'] - m['tp'], transitions=dict(transitions),
                 baseline_recall_retention=safe(m['tp'], metrics['baseline']['tp']))
        metrics[rule] = m
    metrics['candidate_coverage'] = dict(positive_windows=positives,
        gt_pair_present_windows=sum(r['truth'] and r['gt_pair_present'] for r in records),
        top_k_gt_pair_windows=sum(r['truth'] and r['candidate_hit'] for r in records),
        top_k_gt_pair_recall=safe(sum(r['truth'] and r['candidate_hit'] for r in records), positives))
    coverage = metrics['candidate_coverage']
    coverage['top_k_coverage_given_gt_pair_present'] = safe(coverage['top_k_gt_pair_windows'],
                                                          coverage['gt_pair_present_windows'])
    return metrics


def baseline_records(rows, predictions):
    return [crn.aggregate_window(row, predictions[row['window_id']], 4)
            for row in crn.eligible_rows(rows, 4)]


def verify_original(rows, predictions, saved, report):
    records = baseline_records(rows, predictions)
    actual = crn.report(records, rows, saved['threshold_by_horizon'], 4)['per_horizon']['2']
    expected = report['train_oof_2s']
    for key in ('tp', 'fp', 'tn', 'fn'):
        require(actual[key] == expected[key.upper()], f'Original CRN5 OOF {key} mismatch: {actual}')
    for key in ('identifiable_windows', 'correct_gt_actor_pair_hits'):
        require(actual[key] == expected[key], f'Original CRN5 OOF {key} mismatch')
    require(math.isclose(actual['recall'], expected['recall'], abs_tol=1e-12), 'Original OOF recall mismatch')
    selected = crn.select_threshold(records, rows, 4)['2']
    require(math.isclose(selected, saved['threshold_by_horizon']['2'], abs_tol=2e-6),
            'Reproduced OOF threshold differs from frozen training threshold')
    return actual


def run_nested(rows, original_predictions, assignments, cache, args, threshold):
    outer_predictions, audit = [], []
    for outer in range(5):
        training = [r for r in rows if assignments[r['scenario_id']] != outer]
        held = [r for r in rows if assignments[r['scenario_id']] == outer]
        disjoint(training, held)
        inner_seed = args.seed + 100 + outer * 100
        training_predictions = cache.crossfit(training, inner_seed)
        training_data = examples(training, training_predictions, args.top_k)
        calibration = []
        inner_audit = []
        for inner, (fitting, calibration_rows) in enumerate(splits(training, inner_seed)):
            disjoint(fitting, calibration_rows)
            # CRN training features must also exclude this calibration fold.
            fitting_predictions = cache.crossfit(fitting, inner_seed + 10 + inner * 10)
            fitting_data = examples(fitting, fitting_predictions, args.top_k)
            calibration_data = examples(calibration_rows, training_predictions, args.top_k)
            calibration.extend(cache.rerank(fitting_data, calibration_data, args.seed + outer * 5 + inner))
            inner_audit.append(dict(inner_fold=inner,
                fitting_scenarios=sorted({r['scenario_id'] for r in fitting}),
                calibration_scenarios=sorted({r['scenario_id'] for r in calibration_rows})))
        thresholds = {rule: select_threshold(calibration, rule, threshold) for rule in RULES}
        held_data = examples(held, original_predictions, args.top_k)
        predicted = cache.rerank(training_data, held_data, args.seed + outer)
        for r in predicted:
            r.update(outer_fold=outer, **{rule + '_threshold': thresholds[rule] for rule in RULES})
        outer_predictions.extend(predicted)
        fold = dict(outer_fold=outer, fitting_scenarios=sorted({r['scenario_id'] for r in training}),
                    held_scenarios=sorted({r['scenario_id'] for r in held}), inner_folds=inner_audit,
                    thresholds=thresholds, calibration_metrics=evaluate(calibration, threshold, thresholds),
                    held_metrics=evaluate(predicted, threshold))
        audit.append(fold)
        write_json(args.out / f'outer_fold_{outer}.json', dict(audit=fold, predictions=predicted))
        print(f'Finished nested outer fold {outer + 1}/5', flush=True)
    return sorted(outer_predictions, key=lambda r: r['window_id']), audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'out/collision_risk_hazard/train483.jsonl')
    parser.add_argument('--crn5-checkpoint', type=Path, default=ROOT / 'out/collision_risk_hazard/frozen/crn5_a_w01/collision_risk_final.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=ROOT / 'out/experiments' /
                        ('crn5_r1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    args = parser.parse_args(argv)
    require(not args.out.exists(), 'Refusing to overwrite an existing experiment directory')
    require(args.batch_size > 0 and args.top_k > 0, 'batch-size and top-k must be positive')
    require(file_hash(args.crn5_checkpoint) == FROZEN_SHA, 'Expected frozen CRN5-A checkpoint')
    report_path = args.crn5_checkpoint.parent / 'report.json'
    report = json.loads(report_path.read_text())
    config = report['config']
    require(all(config[k] == v for k, v in dict(epochs=10, seed=0, window_loss_weight=0.1, max_horizon=4).items())
            and report['selected_lr'] == 0.001, 'Unexpected saved CRN5-A training configuration')
    crn.verify_manifest(args.train, 'train', smoke=False)
    rows = crn.read_rows(args.train, 'train')
    saved = torch.load(args.crn5_checkpoint, map_location='cpu', weights_only=False)
    require(saved['training_variant'] == 'CRN5-A' and saved['max_horizon'] == 4 and
            saved['lr'] == 0.001 and saved['window_loss_weight'] == 0.1, 'Unexpected CRN5-A metadata')
    assignments = crn.folds_by_scenario(rows, 0)
    require(assignments == report['fold_assignments'], 'Original CRN5-A fold assignments differ')
    require(set(saved['train_scenario_ids']) == set(assignments), 'Collector differs from frozen train cohort')
    sources = {str(p.resolve()): file_hash(p) for p in
               (args.train, args.train.with_suffix('.manifest.json'), args.crn5_checkpoint,
                report_path, Path(saved['v2_checkpoint']), Path(__file__), Path(crn.__file__),
                ROOT / 'traffic_llm/collision_risk_net.py', ROOT / 'traffic_llm/predict_model.py',
                ROOT / 'traffic_llm/predict_nets.py')}
    args.out.mkdir(parents=True, exist_ok=False)
    configuration = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                         original_crn_config=config, reranker=R1_CONFIG, feature_names=FEATURE_NAMES,
                         source_hashes=sources, torch_version=torch.__version__,
                         rules={'filter_only': 'CRN positive AND rank-1 R1 risk >= threshold',
                                'rerank_plus_filter': 'CRN positive AND max top-k R1 risk >= threshold'},
                         missing_gt_pair_policy='No synthetic pair positives; exclude positive window/ranking loss without a GT candidate',
                         threshold_policy='Minimize FPs with TP >= ceil(0.97 * calibration CRN TP); ties favor more TPs then higher threshold',
                         fixed_crn_threshold=saved['threshold_by_horizon']['2'])
    write_json(args.out / 'configuration.json', configuration)
    cache = CRNCache(rows, saved, config, args, sources)
    original = {}
    # The original fold seeds stay 0..4 even when --seed changes R1 folds.
    for fold in range(5):
        fitting = [r for r in rows if assignments[r['scenario_id']] != fold]
        held = [r for r in rows if assignments[r['scenario_id']] == fold]
        original.update(cache.infer(fitting, held, fold))
    pair_records = []
    for row in crn.eligible_rows(rows, 4):
        pairs = itertools.combinations([a['actor_id'] for a in row['actors']], 2)
        pair_records.append(dict(window_id=row['window_id'], scenario_id=row['scenario_id'],
            outer_fold=assignments[row['scenario_id']], pairs=[
                dict(actor_pair=pair, hazards=hazards, cumulative_risks=pair_statistics(hazards)[0],
                     event_probabilities=pair_statistics(hazards)[1])
                for pair, hazards in zip(pairs, original[row['window_id']])]))
    write_json(args.out / 'crn5_original_oof_pairs.json', dict(fold_assignments=assignments, windows=pair_records))
    del pair_records
    baseline = verify_original(rows, original, saved, report)
    write_json(args.out / 'original_oof_baseline.json', baseline)
    print('Original CRN5-A OOF baseline verified', flush=True)
    threshold = saved['threshold_by_horizon']['2']
    predicted, fold_audit = run_nested(rows, original, assignments, cache, args, threshold)
    require(len(predicted) == baseline['identifiable_windows'] and
            len({r['window_id'] for r in predicted}) == len(predicted), 'Nested OOF coverage mismatch')
    final_thresholds = {rule: select_threshold(predicted, rule, threshold) for rule in RULES}
    final_data = examples(rows, original, args.top_k)
    final_model = fit_reranker(final_data, args.device, args.batch_size, args.seed)
    torch.save(dict(state_dict={k: v.cpu() for k, v in final_model.state_dict().items()},
                    architecture='CRN5-R1-residual-hazard-MLP', configuration=configuration,
                    thresholds=final_thresholds, top_k=args.top_k,
                    train_scenario_ids=sorted(assignments), crn5_checkpoint_sha256=FROZEN_SHA),
               args.out / 'crn5_r1_final.pt')
    nested = evaluate(predicted, threshold)
    summary = dict(original_crn5_oof=baseline, nested_unbiased_oof=nested,
                   collector_windows=len(rows),
                   unidentifiable_2s_windows=sum(not crn.identifiable(r, 2) for r in rows),
                   crn_eligible_windows=len(original),
                   final_thresholds=final_thresholds,
                   final_threshold_selection_metrics=evaluate(predicted, threshold, final_thresholds),
                   final_threshold_selection_metrics_are_unbiased=False,
                   nested_retains_97_percent={rule: nested[rule]['baseline_recall_retention'] >= 0.97 for rule in RULES},
                   note='Nested held-out retention is measured, not guaranteed by calibration constraints.',
                   source_hashes=sources, fold_audit=fold_audit)
    write_json(args.out / 'train_oof_summary.json', summary)
    write_json(args.out / 'crn_fit_audit.json', cache.audit)
    write_json(args.out / 'reranker_oof_predictions.json', predicted)
    for r in predicted:
        b = r['crn_risk'] >= threshold
        r['baseline_prediction'] = b
        for rule in RULES:
            p = b and r[rule + '_score'] >= r[rule + '_threshold']
            r[rule + '_prediction'] = p
            r[rule + '_transition'] = outcome(r['truth'], b) + '->' + outcome(r['truth'], p)
            r[rule + '_final_threshold_prediction'] = b and r[rule + '_score'] >= final_thresholds[rule]
    with (args.out / 'per_window_transitions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(predicted[0]))
        writer.writeheader()
        for r in predicted:
            writer.writerow({k: json.dumps(v) if isinstance(v, (tuple, list)) or v is None else v for k, v in r.items()})
    require(all(file_hash(Path(p)) == h for p, h in sources.items()), 'An input source changed during training')
    print(json.dumps(dict(output=str(args.out), nested_unbiased_oof=nested, final_thresholds=final_thresholds), indent=2))


if __name__ == '__main__':
    main()
