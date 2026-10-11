#!/usr/bin/env python3
"""Evaluate frozen CRN5-A and CRN5-R1 on the official val104 collector.

Uses saved top-5 features and thresholds only; no fitting, calibration, replay,
scenario reconstruction, or API calls. Refuses to overwrite output directories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples import train_crn5_r1 as r1
from traffic_llm.collision_risk_net import CollisionRiskNet, load_v2
import torch

COLLECTOR = ROOT / 'out/collision_risk_hazard/run_20261006_window_balanced/val104.jsonl'
CRN5 = ROOT / 'out/collision_risk_hazard/frozen/crn5_a_w01/collision_risk_final.pt'
R1 = ROOT / 'out/experiments/crn5_r1_train483_v1/crn5_r1_final.pt'
EXPECTED = dict(tp=41, fp=29, tn=137, fn=12, correct_gt_pair_hits=31)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--out', type=Path, default=ROOT / 'out/experiments' /
                        ('crn5_r1_val104_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    args = parser.parse_args(argv)
    r1.require(args.batch_size > 0, 'batch-size must be positive')
    r1.require(not args.out.exists(), 'Output directory must be new')
    r1.require(r1.file_hash(CRN5) == r1.FROZEN_SHA, 'Frozen CRN5-A checkpoint changed')
    saved = torch.load(CRN5, map_location='cpu', weights_only=False)
    reranker_saved = torch.load(R1, map_location='cpu', weights_only=False)
    config = reranker_saved['configuration']
    threshold = saved['threshold_by_horizon']['2']
    thresholds = reranker_saved['thresholds']
    r1.require(saved['training_variant'] == 'CRN5-A' and saved['max_horizon'] == 4,
               'Unexpected CRN5-A metadata')
    r1.require(reranker_saved['architecture'] == 'CRN5-R1-residual-hazard-MLP' and
               reranker_saved['crn5_checkpoint_sha256'] == r1.FROZEN_SHA,
               'R1 checkpoint does not match frozen CRN5-A')
    r1.require(reranker_saved['top_k'] == config['arguments']['top_k'] == 5 and
               config['feature_names'] == r1.FEATURE_NAMES and config['reranker'] == r1.R1_CONFIG,
               'Unexpected saved R1 top-5 configuration')
    r1.require(config['fixed_crn_threshold'] == threshold and
               set(thresholds) == set(r1.RULES) and
               all(math.isfinite(t) and t >= 0 for t in thresholds.values()),
               'Invalid saved thresholds')
    paths = (COLLECTOR, COLLECTOR.with_suffix('.manifest.json'), CRN5, R1,
             Path(saved['v2_checkpoint']))
    sources = {str(p.resolve()): r1.file_hash(p) for p in paths}
    r1.crn.verify_manifest(COLLECTOR, 'val', smoke=False)
    rows = r1.crn.read_rows(COLLECTOR, 'val')
    cohort = {r['window_id'] for r in rows if r1.crn.identifiable(r, 2)}
    r1.require(len(cohort) == 219, 'Expected 219 identifiable val104 windows')
    validation_scenarios = {r['scenario_id'] for r in rows}
    for checkpoint in (saved, reranker_saved):
        r1.require(not validation_scenarios & set(checkpoint['train_scenario_ids']),
                   'Train/validation scenario overlap')
    model = CollisionRiskNet(load_v2(saved['v2_checkpoint'], args.device),
                             saved['hidden_dim'], saved['dropout'], max_horizon=4).to(args.device)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    print('Scoring frozen CRN5-A collector pairs', flush=True)
    predictions = r1.pair_predict(model, rows, args.device, args.batch_size)
    data = r1.examples(rows, predictions, reranker_saved['top_k'])
    r1.require(len(data) == 219 and {e['window_id'] for e in data} == cohort,
               'R1 and CRN5-A must evaluate the identical 219 windows')
    reranker = r1.PairReranker(reranker_saved['state_dict']['mean'],
                               reranker_saved['state_dict']['scale']).to(args.device)
    reranker.load_state_dict(reranker_saved['state_dict'], strict=True)
    reranker.eval().requires_grad_(False)
    records = sorted(r1.predict_reranker(reranker, data, args.device), key=lambda r: r['window_id'])
    metrics = r1.evaluate(records, threshold, thresholds)
    coverage = metrics['candidate_coverage']
    coverage['all_pairs_gt_pair_recall'] = (coverage['gt_pair_present_windows'] /
                                           coverage['positive_windows'])
    baseline = metrics['baseline']
    if any(baseline[k] != v for k, v in EXPECTED.items()):
        raise AssertionError(f'CRN5-A baseline mismatch: {baseline}; expected {EXPECTED}')
    examples_by_id = {e['window_id']: e for e in data}
    for record in records:
        e = examples_by_id[record['window_id']]
        b = record['crn_risk'] >= threshold
        record.update(gt_bucket=e['gt_bucket'], candidate_pairs=e['pairs'],
                      candidate_gt_hits=e['hits'], crn_threshold=threshold,
                      baseline_prediction=b, baseline_outcome=r1.outcome(record['truth'], b),
                      baseline_correct_gt_pair_hit=b and record['baseline_hit'])
        for rule in r1.RULES:
            p = b and record[rule + '_score'] >= thresholds[rule]
            hit = record['baseline_hit' if rule == 'filter_only' else 'reranked_hit']
            record.update({rule + '_threshold': thresholds[rule], rule + '_prediction': p,
                           rule + '_outcome': r1.outcome(record['truth'], p),
                           rule + '_correct_gt_pair_hit': p and hit,
                           rule + '_transition': r1.outcome(record['truth'], b) + '->' +
                           r1.outcome(record['truth'], p)})
        record['r1_mode_transition'] = record['filter_only_outcome'] + '->' + record['rerank_plus_filter_outcome']
    r1.require(all(r1.file_hash(Path(p)) == h for p, h in sources.items()),
               'An input artifact changed during evaluation')
    summary = dict(window_count=len(records), collector_windows=len(rows), horizon_s=2,
                   top_k=reranker_saved['top_k'], crn_threshold=threshold, r1_thresholds=thresholds,
                   threshold_policy='Saved training thresholds; no refitting or recalibration',
                   baseline_assertion_passed=True, expected_baseline=EXPECTED, metrics=metrics,
                   r1_mode_transitions=dict(sorted(Counter(r['r1_mode_transition'] for r in records).items())),
                   confusion_matrix_order='[[TN, FP], [FN, TP]]',
                   rules=config['rules'], source_hashes=sources,
                   device=args.device, batch_size=args.batch_size, torch_version=torch.__version__)
    args.out.mkdir(parents=True, exist_ok=False)
    r1.write_json(args.out / 'summary.json', summary)
    with (args.out / 'per_window.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, tuple)) or v is None else v
                             for k, v in record.items()})
    print(json.dumps(dict(output=str(args.out), metrics=metrics), indent=2))


if __name__ == '__main__':
    main()
