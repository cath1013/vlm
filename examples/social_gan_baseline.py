#!/usr/bin/env python3
"""Train official SGAN on official train only, or evaluate the frozen 219 windows.

All outputs require a new directory. No download or pretrained pedestrian weights.
Example: python examples/social_gan_baseline.py train --root DATA --carla-maps MAPS --out NEW
Example: python examples/social_gan_baseline.py evaluate --root DATA --carla-maps MAPS --checkpoint MODEL --out NEW
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_llm.social_gan import (SEED, SAMPLES, POLICY, OBS_TIMES, FUTURE_TIMES,
    GENERATOR, DISCRIMINATOR, HISTORY_POLICY, SocialGANPredictor, stable_seed, scene_tensors, sample_scene)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_hash():
    from traffic_llm._vendor.sgan import models
    return hashlib.sha256(Path(models.__file__).read_bytes()).hexdigest()


def runner_for(args):
    from traffic_llm.config import PipelineConfig
    from traffic_llm.da_runner import DeepAccidentRunner
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = 'sensor3d'
    return DeepAccidentRunner(args.root, cfg)


def snapshots(runner, scenario, maps):
    from traffic_llm.carla_map import find_xodr
    from traffic_llm.da_runner import DeepAccidentRunner
    xodr = find_xodr(scenario.town, [maps])
    if not xodr:
        raise ValueError(f'Missing map for {scenario.town}')
    # Names can repeat across mini/train/val. The legacy resolver searches its
    # root and returns the first match when a scenario_type is supplied. Scope
    # that search to the split root recorded by the official source scan.
    split_runner = DeepAccidentRunner(scenario.root, runner.cfg)
    built = split_runner.build(scenario.scenario, scenario.scenario_type, opendrive_path=xodr)
    if (Path(built.scenario.root).resolve() != Path(scenario.root).resolve() or
            built.scenario.split != scenario.split or
            built.scenario.scenario_id != scenario.scenario_id):
        raise ValueError(
            f'Resolved scenario differs from source manifest: '
            f'expected {scenario.root}/{scenario.scenario_id}, '
            f'got {built.scenario.root}/{built.scenario.scenario_id}')
    # Disable irrelevant legacy prediction; observed actor states are unchanged.
    with patch('traffic_llm.pipeline.predict', return_value=[]):
        snaps = list(built.snapshots(rate_hz=2.))
    return snaps


def training_scenes(snaps, sid, device):
    """All observed actors provide context; future availability masks loss only."""
    import torch
    from traffic_llm.actor_tracks import future_track
    per_actor = {}
    for snap in snaps:
        for actor in snap.actors:
            per_actor.setdefault(actor.actor_id, []).append((snap.t, actor.world_xy))
    for snap in snaps:
        if snap.t - snaps[0].t < 5 or not snap.actors:
            continue
        actors = sorted(snap.actors, key=lambda a: a.actor_id)
        obs, rel, bounds, origin = scene_tensors(actors, snap.t, device)
        targets, valid = [], []
        for actor in actors:
            track, complete = future_track(per_actor, actor.actor_id, snap.t, 2., dt=.5)
            valid.append(complete and len(track) == 5)
            targets.append(track[1:] if valid[-1] else [(0., 0.)]*4)
        mask = torch.tensor(valid, dtype=torch.bool, device=device)
        if not mask.any():
            continue
        future = torch.tensor(targets, dtype=torch.float32, device=device).transpose(0, 1).contiguous() - origin
        yield obs, rel, bounds, future, mask, f'{sid}:{snap.t:.9f}:' + ','.join(a.actor_id for a in actors)


def train(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import torch.nn.functional as F
    from traffic_llm._vendor.sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runner = runner_for(args)
    scenarios = sorted((s for s in runner.list_scenarios() if s.split == 'train'), key=lambda s:s.scenario_id)
    if not scenarios:
        raise ValueError('No official training scenarios')
    val = [s for s in scenarios if stable_seed(s.scenario_id) % 10 == 0]
    fit = [s for s in scenarios if s not in val]
    if not fit or not val:
        raise ValueError('Empty scenario-disjoint development partition')
    # No official-val scenario is built, loaded, or used for checkpoint selection.
    data = {s.scenario_id: list(training_scenes(snapshots(runner, s, args.carla_maps),
                                               s.scenario_id, args.device)) for s in scenarios}
    if not any(data[s.scenario_id] for s in fit) or not any(data[s.scenario_id] for s in val):
        raise ValueError('No supervised training/internal-validation scenes')
    g = TrajectoryGenerator(**GENERATOR).to(args.device)
    d = TrajectoryDiscriminator(**DISCRIMINATOR).to(args.device)
    go = torch.optim.Adam(g.parameters(), lr=5e-4)
    do = torch.optim.Adam(d.parameters(), lr=5e-4)
    best, report = float('inf'), []

    def discriminator_input(obs, future, mask):
        full = torch.cat([obs[:, mask], future[:, mask]])
        relative = torch.zeros_like(full)
        relative[1:] = full[1:] - full[:-1]
        bounds = torch.tensor([[0, int(mask.sum())]], device=args.device)
        # Official local discriminator squeezes singleton batches; retain its
        # learned layers and restore batch dimension through a forward hook.
        return d(full, relative, bounds)

    hook = d.real_classifier.register_forward_pre_hook(
        lambda module, inputs: (inputs[0].reshape(-1, DISCRIMINATOR['h_dim']),))
    for epoch in range(args.epochs):
        g.train(); d.train()
        scenes = [item for s in fit for item in data[s.scenario_id]]
        random.shuffle(scenes)
        for obs, rel, bounds, target, mask, key in scenes:
            for _ in range(2):
                do.zero_grad(set_to_none=True)
                fake = g(obs, rel, bounds).cumsum(0) + obs[-1]
                real_score = discriminator_input(obs, target, mask)
                fake_score = discriminator_input(obs, fake.detach(), mask)
                # Official SGAN soft-label adversarial objective.
                loss = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score)*random.uniform(.7, 1.2))
                loss += F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
                loss.backward(); do.step()
            go.zero_grad(set_to_none=True)
            d.requires_grad_(False)
            samples = [g(obs, rel, bounds).cumsum(0) + obs[-1] for _ in range(SAMPLES)]
            variety = torch.stack([((x[:, mask]-target[:, mask])**2).sum() for x in samples]).min()
            variety = variety / (4 * mask.sum())
            score = discriminator_input(obs, samples[-1], mask)
            loss = variety + F.binary_cross_entropy_with_logits(score, torch.ones_like(score)*random.uniform(.7, 1.2))
            loss.backward(); go.step()
            d.requires_grad_(True)
        g.eval()
        total, count = 0., 0
        with torch.inference_mode():
            for s in val:
                for obs, rel, bounds, target, mask, key in data[s.scenario_id]:
                    prediction = sample_scene(g, obs, rel, bounds, key)
                    errors = (prediction[:, mask]-target[:, mask]).norm(dim=-1)
                    total += errors.sum().item(); count += errors.numel()
        ade = total/count
        report.append(dict(epoch=epoch+1, internal_val_ade_m=ade))
        if ade < best:
            best = ade
            saved = dict(architecture='official-social-gan-v1', generator_config=GENERATOR,
                discriminator_config=DISCRIMINATOR, aggregation=POLICY, seed=SEED,
                history_policy=HISTORY_POLICY,
                observation_offsets_s=list(OBS_TIMES), future_offsets_s=list(FUTURE_TIMES),
                train_scenario_ids=[s.scenario_id for s in fit],
                internal_val_scenario_ids=[s.scenario_id for s in val],
                official_train_scenario_ids=[s.scenario_id for s in scenarios],
                official_models_sha256=source_hash(), epoch=epoch+1, internal_val_ade_m=ade,
                g_state=g.state_dict(), d_state=d.state_dict(),
                generator_optimizer=go.state_dict(), discriminator_optimizer=do.state_dict())
            torch.save(saved, args.out/'best.pt')
        write_json(args.out/'training.json', report)
    hook.remove()


def evaluate(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    from examples import evaluate_frozen_models_api_val104 as frozen
    from examples import evaluate_pair_reranker_v4_0_2 as protocol
    from traffic_llm.accident_qa import WindowConfig, build_windows, window_ground_truth
    from traffic_llm.deepaccident import estimate_collision
    from traffic_llm.predict_model import PredictContext
    from traffic_llm.serialize import rank_actors
    from traffic_llm.swept_path import swept_pair_clearances
    predictor = SocialGANPredictor(args.checkpoint, args.device)
    ledger = frozen.load_cohort()
    runner = runner_for(args)
    scenarios = sorted((s for s in runner.list_scenarios() if s.split == 'val'), key=lambda s:s.scenario_id)
    if len(scenarios) != 104 or predictor.development_scenarios & {s.scenario_id for s in scenarios}:
        raise ValueError('Expected untouched official Val104')
    wc = WindowConfig(window_s=5., stride_s=1., horizon_s=5., snapshot_rate_hz=2.,
                      warmup=False, full_window=False)
    records = {}
    for s in scenarios:
        if not any(r['scenario_id'] == s.scenario_id for r in ledger.values()):
            continue
        snaps = snapshots(runner, s, args.carla_maps)
        collision = estimate_collision(s, runner.cfg.deepaccident)
        windows, _ = build_windows(snaps, wc, collision_time_s=collision.time_s if collision and collision.occurred else None)
        identities = {a:s.meta.agent_id_of(a) for a in s.agents}
        for w in windows:
            wid = f'{s.scenario_id}:{w.index}:{w.t_end:.3f}'
            if wid not in ledger:
                continue
            gt = window_ground_truth(w, wc, collision=collision, agent_carla_ids=identities,
                scenario_id=s.scenario_id, scenario_split=s.scenario_type, dataset_split=s.split,
                town=s.town, data_end_s=snaps[-1].t)
            truth = protocol.evaluation_truth(gt)
            if truth is None:
                raise ValueError(f'Frozen window became censored: {wid}')
            expected_bucket = int(ledger[wid]['gt_bucket'])
            if truth[0] != (expected_bucket if expected_bucket in (1, 2) else None):
                raise ValueError(f'Frozen label mismatch: {wid}')
            # Merge actual observed timestamps from the full window; never future states.
            for a in w.last.actors:
                points = {t:(x,y) for t,x,y in a.track_history if w.t_end-5 <= t <= w.t_end}
                for snap in w.snapshots:
                    for past in snap.actors:
                        if past.actor_id == a.actor_id and w.t_end-5 <= snap.t <= w.t_end:
                            points[snap.t] = past.world_xy
                a.track_history = [(t,*xy) for t,xy in sorted(points.items())]
            contexts = [PredictContext(a, 2., t_s=w.t_end, scenario_id=s.scenario_id) for a in w.last.actors]
            predictions = predictor.predict_scene(contexts)
            for a in w.last.actors:
                a.predictions = [predictions[a.actor_id]]
            actors, _ = rank_actors(w.last, wc.actor_cap(len(w.last.actors)))
            pairs = swept_pair_clearances(actors, horizon_s=2., sample_dt_s=wc.swept_sample_dt_s,
                contact_margin_m=wc.swept_contact_margin_m, exclude_touching_now=wc.swept_exclude_touching_now,
                exclude_static_pairs=wc.swept_exclude_static_pairs)
            contacts = [p for p in pairs if p.predicted_contact]
            predicted = bool(contacts)
            hit = any(protocol.matches_gt_pair(asdict(p), truth[1]) for p in contacts)
            records[wid] = dict(window_id=wid, scenario_id=s.scenario_id, gt_bucket=expected_bucket,
                truth=truth[0] is not None, predicted=predicted, correct_gt_pair_hit=hit,
                predicted_onset_bucket=min((max(1, math.ceil(p.first_contact_s))
                    for p in contacts if p.first_contact_s is not None), default=None),
                contacts=[asdict(p) for p in contacts], trajectories=[dict(actor_id=a.actor_id,
                    timestamps_s=[w.t_end+t for t in FUTURE_TIMES],
                    world_xy=predictions[a.actor_id].waypoints[1:]) for a in w.last.actors])
    if len(records) != 219 or set(records) != set(ledger):
        raise ValueError('SGAN must score exactly the frozen 219 windows')
    rows = [records[k] for k in sorted(records)]
    counts = {name: sum((r['truth'], r['predicted']) == flags for r in rows)
              for name, flags in dict(TP=(True,True), FP=(False,True), TN=(False,False), FN=(True,False)).items()}
    write_json(args.out/'summary.json', dict(model='official Social GAN', seed=SEED,
        aggregation=POLICY, history_policy=HISTORY_POLICY, windows=219, **counts,
        correct_gt_pair_hits=sum(r['truth'] and r['predicted'] and r['correct_gt_pair_hit'] for r in rows),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_sha256=source_hash(), cohort_sha256=hashlib.sha256(Path(frozen.COHORT).read_bytes()).hexdigest(),
        window_config=asdict(wc)))
    with (args.out/'predictions.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('train', 'evaluate'))
    ap.add_argument('--root', required=True)
    ap.add_argument('--carla-maps', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--epochs', type=int, default=200)
    args = ap.parse_args()
    if args.epochs <= 0 or args.command == 'evaluate' and args.checkpoint is None:
        ap.error('Positive epochs required; evaluate requires --checkpoint')
    args.out.mkdir(parents=True, exist_ok=False)
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
