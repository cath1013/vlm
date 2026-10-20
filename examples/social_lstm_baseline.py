#!/usr/bin/env python3
"""Train Social LSTM on official train only, or evaluate the frozen 219 windows.

All outputs require a new directory. No download or pretrained pedestrian weights.
Example: python examples/social_lstm_baseline.py train --root DATA --carla-maps MAPS --out NEW
Example: python examples/social_lstm_baseline.py evaluate --root DATA --carla-maps MAPS --checkpoint MODEL --out NEW

Training caches CPU scenes and packs at most 8 scenes / 128 actors per step.
Use --batch-size 1 for the previous single-scene optimizer-step schedule.
--validation-nll-every 0 skips optional teacher-forced validation likelihood;
ADE/FDE are always computed with deterministic autoregressive predictions.
--resume PREVIOUS/last.pt --out NEW restores the last completed epoch, including
RNG and early-stopping state. An interrupted partial epoch is replayed. --epochs
is the total epoch limit; --patience 0 disables early stopping.
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
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_llm.social_lstm import (SEED, POLICY, OBS_TIMES, FUTURE_TIMES,
    MODEL_CONFIG, ARCHITECTURE, HISTORY_POLICY, SELECTION, ATTRIBUTION,
    SocialLSTM, SocialLSTMPredictor, gaussian_nll, source_hash, stable_seed, scene_tensors)


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def provenance():
    return dict(architecture=ARCHITECTURE, model_config=MODEL_CONFIG,
        prediction_policy=POLICY, history_policy=HISTORY_POLICY, seed=SEED,
        observation_offsets_s=list(OBS_TIMES), future_offsets_s=list(FUTURE_TIMES),
        checkpoint_selection=SELECTION, source_attribution=ATTRIBUTION,
        source_sha256=source_hash(),
        entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        partition_policy='official_train_only; stable_seed(scenario_id) % 10 == 0 for internal_val',
        objective='future_teacher_forced_correlated_bivariate_gaussian_NLL',
        numerical_stability=dict(log_sigma_bounds=[-7., 7.], rho_margin=1e-5),
        optimizer=dict(name='RMSprop', lr=.003, alpha=.9, eps=1e-8,
                       weight_decay=0., gradient_clip_norm=10.))


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


def training_scenes(snaps, sid):
    """All observed actors provide context; future availability masks loss only."""
    device = 'cpu'
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


def scene_batches(scenes, batch_size, max_actors, max_pairs):
    """Pack whole CPU scenes without padding or changing actor order.

    Pair budget bounds sum(N_scene**2); an oversized scene raises rather than
    dropping actors or silently exceeding the requested memory budget.
    """
    batch, actors, pairs = [], 0, 0
    for scene in scenes:
        n = scene[0].shape[1]
        if n > max_actors or n*n > max_pairs:
            raise ValueError(f'Scene {scene[-1]} has {n} actors; increase '
                             '--max-batch-actors/--max-batch-pairs to include it')
        if batch and (len(batch) >= batch_size or actors+n > max_actors or pairs+n*n > max_pairs):
            yield batch
            batch, actors, pairs = [], 0, 0
        batch.append(scene)
        actors += n
        pairs += n*n
    if batch:
        yield batch


def device_batch(scenes, device):
    """Only the current packed batch is transferred; the scene cache stays CPU."""
    import torch
    bounds, weights, start = [], [], 0
    for obs, rel, unused_bounds, target, mask, key in scenes:
        end = start + obs.shape[1]
        bounds.append((start, end))
        # Average each scene's original Gaussian NLL, then average scenes.
        valid_count = int(mask.sum())
        weights.extend([1. / (4 * valid_count * len(scenes))] * valid_count)
        start = end
    obs = torch.cat([s[0] for s in scenes], dim=1)
    target = torch.cat([s[3] for s in scenes], dim=1)
    mask = torch.cat([s[4] for s in scenes])
    indices = mask.nonzero(as_tuple=True)[0]
    tensors = (obs, target, mask, indices, torch.tensor(weights, dtype=obs.dtype))
    if torch.device(device).type == 'cuda':
        tensors = tuple(t.pin_memory() for t in tensors)
    obs, target, mask, indices, weights = (t.to(device, non_blocking=True) for t in tensors)
    return obs, target, mask, indices, weights, bounds


def cpu_tree(value):
    """Checkpoint snapshots must not leave a second model/optimizer in VRAM."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, value):
    import torch
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state(device):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else None)


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def train(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
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
                                               s.scenario_id)) for s in scenarios}
    if not any(data[s.scenario_id] for s in fit) or not any(data[s.scenario_id] for s in val):
        raise ValueError('No supervised training/internal-validation scenes')
    model = SocialLSTM(**MODEL_CONFIG).to(args.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=.003, alpha=.9)
    best, report, best_checkpoint = float('inf'), [], None
    start_epoch, stale_epochs, previous_elapsed = 0, 0, 0.
    metadata = provenance()
    execution = dict(batch_size=args.batch_size, max_batch_actors=args.max_batch_actors,
        max_batch_pairs=args.max_batch_pairs, validation_nll_every=args.validation_nll_every,
        batch_loss='mean_of_scene_mean_Gaussian_NLL', data_cache_device='cpu')
    metadata.update(train_scenario_ids=[s.scenario_id for s in fit],
        internal_val_scenario_ids=[s.scenario_id for s in val],
        official_train_scenario_ids=[s.scenario_id for s in scenarios],
        torch_version=str(torch.__version__), device=args.device, execution=execution)
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        # Epoch-boundary replay requires the same code, partition, batching and device.
        for key, value in metadata.items():
            if saved.get(key) != value:
                raise ValueError(f'Resume provenance/configuration mismatch: {key}')
        if saved.get('resume_format') != 1:
            raise ValueError('--resume requires an optimized last.pt checkpoint')
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        start_epoch = saved['epoch']
        report = saved['history']
        best_checkpoint = saved['best_checkpoint']
        best = best_checkpoint['internal_val_ade_m']
        stale_epochs = saved['stale_epochs']
        previous_elapsed = saved['elapsed_s']
        restore_rng(saved['rng_state'])
        save_checkpoint(args.out/'best.pt', best_checkpoint)
        # Keep the resume point available even if already at the epoch/patience limit.
        save_checkpoint(args.out/'last.pt', saved)
        del saved
        print(f'Resumed epoch {start_epoch}; next epoch {start_epoch+1}; '
              f'best internal-val ADE={best:.6f} m', flush=True)
    fit_scenes = [item for s in fit for item in data[s.scenario_id]]
    val_scenes = [item for s in val for item in data[s.scenario_id]]
    run_started = time.monotonic()
    stopped = args.patience > 0 and stale_epochs >= args.patience
    for epoch in range(start_epoch, args.epochs):
        if stopped:
            break
        epoch_started = time.monotonic()
        model.train()
        scenes = list(fit_scenes)
        random.shuffle(scenes)
        train_totals = torch.zeros(2, device=args.device, dtype=torch.float64)
        train_count = 0
        for batch in scene_batches(scenes, args.batch_size, args.max_batch_actors, args.max_batch_pairs):
            obs, target, mask, indices, weights, bounds = device_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            params = model(obs, target, mask, scene_bounds=bounds)
            nll = gaussian_nll(params, target, indices, reduction='none')
            loss = (nll * weights[None, :]).sum()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=False)
            optimizer.step()
            # Aggregate detached device scalars, synchronize once per epoch.
            train_totals[0] += nll.detach().sum()
            train_totals[1] += (~torch.isfinite(norm)).to(train_totals.dtype)
            train_count += nll.numel()
            del obs, target, mask, indices, weights, params, nll, loss, norm
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        totals = torch.zeros(3, device=args.device, dtype=torch.float64)
        count, actors = 0, 0
        compute_nll = args.validation_nll_every > 0 and (epoch+1) % args.validation_nll_every == 0
        with torch.inference_mode():
            for batch in scene_batches(val_scenes, args.batch_size, args.max_batch_actors, args.max_batch_pairs):
                obs, target, mask, indices, weights, bounds = device_batch(batch, args.device)
                # Encode history once, even when optional teacher-forced NLL is requested.
                params, state = model.encode(obs, bounds)
                prediction = model.decode(params, state, scene_bounds=bounds)[..., :2]
                errors = (prediction[:, indices] - target[:, indices]).norm(dim=-1)
                totals[0] += errors.sum()
                totals[1] += errors[-1].sum()
                count += errors.numel()
                actors += indices.numel()
                if compute_nll:
                    teacher = model.decode(params, state, target, mask, bounds)
                    totals[2] += gaussian_nll(teacher, target, indices) * errors.numel()
                    del teacher
                del obs, target, mask, indices, weights, params, state, prediction, errors
        train_sum, invalid_gradients = train_totals.cpu().tolist()
        total, final_total, nll_total = totals.cpu().tolist()
        ade, fde, train_loss = total/count, final_total/actors, train_sum/train_count
        if invalid_gradients or not all(math.isfinite(v) for v in (ade, fde, train_loss, nll_total)):
            raise ValueError('Nonfinite loss, gradient or internal-validation metrics; last.pt remains at prior epoch')
        row = dict(epoch=epoch+1, train_nll=train_loss,
            internal_val_ade_m=ade, internal_val_fde_m=fde,
            internal_val_actor_trajectories=actors)
        if compute_nll:
            row['internal_val_teacher_forced_nll'] = nll_total/count
        improved = ade < best
        if improved:
            best, stale_epochs = ade, 0
        else:
            stale_epochs += 1
        stopped = args.patience > 0 and stale_epochs >= args.patience
        model_state = cpu_tree(model.state_dict())
        if improved:
            best_checkpoint = dict(**metadata, **row, model_state=model_state)
            save_checkpoint(args.out/'best.pt', best_checkpoint)
        # Times include validation and best-checkpoint I/O; last-checkpoint I/O is
        # included in the next epoch's cumulative elapsed time.
        row.update(epoch_duration_s=time.monotonic()-epoch_started,
                   elapsed_s=previous_elapsed+time.monotonic()-run_started)
        report.append(row)
        last = dict(**metadata, **row, resume_format=1, model_state=model_state,
            optimizer_state=cpu_tree(optimizer.state_dict()), rng_state=rng_state(args.device),
            history=report, best_checkpoint=best_checkpoint, stale_epochs=stale_epochs,
            patience=args.patience)
        save_checkpoint(args.out/'last.pt', last)
        del last, model_state
        write_json(args.out/'training.json', report)
        write_json(args.out/'summary.json', dict(**metadata, completed_epochs=epoch+1,
            best_epoch=best_checkpoint['epoch'], best_internal_val_ade_m=best,
            checkpoint='best.pt', last_checkpoint='last.pt', patience=args.patience,
            stale_epochs=stale_epochs, early_stopped=stopped, elapsed_s=row['elapsed_s'],
            resumed_from=str(args.resume) if args.resume else None))
        print(f"epoch={epoch+1}/{args.epochs} train_nll={train_loss:.6f} "
              f"val_ADE_m={ade:.6f} val_FDE_m={fde:.6f} "
              f"epoch_s={time.monotonic()-epoch_started:.2f} "
              f"elapsed_s={previous_elapsed+time.monotonic()-run_started:.2f} "
              f"stale={stale_epochs}/{args.patience}", flush=True)
    if not report:
        raise ValueError('No completed training epochs')
    # Also materialize reports when a resumed run is already at its limit.
    write_json(args.out/'training.json', report)
    write_json(args.out/'summary.json', dict(**metadata, completed_epochs=report[-1]['epoch'],
        best_epoch=best_checkpoint['epoch'], best_internal_val_ade_m=best,
        checkpoint='best.pt', last_checkpoint='last.pt', patience=args.patience,
        stale_epochs=stale_epochs, early_stopped=stopped,
        elapsed_s=previous_elapsed+time.monotonic()-run_started,
        resumed_from=str(args.resume) if args.resume else None))


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
    predictor = SocialLSTMPredictor(args.checkpoint, args.device)
    ledger = frozen.load_cohort()
    runner = runner_for(args)
    available = runner.list_scenarios()
    official_train = {s.scenario_id for s in available if s.split == 'train'}
    if predictor.development_scenarios != official_train:
        raise ValueError('Checkpoint development manifest differs from official training scenarios')
    scenarios = sorted((s for s in available if s.split == 'val'), key=lambda s:s.scenario_id)
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
        raise ValueError('Social LSTM must score exactly the frozen 219 windows')
    rows = [records[k] for k in sorted(records)]
    counts = {name: sum((r['truth'], r['predicted']) == flags for r in rows)
              for name, flags in dict(TP=(True,True), FP=(False,True), TN=(False,False), FN=(True,False)).items()}
    write_json(args.out/'summary.json', dict(**provenance(), model='Social LSTM', windows=219, **counts,
        checkpoint_epoch=predictor.checkpoint_epoch,
        checkpoint_internal_val_ade_m=predictor.internal_val_ade_m,
        correct_gt_pair_hits=sum(r['truth'] and r['predicted'] and r['correct_gt_pair_hit'] for r in rows),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        cohort_sha256=hashlib.sha256(Path(frozen.COHORT).read_bytes()).hexdigest(),
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
    ap.add_argument('--epochs', type=int, default=200, help='Total epoch limit, including resumed epochs')
    ap.add_argument('--batch-size', type=int, default=8, help='Maximum whole scenes per optimizer step')
    ap.add_argument('--max-batch-actors', type=int, default=128)
    ap.add_argument('--max-batch-pairs', type=int, default=16384, help='Maximum sum of squared scene actor counts')
    ap.add_argument('--validation-nll-every', type=int, default=0, help='Teacher-forced NLL interval; 0 disables this optional metric')
    ap.add_argument('--patience', type=int, default=30, help='Epochs without ADE improvement; 0 disables early stopping')
    ap.add_argument('--resume', type=Path, help='Resume an epoch-boundary last.pt into a NEW output directory')
    args = ap.parse_args()
    if args.epochs <= 0 or args.command == 'evaluate' and args.checkpoint is None:
        ap.error('Positive epochs required; evaluate requires --checkpoint')
    if min(args.batch_size, args.max_batch_actors, args.max_batch_pairs) <= 0:
        ap.error('Batch limits must be positive')
    if args.patience < 0 or args.validation_nll_every < 0:
        ap.error('Patience and NLL interval must be nonnegative')
    if args.resume and (args.command != 'train' or not args.resume.is_file()):
        ap.error('--resume requires train and an existing last.pt')
    args.out.mkdir(parents=True, exist_ok=False)
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
