"""Official SGAN adapter: 5 s observed history, 2 s future, world XY metres.

Histories are interpolated using their timestamps at 2 Hz. Missing leading
history is held at the first available observation (never extrapolated from GT).
Interior gaps > 1 s are held at the preceding observation. This dense-input
policy is shared by training and inference; the SGAN architecture is unchanged.
Prediction aggregation is the medoid of 20 joint-scene samples in XY distance,
with ties resolved by sample index. Neither targets nor collision labels enter it.
"""
from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from pathlib import Path

OBS_TIMES = tuple(-5 + .5 * i for i in range(11))
FUTURE_TIMES = (.5, 1., 1.5, 2.)
SEED = 1729
SAMPLES = 20
POLICY = 'joint_scene_medoid_20_v1'
HISTORY_POLICY = 'timestamp_linear_2hz_leading_hold_gap_over_1s_hold_v1'
GENERATOR = dict(obs_len=11, pred_len=4, embedding_dim=64, encoder_h_dim=64,
                 decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(8,),
                 noise_type='gaussian', noise_mix_type='global', pooling_type='pool_net',
                 pool_every_timestep=True, dropout=0., bottleneck_dim=1024,
                 batch_norm=False)
DISCRIMINATOR = dict(obs_len=11, pred_len=4, embedding_dim=64, h_dim=64,
                     mlp_dim=1024, num_layers=1, batch_norm=False, d_type='local')


def stable_seed(key):
    return (SEED + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)) % (2**31)


def history_xy(actor, cutoff):
    points = {float(t): (float(x), float(y)) for t, x, y in actor.track_history
              if cutoff - 5 <= t <= cutoff and all(math.isfinite(v) for v in (t, x, y))}
    points[float(cutoff)] = tuple(map(float, actor.world_xy))
    times = sorted(points)
    result = []
    for offset in OBS_TIMES:
        t = cutoff + offset
        i = bisect_right(times, t) - 1
        if i < 0:
            xy = points[times[0]]
        elif i == len(times)-1 or times[i+1] - times[i] > 1.:
            xy = points[times[i]]
        else:
            a, b = times[i:i+2]
            w = (t-a)/(b-a)
            xy = tuple(points[a][j]*(1-w) + points[b][j]*w for j in range(2))
        result.append(xy)
    if not all(math.isfinite(v) for xy in result for v in xy):
        raise ValueError('Nonfinite observed XY')
    return result


def scene_tensors(actors, cutoff, device='cpu'):
    import torch
    obs = torch.tensor([history_xy(a, cutoff) for a in actors], dtype=torch.float32,
                       device=device).transpose(0, 1).contiguous()
    # Shared scene origin preserves inter-actor geometry and world axes.
    origin = obs[-1, 0].clone()
    obs = obs - origin
    rel = torch.zeros_like(obs)
    rel[1:] = obs[1:] - obs[:-1]
    bounds = torch.tensor([[0, len(actors)]], dtype=torch.long, device=device)
    return obs, rel, bounds, origin


def sample_scene(generator, obs, rel, bounds, key):
    import torch
    devices = [obs.device.index or 0] if obs.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(stable_seed(key))
        samples = torch.stack([generator(obs, rel, bounds).cumsum(0) + obs[-1]
                               for _ in range(SAMPLES)])
    # Select one coherent joint scene rather than mixing actors across samples.
    distances = torch.cdist(samples.flatten(1), samples.flatten(1))
    return samples[distances.sum(1).argmin()]


class SocialGANPredictor:
    """Drop-in PipelineConfig.predictor implementing the joint-scene contract."""
    def __init__(self, checkpoint, device='cpu'):
        import torch
        from ._vendor.sgan.models import TrajectoryGenerator
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if (saved.get('architecture') != 'official-social-gan-v1' or
                saved.get('generator_config') != GENERATOR or
                saved.get('aggregation') != POLICY or saved.get('seed') != SEED or
                saved.get('history_policy') != HISTORY_POLICY or
                saved.get('observation_offsets_s') != list(OBS_TIMES) or
                saved.get('future_offsets_s') != list(FUTURE_TIMES)):
            raise ValueError('Incompatible SGAN checkpoint/protocol')
        train, val = set(saved['train_scenario_ids']), set(saved['internal_val_scenario_ids'])
        official = set(saved['official_train_scenario_ids'])
        if not train or not val or train & val or train | val != official:
            raise ValueError('Invalid official-train scenario partition')
        from ._vendor.sgan import models
        if saved.get('official_models_sha256') != hashlib.sha256(Path(models.__file__).read_bytes()).hexdigest():
            raise ValueError('SGAN source differs from checkpoint provenance')
        self.development_scenarios = official
        self.device = device
        self.model = TrajectoryGenerator(**GENERATOR).to(device)
        self.model.load_state_dict(saved['g_state'], strict=True)
        self.model.eval().requires_grad_(False)

    def predict_scene(self, contexts):
        import torch
        from .schemas import PredictedPath
        if not contexts:
            return {}
        contexts = sorted(contexts, key=lambda c: c.actor.actor_id)
        cutoff, sid = contexts[0].t_s, contexts[0].scenario_id
        if cutoff is None or any(c.t_s != cutoff or c.scenario_id != sid for c in contexts):
            raise ValueError('SGAN requires a single timestamped scene')
        actors = [c.actor for c in contexts]
        if len({a.actor_id for a in actors}) != len(actors):
            raise ValueError('Duplicate actor IDs')
        with torch.inference_mode():
            obs, rel, bounds, origin = scene_tensors(actors, cutoff, self.device)
            key = f'{sid}:{cutoff:.9f}:' + ','.join(a.actor_id for a in actors)
            xy = (sample_scene(self.model, obs, rel, bounds, key) + origin).cpu().tolist()
        return {a.actor_id: PredictedPath('Social GAN', 1.,
                    [tuple(a.world_xy)] + [tuple(step[i]) for step in xy], 2.,
                    waypoint_times_s=[0., *FUTURE_TIMES]) for i, a in enumerate(actors)}

    def predict_records(self, contexts):
        """Export stable IDs, world XY and absolute future timestamps."""
        paths = self.predict_scene(contexts)
        return [dict(actor_id=c.actor.actor_id, scenario_id=c.scenario_id,
                     timestamps_s=[c.t_s+t for t in FUTURE_TIMES],
                     world_xy=paths[c.actor.actor_id].waypoints[1:])
                for c in sorted(contexts, key=lambda c: c.actor.actor_id)]
