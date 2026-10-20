"""Social LSTM (Alahi et al., CVPR 2016), independently implemented.

Architecture reference: https://github.com/quancore/social-lstm (model.py,
getSocialTensor and Gaussian2DLikelihood in helper.py). No upstream code or
weights are vendored. Positions are scene-translated world XY in metres;
neighbor grids use world axes and metre units, rather than image coordinates.
History preparation matches examples/social_gan_baseline.py exactly.
"""
from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from pathlib import Path

import torch
from torch import nn

OBS_TIMES = tuple(-5 + .5 * i for i in range(11))
FUTURE_TIMES = (.5, 1., 1.5, 2.)
SEED = 1729
POLICY = 'autoregressive_conditional_mean_v1'
HISTORY_POLICY = 'timestamp_linear_2hz_leading_hold_gap_over_1s_hold_v1'
ARCHITECTURE = 'social-lstm-grid-gaussian-v1'
MODEL_CONFIG = dict(hidden_dim=128, embedding_dim=64, grid_size=8,
                    neighborhood_size_m=32., dropout=0.)
SELECTION = 'minimum_internal_validation_autoregressive_ADE_m; earliest_epoch_on_tie'
ATTRIBUTION = dict(paper='Alahi et al., Social LSTM: Human Trajectory Prediction in Crowded Spaces, CVPR 2016',
    paper_url='https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf',
    reference_url='https://github.com/quancore/social-lstm',
    implementation='Independent PyTorch implementation; no pretrained weights',
    adaptations='World-XY metre grid; shared scene translation; 11 observations and 4 future steps; deterministic conditional means')


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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


class SocialLSTM(nn.Module):
    """Shared actor LSTM with synchronous, per-step grid social pooling.

    Each cell sums other actors' previous hidden states. All actors update
    simultaneously; self is excluded. Grids are rebuilt during rollout.
    Outputs are (mu_x, mu_y, log_sigma_x, log_sigma_y, atanh_rho).
    """
    def __init__(self, hidden_dim=128, embedding_dim=64, grid_size=8,
                 neighborhood_size_m=32., dropout=0.):
        super().__init__()
        if grid_size <= 0 or neighborhood_size_m <= 0:
            raise ValueError('Positive grid and neighborhood sizes required')
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.neighborhood_size_m = neighborhood_size_m
        self.position_embedding = nn.Linear(2, embedding_dim)
        self.social_embedding = nn.Linear(grid_size * grid_size * hidden_dim, embedding_dim)
        self.cell = nn.LSTMCell(2 * embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 5)
        self.activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))

    def social_tensor(self, xy, hidden):
        n = xy.shape[0]
        # Bound temporary pair/grid tensors even for a crowded single scene.
        return torch.cat([self._pool_rows(xy, hidden, start, min(start + 16, n))
                          for start in range(0, n, 16)], dim=0)

    def _pool_rows(self, xy, hidden, start, end):
        n, g = xy.shape[0], self.grid_size
        # [focal actor, neighbor, XY]; half-open square neighborhood.
        offsets = xy[None, :, :] - xy[start:end, None, :]
        cells = torch.floor((offsets / self.neighborhood_size_m + .5) * g).long()
        inside = ((cells >= 0) & (cells < g)).all(-1)
        inside = inside & (torch.arange(start, end, device=xy.device)[:, None]
                           != torch.arange(n, device=xy.device)[None, :])
        # Matrix multiplication avoids nondeterministic CUDA scatter reductions.
        index = cells[..., 1] * g + cells[..., 0]
        grid = (index[..., None] == torch.arange(g*g, device=xy.device)) & inside[..., None]
        return torch.matmul(grid.to(hidden.dtype).transpose(1, 2), hidden).reshape(end-start, -1)

    def step(self, xy, state, scene_bounds=None, emit=True):
        hidden, cell = state
        # Bounds are CPU Python integers: no device-to-host synchronization.
        # Pool within each scene, then share the embedding and recurrent kernels.
        social = (self.social_tensor(xy, hidden) if scene_bounds is None else
                  torch.cat([self.social_tensor(xy[a:b], hidden[a:b])
                             for a, b in scene_bounds], dim=0))
        embedded = torch.cat((self.activation(self.position_embedding(xy)),
                              self.activation(self.social_embedding(social))), -1)
        hidden, cell = self.cell(embedded, (hidden, cell))
        return self.output(hidden) if emit else None, (hidden, cell)

    def forward(self, obs, target=None, valid=None, scene_bounds=None):
        """Return four Gaussian parameter frames.

        Training teacher-forces only actors with complete future labels. Other
        observed actors remain social context using predicted conditional means,
        never zero-filled future targets. Inference consumes no future labels.
        Packed scenes supply a sequence of (start, end) Python integer bounds;
        pooling is isolated within these slices at every recurrent step.
        """
        if obs.ndim != 3 or obs.shape[0] != len(OBS_TIMES) or obs.shape[-1] != 2:
            raise ValueError('Expected [11, actors, 2] observations')
        if target is not None and (target.shape != (4, obs.shape[1], 2) or valid is None):
            raise ValueError('Teacher forcing requires four targets and an actor mask')
        params, state = self.encode(obs, scene_bounds)
        return self.decode(params, state, target, valid, scene_bounds)

    def encode(self, obs, scene_bounds=None):
        state = (obs.new_zeros(obs.shape[1], self.hidden_dim),
                 obs.new_zeros(obs.shape[1], self.hidden_dim))
        for t, xy in enumerate(obs):
            params, state = self.step(xy, state, scene_bounds, emit=t == len(OBS_TIMES)-1)
        return params, state

    def decode(self, params, state, target=None, valid=None, scene_bounds=None):
        outputs = [params]
        for t in range(3):
            xy = params[..., :2]
            if target is not None:
                xy = torch.where(valid[:, None], target[t], xy)
            params, state = self.step(xy, state, scene_bounds)
            outputs.append(params)
        return torch.stack(outputs)


def gaussian_nll(params, target, valid, reduction='mean'):
    """Mean correlated bivariate Gaussian negative log likelihood in log space.

    Log standard deviations and correlation are bounded for numerical stability;
    the full normalization term is retained. Loss covers valid future actor steps.
    valid may be a boolean mask or integer actor indices. reduction='none'
    returns [future steps, valid actors] for scene-balanced batch reductions.
    """
    # Callers can pass CPU-prepared integer indices to avoid CUDA nonzero sync.
    selected = params[:, valid]
    delta = target[:, valid] - selected[..., :2]
    log_sigma = selected[..., 2:4].clamp(-7., 7.)
    rho = selected[..., 4].tanh().clamp(-1 + 1e-5, 1 - 1e-5)
    normalized = delta * torch.exp(-log_sigma)
    x, y = normalized.unbind(-1)
    one_minus_rho2 = 1 - rho.square()
    nll = (math.log(2 * math.pi) + log_sigma.sum(-1) + .5 * one_minus_rho2.log()
           + (x.square() + y.square() - 2 * rho * x * y) / (2 * one_minus_rho2))
    if nll.numel() == 0:
        raise ValueError('No valid future labels')
    if reduction == 'none':
        return nll
    if reduction != 'mean':
        raise ValueError('Unsupported NLL reduction')
    return nll.mean()


class SocialLSTMPredictor:
    """Drop-in PipelineConfig.predictor implementing the joint-scene contract."""
    def __init__(self, checkpoint, device='cpu'):
        import torch
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        expected = dict(architecture=ARCHITECTURE, model_config=MODEL_CONFIG,
                        prediction_policy=POLICY, seed=SEED, history_policy=HISTORY_POLICY,
                        observation_offsets_s=list(OBS_TIMES), future_offsets_s=list(FUTURE_TIMES),
                        checkpoint_selection=SELECTION, source_sha256=source_hash())
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Incompatible Social LSTM checkpoint/protocol/source')
        train, val = set(saved['train_scenario_ids']), set(saved['internal_val_scenario_ids'])
        official = set(saved['official_train_scenario_ids'])
        if not train or not val or train & val or train | val != official:
            raise ValueError('Invalid official-train scenario partition')
        if val != {sid for sid in official if stable_seed(sid) % 10 == 0}:
            raise ValueError('Checkpoint internal-validation partition differs from policy')
        self.development_scenarios = official
        self.checkpoint_epoch = saved['epoch']
        self.internal_val_ade_m = saved['internal_val_ade_m']
        self.device = device
        self.model = SocialLSTM(**MODEL_CONFIG).to(device)
        self.model.load_state_dict(saved['model_state'], strict=True)
        self.model.eval().requires_grad_(False)

    def predict_scene(self, contexts):
        import torch
        from .schemas import PredictedPath
        if not contexts:
            return {}
        contexts = sorted(contexts, key=lambda c: c.actor.actor_id)
        cutoff, sid = contexts[0].t_s, contexts[0].scenario_id
        if cutoff is None or any(c.t_s != cutoff or c.scenario_id != sid for c in contexts):
            raise ValueError('Social LSTM requires a single timestamped scene')
        actors = [c.actor for c in contexts]
        if len({a.actor_id for a in actors}) != len(actors):
            raise ValueError('Duplicate actor IDs')
        with torch.inference_mode():
            obs, rel, bounds, origin = scene_tensors(actors, cutoff, self.device)
            predicted = self.model(obs)[..., :2] + origin
            if not torch.isfinite(predicted).all():
                raise ValueError('Nonfinite Social LSTM trajectory')
            xy = predicted.cpu().tolist()
        return {a.actor_id: PredictedPath('Social LSTM', 1.,
                    [tuple(a.world_xy)] + [tuple(step[i]) for step in xy], 2.,
                    waypoint_times_s=[0., *FUTURE_TIMES]) for i, a in enumerate(actors)}

    def predict_records(self, contexts):
        """Export stable IDs, world XY and absolute future timestamps."""
        paths = self.predict_scene(contexts)
        return [dict(actor_id=c.actor.actor_id, scenario_id=c.scenario_id,
                     timestamps_s=[c.t_s+t for t in FUTURE_TIMES],
                     world_xy=paths[c.actor.actor_id].waypoints[1:])
                for c in sorted(contexts, key=lambda c: c.actor.actor_id)]
