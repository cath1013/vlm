"""Observation-only, unordered pair collision timing on a frozen JointScene V2."""
from __future__ import annotations

import itertools

import torch
from torch import nn

from .predict_nets import JointSceneMotionNetV2


def valid_pairs(actor_mask: torch.Tensor):
    """Upper-triangle indices and validity [B,P], including the empty case."""
    i, j = torch.triu_indices(actor_mask.shape[1], actor_mask.shape[1], 1,
                              device=actor_mask.device)
    return i, j, actor_mask[:, i] & actor_mask[:, j]


def ground_truth_bucket(gt):
    """The window label remains defined even with no visible collision pair."""
    positive = [r for r in gt["expected"] if r["accident_expected"]]
    if len(positive) > 1:
        raise ValueError("Expected one collision bucket per window")
    return int(positive[0]["k"]) if positive else 0


def pair_labels(actor_ids, gt):
    """Assign the exact GT bucket to each visible unordered pair.

    ``window_ground_truth`` groups registry actor IDs by stable source/CARLA
    identity. A pair is positive only when its two actors belong to distinct
    involved-vehicle groups in the expected collision bucket.
    """
    positive = [r for r in gt["expected"] if r["accident_expected"]]
    bucket = ground_truth_bucket(gt)
    groups = [set(v["actor_ids"]) for v in positive[0]["involved_vehicles"]] if positive else []
    labels = []
    for a, b in itertools.combinations(actor_ids, 2):
        hit = any((a in x and b in y) or (b in x and a in y)
                  for x, y in itertools.combinations(groups, 2))
        labels.append(bucket if hit else 0)
    return labels


class CollisionRiskNet(nn.Module):
    """Six logits per pair; no predicted future state enters the representation."""

    def __init__(self, backbone: JointSceneMotionNetV2, hidden_dim=128,
                 dropout=0.1, freeze_backbone=True):
        super().__init__()
        if not isinstance(backbone, JointSceneMotionNetV2) or backbone.spec.get("architecture") != "JointSceneMotionNetV2":
            raise TypeError("A JointSceneMotionNetV2 checkpoint is required")
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)
            backbone.eval()
        h = backbone.hidden_dim
        # sum and absolute difference are invariant under actor exchange.
        # Directed physical edges are symmetrized in the same way.
        self.head = nn.Sequential(nn.Linear(2 * h + 2 * backbone.EDGE_DIM, hidden_dim),
                                  nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, 6))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, global_features, candidates, candidate_mask, history,
                interactions, interaction_mask, actor_mask, origin):
        b = self.backbone
        state, has_route = b._encode_actors(global_features, candidates, candidate_mask,
                                             history, interactions, interaction_mask, actor_mask)
        velocity = b.initial_velocity(global_features, actor_mask)
        edge = b.edge_features(origin, velocity, has_route, global_features)
        norm_edge = b.normalize_edges(edge)
        for block in b.blocks:
            state = block(state, norm_edge, actor_mask)
        i, j, mask = valid_pairs(actor_mask)
        if i.numel() == 0:
            return state.new_empty((state.shape[0], 0, 6)), mask
        ab, ba = norm_edge[:, i, j], norm_edge[:, j, i]
        features = torch.cat((state[:, i] + state[:, j],
                              (state[:, i] - state[:, j]).abs(),
                              ab + ba, (ab - ba).abs()), dim=-1)
        logits = self.head(features)
        return logits.masked_fill(~mask.unsqueeze(-1), 0.0), mask


def load_v2(path, device="cpu"):
    model = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(model, JointSceneMotionNetV2) or model.spec.get("architecture") != "JointSceneMotionNetV2":
        raise TypeError("Checkpoint is not JointSceneMotionNetV2")
    return model
