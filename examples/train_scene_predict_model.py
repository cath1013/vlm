"""Train JointSceneMotionNet on ``make_scene_predict_dataset.py`` output."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from traffic_llm.predict_model import (MANEUVER_CLASSES, N_CANDIDATE_FEATURES,
                                       N_GLOBAL_FEATURES, N_HISTORY_FEATURES,
                                       N_HISTORY_STEPS, N_INTERACTION_FEATURES,
                                       OFF_CANDIDATE_LABEL)
from traffic_llm.predict_nets import JointSceneMotionNet, JointSceneMotionNetV2


def scenario_bucket(scenario_id: str, n: int = 100) -> int:
    return zlib.crc32(scenario_id.encode("utf-8")) % n


def read_scenes(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_scenes(scenes, mode="dataset", val_frac=.1, test_frac=.2):
    supported = {"train", "val"}
    unsupported = [s for s in scenes if s.get("dataset_split") not in supported]
    if unsupported:
        counts = {}
        for scene in unsupported:
            split = scene.get("dataset_split", "<missing>")
            counts[split] = counts.get(split, 0) + 1
        print(f"warning: unsupported scene splits excluded: {counts}", file=sys.stderr)
    scenes = [s for s in scenes if s.get("dataset_split") in supported]
    if mode == "dataset":
        # Official train is the only development pool.  Validation is carved
        # deterministically below; official val is never used for selection.
        train = [s for s in scenes if s["dataset_split"] == "train"]
        test = [s for s in scenes if s["dataset_split"] == "val"]
    else:
        cut = int(round(test_frac * 100))
        test = [s for s in scenes if scenario_bucket(s["scenario_id"]) < cut]
        train = [s for s in scenes if scenario_bucket(s["scenario_id"]) >= cut]
    val = []
    if val_frac > 0:
        lo = int(round(test_frac * 100)) if mode != "dataset" else 0
        hi = lo + int(round(val_frac * 100))
        val = [s for s in train if lo <= scenario_bucket(s["scenario_id"]) < hi]
        train = [s for s in train if not (lo <= scenario_bucket(s["scenario_id"]) < hi)]
    return train, val, test


class SceneDataset(Dataset):
    def __init__(self, scenes):
        self.scenes = scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        return self.scenes[index]


def scene_collate(scenes):
    """Pad actor/candidate/interaction axes only to this batch's maxima."""
    B, K = len(scenes), 5
    A = max(len(s["actors"]) for s in scenes)
    N = max(1, max((len(a["candidates"]) for s in scenes for a in s["actors"]), default=0))
    I = max(1, max((len(a["interactions"]) for s in scenes for a in s["actors"]), default=0))
    out = {
        "global": torch.zeros(B, A, N_GLOBAL_FEATURES),
        "candidates": torch.zeros(B, A, N, N_CANDIDATE_FEATURES),
        "candidate_mask": torch.zeros(B, A, N, dtype=torch.bool),
        "history": torch.zeros(B, A, N_HISTORY_STEPS, N_HISTORY_FEATURES),
        "interactions": torch.zeros(B, A, I, N_INTERACTION_FEATURES),
        "interaction_mask": torch.zeros(B, A, I, dtype=torch.bool),
        "actor_mask": torch.zeros(B, A, dtype=torch.bool),
        "origin": torch.zeros(B, A, 2),
        "target": torch.zeros(B, A, K, 2),
        "target_mask": torch.zeros(B, A, K, dtype=torch.bool),
        "maneuver": torch.full((B, A), -100, dtype=torch.long),
        "scenes": scenes,
    }
    maneuver_index = {m: i for i, m in enumerate(MANEUVER_CLASSES)}
    off_index = maneuver_index[OFF_CANDIDATE_LABEL]
    for b, scene in enumerate(scenes):
        for aidx, actor in enumerate(scene["actors"]):
            out["actor_mask"][b, aidx] = True
            out["global"][b, aidx] = torch.tensor(actor["global"])
            out["history"][b, aidx] = torch.tensor(actor["history"])
            out["origin"][b, aidx] = torch.tensor(actor["origin_enu"])
            out["target"][b, aidx] = torch.tensor(actor["target_offsets"])
            out["target_mask"][b, aidx] = torch.tensor(actor["target_mask"])
            nc, ni = len(actor["candidates"]), len(actor["interactions"])
            if nc:
                out["candidates"][b, aidx, :nc] = torch.tensor(actor["candidates"])
                out["candidate_mask"][b, aidx, :nc] = True
            if ni:
                out["interactions"][b, aidx, :ni] = torch.tensor(actor["interactions"])
                out["interaction_mask"][b, aidx, :ni] = True
            if actor["has_route_candidate"] and not actor["maneuver_ambiguous"]:
                out["maneuver"][b, aidx] = (maneuver_index.get(actor["maneuver"], -100)
                                              if actor["matched"] else off_index)
    return out


def to_device(batch, device):
    non_blocking = torch.device(device).type == "cuda"
    return {k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def autocast_context(device, amp_dtype):
    """Return AMP context, or a no-op autocast context when disabled."""
    kind = torch.device(device).type
    return torch.amp.autocast(kind, dtype=amp_dtype, enabled=amp_dtype is not None)


def amp_name(amp_dtype):
    if amp_dtype is None:
        return "disabled"
    return "bf16" if amp_dtype == torch.bfloat16 else "fp16"


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def scene_loss(model, batch, man_weight=1.0):
    out = model(batch["global"], batch["candidates"], batch["candidate_mask"],
                batch["history"], batch["interactions"], batch["interaction_mask"],
                batch["actor_mask"], batch["origin"])
    pred, logits = out if isinstance(out, tuple) else (out, None)
    per = nn.functional.huber_loss(pred, batch["target"], reduction="none", delta=2.0).sum(-1)
    valid = batch["actor_mask"].unsqueeze(-1) & batch["target_mask"]
    loss = (per * valid).sum() / valid.sum().clamp(min=1)
    if logits is not None and man_weight:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_target = batch["maneuver"].reshape(-1)
        valid_man = flat_target != -100
        if valid_man.any():
            loss = loss + man_weight * nn.functional.cross_entropy(
                flat_logits[valid_man], flat_target[valid_man]
            )
    return loss, pred, valid


def train_one_epoch(model, loader, optimizer, device, amp_dtype=None, scaler=None,
                    grad_accum_steps=1, loss_fn=scene_loss):
    """Optimize one epoch with fixed-size microbatch accumulation.

    ``loss_fn`` is injectable only for focused unit tests; normal training uses
    ``scene_loss`` unchanged. A final short window uses its actual count as
    divisor, so it is never under-scaled.
    """
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    n_microbatches = len(loader)
    optimization_loss = 0.0
    optimizer_steps = 0
    for batch_index, raw in enumerate(loader):
        if batch_index % grad_accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
            window_size = min(grad_accum_steps, n_microbatches - batch_index)
        batch = to_device(raw, device)
        with autocast_context(device, amp_dtype):
            loss, _, _ = loss_fn(model, batch)
            scaled_loss = loss / window_size
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        # Keep epoch reporting in original scene-loss units, not scaled units.
        optimization_loss += loss.detach().float().item()
        if ((batch_index + 1) % grad_accum_steps == 0
                or batch_index + 1 == n_microbatches):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer_steps += 1
    return {"optimization_loss": optimization_loss / max(1, n_microbatches),
            "microbatches": n_microbatches, "optimizer_steps": optimizer_steps}


@torch.no_grad()
def fit_normalizer(model, loader, device):
    stats = {name: [None, None, 0] for name in ("g", "c", "i")}
    specs = (("g", "global", "actor_mask"), ("c", "candidates", "candidate_mask"),
             ("i", "interactions", "interaction_mask"))
    for raw in loader:
        batch = to_device(raw, device)
        for name, value, mask in specs:
            x = batch[value][batch[mask]]
            if not x.numel():
                continue
            sm, sq, count = stats[name]
            stats[name] = [x.sum(0) if sm is None else sm + x.sum(0),
                           (x * x).sum(0) if sq is None else sq + (x * x).sum(0),
                           count + x.shape[0]]
        # V2-only: raw observation-time directed edges, excluding self and
        # padded actor pairs.  This path deliberately has no future targets or
        # rollout predictions, and the V1 model has no e_mu/e_sd buffers.
        if hasattr(model, "e_mu"):
            velocity = model.initial_velocity(batch["global"], batch["actor_mask"])
            has_route = batch["candidate_mask"].any(-1)
            edge = model.edge_features(batch["origin"], velocity, has_route, batch["global"])
            edge_mask = model.edge_valid_mask(batch["actor_mask"])
            x = edge[edge_mask]
            if x.numel():
                sm, sq, count = stats.setdefault("e", [None, None, 0])
                stats["e"] = [x.sum(0) if sm is None else sm + x.sum(0),
                              (x * x).sum(0) if sq is None else sq + (x * x).sum(0),
                              count + x.shape[0]]
    for name, mu, sd in (("g", model.g_mu, model.g_sd), ("c", model.c_mu, model.c_sd),
                         ("i", model.i_mu, model.i_sd)):
        sm, sq, count = stats[name]
        if count:
            mean = sm / count
            mu.copy_(mean)
            sd.copy_((sq / count - mean.square()).clamp(min=1e-6).sqrt().clamp(min=1e-3))
    if hasattr(model, "e_mu"):
        sm, sq, count = stats.get("e", [None, None, 0])
        if count:
            mean = sm / count
            model.e_mu.copy_(mean)
            model.e_sd.copy_((sq / count - mean.square()).clamp(min=1e-6).sqrt().clamp(min=1e-3))


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None):
    model.eval()
    total_ade = total_fde = 0.0
    n_points = n_traj = n_scenes = 0
    for raw in loader:
        batch = to_device(raw, device)
        with autocast_context(device, amp_dtype):
            _, pred, valid = scene_loss(model, batch)
        dist = torch.linalg.vector_norm(pred - batch["target"], dim=-1)
        counts = valid.sum(-1)
        has = counts > 0
        actor_ade = (dist * valid).sum(-1) / counts.clamp(min=1)
        total_ade += actor_ade[has].sum().item()
        n_points += valid.sum().item()
        last = (valid.float() * torch.arange(valid.shape[-1], device=device)).argmax(-1)
        fde = dist.gather(-1, last.unsqueeze(-1)).squeeze(-1)
        total_fde += fde[has].sum().item()
        n_traj += has.sum().item()
        n_scenes += len(raw["scenes"])
    return {"ade": total_ade / max(1, n_traj), "fde": total_fde / max(1, n_traj),
            "scenes": n_scenes, "trajectories": n_traj, "points": n_points}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train JointSceneMotionNet")
    ap.add_argument("--data", default="out/predict_dataset_scene_v1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--architecture", choices=["v1", "v2"], default="v1",
                    help="scene model architecture (v1 is the reproducible baseline)")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--scene-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    amp_group = ap.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true",
                           help="enable CUDA automatic mixed precision")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false",
                           help="disable CUDA automatic mixed precision")
    ap.set_defaults(amp=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="number of scene microbatches per optimizer update")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--train-eval-every", type=int, default=10)
    ap.add_argument("--compile", action="store_true",
                    help="compile only the model-forward wrapper (off by default)")
    ap.add_argument("--evaluate-test", action="store_true",
                    help="evaluate the reserved official DeepAccident val split")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out", default=None,
                    help="output directory (defaults separately for v1 and v2)")
    ap.add_argument("--split-by", choices=["dataset", "scenario"], default="dataset")
    ap.add_argument("--val-frac", type=float, default=.1)
    ap.add_argument("--test-frac", type=float, default=.2)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = ("out/predict_model_joint_scene_v1" if args.architecture == "v1"
                    else "out/predict_model_joint_scene_v2")
    if (args.save_every < 0 or args.batch_size <= 0 or args.num_workers < 0
            or args.grad_accum_steps < 1
            or args.eval_every <= 0 or args.train_eval_every <= 0):
        ap.error("--save-every>=0, --batch-size>0, --num-workers>=0, "
                 "--grad-accum-steps>=1, --eval-every>0, --train-eval-every>0 이어야 합니다")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device_type = torch.device(device).type
    if args.amp and device_type != "cuda":
        ap.error("--amp 는 CUDA device에서만 사용할 수 있습니다")
    amp_enabled = device_type == "cuda" if args.amp is None else args.amp
    amp_dtype = None
    if amp_enabled:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # BF16 has sufficient exponent range and deliberately needs no scaler.
    scaler = (torch.amp.GradScaler("cuda")
              if amp_dtype == torch.float16 else None)
    torch.manual_seed(args.seed)
    print(f"device: {device}")
    print(f"AMP: {amp_name(amp_dtype)}")
    print(f"physical batch size: {args.batch_size}")
    print(f"grad accumulation steps: {args.grad_accum_steps}")
    print(f"effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"num workers: {args.num_workers}")
    print("official test evaluation: " + ("ENABLED" if args.evaluate_test else "disabled"))
    scenes = read_scenes(os.path.join(args.data, "scenes.jsonl"))
    train, val, test = split_scenes(scenes, args.split_by, args.val_frac, args.test_frac)
    if not train or not test:
        raise SystemExit("학습/시험 scene 분할이 비어 있습니다")
    loader_args = dict(batch_size=args.batch_size, collate_fn=scene_collate,
                       num_workers=args.num_workers,
                       pin_memory=device_type == "cuda")
    if args.num_workers > 0:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    norm_loader = DataLoader(SceneDataset(train), shuffle=False, **loader_args)
    train_loader = DataLoader(SceneDataset(train), shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), **loader_args)
    val_loader = DataLoader(SceneDataset(val), shuffle=False, **loader_args) if val else None
    test_loader = DataLoader(SceneDataset(test), shuffle=False, **loader_args)
    model_cls = JointSceneMotionNet if args.architecture == "v1" else JointSceneMotionNetV2
    checkpoint_stem = ("joint_scene_motionnet" if args.architecture == "v1"
                       else "joint_scene_motionnet_v2")
    model = model_cls(args.hidden_dim, args.dropout, args.scene_layers,
                      args.num_heads).to(device)
    fit_normalizer(model, norm_loader, device)
    # Keep ``model`` as the serializable base module; the compiled wrapper is
    # only used to dispatch forwards and shares its parameters/buffers.
    forward_model = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    os.makedirs(args.out, exist_ok=True)
    best, best_epoch, history = float("inf"), 0, []
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        forward_model.train()
        synchronize(device)
        train_started = time.perf_counter()
        epoch_optimization = train_one_epoch(
            forward_model, train_loader, opt, device, amp_dtype, scaler,
            args.grad_accum_steps,
        )
        optimization_loss = epoch_optimization["optimization_loss"]
        synchronize(device)
        train_time = time.perf_counter() - train_started
        run_val = epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0
        run_train = epoch % args.train_eval_every == 0
        tr = None
        train_eval_time = None
        if run_train:
            synchronize(device)
            started = time.perf_counter()
            tr = evaluate(forward_model, norm_loader, device, amp_dtype)
            synchronize(device)
            train_eval_time = time.perf_counter() - started
        va = None
        val_time = None
        if run_val:
            synchronize(device)
            started = time.perf_counter()
            va = evaluate(forward_model, val_loader if val_loader else norm_loader,
                          device, amp_dtype)
            synchronize(device)
            val_time = time.perf_counter() - started
        history.append({"epoch": epoch,
                        "optimization_loss": optimization_loss,
                        "train": tr, "val": va,
                        "train_time_s": train_time,
                        "train_eval_time_s": train_eval_time,
                        "validation_time_s": val_time})
        if va is not None and va["ade"] < best:
            best, best_epoch = va["ade"], epoch
            torch.save(model, os.path.join(args.out, f"{checkpoint_stem}_best.pt"))
        if args.save_every and epoch % args.save_every == 0:
            torch.save(model, os.path.join(args.out, f"{checkpoint_stem}_epoch{epoch:03d}.pt"))
        text = (f"epoch {epoch:3d}  loss {optimization_loss:.4f}  "
                f"train time {train_time:.1f}s")
        if tr is not None:
            text += f"  train ADE {tr['ade']:.3f} FDE {tr['fde']:.3f} ({train_eval_time:.1f}s)"
        if va is not None:
            text += f"  val ADE {va['ade']:.3f} FDE {va['fde']:.3f} ({val_time:.1f}s)"
        print(text)
    torch.save(model, os.path.join(args.out, f"{checkpoint_stem}_final.pt"))
    final_history = history[-1]
    final_metrics = {"epoch": args.epochs,
                     "train": (final_history["train"] if final_history["train"] is not None
                               else evaluate(model, norm_loader, device, amp_dtype)),
                     # Validation always runs on the final epoch, so this is
                     # the exact final-model metric without a duplicate pass.
                     "val": final_history["val"],
                     "test": (evaluate(model, test_loader, device, amp_dtype)
                              if args.evaluate_test else None)}
    best_model = torch.load(os.path.join(args.out, f"{checkpoint_stem}_best.pt"),
                            map_location=device, weights_only=False)
    best_metrics = {"epoch": best_epoch, "train": evaluate(best_model, norm_loader, device, amp_dtype),
                    "val": evaluate(best_model, val_loader, device, amp_dtype) if val_loader else evaluate(best_model, norm_loader, device, amp_dtype),
                    "test": (evaluate(best_model, test_loader, device, amp_dtype)
                             if args.evaluate_test else None)}
    def split_counts(items):
        actors = [a for scene in items for a in scene["actors"]]
        return {"scenes": len(items), "actors": len(actors),
                "with_route": sum(a["has_route_candidate"] for a in actors),
                "without_route": sum(not a["has_route_candidate"] for a in actors)}
    report = {"architecture": type(model).__name__, "dataset": args.data, "seed": args.seed,
              "normalization_source": "training only", "hyperparameters": vars(args),
              "grad_accum_steps": args.grad_accum_steps,
              "effective_batch_size": args.batch_size * args.grad_accum_steps,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "split": {"train": split_counts(train), "val": split_counts(val),
                        "test": split_counts(test)}, "best_epoch": best_epoch,
              "best_validation_ade": best, "best_checkpoint": best_metrics,
              "final_checkpoint": final_metrics, "history": history,
              "amp": amp_name(amp_dtype),
              "cuda_peak_memory_bytes": (torch.cuda.max_memory_allocated(device)
                                          if device_type == "cuda" else None)}
    if args.architecture == "v2":
        report["edge_normalization"] = model.spec["edge_normalization"]
        report["edge_dim"] = model.spec["edge_dim"]
        report["edge_attention"] = model.spec["edge_attention"]
        report["num_heads_note"] = model.spec["num_heads_note"]
    with open(os.path.join(args.out, "train_report_joint_scene.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    if device_type == "cuda":
        print(f"CUDA peak memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
