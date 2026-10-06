#!/usr/bin/env python3
"""Plot frozen V4/CRN-4 cases without recomputing either classifier's decisions.

ID-only CSVs are enriched from the saved final-analysis ledger; every case also
joins the frozen V4 per-window ledger. Case IDs follow a seeded shuffle, and
only case_id_mapping.csv exports their window IDs. Clearances are
minima over the union of the listed CRN, V4 and GT actor pairs (never arbitrary
cross-pairs), using the validation audit's predictor/GT XY, raw CARLA yaw and
exact instance sizes, zero contact margin and 0.1-s samples. Intervals are
(0, 2] and (2, 5]; t=0 contact onset uses the audit's history-aware semantics.
All predictor modes are drawn; clearance uses the audit's minimum-clearance
path combination per pair. Missing actors, identity or XY trajectories fail
explicitly. Missing raw geometry is reported as incomplete coverage, with XY
trajectories retained. GT paths ending before 5 s show their actual coverage;
clearances then describe only available samples, without extrapolation.

Outputs must be new: existing PNGs and CSVs are never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import math
import random
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples import audit_gt_carla_boxes as boxes  # noqa: E402
from examples import audit_gt_heading_source_ablation as geometry  # noqa: E402
from examples.audit_predictor_trajectory_contacts import replay_windows  # noqa: E402
from traffic_llm.deepaccident import estimate_collision  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.swept_path import _footprint_clearance, _motion_fn  # noqa: E402

ANALYSIS = ROOT / "out/collision_risk_hazard/run_20261006_window_balanced/final_analysis"
PREDICTOR = ROOT / "out/predict_model_joint_scene_v2/joint_scene_motionnet_v2_best.pt"
V4_LEDGER = ROOT / "out/pair_reranker_v4/eval_0_2_val104_with_rows_windows.jsonl"
FIELDS = ("case_id scenario_id comparison_group gt_bucket "
          "collision_within_2s collision_after_2s crn_risk_2s crn_threshold_2s "
          "crn_selected_pair v4_selected_pairs gt_pair predicted_min_clearance_m "
          "predicted_min_clearance_time_s gt_min_clearance_0_2_m "
          "gt_min_clearance_0_2_time_s gt_min_clearance_2_5_m "
          "gt_min_clearance_2_5_time_s gt_first_contact_time_s "
          "listed_pairs_gt_first_contact_time_s v4_gt_pair_candidate_covered "
          "v4_baseline_predicted v4_correct_gt_pair_hit crn_correct_gt_pair_hit "
          "predicted_geometry_complete_0_2 gt_geometry_complete_0_2 gt_geometry_complete_2_5 "
          "predicted_missing_geometry_times_s gt_missing_geometry_times_s "
          "gt_collision_pair_observable_at_t0").split()
PAIR_FIELDS = ("case_id actor_pair predicted_min_clearance_m predicted_min_clearance_time_s "
               "gt_min_clearance_0_2_m gt_min_clearance_0_2_time_s "
               "gt_min_clearance_2_5_m gt_min_clearance_2_5_time_s "
               "predicted_contact_event predicted_contact_time_s gt_contact_event gt_contact_time_s").split()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_pairs(value, *, single=False):
    if value in (None, "", "None", "null"):
        return []
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    pairs = [parsed] if single and parsed else parsed
    if not isinstance(pairs, (list, tuple)):
        raise ValueError(f"invalid pair list: {value!r}")
    result = []
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(a, str) and a for a in pair) or pair[0] == pair[1]):
            raise ValueError(f"expected two distinct actor IDs: {pair!r}")
        result.append(tuple(pair))
    return result


def load_cases(path, limit=None, blind_seed=20261006, per_group_limit=None, groups=None):
    if limit is not None and per_group_limit is not None:
        raise ValueError("--limit and --per-group-limit cannot be used together")
    if per_group_limit is not None and per_group_limit <= 0:
        raise ValueError("--per-group-limit must be positive")
    ledger_path = ANALYSIS / "v4_vs_crn4_2s_all_windows.csv"
    ledger = {r["window_id"]: r for r in read_csv(ledger_path)} if ledger_path.exists() else {}
    v4_ledger = {}
    with V4_LEDGER.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record["window_id"] in v4_ledger:
                raise ValueError(f"duplicate V4 ledger window: {record['window_id']}")
            v4_ledger[record["window_id"]] = record
    cases = read_csv(path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        if groups is None:
            cases = cases[:limit]
    if not cases:
        raise ValueError("no cases requested")
    result = []
    for case in cases:
        wid = case.get("window_id", "")
        scenario, index, cutoff = wid.rsplit(":", 2)
        if wid != f"{scenario}:{int(index)}:{float(cutoff):.3f}":
            raise ValueError(f"noncanonical window_id: {wid}")
        if case.get("scenario_id") != scenario:
            raise ValueError(f"scenario_id mismatch for {wid}")
        if wid not in v4_ledger:
            raise KeyError(f"window missing from frozen V4 ledger: {wid}")
        frozen = v4_ledger[wid]
        row = {**ledger.get(wid, {}), **case}
        row.update(v4_selected_pairs=frozen["selected_pairs"],
                   v4_gt_pair_candidate_covered=frozen["gt_pair_covered"],
                   v4_baseline_predicted=frozen["baseline_predicted"],
                   v4_correct_gt_pair_hit=frozen["correct_pair_hit"])
        crn_hit = row.get("crn_correct_gt_pair_hit", row.get("crn4_correct_pair_hit", ""))
        if crn_hit not in (True, False, "True", "False", "true", "false", "1", "0"):
            raise ValueError(f"saved CRN correct-pair hit unavailable for {wid}")
        row["crn_correct_gt_pair_hit"] = str(crn_hit).lower() in ("true", "1")
        for canonical, aliases in {
            "comparison_group": ("analysis_group", "transition"),
            "crn_risk_2s": ("crn4_risk_2s",),
            "crn_threshold_2s": ("crn4_threshold_2s",),
            "crn_selected_pair": ("crn4_selected_pair",),
        }.items():
            if canonical not in row:
                row[canonical] = next((row[a] for a in aliases if a in row), "")
        if "v4_selected_pairs" not in row or not row["crn_selected_pair"]:
            raise ValueError(f"saved selected pairs unavailable for {wid}; supply pair columns")
        row["v4_pairs"] = parse_pairs(row["v4_selected_pairs"])
        row["crn_pairs"] = parse_pairs(row["crn_selected_pair"], single=True)
        for key in ("crn_risk_2s", "crn_threshold_2s"):
            if row[key] != "":
                row[key] = float(row[key])
        result.append(row)
    if groups is not None:
        requested_groups = set(groups)
        missing = requested_groups - {row["comparison_group"] for row in result}
        if missing:
            raise ValueError(f"requested groups have no cases: {', '.join(sorted(missing))}")
        result = [row for row in result if row["comparison_group"] in requested_groups]
        if limit is not None:
            result = result[:limit]
    rng = random.Random(blind_seed)
    if per_group_limit is not None:
        groups = {}
        for row in result:
            groups.setdefault(row["comparison_group"], []).append(row)
        result = []
        for group in sorted(groups):
            rng.shuffle(groups[group])
            result.extend(groups[group][:per_group_limit])
    rng.shuffle(result)
    return result


def checked_paths(actor, heading, horizon, require_full):
    paths = []
    for path in actor.predictions:
        if path.waypoint_times_s is not None and (
                len(path.waypoint_times_s) < 2 or path.waypoint_times_s[-1] <= 0):
            raise ValueError(f"missing timestamped future path for {actor.actor_id}")
        motion = geometry._future_pose(actor, path, horizon, heading)
        if motion is None or (require_full and motion[1] < horizon - 1e-8) or motion[2]:
            raise ValueError(f"missing/truncated/gapped {horizon:g}-s path for {actor.actor_id}")
        paths.append((path, motion[0], motion[1]))
    if not paths:
        raise ValueError(f"no future trajectory for {actor.actor_id}")
    return paths


def analyze_pairs(actors, pairs, identities, frame_boxes, cutoff, horizon):
    heading = geometry._HeadingLookup(frame_boxes, identities, cutoff)
    size = geometry._ExactFootprintLookup(frame_boxes, identities, cutoff)
    paths = {aid: checked_paths(actor, heading, horizon, require_full=horizon == 2.)
             for aid, actor in actors.items()}
    xy_paths = {aid: [(path, _motion_fn(actors[aid], path, horizon)[0], available)
                      for path, _, available in actor_paths] for aid, actor_paths in paths.items()}
    missing_times = set()
    for aid, actor in actors.items():
        available = max(available for _, _, available in paths[aid])
        for step in range(math.floor(available * 10 + 1e-8) + 1):
            t = step / 10
            if heading(aid, t) is None or size(actor, t) is None:
                missing_times.add(t)
    samples, events, chosen, missing_intervals, available_by_pair = {}, {}, {}, {}, {}
    for a, b in pairs:
        best = None
        for (_, pa, available_a), (_, pb, available_b) in itertools.product(paths[a], paths[b]):
            available = min(available_a, available_b, horizon)
            values, missing = [], []
            for step in range(1, math.floor(available * 10 + 1e-8) + 1):
                t = step / 10
                aa, bb, ea, eb = pa(t), pb(t), size(actors[a], t), size(actors[b], t)
                if aa is None or bb is None or ea is None or eb is None:
                    missing.append((max(0., t - .05), min(available, t + .05)))
                else:
                    values.append((t, _footprint_clearance(aa, ea, bb, eb)))
            rank = min(((gap, t) for t, gap in values), default=(math.inf, math.inf))
            if best is None or rank < best[0]:
                best = rank, values, pa, pb, available, missing
        _, values, pa, pb, available, missing = best
        samples[(a, b)] = values
        chosen[(a, b)] = (pa, pb)
        missing_intervals[(a, b)] = missing
        available_by_pair[(a, b)] = available
        aa, bb, ea, eb = pa(0), pb(0), size(actors[a], 0), size(actors[b], 0)
        if aa is None or bb is None or ea is None or eb is None:
            events[(a, b)] = ("GEOMETRY_UNCERTAIN", None)
            continue
        now = _footprint_clearance(aa, ea, bb, eb)
        before, gapped = geometry._raw_history_before(actors[a], actors[b], heading, size, 0.)
        events[(a, b)] = geometry._contact_event(
            now <= 0., before, gapped, [(t, gap <= 0.) for t, gap in values],
            available >= horizon - 1e-8 and not missing, missing)
        # The audit helper's NONE means no observed contact; incomplete
        # geometry cannot establish absence of contact across the gap.
        if missing and events[(a, b)][0] in {"NONE", "PREEXISTING_PERSISTENT"}:
            events[(a, b)] = ("GEOMETRY_UNCERTAIN", None)
    return {"paths": paths, "samples": samples, "events": events,
            "chosen": chosen, "size": size, "xy_paths": xy_paths,
            "missing_geometry_times": sorted(missing_times), "missing_intervals": missing_intervals,
            "available_by_pair": available_by_pair}


def geometry_complete(analysis, start, end):
    return (all(available >= end - 1e-8 for available in analysis["available_by_pair"].values())
            and not any((start < t <= end) or (start == 0 and t == 0)
                        for t in analysis["missing_geometry_times"]))


def minimum(analysis, start, end):
    values = [(gap, t) for samples in analysis["samples"].values()
              for t, gap in samples if start < t <= end]
    return min(values) if values else (None, None)


def pair_metadata(case_id, predicted, truth):
    """Export existing per-pair events and available-sample clearance minima."""
    records = []
    for pair in sorted(predicted["samples"]):
        record = {"case_id": case_id, "actor_pair": json.dumps(pair)}
        for prefix, analysis, start, end in (
                ("predicted_min_clearance", predicted, 0, 2),
                ("gt_min_clearance_0_2", truth, 0, 2),
                ("gt_min_clearance_2_5", truth, 2, 5)):
            gap, at = min(((gap, t) for t, gap in analysis["samples"][pair] if start < t <= end),
                          default=(None, None))
            record[prefix + "_m"], record[prefix + "_time_s"] = gap, at
        for prefix, analysis in (("predicted", predicted), ("gt", truth)):
            event, at = analysis["events"][pair]
            record[prefix + "_contact_event"], record[prefix + "_contact_time_s"] = event, at
        records.append(record)
    return records


def interaction_inset(ax, actors, analysis, colors, style):
    from matplotlib.patches import Polygon

    focus = min(((gap, t, pair) for pair, values in analysis["samples"].items()
                 for t, gap in values), default=None)
    if focus is None:
        ax.text(.02, .98, "Interaction inset unavailable: no valid pair geometry",
                transform=ax.transAxes, va="top", fontsize=7)
        return
    gap, at, pair = focus
    inset = ax.inset_axes([.53, .55, .44, .4])
    footprints = []
    for aid, pose in zip(pair, analysis["chosen"][pair]):
        index = next(i for i, (_, raw_pose, _) in enumerate(analysis["paths"][aid]) if raw_pose is pose)
        _, xy_pose, available = analysis["xy_paths"][aid][index]
        if at > available:
            continue
        p, dimensions = pose(at), analysis["size"](actors[aid], at)
        if p is None or dimensions is None:
            continue
        footprint = boxes.oriented_footprint(p[:2], p[2:], *dimensions)
        footprints.extend(footprint)
        inset.add_patch(Polygon(footprint, fill=False, edgecolor=colors[aid], linewidth=1.4))
        times = [step / 10 for step in range(math.floor(available * 10 + 1e-8) + 1)
                 if abs(step / 10 - at) <= .8 + 1e-8]
        xy = [xy_pose(t)[:2] for t in times]
        if xy:
            inset.plot(*zip(*xy), style, marker=".", markersize=2, color=colors[aid])
        inset.plot(*p[:2], "o", color=colors[aid], markersize=3)
    if footprints:
        xmin, xmax = min(x for x, y in footprints), max(x for x, y in footprints)
        ymin, ymax = min(y for x, y in footprints), max(y for x, y in footprints)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        radius = max(5., max(xmax - xmin, ymax - ymin) / 2 + 2.)
        inset.set(xlim=(cx-radius, cx+radius), ylim=(cy-radius, cy+radius))
    inset.set_aspect("equal", adjustable="box")
    inset.tick_params(labelsize=6)
    inset.grid(alpha=.2)
    label = f"{pair[0]} / {pair[1]}\n+{at:.1f} s, available-sample min {gap:.2f} m"
    label = "\n".join(textwrap.fill(line, width=25) for line in label.splitlines())
    inset.text(.02, .98, label, transform=inset.transAxes, va="top", fontsize=5,
               bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))


def prepare_case(case, window, gt_window, agent_ids, collision, frame_boxes):
    identities = geometry.identities(window, agent_ids)
    actors = {a.actor_id: a for a in window.last.actors}
    gt_actors = {a.actor_id: a for a in gt_window.last.actors}
    gt_pairs = []
    gt_observable = False
    gt_ids = list(collision.carla_ids or []) if collision and collision.occurred else []
    if gt_ids:
        if len(gt_ids) != 2:
            raise ValueError(f"expected two GT collision identities, got {gt_ids}")
        aliases = [[aid for aid in actors if cid in identities.get(aid, set())] for cid in gt_ids]
        gt_observable = all(aliases)
        if gt_observable:
            gt_pairs = list(itertools.product(*aliases))
    pairs = list(dict.fromkeys(tuple(sorted(p)) for p in
                              case["crn_pairs"] + case["v4_pairs"] + gt_pairs))
    relevant = sorted({aid for pair in pairs for aid in pair})
    if not relevant:
        raise ValueError(f"no relevant pair for {case['window_id']}")
    for aid in relevant:
        if aid not in actors or aid not in gt_actors:
            raise KeyError(f"requested actor absent from predictor/GT window: {aid}")
        if len(identities.get(aid, set())) != 1:
            raise ValueError(f"missing/ambiguous CARLA identity for {aid}: {identities.get(aid)}")
    actors = {aid: actors[aid] for aid in relevant}
    gt_actors = {aid: gt_actors[aid] for aid in relevant}
    predicted = analyze_pairs(actors, pairs, identities, frame_boxes, window.t_end, 2.)
    truth = analyze_pairs(gt_actors, pairs, identities, frame_boxes, window.t_end, 5.)
    pred_min, gt_early, gt_late = minimum(predicted, 0, 2), minimum(truth, 0, 2), minimum(truth, 2, 5)
    listed_contact = min((t for event, t in truth["events"].values()
                   if event in {"NEW_CONTACT", "BOUNDARY_ONSET", "RECONTACT"} and t is not None),
                  default=None)
    gt_pair_keys = {tuple(sorted(pair)) for pair in gt_pairs}
    contact = min((t for pair, (event, t) in truth["events"].items()
                   if pair in gt_pair_keys and event in {"NEW_CONTACT", "BOUNDARY_ONSET", "RECONTACT"}
                   and t is not None), default=None)
    ttc = collision.time_s - window.t_end if gt_ids and collision.time_s is not None else None
    bucket = math.ceil(ttc - 1e-9) if ttc is not None and 1e-9 < ttc <= 5 else 0
    if case.get("gt_bucket", "") != "" and int(case["gt_bucket"]) != bucket:
        raise ValueError(f"saved/reconstructed GT bucket mismatch for {case['window_id']}")
    metadata = {key: case.get(key, "") for key in FIELDS}
    metadata["window_id"] = case["window_id"]  # Only the mapping CSV exports this identity.
    metadata.update(gt_bucket=bucket, collision_within_2s=bool(bucket in (1, 2)),
                    gt_collision_pair_observable_at_t0=gt_observable,
                    collision_after_2s=bool(bucket in (3, 4, 5)),
                    crn_selected_pair=json.dumps(case["crn_pairs"][0] if case["crn_pairs"] else []),
                    v4_selected_pairs=json.dumps(case["v4_pairs"]), gt_pair=json.dumps(gt_ids),
                    predicted_min_clearance_m=pred_min[0], predicted_min_clearance_time_s=pred_min[1],
                    gt_min_clearance_0_2_m=gt_early[0], gt_min_clearance_0_2_time_s=gt_early[1],
                    gt_min_clearance_2_5_m=gt_late[0], gt_min_clearance_2_5_time_s=gt_late[1],
                    gt_first_contact_time_s=contact,
                    listed_pairs_gt_first_contact_time_s=listed_contact,
                    predicted_geometry_complete_0_2=geometry_complete(predicted, 0, 2),
                    gt_geometry_complete_0_2=geometry_complete(truth, 0, 2),
                    gt_geometry_complete_2_5=geometry_complete(truth, 2, 5),
                    predicted_missing_geometry_times_s=json.dumps(predicted["missing_geometry_times"]),
                    gt_missing_geometry_times_s=json.dumps(truth["missing_geometry_times"]))
    return actors, identities, predicted, truth, metadata, ttc


def render_case(case_id, actors, identities, predicted, truth, metadata, ttc, out, observed_window=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    points = []
    colors = {}
    for i, (aid, actor) in enumerate(actors.items()):
        color = plt.get_cmap("tab20")((2 * i + i // 10) % 20)
        colors[aid] = color
        label = f"{aid} | source_track_ids={actor.source_track_ids} | CARLA={sorted(identities[aid])}"
        cutoff = float(metadata["window_id"].rsplit(":", 1)[1])
        start = observed_window.t_start if observed_window is not None else -math.inf
        history = {t: (x, y) for t, x, y in actor.track_history if start <= t <= cutoff + 1e-8}
        if observed_window is not None:
            for snapshot in observed_window.snapshots:
                for observed in snapshot.actors:
                    if observed.actor_id == aid:
                        history[snapshot.t] = observed.world_xy
        if not history:
            raise ValueError(f"observed history unavailable for {aid}")
        xy = [xy for t, xy in sorted(history.items())] + [actor.world_xy]
        axes[0].plot(*zip(*xy), ".-", color=color, label=label)
        points.extend(xy)
        for ax, analysis, horizon, style in ((axes[1], predicted, 2, "--"), (axes[2], truth, 5, "-")):
            for path, pose, available in analysis["xy_paths"][aid]:
                times = [step / 10 for step in range(math.floor(available * 10 + 1e-8) + 1)]
                xy = [pose(t)[:2] for t in times]
                ax.plot(*zip(*xy), style, marker=".", markersize=2, color=color, alpha=.7)
                points.extend(xy)
                for t in range(1, math.floor(available + 1e-8) + 1):
                    x, y = pose(t)[:2]
                    ax.plot(x, y, ".", color=color)
                    ax.annotate(f"+{t}s", (x, y), fontsize=7, color=color)
            for pair, values in analysis["samples"].items():
                if aid in pair:
                    draw_times = {0.}
                    if values:
                        draw_times.add(min(values, key=lambda v: (v[1], v[0]))[0])
                    event, at = analysis["events"][pair]
                    if at is not None:
                        draw_times.add(at)
                    pose = analysis["chosen"][pair][pair.index(aid)]
                    available = next(coverage for _, path_pose, coverage in analysis["paths"][aid]
                                     if path_pose is pose)
                    for t in sorted(draw_times):
                        if t > available:
                            continue
                        p = pose(t)
                        dimensions = analysis["size"](actor, t)
                        if p is None or dimensions is None:
                            continue
                        footprint = boxes.oriented_footprint(p[:2], p[2:], *dimensions)
                        ax.add_patch(Polygon(footprint, fill=False, edgecolor=color, alpha=.5))
                        points.extend(footprint)
        for ax in axes:
            ax.plot(*actor.world_xy, "o", color=color, markersize=5)
            ax.annotate(f"{aid} t=0", actor.world_xy, fontsize=8, color=color)
        pose = predicted["paths"][aid][0][1](0)
        dimensions = predicted["size"](actor, 0)
        if pose is not None and dimensions is not None:
            footprint = boxes.oriented_footprint(pose[:2], pose[2:], *dimensions)
            axes[0].add_patch(Polygon(footprint, fill=False, edgecolor=color))
    xmin, xmax = min(x for x, y in points), max(x for x, y in points)
    ymin, ymax = min(y for x, y in points), max(y for x, y in points)
    padding = max(3., .08 * max(xmax - xmin, ymax - ymin))
    for ax, title in zip(axes, ("Observed history (t ≤ 0)", "JointScene V2 predicted future (0–2 s)",
                              "GT future (available data up to 5 s)")):
        ax.set(title=title, xlabel="World ENU east (m)", ylabel="World ENU north (m)",
               xlim=(xmin-padding, xmax+padding), ylim=(ymin-padding, ymax+padding))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, loc="best")
    interaction_inset(axes[1], actors, predicted, colors, "--")
    interaction_inset(axes[2], actors, truth, colors, "-")
    def display(value):
        return "none" if value is None else f"{value:.2f}"
    info = "Min footprint clearance over listed interaction pairs (0.1-s audit samples, raw CARLA yaw/size, zero margin):\n"
    for prefix, title, complete_key in (
            ("predicted_min_clearance", "Predicted 0–2 s", "predicted_geometry_complete_0_2"),
            ("gt_min_clearance_0_2", "GT 0–2 s", "gt_geometry_complete_0_2"),
            ("gt_min_clearance_2_5", "GT 2–5 s", "gt_geometry_complete_2_5")):
        qualifier = "minimum" if metadata[complete_key] else "available-sample minimum (coverage incomplete)"
        info += f"{title} {qualifier}: {display(metadata[prefix+'_m'])} m at +{display(metadata[prefix+'_time_s'])} s; "
    missing = sorted(set(predicted["missing_geometry_times"] + truth["missing_geometry_times"]))
    if missing:
        info += "\nGeometry coverage incomplete: " + ", ".join(f"+{t:.1f} s" for t in missing)
    contact = metadata["gt_first_contact_time_s"]
    contact_bucket = max(1, math.ceil(contact-1e-9)) if contact is not None else "none"
    info += (f"\nGT collision-pair geometry first new/boundary/recontact: +{display(contact)} s (bucket {contact_bucket}); "
             f"listed-pair geometry first contact: +{display(metadata['listed_pairs_gt_first_contact_time_s'])} s; "
             f"dataset collision: +{display(ttc)} s (bucket {metadata['gt_bucket']})")
    info += "\nGT sample coverage: " + "; ".join(
        f"{aid}: 0–{min(p[2] for p in paths):.1f} s"
        for aid, paths in truth["paths"].items()) + ". Minima/contact describe available samples only."
    info += "\nGT contact events: " + "; ".join(
        f"{a}/{b}: {event}" for (a, b), (event, at) in truth["events"].items())
    diagnostic = (f"{case_id} | {metadata['comparison_group']}\n"
                  f"CRN risk={metadata['crn_risk_2s']} threshold={metadata['crn_threshold_2s']} pair={metadata['crn_selected_pair']}\n"
                  f"V4 accepted={metadata['v4_selected_pairs']} | GT CARLA pair={metadata['gt_pair']} bucket={metadata['gt_bucket']}\n"
                  f"GT collision pair observable at t=0: {metadata['gt_collision_pair_observable_at_t0']}")
    def wrap(text):
        return "\n".join(textwrap.fill(line, width=175) for line in text.splitlines())
    info, diagnostic = wrap(info), wrap(diagnostic)
    fig.text(.02, .02, info, fontsize=9)
    fig.suptitle(case_id)
    fig.tight_layout(rect=(0, .05 + .016 * len(info.splitlines()),
                          1, .97 - .018 * len(diagnostic.splitlines())))
    try:
        fig.savefig(out / "blind" / f"{case_id}.png", dpi=150)
        fig.suptitle(diagnostic, fontsize=10)
        fig.savefig(out / "diagnostic" / f"{case_id}.png", dpi=150)
    finally:
        plt.close(fig)


def export_case_pdfs(out, case_ids):
    """Embed completed PNGs at native pixel resolution, one image per page."""
    from PIL import Image
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    import matplotlib

    ordered = sorted(case_ids, key=lambda case_id: int(case_id.removeprefix("case_")))
    if not ordered:
        raise ValueError("no case PNGs requested for PDF export")
    outputs = [(out / f"{kind}_cases.pdf", [out / kind / f"{case_id}.png" for case_id in ordered])
               for kind in ("blind", "diagnostic")]
    for pdf, pngs in outputs:
        if pdf.exists():
            raise FileExistsError(f"PDF output already exists: {pdf}")
        for png in pngs:
            if not png.is_file():
                raise FileNotFoundError(f"missing case PNG for PDF export: {png}")
    with matplotlib.rc_context({"savefig.bbox": None}):
        for pdf, pngs in outputs:
            with PdfPages(pdf) as document:
                for png in pngs:
                    with Image.open(png) as source:
                        pixels = np.asarray(source.convert("RGBA"))
                        width, height = source.size
                    fig = Figure(figsize=(width / 72, height / 72), dpi=72)
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.imshow(pixels, interpolation="none", aspect="auto")
                    ax.set_axis_off()
                    document.savefig(fig, dpi=72, bbox_inches=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ANALYSIS / "priority_error_cases.csv")
    parser.add_argument("--root", required=True)
    parser.add_argument("--carla-maps", required=True)
    parser.add_argument("--predictor", type=Path, default=PREDICTOR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=ANALYSIS / "trajectory_plots")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-group-limit", type=int,
                        help="maximum cases per comparison_group, sampled using --blind-seed")
    parser.add_argument("--groups", nargs="+", metavar="GROUP",
                        help="comparison_group values to include before sampling")
    parser.add_argument("--blind-seed", type=int, default=20261006,
                        help="seed for shuffling requested cases before assigning IDs")
    parser.add_argument("--no-pdf", action="store_true", help="disable assembling PNGs into case PDFs")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="record individual case failures, finish remaining cases, then exit nonzero without PDFs")
    args = parser.parse_args(argv)
    cases = load_cases(args.cases, args.limit, args.blind_seed, args.per_group_limit, args.groups)
    targets = [args.out / name for name in ("case_plot_metadata.csv", "case_id_mapping.csv", "case_pair_metadata.csv")]
    targets += [args.out / kind / f"case_{i:03d}.png" for i in range(1, len(cases)+1)
                for kind in ("blind", "diagnostic")]
    if not args.no_pdf:
        targets += [args.out / f"{kind}_cases.pdf" for kind in ("blind", "diagnostic")]
    if args.continue_on_error:
        targets.append(args.out / "failed_cases.csv")
    if any(path.exists() for path in targets):
        raise FileExistsError("requested plot/CSV output already exists; choose a new --out")
    cfg = PipelineConfig()
    runner = DeepAccidentRunner(args.root, cfg)
    validation = sorted((scenario for scenario in runner.list_scenarios() if scenario.split == "val"),
                        key=lambda scenario: scenario.scenario_id)
    if len(validation) != 104:
        raise ValueError(f"official val104 requires 104 scenarios; found {len(validation)}")
    scenarios = {}
    for scenario in validation:
        if scenario.scenario_id in scenarios:
            raise ValueError(f"duplicate validation scenario_id: {scenario.scenario_id}")
        scenarios[scenario.scenario_id] = scenario
    for case in cases:
        if not args.continue_on_error and case["scenario_id"] not in scenarios:
            raise KeyError(f"scenario not found: {case['scenario_id']}")
    for kind in ("blind", "diagnostic"):
        (args.out / kind).mkdir(parents=True, exist_ok=True)
    conf = dict(mode="sensor3d", window_s=5., stride_s=1., horizon_s=5., history_stride_s=1.)
    cache, metadata, pair_records, failures = {}, [], [], []
    for i, case in enumerate(cases, 1):
        case_id = f"case_{i:03d}"
        try:
            scenario = scenarios[case["scenario_id"]]
            if scenario.scenario_id not in cache:
                print(f"Replaying {scenario.scenario_id} (sensor3d, {args.device})", flush=True)
                row = dict(town=scenario.town, scenario=scenario.scenario, scenario_type=scenario.scenario_type)
                pred, gt, agent_ids, _ = replay_windows(scenario.root, row, conf, args.carla_maps,
                                                       str(args.predictor), "joint_scene", args.device)
                frame_boxes = {frame: boxes.merge_frame_boxes(scenario, frame)[0]
                               for frame in sorted({f for series in scenario.agents.values() for f in series.frames})}
                cache[scenario.scenario_id] = pred, gt, agent_ids, frame_boxes, estimate_collision(scenario, cfg.deepaccident)
            pred, gt, agent_ids, frame_boxes, collision = cache[scenario.scenario_id]
            matches = [w for w in pred.values() if f"{scenario.scenario_id}:{w.index}:{w.t_end:.3f}" == case["window_id"]]
            if len(matches) != 1 or matches[0].label not in gt:
                raise KeyError("requested predictor/GT window cannot be reconstructed")
            window = matches[0]
            actors, identities, predicted, truth, record, ttc = prepare_case(
                case, window, gt[window.label], agent_ids, collision, frame_boxes)
            record["case_id"] = case_id
            render_case(case_id, actors, identities, predicted, truth, record, ttc, args.out,
                        observed_window=window)
            case_pairs = pair_metadata(case_id, predicted, truth)
            metadata.append(record)
            pair_records.extend(case_pairs)
            print(f"[{i}/{len(cases)}] {case_id}", flush=True)
        except Exception as error:
            if not args.continue_on_error:
                raise RuntimeError(f"{case_id} {case['window_id']}: {error}") from error
            failures.append({"case_id": case_id, "window_id": case["window_id"],
                             "exception_type": type(error).__name__, "error_message": str(error)})
            print(f"FAILED {case_id} {case['window_id']}: {type(error).__name__}: {error}", flush=True)
    for path, fields, records in ((targets[0], FIELDS, metadata),
                                  (targets[1], ["case_id", "window_id"], metadata),
                                  (targets[2], PAIR_FIELDS, pair_records)):
        with path.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    if failures:
        failed_out = args.out / "failed_cases.csv"
        with failed_out.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["case_id", "window_id", "exception_type", "error_message"])
            writer.writeheader()
            writer.writerows(failures)
        raise RuntimeError(f"{len(failures)} case(s) failed; PDFs skipped; see {failed_out}")
    if not args.no_pdf:
        export_case_pdfs(args.out, [record["case_id"] for record in metadata])


if __name__ == "__main__":
    main()
