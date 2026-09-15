"""Replay only saved scorable-FP windows and separate pair-selection exposure.

No LLM call is made.  A V2-selected contact pair that is outside raw clearance
top-12 is *reranker-introduced exposure*, not proof that V2 caused Gemini's
verdict.  Conversely, a raw top-12 pair would have been exposed without V2.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm.accident_qa import WindowConfig, build_windows  # noqa: E402
from traffic_llm.carla_map import find_xodr  # noqa: E402
from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.deepaccident import estimate_collision  # noqa: E402
from traffic_llm.pair_reranker_v2 import PairRerankerV2, pair_features_v2  # noqa: E402
from traffic_llm.predict_model import TorchPredictor  # noqa: E402
from traffic_llm.serialize import rank_actors  # noqa: E402
from traffic_llm.swept_path import swept_pair_clearances  # noqa: E402


def pair_key(row):
    return (row["actor_a"], row["actor_b"], row["interval_index"])


def contact_rows(scene):
    return [dict(zip(scene["predicted_pair_columns"], row))
            for row in scene["predicted_closest_pairs"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--carla-maps", required=True)
    ap.add_argument("--fp-json", default="out/analysis/fp_cases_20260914.json")
    ap.add_argument("--experiment", default=(
        "out/experiments/waypointnet_compact_v2_fixed_val104_gemini3flash_20260909"))
    ap.add_argument("--predictor", default="out/predict_model/waypointnet_best.pt")
    ap.add_argument("--reranker", default="out/pair_reranker_v2_full_model/pair_reranker_v2.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-scenarios", type=int, default=None)
    args = ap.parse_args(argv)

    fp_doc = json.loads(Path(args.fp_json).read_text(encoding="utf-8"))
    cases = fp_doc["cases"]
    by_scene = defaultdict(list)
    for case in cases:
        by_scene[(case["scenario"], case["outcome"])].append(case)
    manifest = json.loads((Path(args.experiment) / "batch_manifest.json").read_text())
    metadata = {(row["scenario"], row["outcome"]): row
                for row in manifest["scenarios"]}
    wanted = sorted(by_scene, key=lambda x: (x[1], x[0]))
    if args.limit_scenarios:
        wanted = wanted[:args.limit_scenarios]

    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = "sensor3d"
    cfg.predictor = TorchPredictor(args.predictor, mode="waypoints", device=args.device)
    runner = DeepAccidentRunner(args.root, cfg)
    reranker = PairRerankerV2.load(args.reranker)
    wcfg = WindowConfig(window_s=5, stride_s=1, horizon_s=5, snapshot_rate_hz=2,
                        history_stride_s=1, warmup=False, full_window=False)
    out_cases, errors = [], []
    for ordinal, (scenario_name, outcome) in enumerate(wanted, 1):
        meta = metadata.get((scenario_name, outcome))
        if not meta:
            errors.append({"scenario": scenario_name, "error": "manifest mismatch"})
            continue
        try:
            xodr = find_xodr(meta["town"], [args.carla_maps])
            built = runner.build(scenario_name, meta["scenario_type"], opendrive_path=xodr)
            snapshots = list(built.snapshots(rate_hz=2))
            collision = estimate_collision(built.scenario, cfg.deepaccident)
            windows, _ = build_windows(snapshots, wcfg, collision_time_s=(
                collision.time_s if collision and collision.occurred else None))
            replay = {w.label: w for w in windows}
            for case in by_scene[(scenario_name, outcome)]:
                win = replay.get(case["window"])
                if win is None:
                    errors.append({"case_id": case["case_id"], "error": "window absent"})
                    continue
                actors, _ = rank_actors(win.last, wcfg.actor_cap(len(win.last.actors)))
                actor_by_id = {a.actor_id: a for a in actors}
                raw = swept_pair_clearances(
                    actors, horizon_s=wcfg.horizon_s, sample_dt_s=wcfg.swept_sample_dt_s,
                    contact_margin_m=wcfg.swept_contact_margin_m,
                    exclude_touching_now=wcfg.swept_exclude_touching_now,
                    exclude_static_pairs=wcfg.swept_exclude_static_pairs,
                )[:wcfg.swept_candidate_pool_cap]
                scored = sorted(((reranker.predict_features(pair_features_v2(
                    actor_by_id[p.actor_a], actor_by_id[p.actor_b], p, wcfg.horizon_s)), p)
                    for p in raw), key=lambda x: (-x[0], x[1].minimum_clearance_m,
                    x[1].time_after_observation_s, x[1].actor_a, x[1].actor_b))
                raw_rank = {pair_key({"actor_a": p.actor_a, "actor_b": p.actor_b,
                                      "interval_index": p.interval_index}): i + 1
                            for i, p in enumerate(raw)}
                v2_rank = {pair_key({"actor_a": p.actor_a, "actor_b": p.actor_b,
                                     "interval_index": p.interval_index}): i + 1
                           for i, (_, p) in enumerate(scored)}
                saved_path = Path(case["response"]).parent.parent / ("llm_payload_" + case["window"] + ".json")
                saved_payload = json.loads(saved_path.read_text())
                saved_scene = json.loads(saved_payload["contents"][0]["parts"][0]["text"])
                saved_pairs = contact_rows(saved_scene)
                claimed_ids = set(case["prediction"]["involved_actor_ids"])
                claimed = [p for p in raw if p.predicted_contact
                           and {p.actor_a, p.actor_b} <= claimed_ids]
                rows = []
                for p in claimed:
                    key = pair_key({"actor_a": p.actor_a, "actor_b": p.actor_b,
                                    "interval_index": p.interval_index})
                    rows.append({"actor_a": p.actor_a, "actor_b": p.actor_b,
                                 "interval_index": p.interval_index,
                                 "raw_rank": raw_rank[key], "v2_rank": v2_rank[key],
                                 "v2_score": round(float(reranker.predict_features(pair_features_v2(
                                     actor_by_id[p.actor_a], actor_by_id[p.actor_b], p, wcfg.horizon_s))), 8),
                                 "raw_top12": raw_rank[key] <= 12,
                                 "v2_top12": v2_rank[key] <= 12,
                                 "same_bucket": p.interval_index == case["prediction"]["k"]})
                saved_keys = {(p["actor_a"], p["actor_b"], p["interval_index"])
                              for p in saved_pairs}
                replay_keys = {key for key, rank in v2_rank.items() if rank <= 12}
                out_cases.append({"case_id": case["case_id"], "scenario": scenario_name,
                                  "window": case["window"], "k": case["prediction"]["k"],
                                  "claimed_contact_candidates": rows,
                                  "saved_v2_selection_matches_replay": saved_keys == replay_keys,
                                  "saved_count": len(saved_keys), "replay_count": len(replay_keys)})
        except Exception as exc:
            errors.append({"scenario": scenario_name, "error": f"{type(exc).__name__}: {exc}"})
        print(f"[{ordinal}/{len(wanted)}] {scenario_name} cases={len(out_cases)} errors={len(errors)}", flush=True)
    counts = Counter()
    for case in out_cases:
        rows = case["claimed_contact_candidates"]
        counts["fp_cases_replayed"] += 1
        counts["replay_exact_selection_match"] += case["saved_v2_selection_matches_replay"]
        if not rows:
            category = "no_contact_candidate_in_raw50"
        elif any(r["v2_top12"] and not r["raw_top12"] for r in rows):
            category = "reranker_introduced_contact_exposure"
        elif any(r["v2_top12"] and r["same_bucket"] for r in rows):
            category = "v2_selected_same_bucket_raw_also_exposed"
        elif any(r["v2_top12"] for r in rows):
            category = "v2_selected_other_bucket_raw_also_exposed"
        elif any(r["raw_top12"] for r in rows):
            category = "raw_only_contact_exposure"
        else:
            category = "contact_not_exposed_by_top12"
        case["exposure_category"] = category
        counts[category] += 1
    doc = {"definition": __doc__, "counts": dict(counts), "errors": errors,
           "cases": out_cases}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": doc["counts"], "errors": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
