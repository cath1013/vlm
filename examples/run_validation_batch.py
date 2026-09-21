"""Generate and optionally evaluate a balanced DeepAccident validation batch.

This is the reproducible bridge between the one-scenario window generator and a
full experiment.  Every scenario gets its own directory so payloads from two
scenarios can never overwrite one another.

Examples
--------
Generate and audit 10 accident + 10 normal scenarios::

    .venv/bin/python examples/run_validation_batch.py \
        --root /home/sryu/inclab-nas/DeepAccident --carla-maps carla_map \
        --out out/experiments/waypointnet_audit20

Call the provider after the audit succeeds (safe to resume)::

    .venv/bin/python examples/run_validation_batch.py \
        --root /home/sryu/inclab-nas/DeepAccident --carla-maps carla_map \
        --out out/experiments/waypointnet_audit20 --skip-generate --call-llm
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.accident_qa import aggregate_modes  # noqa: E402


def _pick_balanced(scenarios, n_each: int, seed: int):
    """Select a deterministic but non-lexicographic balanced cohort."""
    rng = random.Random(seed)
    selected = []
    for outcome in ("accident", "normal"):
        pool = [s for s in scenarios if s.split == "val" and s.outcome == outcome]
        rng.shuffle(pool)
        if len(pool) < n_each:
            raise SystemExit(
                f"val/{outcome}: requested {n_each}, but only {len(pool)} available"
            )
        selected.extend(pool[:n_each])
    # Keep generation order stable and interleave outcomes for easier monitoring.
    acc = sorted((s for s in selected if s.outcome == "accident"),
                 key=lambda s: (s.town, s.scenario_type, s.scenario))
    nor = sorted((s for s in selected if s.outcome == "normal"),
                 key=lambda s: (s.town, s.scenario_type, s.scenario))
    return [x for pair in zip(acc, nor) for x in pair]


def _pick_all_validation(scenarios):
    """Select every validation scenario in a deterministic order."""
    selected = [s for s in scenarios if s.split == "val"]
    selected.sort(key=lambda s: (s.scenario_type, s.scenario))
    return selected


def _scenario_dir(base: Path, sc) -> Path:
    return base / sc.split / sc.outcome / sc.scenario


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    if proc.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        raise RuntimeError(
            f"command failed ({proc.returncode}); log={log_path}\n" + "\n".join(tail)
        )


def _write_batch_manifest(path: Path, args, selected) -> dict:
    doc = {
        "created_at_unix": time.time(),
        "dataset_root": os.path.abspath(args.root),
        "dataset_split": "val",
        "selection": {
            "seed": args.seed,
            "n_each": args.n_each,
            "all_validation": args.all_validation,
            "n_total": len(selected) if args.all_validation else None,
        },
        "config": {
            "mode": args.mode,
            "window_s": args.window,
            "stride_s": args.stride,
            "horizon_s": args.horizon,
            "contact_margin_m": args.contact_margin_m,
            "future_source": args.future_source,
            # The comparison experiment uses fixed-length observation windows.
            # Keep this explicit in the cohort manifest so an expanding-prefix
            # run cannot be mistaken for the intended setup.
            "warmup": False,
            "full_window": False,
            "history_stride_s": args.history_stride,
            "payload_profile": args.payload_profile,
            "predictor": os.path.abspath(args.predictor),
            "predictor_mode": args.predictor_mode,
            "pair_reranker_model": (
                os.path.abspath(args.pair_reranker_model)
                if args.pair_reranker_model else None
            ),
            "language": args.language,
            "provider": args.provider,
            "model": args.model,
            "early_credit_s": args.early_credit,
            "score_modes": [
                "binary_scenario", "binary_window", "strict", "early",
                "weighted", "binary_bucket",
            ],
            "late_decay": args.late_decay,
            "early_decay": args.early_decay,
        },
        "scenarios": [
            {
                "scenario": s.scenario,
                "scenario_type": s.scenario_type,
                "dataset_split": s.split,
                "outcome": s.outcome,
                "town": s.town,
                "directory": str(_scenario_dir(path.parent, s).relative_to(path.parent)),
            }
            for s in selected
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


def generate(args, selected) -> None:
    py = sys.executable
    script = str(ROOT / "examples" / "run_deepaccident.py")
    for i, sc in enumerate(selected, 1):
        target = _scenario_dir(Path(args.out), sc)
        manifest = target / "manifest.json"
        if args.skip_existing and manifest.is_file():
            print(f"[{i:02d}/{len(selected)}] reuse {sc.outcome:8s} {sc.scenario}", flush=True)
            continue
        print(f"[{i:02d}/{len(selected)}] build {sc.outcome:8s} {sc.scenario}", flush=True)
        cmd = [
            py, script,
            "--root", args.root,
            "--scenario", sc.scenario,
            "--type", sc.scenario_type,
            "--mode", args.mode,
            "--carla-maps", args.carla_maps,
            "--windows-dir", args.out,
            "--out-label", sc.scenario,
            "--predictor", args.predictor,
            "--predictor-device", args.predictor_device,
            "--provider", args.provider,
            "--model", args.model,
            "--language", args.language,
            "--window", str(args.window),
            "--stride", str(args.stride),
            "--horizon", str(args.horizon),
            "--contact-margin-m", str(args.contact_margin_m),
            "--future-source", args.future_source,
            "--history-stride", str(args.history_stride),
            "--payload-profile", args.payload_profile,
            "--early-credit", str(args.early_credit),
            "--no-warmup",
            "--no-full-window",
        ]

        if args.predictor_mode:
            cmd.extend(["--predictor-mode", args.predictor_mode])
        if args.pair_reranker_model:
            cmd.extend(["--pair-reranker-model", args.pair_reranker_model])
        _run(cmd, Path(args.out) / "logs" / f"generate_{sc.scenario}.log")
        if not manifest.is_file():
            raise RuntimeError(f"generator did not create {manifest}")


def _iter_manifests(base: Path, cohort: dict) -> Iterable[tuple[dict, Path, dict]]:
    for row in cohort["scenarios"]:
        d = base / row["directory"]
        p = d / "manifest.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        yield row, d, json.loads(p.read_text(encoding="utf-8"))


def audit(base: Path, cohort: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    windows = accident_windows = chars = unobserved = 0
    outcome_counts = Counter()
    town_counts = Counter()

    for row, directory, man in _iter_manifests(base, cohort):
        scenario = row["scenario"]
        outcome_counts[row["outcome"]] += 1
        town_counts[row["town"]] += 1
        msc = man.get("scenario") or {}
        cfg = man.get("config") or {}
        if msc.get("id", "").split("/")[-1] != scenario:
            errors.append(f"{scenario}: manifest scenario mismatch: {msc.get('id')}")
        if msc.get("outcome") != row["outcome"]:
            errors.append(f"{scenario}: outcome mismatch")
        if cfg.get("full_window") is not False:
            errors.append(f"{scenario}: full_window must be false")
        expected_warmup = bool(cohort.get("config", {}).get("warmup", False))
        if bool(cfg.get("warmup", False)) != expected_warmup:
            errors.append(
                f"{scenario}: warmup mismatch: {cfg.get('warmup')} != "
                f"{expected_warmup}"
            )
        expected_profile = cohort.get("config", {}).get(
            "payload_profile", "standard"
        )
        if cfg.get("payload_profile", "standard") != expected_profile:
            errors.append(
                f"{scenario}: payload profile mismatch: "
                f"{cfg.get('payload_profile')} != {expected_profile}"
            )
        expected_future_source = cohort.get("config", {}).get(
            "future_source", "predictor"
        )
        if cfg.get("future_source", "predictor") != expected_future_source:
            errors.append(
                f"{scenario}: future source mismatch: "
                f"{cfg.get('future_source')} != {expected_future_source}"
            )
        predictor_name = str(cfg.get("predictor", "")).lower()
        if not ("waypointnet" in predictor_name or "jointscene" in predictor_name
                or "joint_scene" in predictor_name):
            errors.append(f"{scenario}: expected waypoint or JointScene predictor not recorded")
        expected_reranker = cohort.get("config", {}).get("pair_reranker_model")
        if expected_reranker:
            recorded_reranker = str(cfg.get("pair_reranker", ""))
            if Path(expected_reranker).name not in recorded_reranker:
                errors.append(
                    f"{scenario}: pair re-ranker mismatch: "
                    f"{recorded_reranker} != {expected_reranker}"
                )
        if not str(man.get("scenario", {}).get("town", "")).startswith("Town"):
            errors.append(f"{scenario}: missing town")
        if not str(cfg.get("endpoint", "")).startswith("https://"):
            errors.append(f"{scenario}: provider endpoint missing")

        for w in man.get("windows") or []:
            windows += 1
            chars += int(w.get("input_chars") or 0)
            accident_windows += int(bool(w.get("has_accident_in_horizon")))
            pp = directory / w["payload"]
            gp = directory / w["ground_truth"]
            if not pp.is_file() or not gp.is_file():
                errors.append(f"{scenario}/{w.get('label')}: payload or GT missing")
                continue
            payload = json.loads(pp.read_text(encoding="utf-8"))
            gt = json.loads(gp.read_text(encoding="utf-8"))
            if gt.get("window", {}).get("t_end_s") != w.get("t_end_s"):
                errors.append(f"{scenario}/{w.get('label')}: cutoff mismatch")
            if gt.get("collision", {}).get("time_s") is not None:
                ct = float(gt["collision"]["time_s"])
                if ct <= float(w["t_end_s"]):
                    errors.append(f"{scenario}/{w.get('label')}: collision is observed, not future")
            expected = gt.get("expected") or []
            if len(expected) != int(round(float(cohort["config"]["horizon_s"]))):
                errors.append(f"{scenario}/{w.get('label')}: wrong bucket count")
            unobserved += len(gt.get("unobserved_carla_ids") or [])
            # The answer schema must be present in every provider payload.  This
            # catches accidentally generating a generic scene-QA request.
            blob = json.dumps(payload, ensure_ascii=False)
            if "accident_expected" not in blob or "predictions" not in blob:
                errors.append(f"{scenario}/{w.get('label')}: prediction schema absent")
        if not man.get("windows"):
            warnings.append(f"{scenario}: no scorable pre-collision windows")

    selection = cohort["selection"]
    if selection.get("all_validation"):
        expected_total = int(selection.get("n_total", 0))
        if sum(outcome_counts.values()) != expected_total:
            errors.append(
                f"validation cohort size mismatch: "
                f"{sum(outcome_counts.values())} != {expected_total}"
            )
    else:
        expected_each = int(selection["n_each"])
        if outcome_counts != Counter({"accident": expected_each, "normal": expected_each}):
            errors.append(f"cohort is not balanced: {dict(outcome_counts)}")
    report = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_scenarios": sum(outcome_counts.values()),
        "outcomes": dict(outcome_counts),
        "towns": dict(sorted(town_counts.items())),
        "n_windows": windows,
        "n_accident_windows": accident_windows,
        "n_normal_windows": windows - accident_windows,
        "input_chars_mean": round(chars / windows) if windows else None,
        "unobserved_collision_actor_mentions": unobserved,
    }
    (base / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report


def call_llm(args, cohort: dict) -> None:
    py = sys.executable
    script = str(ROOT / "examples" / "ask_llm.py")
    manifest_rows = list(_iter_manifests(Path(args.out), cohort))
    # A fixed-window run can legitimately produce zero windows when the
    # recording is shorter than the requested observation.  Such scenarios
    # have no payload to send and must not abort the entire batch.
    jobs = [job for job in manifest_rows if job[2].get("windows")]
    n_skipped = len(manifest_rows) - len(jobs)
    if n_skipped:
        print(f"Skip {n_skipped} scenarios with no generated windows.", flush=True)

    def one(job):
        row, directory, _ = job
        cmd = [
            py, script, str(directory), "--score", "--modes", "all",
            "--provider", args.provider, "--skip-existing",
            "--late-decay", str(args.late_decay),
            "--early-decay", str(args.early_decay),
        ]
        _run(cmd, Path(args.out) / "logs" / f"llm_{row['scenario']}.log")
        return row["scenario"]

    # Each ask_llm process remains serial within one scenario, preserving its
    # straightforward resume semantics.  Parallelism is bounded across scenarios.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, job): job[0]["scenario"] for job in jobs}
        done = 0
        for fut in as_completed(futures):
            scenario = futures[fut]
            fut.result()
            done += 1
            print(f"[{done:02d}/{len(jobs)}] LLM complete {scenario}", flush=True)


def aggregate_batch(base: Path, cohort: dict) -> dict:
    """Merge scenario score files into a single paired-record result."""
    modes = tuple(cohort["config"]["score_modes"])
    records = []
    missing = []
    no_windows = []
    for row, directory, manifest in _iter_manifests(base, cohort):
        if not manifest.get("windows"):
            no_windows.append(row["directory"])
            continue
        path = directory / "responses" / "scores_modes.json"
        if not path.is_file():
            missing.append(row["scenario"])
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for score in doc.get("per_window") or []:
            enriched = {
                "scenario": row["scenario"],
                "scenario_type": row["scenario_type"],
                "dataset_split": row["dataset_split"],
                "outcome": row["outcome"],
                "town": row["town"],
                **score,
            }
            records.append(enriched)
    result = {
        "provider": cohort["config"]["provider"],
        "model": cohort["config"]["model"],
        "modes": list(modes),
        "n_scenarios_total": len(cohort["scenarios"]),
        "n_scenarios_with_windows": len(cohort["scenarios"]) - len(no_windows),
        "n_scenarios_no_windows": len(no_windows),
        "n_scenarios_complete": len(cohort["scenarios"]) - len(no_windows) - len(missing),
        "n_windows": len(records),
        "no_window_scenarios": no_windows,
        "missing_scenarios": missing,
        "aggregate": aggregate_modes(records, modes) if records else {},
    }
    (base / "scores_modes.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (base / "records.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--carla-maps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-each", type=int, default=10)
    ap.add_argument("--all-validation", action="store_true",
                    help="use every scenario in the validation split")
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--mode", choices=("sensor3d", "camera"), default="sensor3d")
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--contact-margin-m", type=float, default=0.0,
                    help="swept-path contact threshold in metres (default: 0.0)")
    ap.add_argument("--future-source", choices=("predictor", "ground_truth"),
                    default="predictor")
    ap.add_argument("--history-stride", type=float, default=1.0)
    ap.add_argument("--payload-profile", choices=("standard", "compact"),
                    default="compact",
                    help="payload format (default: compact; standard disables pair re-ranking)")
    ap.add_argument("--early-credit", type=float, default=5.0,
                    help="early credit in seconds (main protocol default: 5)")
    ap.add_argument("--late-decay", type=float, default=0.2)
    ap.add_argument("--early-decay", type=float, default=0.0)
    ap.add_argument("--predictor", default="out/predict_model/waypointnet_best.pt")
    ap.add_argument(
         "--predictor-mode",
         choices=("rank", "waypoints", "joint_scene"),
         default=None,
         help="predictor mode; omitted = infer automatically from checkpoint",
    )
    reranker = ap.add_mutually_exclusive_group()
    reranker.add_argument("--pair-reranker-model", default=None,
                         help="override the default V2 re-ranker JSON checkpoint")
    reranker.add_argument("--no-pair-reranker", action="store_true",
                         help="disable pair re-ranking (raw compact payload)")
    ap.add_argument("--predictor-device", default="cpu")
    ap.add_argument("--provider", choices=("claude", "openai", "gemini"),
                    default="gemini")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--language", choices=("ko", "en"), default="en")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--call-llm", action="store_true")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent scenario-level LLM jobs (default: 4)")
    args = ap.parse_args(argv)
    if args.payload_profile == "standard" and args.pair_reranker_model:
        ap.error("--pair-reranker-model requires --payload-profile compact")
    if args.payload_profile == "compact" and not args.no_pair_reranker:
        args.pair_reranker_model = args.pair_reranker_model or str(
            ROOT / "out/pair_reranker_v2_full_model/pair_reranker_v2.json"
        )
    if not args.skip_generate and args.pair_reranker_model:
        if not Path(args.pair_reranker_model).is_file():
            ap.error(f"missing pair re-ranker checkpoint: {args.pair_reranker_model}; "
                     "provide --pair-reranker-model or use --no-pair-reranker")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    base = Path(args.out).resolve()
    cohort_path = base / "batch_manifest.json"
    if args.skip_generate:
        if not cohort_path.is_file():
            raise SystemExit(f"missing cohort manifest: {cohort_path}")
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    else:
        runner = DeepAccidentRunner(args.root, PipelineConfig())
        if args.all_validation:
            selected = _pick_all_validation(runner.list_scenarios())
        else:
            selected = _pick_balanced(runner.list_scenarios(), args.n_each, args.seed)
        cohort = _write_batch_manifest(cohort_path, args, selected)
        generate(args, selected)

    report = audit(base, cohort)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not report["passed"]:
        raise SystemExit("audit failed; LLM calls were not started")
    if args.call_llm:
        call_llm(args, cohort)
        combined = aggregate_batch(base, cohort)
        print(json.dumps(combined, ensure_ascii=False, indent=1))
    else:
        complete = all(
            not manifest.get("windows")
            or (directory / "responses" / "scores_modes.json").is_file()
            for _, directory, manifest in _iter_manifests(base, cohort)
        )
        if complete:
            combined = aggregate_batch(base, cohort)
            print(json.dumps(combined, ensure_ascii=False, indent=1))
        else:
            print("Audit passed. Re-run with --skip-generate --call-llm to call the provider.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
