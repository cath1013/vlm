"""Audit every response in a validation-batch experiment.

Produces a detailed JSON record and a Markdown report.  The analysis is
deterministic: it never calls another LLM to judge the first LLM.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CUES = {
    "linear_contact": re.compile(r"linear[- ]extrapolat|closing rate|contact in", re.I),
    "ttc": re.compile(r"\bttc\b|time[- ]to[- ]collision", re.I),
    "junction_conflict": re.compile(
        r"junction|intersection|orthogonal|simultaneous(?:ly)? entr", re.I
    ),
    "predicted_path": re.compile(r"predicted path|projected path|trajector", re.I),
    "following": re.compile(r"rear[- ]end|car following|headway", re.I),
    "lane_or_opposing": re.compile(r"lane[- ]change|head[- ]on|sideswipe|opposing", re.I),
    "braking_or_stop": re.compile(r"brak|decelerat|\bstop(?:ped|ping)?\b", re.I),
    "uncertainty": re.compile(r"uncertain|limitation|signal state|cannot determine", re.I),
}

FALSE_DANGER = re.compile(
    r"accident (?:is |was )?(?:highly )?(?:expected|likely)|"
    r"collision (?:is |was )?(?:highly )?(?:expected|likely)|"
    r"(?:will|would) (?:still )?(?:collide|result in .*collision)|"
    r"lead(?:ing|s)? to (?:an? )?(?:rear-end |head-on )?collision|"
    r"making (?:an? )?collision (?:highly )?(?:probable|likely)",
    re.I,
)
TRUE_SAFE = re.compile(
    r"no (?:accident|collision) (?:is )?expected|collision (?:is )?unlikely|"
    r"no credible collision|remain safely separated",
    re.I,
)


def load_records(base: Path):
    return [json.loads(line) for line in (base / "records.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]


def response_path(base: Path, record: dict) -> Path:
    return (base / record["dataset_split"] / record["outcome"] /
            record["scenario"] / "responses" /
            f"response_{record['window']['label']}.json")


def analyze(base: Path) -> dict:
    records = load_records(base)
    detail = []
    statuses = Counter()
    confidence = defaultdict(Counter)
    cues_by_status = defaultdict(Counter)
    positive_bucket_hist = Counter()
    scenario_patterns = defaultdict(lambda: Counter(windows=0, positive=0))
    # These are not automatically contradictions: a negative bucket may validly
    # mention a collision in another bucket.  Preserve them for manual review.
    negative_danger_references = []
    positive_safe_references = []
    all_positive = 0
    all_negative = 0

    for rec in records:
        path = response_path(base, rec)
        saved = json.loads(path.read_text(encoding="utf-8"))
        answer = saved["answer"]
        predictions = sorted(answer.get("predictions") or [], key=lambda x: x.get("k", 0))
        positives = [p for p in predictions if p.get("accident_expected")]
        negatives = [p for p in predictions if not p.get("accident_expected")]
        # New reports expose the one-row-per-window axis explicitly.  Fall back
        # to the historical bucket-level alias for old records.
        binary = rec.get("binary_window") or rec.get("binary") or {}
        status = binary.get("status", "UNSCORABLE")
        statuses[status] += 1
        positive_bucket_hist[len(positives)] += 1
        all_positive += int(len(positives) == len(predictions) and bool(predictions))
        all_negative += int(not positives)
        sp = scenario_patterns[rec["scenario"]]
        sp["windows"] += 1
        sp["positive"] += int(bool(positives))

        pos_text = " ".join(str(p.get("reason", "")) for p in positives)
        all_text = " ".join(str(p.get("reason", "")) for p in predictions)
        cue_names = [name for name, rx in CUES.items() if rx.search(pos_text)]
        for name in cue_names:
            cues_by_status[status][name] += 1
        for p in predictions:
            confidence["positive" if p.get("accident_expected") else "negative"][
                str(p.get("confidence", "missing"))
            ] += 1

        contradictions = []
        for p in negatives:
            reason = str(p.get("reason", ""))
            if FALSE_DANGER.search(reason):
                item = {"k": p.get("k"), "reason": reason}
                contradictions.append({"false_but_danger": item})
                negative_danger_references.append({
                    "scenario": rec["scenario"], "window": rec["window"]["label"], **item
                })
        for p in positives:
            reason = str(p.get("reason", ""))
            if TRUE_SAFE.search(reason):
                item = {"k": p.get("k"), "reason": reason}
                contradictions.append({"true_but_safe": item})
                positive_safe_references.append({
                    "scenario": rec["scenario"], "window": rec["window"]["label"], **item
                })

        detail.append({
            "scenario": rec["scenario"],
            "outcome": rec["outcome"],
            "town": rec["town"],
            "window": rec["window"]["label"],
            "binary_status": status,
            "truth": binary.get("truth"),
            "pred": binary.get("pred"),
            "positive_ks": [p.get("k") for p in positives],
            "positive_confidence": [p.get("confidence") for p in positives],
            "cues_in_positive_reasons": cue_names,
            "contradictions": contradictions,
            "overall_assessment": answer.get("overall_assessment", ""),
            "data_limitations": answer.get("data_limitations", ""),
            "predictions": predictions,
            "response_path": str(path.relative_to(base)),
        })

    cue_table = {
        status: {name: counts.get(name, 0) for name in CUES}
        for status, counts in cues_by_status.items()
    }
    always_by_scenario = []
    for scenario, counts in scenario_patterns.items():
        always_by_scenario.append({
            "scenario": scenario,
            "windows": counts["windows"],
            "windows_with_warning": counts["positive"],
            "warning_rate": round(counts["positive"] / counts["windows"], 3),
        })
    always_by_scenario.sort(key=lambda x: (-x["warning_rate"], x["scenario"]))

    return {
        "n_responses": len(detail),
        "binary_statuses": dict(statuses),
        "positive_bucket_count_histogram": dict(sorted(positive_bucket_hist.items())),
        "all_five_buckets_positive": all_positive,
        "no_bucket_positive": all_negative,
        "confidence": {k: dict(v) for k, v in confidence.items()},
        "reason_cues_by_binary_status": cue_table,
        "negative_bucket_reason_references_danger": negative_danger_references,
        "positive_bucket_reason_references_safety": positive_safe_references,
        "scenario_warning_rates": always_by_scenario,
        "responses": detail,
    }


def markdown(doc: dict) -> str:
    lines = [
        "# LLM response audit — WaypointNet pilot 20",
        "",
        f"All **{doc['n_responses']}** parsed responses were inspected deterministically.",
        "",
        "## Summary",
        "",
        f"- Binary statuses: `{doc['binary_statuses']}`",
        f"- No positive bucket: **{doc['no_bucket_positive']}** responses",
        f"- All five buckets positive: **{doc['all_five_buckets_positive']}** responses",
        f"- Negative buckets whose explanation references danger (often in another interval): "
        f"**{len(doc['negative_bucket_reason_references_danger'])}** intervals",
        f"- Positive buckets whose explanation references safety: "
        f"**{len(doc['positive_bucket_reason_references_safety'])}** intervals",
        "",
        "### Positive buckets per response",
        "",
        "| Positive buckets | Responses |",
        "|---:|---:|",
    ]
    for n, count in doc["positive_bucket_count_histogram"].items():
        lines.append(f"| {n} | {count} |")
    lines.extend(["", "### Confidence", "", "| Boolean | low | medium | high |", "|---|---:|---:|---:|"])
    for key in ("positive", "negative"):
        c = doc["confidence"].get(key, {})
        lines.append(f"| {key} | {c.get('low',0)} | {c.get('medium',0)} | {c.get('high',0)} |")
    lines.extend(["", "### Cues found in positive reasons", "", "Counts are response-level: a cue is counted once when it appears in any positive interval.", "", "| Status | " + " | ".join(CUES) + " |", "|---|" + "---:|" * len(CUES)])
    for status in ("TP", "FP", "TN", "FN", "UNSCORABLE"):
        if status not in doc["reason_cues_by_binary_status"]:
            continue
        c = doc["reason_cues_by_binary_status"][status]
        lines.append("| " + status + " | " + " | ".join(str(c.get(k, 0)) for k in CUES) + " |")
    lines.extend([
        "",
        "## Response-by-response index",
        "",
        "The JSON audit contains every interval reason verbatim. This index shows every response and its diagnostic classification.",
        "",
        "| # | Scenario | Window | Outcome | Status | Positive k | Cues | Contradictions |",
        "|---:|---|---:|---|---|---|---|---:|",
    ])
    for i, r in enumerate(doc["responses"], 1):
        lines.append(
            f"| {i} | {r['scenario']} | {r['window']} | {r['outcome']} | "
            f"{r['binary_status']} | {','.join(map(str,r['positive_ks'])) or '—'} | "
            f"{', '.join(r['cues_in_positive_reasons']) or '—'} | {len(r['contradictions'])} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    args = ap.parse_args(argv)
    base = Path(args.experiment).resolve()
    doc = analyze(base)
    (base / "response_audit.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (base / "response_audit.md").write_text(markdown(doc), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k not in ("responses",)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
