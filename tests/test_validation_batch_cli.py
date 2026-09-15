from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from examples.run_validation_batch import ROOT, aggregate_batch, call_llm, parse_args


BASE = ["--root", "/dataset", "--carla-maps", "carla_map", "--out", "out/test"]


def test_defaults_enable_compact_and_v2():
    with patch.object(Path, "is_file", return_value=True):
        args = parse_args(BASE)
    assert args.payload_profile == "compact"
    assert args.pair_reranker_model == str(
        ROOT / "out/pair_reranker_v2_full_model/pair_reranker_v2.json"
    )


def test_standard_disables_reranker():
    args = parse_args(BASE + ["--payload-profile", "standard"])
    assert args.payload_profile == "standard"
    assert args.pair_reranker_model is None


def test_raw_compact_opt_out():
    args = parse_args(BASE + ["--no-pair-reranker"])
    assert args.payload_profile == "compact"
    assert args.pair_reranker_model is None


def test_custom_checkpoint(tmp_path):
    checkpoint = tmp_path / "custom.json"
    checkpoint.touch()
    args = parse_args(BASE + ["--pair-reranker-model", str(checkpoint)])
    assert args.pair_reranker_model == str(checkpoint)


@pytest.mark.parametrize("flags", [
    ["--no-pair-reranker", "--pair-reranker-model", "custom.json"],
    ["--payload-profile", "standard", "--pair-reranker-model", "custom.json"],
])
def test_conflicting_options_rejected(flags):
    with pytest.raises(SystemExit):
        parse_args(BASE + flags)


def test_missing_checkpoint_fails_before_generation():
    with patch.object(Path, "is_file", return_value=False):
        with pytest.raises(SystemExit):
            parse_args(BASE)


def test_resume_does_not_require_local_checkpoint():
    with patch.object(Path, "is_file", return_value=False):
        assert parse_args(BASE + ["--skip-generate"]).skip_generate


def _cohort(tmp_path):
    rows = []
    for name, windows in (("empty", []), ("ready", [{"label": "0-5"}])):
        directory = f"val/accident/{name}"
        target = tmp_path / directory
        target.mkdir(parents=True)
        (target / "manifest.json").write_text(
            __import__("json").dumps({"windows": windows}), encoding="utf-8"
        )
        rows.append({
            "scenario": name, "scenario_type": "type", "dataset_split": "val",
            "outcome": "accident", "town": "Town", "directory": directory,
        })
    return {
        "config": {
            "provider": "gemini", "model": "model", "late_decay": 0.2,
            "early_decay": 0.0, "score_modes": ["binary_window"],
        },
        "scenarios": rows,
    }


def test_call_llm_skips_scenario_with_no_windows(tmp_path):
    args = SimpleNamespace(out=str(tmp_path), provider="gemini", workers=1,
                           late_decay=0.2, early_decay=0.0)
    with patch("examples.run_validation_batch._run") as run:
        call_llm(args, _cohort(tmp_path))
    assert run.call_count == 1
    assert "ready" in str(run.call_args)


def test_aggregate_reports_no_window_scenario_separately(tmp_path):
    cohort = _cohort(tmp_path)
    result = aggregate_batch(tmp_path, cohort)
    assert result["n_scenarios_total"] == 2
    assert result["n_scenarios_with_windows"] == 1
    assert result["n_scenarios_no_windows"] == 1
    assert result["n_scenarios_complete"] == 0
    assert result["missing_scenarios"] == ["ready"]
