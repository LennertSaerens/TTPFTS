"""Regression gate for native-classification-aware doubling recommendations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from experiments.information_matched import classification as experiment
from experiments.information_matched.config import (
    load_config, sha256_file, verify_protected, verify_sources,
)
from experiments.information_matched.fileio import atomic_csv, atomic_json, read_csv
from experiments.information_matched.run_empirical import SampleRecorder
from momab_main import EGE_ALGORITHMS, ENVIRONMENTS, _construct_environment
from runner.seeding import seed_rngs


def test_partial_rule():
    means = np.array([[0.0, 0.0], [1.0, 1.0], [100.0, 100.0]])
    accepted = np.array([True, False, False])
    active = np.array([False, True, False])
    assert experiment.recommend(means, active, accepted).tolist() == [0, 1]
    assert experiment.recommend(means, np.zeros(3, bool), accepted).tolist() == [0]
    arms = np.array([0, 1, 0])
    rewards = np.array([[2.0, 2.0], [3.0, 3.0], [-10.0, -10.0]])
    events = [
        {"arm": 2, "decision": "reject", "sample_offset": 2},
        {"arm": 0, "decision": "accept", "sample_offset": 3},
        {"arm": 1, "decision": "accept_final", "sample_offset": 3},
    ]
    states = experiment.progressive_rows(arms, rewards, events, [0, 1, 2, 3, 8], 3)
    assert [json.loads(state["recommendation"]) for state in states] == [[0, 1, 2], [0], [1], [0, 1], [0, 1]]
    assert states[1]["rejected"] == "[]"  # No lookahead into the next completed round.
    assert states[2]["rejected"] == "[2]"
    assert states[3]["active"] == "[]"
    assert states[3] == states[4]  # Hold native output throughout unused nominal-budget slots.
    zero_events = [dict(event, sample_offset=0) for event in events]
    prior, first, endpoint = experiment.progressive_rows(
        np.empty(0, int), np.empty((0, 2)), zero_events, [0, 1, 22], 3
    )
    assert prior["recommendation"] == "[0,1,2]"
    assert first["recommendation"] == endpoint["recommendation"] == "[0,1]"
    print("PASS: accepted/rejected invariants, no future decisions, idle tail, and zero-sample timing")


def test_trace_is_passive():
    env = _construct_environment(ENVIRONMENTS["EgeExp7"][0], "gaussian")
    for base, function in EGE_ALGORITHMS.items():
        for budget in (22, 44, 2816):
            seed_rngs(0, "EgeExp7|gaussian", base, 0, "budget", budget)
            original = SampleRecorder(env)
            native = function(T=budget, K=22, D=2, environment=original)
            rng = np.random.get_state()
            seed_rngs(0, "EgeExp7|gaussian", base, 0, "budget", budget)
            recorder = experiment.TimedRecorder(env)
            trace = experiment.TimedTrace(recorder)
            traced = function(T=budget, K=22, D=2, environment=recorder, trace=trace)
            assert native == traced
            assert len(trace) == 22
            assert np.array_equal(rng[1], np.random.get_state()[1])
            assert rng[2:] == np.random.get_state()[2:]
            assert len(original.blocks) == len(recorder.blocks)
            for before, after in zip(original.blocks, recorder.blocks):
                assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    print("PASS: diagnostic timing leaves rewards, native recommendations and RNG unchanged")


def test_end_to_end():
    config = experiment.load_classification_config()
    assert config["labels"] == load_config()["labels"]
    assert all(len(algs) == 7 for algs in config["figure_sets"].values())
    verify_sources(config)
    verify_protected(config)
    with tempfile.TemporaryDirectory() as directory:
        root, serial = Path(directory) / "parallel", Path(directory) / "serial"
        names, reps = ["EgeExp7"], [0, 1]
        experiment.run(root, config, names, reps, 2, allow_dirty=True)
        shard_files = list((root / "classification" / "shards").rglob("*.csv"))
        stamps = {path: path.stat().st_mtime_ns for path in shard_files}
        experiment.run(root, config, names, reps, 2, allow_dirty=True)
        assert all(path.stat().st_mtime_ns == stamp for path, stamp in stamps.items())
        experiment.run(serial, config, names, reps, 1, allow_dirty=True)
        for kind in ("trajectory", "stages", "events"):
            assert experiment.result_path(root, names[0], kind).read_bytes() == experiment.result_path(
                serial, names[0], kind
            ).read_bytes()
        experiment.build_inputs(root, config, names)
        report = experiment.validate(root, config, names, reps, require_clean=False)
        assert report["summaries"]["EgeExp7"]["endpoint_equalities"] == 2 * 2 * 8
        assert report["summaries"]["EgeExp7"]["independent_replay_points"] > 0
        path = experiment.result_path(root, names[0])
        columns, rows = read_csv(path)
        original = path.read_bytes()
        target = next(i for i, row in enumerate(rows) if row["arm_pull"] == "5610")
        rows[target]["recommendation"] = "[]"
        atomic_csv(path, columns, rows)
        manifest_path = root / "classification" / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        manifest = json.loads(original_manifest)
        manifest["files"][str(path.relative_to(root))] = sha256_file(path)
        atomic_json(manifest_path, manifest)
        try:
            experiment.validate(root, config, names, reps, require_clean=False)
        except ValueError as error:
            assert "endpoint differs" in str(error)
        else:
            raise AssertionError("validator failed to reject an inconsistent endpoint")
        path.write_bytes(original)
        manifest_path.write_bytes(original_manifest)
        experiment.validate(root, config, names, reps, require_clean=False)
        figures = Path(directory) / "figures"
        experiment.plot.render(root, config, names, figures, allow_dirty=True, verify_style=False)
        experiment.validate(root, config, names, reps, require_clean=False, check_figures=True, figures_dir=figures)
        assert len(list(figures.glob("*.png"))) == len(list(figures.glob("*_plot_table.csv"))) == 6
        try:
            experiment.plot.render(root, config, names, figures, allow_dirty=True, verify_style=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("renderer overwrote an existing figure without approval")
    verify_sources(config)
    verify_protected(config)
    print("PASS: source preservation, resume, parallel identity, 32 exact endpoints, independent replay, "
          "corruption rejection, six figures, and unchanged labels/baselines")


if __name__ == "__main__":
    test_partial_rule()
    test_trace_is_passive()
    test_end_to_end()
    print("ALL CLASSIFICATION-AWARE DOUBLING TESTS PASSED")
