"""Classification-aware anytime recommendations for native doubling EGE.

Run ``python -m experiments.information_matched.classification --n-jobs 8``.
Add ``--pool-samples`` to retain observations across stages for the active
arms' empirical means, while native EGE selection and classification remain
stage-local. Each mode has a separate immutable binding and output directory.
The new figures retain every existing display label: only the solid D-EGE
curves change. All other curves are copied from the original plot inputs.

Within each fresh stage, recommend accepted arms union the empirical Pareto
set of still-active arms. Rejected arms cannot return and accepted arms cannot
disappear. Apply native classification events after their round's observations,
including final acceptance. Thus the recommendation equals the native EGE
output throughout its nominal-budget idle tail and at every completed endpoint.

By default means/counts and classifications reset strictly after an endpoint;
with pooling only classifications reset. At t=0
retain the original zero-information recommendation (all arms). Zero-sample
classification events are observable on the first integer pull of their
stage, never at the preceding stage's endpoint. Future events from a replayed
incomplete stage never enter recommendations before their observation time.

Native EGE functions, sampling, budget schedules, seeds and tie-breaking are
unchanged. Recording rewards and the existing optional decision trace is
passive; round-robin within-round order is the same as the earlier experiments.
This is a new interim output rule, not a claim of an anytime error guarantee.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bandits.APE import _pareto_optimal_arms_mask
from momab_main import EGE_ALGORITHMS, ENVIRONMENTS, _construct_environment
from runner.ege_allocation import expected_ege_draws
from runner.instance import compute_instance_hash
from runner.schema import SCHEMA_COLUMNS, build_row, encode_int_set
from runner.seeding import seed_rngs

from . import plot
from .build import PLOT_COLUMNS, plot_input_path
from .config import (
    EMPIRICAL_ALGORITHMS, METRICS, REPO_ROOT, assert_safe_output, config_hash, environment_specs,
    load_config, sha256_file, source_provenance, verify_protected, verify_sources,
)
from .fileio import atomic_csv, atomic_json, read_csv
from .run_empirical import SampleRecorder, STAGE_COLUMNS, evaluation_points, round_robin, stage_plan
from .validate import Problems, figure_paths, report_path, validate_figures

CONFIG_PATH = Path(__file__).with_name("classification_config.json")
POOLED_CONFIG_PATH = Path(__file__).with_name("classification_pooled_config.json")
RULE = "native accepted arms union empirical Pareto of current-stage active arms; rejected arms excluded"
POOLED_RULE = "native accepted arms union empirical Pareto of active arms using all stages' observations; rejected arms excluded"
COLUMNS = [
    *SCHEMA_COLUMNS, "base_algorithm", "active_stage", "samples_drawn", "stage_samples",
    "unsampled_arms", "accepted", "rejected", "active", "config_hash",
]
EVENT_COLUMNS = [
    "environment", "algorithm", "repetition", "stage", "round", "arm", "decision",
    "sample_offset", "global_time", "samples_at_decision",
]


def load_classification_config(pool_samples: bool = False) -> dict:
    with (POOLED_CONFIG_PATH if pool_samples else CONFIG_PATH).open() as handle:
        binding = json.load(handle)
    config = copy.deepcopy(load_config())
    rule = POOLED_RULE if pool_samples else RULE
    if binding["base_config_hash"] != config_hash(config) or binding["recommendation"] != rule:
        raise ValueError("classification-aware configuration differs from its frozen specification")
    if pool_samples and binding.get("sample_memory") != "all_stages":
        raise ValueError("pooled classification-aware configuration must retain all earlier observations")
    config["classification_binding"] = binding
    config["empirical_doubling"].update({
        "recommendation": rule,
        "sample_memory": "all_stages" if pool_samples else "current_stage",
        "classification_timing": "native round-end trace offsets; post-decision recommendation",
        "evaluation_points": "shared grid, completed endpoints and first post-restart pull",
    })
    config["figures_dir"] = binding["figures_dir"]
    config["protected_files"].update(binding["additional_protected_files"])
    suffix = "classification_pooled" if pool_samples else "classification"
    config["figure_sets"] = {f"{name}_{suffix}": config["figure_sets"][name] for name in ("uv_both", "kv_both")}
    config["plot"]["legend_kwargs"] = {
        f"{name}_{suffix}": config["plot"]["legend_kwargs"][name] for name in ("uv_both", "kv_both")
    }
    return config


def output_root(config: dict) -> Path:
    pooled = "_pooled" if config["empirical_doubling"]["sample_memory"] == "all_stages" else ""
    return REPO_ROOT / f"results_information_matched_classification{pooled}_{config_hash(config)[:12]}"


def source_root(config: dict) -> Path:
    return REPO_ROOT / config["classification_binding"]["source_root"]


def task_dir(root: Path, name: str, algorithm: str, repetition: int) -> Path:
    return root / "classification" / "shards" / name / algorithm / f"rep{repetition:05d}"


def result_path(root: Path, name: str, kind: str = "trajectory") -> Path:
    suffix = {"trajectory": "", "stages": "_stages", "events": "_events"}[kind]
    return root / "classification" / f"{name}{suffix}.csv"


class TimedRecorder(SampleRecorder):
    def __init__(self, environment):
        super().__init__(environment)
        self.draws = 0

    def sample(self, arms):
        rewards = super().sample(arms)
        self.draws += len(arms)
        return rewards


class TimedTrace(list):
    """Attach an observation clock to the passive native classification trace."""

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def append(self, event):
        super().append({**event, "sample_offset": self.recorder.draws})


def recommend(means: np.ndarray, active: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(active)
    tentative = indices[_pareto_optimal_arms_mask(means[indices])] if indices.size else np.empty(0, int)
    return np.union1d(np.flatnonzero(accepted), tentative)


def progressive_rows(arms: np.ndarray, rewards: np.ndarray, events: list[dict],
                     offsets: list[int], num_arms: int, previous_sums: np.ndarray | None = None,
                     previous_counts: np.ndarray | None = None) -> list[dict]:
    """Evaluate accepted union Pareto(active), optionally pooling earlier rewards.

    Classification state always starts fresh. Supplied sufficient statistics
    are copied and never modified; current/future samples cannot enter the
    preceding stage. ``stage_samples`` counts only the current stage.
    """
    if offsets != sorted(set(offsets)) or any(t < 0 for t in offsets):
        raise ValueError("stage observation offsets must be unique, increasing and non-negative")
    if any(int(a["sample_offset"]) > int(b["sample_offset"]) for a, b in zip(events, events[1:])):
        raise ValueError("classification trace is not in observation order")
    if (previous_sums is None) != (previous_counts is None):
        raise ValueError("pooled sums and counts must be supplied together")
    if previous_sums is not None:
        if previous_sums.shape != (num_arms, rewards.shape[1]) or previous_counts.shape != (num_arms,):
            raise ValueError("pooled sufficient-statistic shapes do not match the environment")
        if np.any(previous_counts < 0) or not np.all(np.isfinite(previous_sums)):
            raise ValueError("invalid pooled sufficient statistics")
    sums = np.zeros((num_arms, rewards.shape[1])) if previous_sums is None else previous_sums.copy()
    counts = np.zeros(num_arms, int) if previous_counts is None else previous_counts.copy()
    active = np.ones(num_arms, bool)
    accepted = np.zeros(num_arms, bool)
    rejected = np.zeros(num_arms, bool)
    position = decision_index = 0
    rows = []
    for offset in offsets:
        upto = min(offset, len(arms))
        np.add.at(sums, arms[position:upto], rewards[position:upto])
        np.add.at(counts, arms[position:upto], 1)
        position = upto
        # offset=0 is the pre-run recommendation. A zero-draw stage still
        # classifies on its first positive nominal pull, not at its start.
        while offset > 0 and decision_index < len(events) and int(events[decision_index]["sample_offset"]) <= offset:
            event = events[decision_index]
            arm, action = int(event["arm"]), event["decision"]
            if not active[arm]:
                raise ValueError(f"arm {arm} classified twice")
            if action not in ("accept", "accept_final", "reject"):
                raise ValueError(f"unknown native classification: {action}")
            active[arm] = False
            accepted[arm] = action != "reject"
            rejected[arm] = action == "reject"
            decision_index += 1
        means = np.zeros_like(sums)
        sampled = counts > 0
        means[sampled] = sums[sampled] / counts[sampled, None]
        rows.append({
            "recommendation": encode_int_set(recommend(means, active, accepted)),
            "stage_samples": position,
            "unsampled_arms": int((~sampled).sum()),
            "accepted": encode_int_set(np.flatnonzero(accepted)),
            "rejected": encode_int_set(np.flatnonzero(rejected)),
            "active": encode_int_set(np.flatnonzero(active)),
        })
    return rows


def task_complete(directory: Path, digest: str) -> bool:
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    with marker.open() as handle:
        payload = json.load(handle)
    if payload["config_hash"] != digest:
        raise ValueError(f"stale classification task at {directory}")
    for name, expected in payload["files"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"corrupt completed classification task: {directory / name}")
    return True


def run_task(root: str, spec: dict, algorithm: str, repetition: int, config: dict) -> None:
    directory = task_dir(Path(root), spec["name"], algorithm, repetition)
    digest = config_hash(config)
    if task_complete(directory, digest):
        return
    name, K, B = spec["name"], spec["num_arms"], spec["budget"]
    base = EMPIRICAL_ALGORITHMS[algorithm]
    env_tag = f"{name}|{config['dist']}"
    environment = _construct_environment(ENVIRONMENTS[name][0], config["dist"])
    seed_rngs(config["master_seed"], env_tag, "env", repetition)
    environment.reset()
    if compute_instance_hash(environment) != spec["instance_hash"]:
        raise ValueError(f"{name}: instance differs from the reused experiment")
    truth = environment.get_top_arms()
    points = evaluation_points(spec, include_restarts=True)
    plan = stage_plan(K, B)
    trajectory, stages, event_rows = [], [], []
    previous_draws = 0
    pool = config["empirical_doubling"]["sample_memory"] == "all_stages"
    history_sums = np.zeros((K, spec["num_objectives"]))
    history_counts = np.zeros(K, int)
    for item in plan:
        seed_rngs(config["master_seed"], env_tag, base, repetition, "budget", item["budget"])
        recorder = TimedRecorder(environment)
        trace = TimedTrace(recorder)
        native = EGE_ALGORITHMS[base](
            T=item["budget"], K=K, D=spec["num_objectives"], environment=recorder, trace=trace
        )
        if recorder.draws != expected_ege_draws(base, item["budget"], K):
            raise ValueError(f"{name}/{algorithm}: native draw count mismatch")
        if len(trace) != K or set(event["arm"] for event in trace) != set(range(K)):
            raise ValueError(f"{name}/{algorithm}: incomplete classification trace")
        blocks = [round_robin(*block) for block in recorder.blocks]
        arms = np.concatenate([block[0] for block in blocks]) if blocks else np.empty(0, int)
        rewards = np.concatenate([block[1] for block in blocks]) if blocks else np.empty((0, spec["num_objectives"]))
        used = min(recorder.draws, item["end"] - item["start"])
        stage_points = [t for t in points if item["start"] < t <= item["end"] or (t == 0 and item["stage"] == 1)]
        states = progressive_rows(
            arms, rewards, trace, [t - item["start"] for t in stage_points], K,
            previous_sums=history_sums if pool else None,
            previous_counts=history_counts if pool else None,
        )
        for t, state in zip(stage_points, states):
            recommendation = json.loads(state["recommendation"])
            row = build_row(
                environment=name, instance_hash=spec["instance_hash"], algorithm=algorithm,
                posterior_model="", repetition=repetition, arm_pull=t, max_budget=B,
                evaluation_grid=points, rho=None, master_seed=config["master_seed"],
                recommendation=recommendation, true_pareto=truth,
                bernoulli=environment.bernoulli_metric(recommendation),
                jaccard=environment.jaccard_metric(recommendation),
                misidentification=environment.mis_id_metric(recommendation),
            )
            row.update({
                **{field: value for field, value in state.items() if field != "recommendation"},
                "base_algorithm": base, "active_stage": item["stage"] if t else 0,
                "samples_drawn": previous_draws + state["stage_samples"], "config_hash": digest,
            })
            trajectory.append(row)
        if item["completed"] and states[-1]["recommendation"] != encode_int_set(native):
            raise ValueError(f"{name}/{algorithm}/stage{item['stage']}: endpoint does not equal native EGE")
        for event in trace:
            event_rows.append({
                "environment": name, "algorithm": algorithm, "repetition": repetition,
                "stage": item["stage"], "round": event["round"], "arm": event["arm"],
                "decision": event["decision"], "sample_offset": event["sample_offset"],
                "global_time": item["start"] + max(1, event["sample_offset"]),
                "samples_at_decision": event["samples_at_decision"],
            })
        stages.append({
            "environment": name, "algorithm": algorithm, "base_algorithm": base,
            "repetition": repetition, "stage": item["stage"], "stage_budget": item["budget"],
            "stage_start": item["start"], "stage_end": item["end"],
            "completed": int(item["completed"]), "actual_draws": recorder.draws,
            "used_draws": used, "recommendation": encode_int_set(native), "config_hash": digest,
        })
        previous_draws += used
        if pool:
            np.add.at(history_sums, arms[:used], rewards[:used])
            np.add.at(history_counts, arms[:used], 1)
    if [row["arm_pull"] for row in trajectory] != points:
        raise ValueError(f"{name}/{algorithm}: trajectory grid mismatch")
    files = {}
    for filename, columns, rows in (
        ("trajectory.csv", COLUMNS, trajectory),
        ("stages.csv", STAGE_COLUMNS, stages),
        ("events.csv", EVENT_COLUMNS, event_rows),
    ):
        atomic_csv(directory / filename, columns, rows)
        files[filename] = sha256_file(directory / filename)
    atomic_json(directory / "complete.json", {"config_hash": digest, "files": files})


def run(root: Path, config: dict, names: list[str], repetitions: list[int], n_jobs: int,
        allow_dirty: bool = False) -> None:
    assert_safe_output(root, config)
    verify_sources(config)
    verify_protected(config)
    provenance = source_provenance(allow_dirty)
    specs = sorted(
        [spec for spec in environment_specs(config) if spec["name"] in names],
        key=lambda spec: -spec["budget"],
    )
    if len(specs) != len(set(names)):
        raise ValueError("unknown or duplicate environments")
    tasks = [(spec, algorithm, rep) for spec in specs for algorithm in EMPIRICAL_ALGORITHMS for rep in repetitions]
    Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(run_task)(str(root), spec, algorithm, rep, config) for spec, algorithm, rep in tasks
    )
    manifest = {"config_hash": config_hash(config), "repetitions": repetitions, "files": {}, **provenance}
    for spec in specs:
        name = spec["name"]
        counts = []
        for kind, columns in (("trajectory", COLUMNS), ("stages", STAGE_COLUMNS), ("events", EVENT_COLUMNS)):
            rows = []
            for algorithm in EMPIRICAL_ALGORITHMS:
                for rep in repetitions:
                    directory = task_dir(root, name, algorithm, rep)
                    if not task_complete(directory, config_hash(config)):
                        raise ValueError(f"incomplete task: {directory}")
                    header, shard_rows = read_csv(directory / f"{kind}.csv")
                    if header != columns:
                        raise ValueError(f"wrong schema: {directory}/{kind}.csv")
                    rows.extend(shard_rows)
            path = result_path(root, name, kind)
            atomic_csv(path, columns, rows)
            manifest["files"][str(path.relative_to(root))] = sha256_file(path)
            counts.append(len(rows))
        print(name, dict(zip(("trajectory", "stages", "events"), counts)), flush=True)
    atomic_json(root / "classification" / "manifest.json", manifest)
    verify_protected(config)


def build_inputs(root: Path, config: dict, names: list[str]) -> None:
    assert_safe_output(root, config)
    for name in names:
        source = pd.read_csv(plot_input_path(source_root(config), name), dtype=str, keep_default_na=False)
        fresh = pd.read_csv(result_path(root, name), dtype=str, keep_default_na=False)
        inputs = pd.concat(
            [source[~source.algorithm.isin(EMPIRICAL_ALGORITHMS)], fresh[PLOT_COLUMNS]], ignore_index=True
        )
        atomic_csv(plot_input_path(root, name), PLOT_COLUMNS, inputs.to_dict("records"))


def independent_replay(spec: dict, config: dict, algorithm: str, repetition: int, points: list[int]) -> dict[int, str]:
    """Scalar replay with a separate recorder, trace list and Pareto implementation."""
    from paretoset import paretoset

    base, K, D = EMPIRICAL_ALGORITHMS[algorithm], spec["num_arms"], spec["num_objectives"]
    tag = f"{spec['name']}|{config['dist']}"
    environment = _construct_environment(ENVIRONMENTS[spec["name"]][0], config["dist"])
    seed_rngs(config["master_seed"], tag, "env", repetition)
    environment.reset()
    results = {}
    pool = config["empirical_doubling"]["sample_memory"] == "all_stages"
    sums, counts = np.zeros((K, D)), np.zeros(K, int)
    for item in stage_plan(K, spec["budget"]):
        calls, sample_counts, trace = [], [], []

        class Recorder:
            def sample(self, indices):
                observations = environment.sample(indices)
                calls.append((np.asarray(indices).copy(), observations.copy()))
                sample_counts.append(len(indices))
                return observations

        class Trace:
            def append(self, event):
                trace.append((sum(sample_counts), event.copy()))

        seed_rngs(config["master_seed"], tag, base, repetition, "budget", item["budget"])
        EGE_ALGORITHMS[base](T=item["budget"], K=K, D=D, environment=Recorder(), trace=Trace())
        sequence = []
        for indices, observations in calls:
            by_arm = {}
            for arm, reward in zip(indices, observations):
                by_arm.setdefault(int(arm), []).append(reward)
            for k in range(len(next(iter(by_arm.values())))):
                sequence.extend((arm, values[k]) for arm, values in by_arm.items())
        if not pool:
            sums, counts = np.zeros((K, D)), np.zeros(K, int)
        active, accepted = set(range(K)), set()
        position = decision = 0
        for t in points:
            if not (item["start"] < t <= item["end"] or t == 0 and item["stage"] == 1):
                continue
            offset = t - item["start"]
            while position < min(offset, len(sequence)):
                arm, reward = sequence[position]
                sums[arm] += reward
                counts[arm] += 1
                position += 1
            while offset > 0 and decision < len(trace) and trace[decision][0] <= offset:
                event = trace[decision][1]
                active.remove(int(event["arm"]))
                if event["decision"] != "reject":
                    accepted.add(int(event["arm"]))
                decision += 1
            indices = sorted(active)
            means = np.array([sums[arm] / counts[arm] if counts[arm] else np.zeros(D) for arm in indices])
            tentative = set(np.array(indices)[paretoset(means, sense=["max"] * D, distinct=False)].tolist()) if indices else set()
            results[t] = encode_int_set(accepted | tentative)
    return results


def validate(root: Path, config: dict, names: list[str], repetitions: list[int],
             require_clean: bool = True, check_figures: bool = False,
             figures_dir: Path | None = None) -> dict:
    from runner.schema import recompute_metrics

    assert_safe_output(root, config)
    verify_sources(config)
    verify_protected(config)
    digest = config_hash(config)
    with (root / "classification" / "manifest.json").open() as handle:
        manifest = json.load(handle)
    if manifest["config_hash"] != digest or manifest["repetitions"] != repetitions:
        raise ValueError("classification manifest config or repetition binding mismatch")
    if require_clean and manifest["source_dirty"]:
        raise ValueError("classification data was generated from uncommitted source")
    for path, expected in manifest["files"].items():
        if sha256_file(root / path) != expected:
            raise ValueError(f"classification file checksum mismatch: {path}")
    problems = Problems()
    summaries = {}
    specs = [spec for spec in environment_specs(config) if spec["name"] in names]
    for spec in specs:
        name, K, B = spec["name"], spec["num_arms"], spec["budget"]
        points = evaluation_points(spec, include_restarts=True)
        plan = {item["stage"]: item for item in stage_plan(K, B)}
        trajectory_header, rows = read_csv(result_path(root, name))
        event_header, events = read_csv(result_path(root, name, "events"))
        stage_header, stages = read_csv(result_path(root, name, "stages"))
        if trajectory_header != COLUMNS or event_header != EVENT_COLUMNS or stage_header != STAGE_COLUMNS:
            raise ValueError(f"{name}: classification schema mismatch")
        environment = _construct_environment(ENVIRONMENTS[name][0], config["dist"])
        truth = set(environment.get_top_arms())
        old = pd.read_csv(source_root(config) / "empirical" / f"{name}_stages.csv", dtype=str, keep_default_na=False)
        old = old[old.repetition.isin([str(rep) for rep in repetitions])]
        columns = [column for column in old.columns if column != "config_hash"]
        if not old[columns].reset_index(drop=True).equals(pd.DataFrame(stages)[columns].reset_index(drop=True)):
            problems.add(f"{name}: native stages differ from prior runs")
        stage_index = {(r["algorithm"], int(r["repetition"]), int(r["stage"])): r for r in stages}
        if len(stage_index) != len(stages) or set(stage_index) != {
                (alg, rep, s) for alg in EMPIRICAL_ALGORITHMS for rep in repetitions for s in plan}:
            problems.add(f"{name}: stage coverage mismatch")
        event_index = {}
        for event in events:
            key = (event["algorithm"], int(event["repetition"]), int(event["stage"]))
            item = plan[key[2]]
            if int(event["global_time"]) != item["start"] + max(1, int(event["sample_offset"])):
                problems.add(f"{name}: mis-timed classification event")
            event_index.setdefault(key, []).append(event)
        for key, group in event_index.items():
            if len(group) != K or {int(e["arm"]) for e in group} != set(range(K)):
                problems.add(f"{name}/{key}: not exactly one classification per arm")
            actual = expected_ege_draws(EMPIRICAL_ALGORITHMS[key[0]], plan[key[2]]["budget"], K)
            if any(e["decision"] not in ("accept", "accept_final", "reject")
                   or not 0 <= int(e["sample_offset"]) <= actual
                   or int(e["samples_at_decision"]) < 0 for e in group):
                problems.add(f"{name}/{key}: invalid classification event")
            if any(int(a["sample_offset"]) > int(b["sample_offset"]) for a, b in zip(group, group[1:])):
                problems.add(f"{name}/{key}: decisions out of order")
            native = stage_index[key]
            accepted = {int(e["arm"]) for e in group if e["decision"] != "reject"}
            if accepted != set(json.loads(native["recommendation"])):
                problems.add(f"{name}/{key}: trace acceptances differ from native output")
        if set(event_index) != set(stage_index):
            problems.add(f"{name}: decision trace coverage mismatch")
        index, checked_endpoints, replayed = {}, 0, 0
        for row in rows:
            algorithm, rep, t = row["algorithm"], int(row["repetition"]), int(row["arm_pull"])
            key = (algorithm, rep, t)
            if key in index:
                problems.add(f"{name}: duplicate trajectory row")
            index[key] = row
            s = next((s for s, item in plan.items() if item["start"] < t <= item["end"]), 1)
            item = plan[s]
            offset = t - item["start"]
            visible = [event for event in event_index[(algorithm, rep, s)] if t > 0 and int(event["sample_offset"]) <= offset]
            accepted = {int(e["arm"]) for e in visible if e["decision"] != "reject"}
            rejected = {int(e["arm"]) for e in visible if e["decision"] == "reject"}
            active = set(range(K)) - accepted - rejected
            recommendation = set(json.loads(row["recommendation"]))
            if not accepted <= recommendation <= accepted | active or recommendation & rejected:
                problems.add(f"{name}/{key}: violates native accept/reject decisions")
            for field, want in (("accepted", accepted), ("rejected", rejected), ("active", active)):
                if set(json.loads(row[field])) != want:
                    problems.add(f"{name}/{key}: classification state mismatch")
            actual = expected_ege_draws(EMPIRICAL_ALGORITHMS[algorithm], item["budget"], K)
            past = sum(min(expected_ege_draws(EMPIRICAL_ALGORITHMS[algorithm], p["budget"], K),
                           p["end"] - p["start"]) for ss, p in plan.items() if ss < s)
            if int(row["stage_samples"]) != min(offset, actual) or int(row["samples_drawn"]) != past + min(offset, actual):
                problems.add(f"{name}/{key}: sample accounting mismatch")
            if int(row["active_stage"]) != (s if t else 0):
                problems.add(f"{name}/{key}: wrong current stage")
            if (row["instance_hash"] != spec["instance_hash"] or int(row["max_budget"]) != B
                    or int(row["master_seed"]) != config["master_seed"] or row["environment"] != name
                    or row["config_hash"] != digest or set(json.loads(row["true_pareto"])) != truth
                    or json.loads(row["evaluation_grid"]) != points):
                problems.add(f"{name}/{key}: trajectory binding mismatch")
            metrics = recompute_metrics(recommendation, truth, K)
            if any(abs(float(row[m]) - metrics[m]) > 1e-12 for m in metrics):
                problems.add(f"{name}/{key}: metric recomputation mismatch")
            if item["completed"] and t == item["end"]:
                checked_endpoints += 1
                if row["recommendation"] != stage_index[(algorithm, rep, s)]["recommendation"] or active:
                    problems.add(f"{name}/{key}: endpoint differs from native output")
        expected_keys = {(alg, rep, t) for alg in EMPIRICAL_ALGORITHMS for rep in repetitions for t in points}
        if set(index) != expected_keys:
            problems.add(f"{name}: trajectory coverage mismatch")
        naive = pd.read_csv(REPO_ROOT / config["doubling_results_dir"] / "doubling" / f"{name}.csv", dtype=str, keep_default_na=False)
        for r in naive.itertuples():
            rep, t = int(r.repetition), int(r.arm_pull)
            if rep in repetitions and t in {item["end"] for item in plan.values() if item["completed"]}:
                if index[(r.algorithm, rep, t)]["recommendation"] != r.recommendation:
                    problems.add(f"{name}: progressive endpoint differs from stored naive recommendation")
        for algorithm in EMPIRICAL_ALGORITHMS:
            for rep in config["validation_samples"]["empirical_recompute"]["repetitions"]:
                if rep not in repetitions:
                    continue
                for t, recommendation in independent_replay(spec, config, algorithm, rep, points).items():
                    replayed += 1
                    if index[(algorithm, rep, t)]["recommendation"] != recommendation:
                        problems.add(f"{name}/{algorithm}/rep{rep}/t={t}: scalar replay mismatch")
        original = pd.read_csv(plot_input_path(source_root(config), name), dtype=str, keep_default_na=False)
        inputs = pd.read_csv(plot_input_path(root, name), dtype=str, keep_default_na=False)
        other = lambda frame: frame[~frame.algorithm.isin(EMPIRICAL_ALGORITHMS)].reset_index(drop=True)
        if not other(inputs).equals(other(original)):
            problems.add(f"{name}: reused plot rows changed")
        if not inputs[inputs.algorithm.isin(EMPIRICAL_ALGORITHMS)].reset_index(drop=True).equals(pd.DataFrame(rows)[PLOT_COLUMNS]):
            problems.add(f"{name}: progressive plot rows changed")
        summaries[name] = {
            "endpoint_equalities": checked_endpoints, "independent_replay_points": replayed,
            "trajectory_rows": len(rows), "event_rows": len(events),
            "plot_input_sha256": sha256_file(plot_input_path(root, name)),
        }
    files = None
    if check_figures:
        directory = figures_dir or REPO_ROOT / config["figures_dir"]
        files = validate_figures(root, directory,
                                 specs, config, len(repetitions), problems, reused_algorithms=("APE_UV",))
        for set_name in config["figure_sets"]:
            original_set = set_name.split("_classification")[0]
            for metric in METRICS:
                original = pd.read_csv(
                    REPO_ROOT / "figures/publication_information_matched"
                    / f"publication_{metric}_{original_set}_plot_table.csv"
                )
                current = pd.read_csv(figure_paths(directory, set_name, metric)[1])
                def unchanged(frame):
                    return frame[
                        ~frame.algorithm.isin(EMPIRICAL_ALGORITHMS) & frame.environment.isin(names)
                    ].sort_values(["environment", "algorithm", "arm_pull"]).reset_index(drop=True)
                try:
                    pd.testing.assert_frame_equal(unchanged(current), unchanged(original), check_exact=True)
                except AssertionError:
                    problems.add(f"{set_name}/{metric}: one of the five reused curves changed")
    report = {
        "status": "pass" if not problems else "fail", "config_hash": config_hash(config),
        "environments": names, "repetitions": repetitions, "summaries": summaries,
        "figures_checked": check_figures, "figure_files": files, "problems": list(problems),
        "recommendation": config["empirical_doubling"]["recommendation"],
        "sample_memory": config["empirical_doubling"]["sample_memory"],
    }
    atomic_json(report_path(root), report)
    if problems:
        raise ValueError("classification-aware validation failed: " + "; ".join(problems[:20]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["run", "build", "validate", "plot", "all"], default="all")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--pool-samples", action="store_true",
                        help="Pool past observations for active-arm means; native decisions stay stage-local.")
    args = parser.parse_args()
    config = load_classification_config(pool_samples=args.pool_samples)
    root = output_root(config)
    names = [spec["name"] for spec in environment_specs(config)]
    repetitions = list(range(config["num_runs"]))
    if args.phase in ("all", "run"):
        run(root, config, names, repetitions, args.n_jobs)
    if args.phase in ("all", "build"):
        build_inputs(root, config, names)
    if args.phase in ("all", "validate"):
        report = validate(root, config, names, repetitions)
        print("Native endpoint equalities:", sum(s["endpoint_equalities"] for s in report["summaries"].values()))
    if args.phase in ("all", "plot"):
        plot.render(root, config=config)
        validate(root, config, names, repetitions, check_figures=True)
        print(f"Six classification-aware figures created in {config['figures_dir']}")
    verify_protected(config)
    print(root)


if __name__ == "__main__":
    main()
