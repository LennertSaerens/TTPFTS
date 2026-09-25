# MOMAB final experiment preregistration

Status: preregistered artifacts prepared; HPC launch is intentionally pending user review.

## Frozen choices

- Source of truth: `experiments/final_config.json` (schema `momab-final-experiment-v1`). Its canonical hash can be printed with `PYTHONPATH=. ./.venv/bin/python -m experiments.final_config_loader --hash`.
- Environments: EgeExp1--EgeExp8, Gaussian rewards only.
- Budgets: EgeExp1 15218, EgeExp2 2544, EgeExp3 685036, EgeExp4 15977, EgeExp5 197373, EgeExp6 51966, EgeExp7 7000, EgeExp8 237958.
- Budget rule: `round(H(nu))` from `environments/reports/hardness_summary.csv`. This was chosen autonomously as the transparent plan default because the user was unavailable; it remains pending explicit user confirmation before launch.
- Frozen instances: EgeExp1, EgeExp4, and EgeExp5 use `environments/instances/*.csv` with hashes recorded in `manifest.json`, `CHECKSUMS.sha256`, and the final config. Other EgeExp environments are deterministic code-defined instances; all runtime instance hashes are recorded in the config.
- Algorithms: EGE_SH, EGE_SR, Uniform, APE, TTPFTS_KNOWN_VAR, TTPFTS_UNKNOWN_VAR.
- Repetitions and seed: 100 repetitions, master seed 0.
- TTPFTS rho: 0.5.
- Reward noise: Gaussian variance 0.25 (std 0.5).
- Evaluation grid: normalized endpoint-inclusive grid with `--eval_points 100`, i.e. 101 normalized points k/100. Since all B values exceed 100, every environment/algorithm has 101 grid rows per repetition.
- Expected rows: 10,100 per environment/algorithm; 60,600 per environment; 484,800 total.

## Why these choices

The configuration aligns budgets with measured environment hardness instead of the obsolete flat 5000-pull budget, keeps all environments on the same normalized x-axis, and separates known-variance and unknown-variance TTPFTS while retaining APE, Uniform, and both EGE baselines. The frozen random environments and config hash are intended to make the scientific run auditable and reproducible.

## Launch artifacts

- Config loader: `experiments/final_config_loader.py`.
- Runtime projections: `experiments/reports/runtime_feasibility.md`, `runtime_projection.csv`, `ege_timing_measurements.csv`, and `anytime_per_pull_costs.csv`.
- SLURM script: `hpc/run_preregistered_final.slurm`.

## Pending user confirmation

1. Confirm or override the autonomous `round(H)` budgets, especially the large EgeExp3/EgeExp5/EgeExp8 budgets.
2. Confirm that the 101-point normalized grid is acceptable for final figures and tables.
3. Confirm launch timing/resource policy. No HPC job has been submitted by this preregistration step.
