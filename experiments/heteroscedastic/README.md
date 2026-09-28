# Paired heteroscedastic random-instance MOMAB experiment

This experiment studies **within-instance** reward-noise heterogeneity without
changing an instance's mean matrix, Pareto set, or mean-based hardness.

For each candidate mean matrix, it freezes two Gaussian conditions:

1. `heterogeneous`: every arm-objective standard deviation is independently
   drawn from `{0.1, 0.2, 0.3, 0.4, 0.5}`;
2. `homoscedastic_rms`: every coordinate has the RMS standard deviation of the
   paired heterogeneous matrix.

The paired difference in an algorithm's metric is therefore attributable to
the distribution of noise at fixed average variance, mean instance, and
evaluation budget.

## Current status

`candidate_config.json` and `instances/` define and checksum 100 candidate
pairs. They are **not** an executed performance experiment. Candidate
hardness has a heavy tail, so no full-budget trajectory may be launched until
an explicit hardness admissibility rule is approved and recorded in a separate
pilot configuration.

Generate or independently reproduce candidate artifacts:

```bash
python -m experiments.heteroscedastic.generate_candidates
python test_heteroscedastic_candidates.py
python -m experiments.heteroscedastic.runtime_probe
```

The candidate and runtime reports in `reports/` are feasibility evidence only.
