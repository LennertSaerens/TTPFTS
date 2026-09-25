# Config-loader dry run

Command was generated with `experiments.final_config_loader.build_momab_args` from `experiments/final_config.json` (config hash `ff29cc456351571808480961d67fd334af454042e81f9094de3c937a7d845d47`).

Dry-run command:

```bash
PYTHONPATH=. ./.venv/bin/python momab_main.py --algorithms EGE_SH,EGE_SR,Uniform,APE,TTPFTS_KNOWN_VAR,TTPFTS_UNKNOWN_VAR --environments EgeExp2 --budgets 20 --num_runs 2 --eval_points 100 --dist gaussian --seed 0 --n_jobs 1 --results_dir experiments/reports/dry_run_results --run_dir experiments/reports/dry_run_results/run
```

Result: completed in 3.93 s, wrote 252 rows (`6 algorithms * 2 reps * 21 unique budget points`) with algorithms `APE`, `EGE_SH`, `EGE_SR`, `TTPFTS_KNOWN_VAR`, `TTPFTS_UNKNOWN_VAR`, and `Uniform`; arm-pull range was 0--20.

The temporary dry-run output directory was removed after validation.
