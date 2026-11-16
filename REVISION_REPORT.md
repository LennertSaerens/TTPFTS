# Revision report: EgeExp MOMAB Pareto-set-identification experiments

Final scientific handoff for the revised eight-environment benchmark suite.
Everything below is reproducible from the committed source tree and the
preregistered configuration; no claim rests on the superseded 5,000-budget
results, an uncommitted tree, an undocumented default, or a failed check.

- **Configuration hash:** `ff29cc456351571808480961d67fd334af454042e81f9094de3c937a7d845d47`
- **Results directory:** `results_preregistered_final_ff29cc456351`
- **Dataset:** 484,800 rows = 8 environments x 6 algorithms x 100 repetitions
  x 101 evaluation points
- **Validation:** `validate_results.py` passes on Linux (HPC) and macOS

## 1. APE: reference equivalence and the fixed-budget adaptation

The Python APE implementation was aligned with the supplied C++ reference
(`8606_.../bandit_lib_c++/src/cpp/`). Equivalence is enforced by
`test_ape_equivalence.py`, which compiles the *unmodified* vendor `utils.cxx`
in a temporary directory and drives both implementations from a shared reward
tape, so the comparison isolates algorithm semantics from RNG differences.
Agreement is required exactly for pull counts, empirical means, empirical
Pareto/OPT masks, `b_t`, `c_t`, the selected arm, and tie resolution.

`sigma` is now a **required keyword-only argument** rather than an unsafe
default, and the runner derives it from the environment's validated reward
standard deviation (`max(environment.stds)` = 0.5).

**Performance.** APE was vectorised: `_pairwise_dom`/`_pairwise_non_dom`,
`_pareto_optimal_arms_mask`, `_all_h`/`_all_g`, the stopping diagnostics and
the `b_t`/`c_t` selection replaced O(K^2) Python loops that issued ~40,000 tiny
NumPy calls per decision. At K=200 this is a **146x speedup**
(525.8 ms -> 3.61 ms per pull). The rewrite is provably safe because every
operation is elementwise arithmetic plus min/max reductions, with no
reassociation of floating-point sums; it was verified **bitwise** against the
scalar formulation on 300 random states and the C++ oracle gate still passes
unchanged.

The production benchmark is **fixed-budget**: the validated sampling rule runs
to the per-environment maximum and the empirical Pareto set is recommended at
each evaluation budget. The original confidence-based stopping rule is
validated separately but does not stop the benchmark.

## 2. Reward law and both posteriors

Rewards are Gaussian with **covariance 0.25 I_D** (standard deviation 0.5),
matching the manuscript claim. Independence across objectives and homogeneity
across arms are asserted for all eight environments, and verified
statistically against preregistered Monte Carlo bounds.

- **Known variance:** flat prior on the mean, known observation variance 0.25,
  giving `mu | x ~ N(xbar, 0.25/n)` for n >= 1.
- **Unknown variance:** the location-scale reference prior
  `p(mu, sigma^2) ~ 1/sigma^2`, giving, with `S = sum_i (x_i - xbar)^2`,
  `mu | x ~ t_{n-1}(xbar, sqrt(S/[n(n-1)]))`, using a numerically stable
  Welford update. Ad hoc degree-of-freedom clamping and the dimensionally
  inconsistent scale calculation were removed, and `n < 2` is treated
  explicitly as an unavailable posterior rather than a degenerate sample.

Both variants share the same explicit **4-pull-per-arm warm-up** (counted
inside the budget) so the comparison isolates the posterior model, and both
use rho = 0.5. Labels are `TTPFTS_KNOWN_VAR` and `TTPFTS_UNKNOWN_VAR`.

### Follow-up audit: Student-t scale versus posterior standard deviation

The reported missing square root is present in the historical `TPosterior`
before commit `2f3e8f5`: it sampled with `S / sqrt(n * df)` even though `S`
stored squared deviations. That commit corrected the scale and the online
sum-of-squares update. The current `bandits/Posteriors.py` is byte-identical
to the version in both `2f3e8f5` and `5bc0e3f` (the source used for the
preregistered benchmark). No second square-root change is needed in the
current TTPFTS sampler.

For `nu = n + 2*alpha - 1` and centered sum of squares `S`, the contracts are:

- sampling scale: `sqrt(S / (n * nu))`;
- posterior standard deviation: `sqrt(S / (n * (nu - 2)))` when `nu > 2`,
  or infinite when `0 < nu <= 2` and `S > 0`.

Scale and standard deviation are different Student-t parameters; the sampler
must not substitute one for the other. The UQ replay reconstructs the actual
posterior standard deviation and uses its square as the covariance diagonal.
APE-UV's frequentist Student-t confidence bounds also already take the square
root of the online sample variance. Notebook code cells contain no separate
Student-t sampler.

The audit did find the analogous live error in the **legacy**
`NormalPTSBandit` and `NormalLSTSBandit` in `bandits/ThompsonSampling.py`.
An inverse-gamma draw is an observation **variance**, so the conditional
normal sampler must use `sqrt(variance / precision)`, not
`variance / precision`. Their conjugate `beta` updates also used the
already-updated mean, understating the scatter; the update now uses the
previous mean and precision. These two classes are not registered in
`momab_main.py` and were not used for the current EGE/Uniform/APE/TTPFTS
publication experiments. Historical results produced with these legacy
classes require recomputation before scientific reuse; preserved archives
are not rewritten.

`validate_posteriors.py` now independently checks the scale/standard-deviation
relationship, standardized sampling transform, logged summaries and UQ
reconstruction for `alpha=-0.5,0,0.5,2`, plus linear response to changing
reward units. It also covers all three Normal-Inverse-Gamma sampling paths
and the legacy conjugate statistics against batch formulas. The two legacy
regressions failed before the fixes; the current Student-t checks already
passed. The existing 200,000-draw Student-t goodness-of-fit gate remains in
place. Current corrected TTPFTS trajectories are not changed or rerun by
this audit; pre-correction archived trajectories remain superseded.

## 3. Environments, instances and the EgeExp3 repair

EgeExp1, EgeExp4 and EgeExp5 are frozen to committed, checksummed arm-mean
matrices; `reset()` resets reward state only and never redraws means.

### EgeExp3 had infinite hardness, and why

Arms lie on a circle, `(cx + r cos t, cy + r sin t)`. Two arms therefore tie in
objective 1 iff `t_i = -t_j (mod 360)` and in objective 2 iff
`t_i = 180 - t_j (mod 360)`: **ties are exactly mirror reflections about the
axes**. A dominated arm at angle phi is *strictly* dominated iff
`180 - t_max^P < phi < 360 - t_min^P`, i.e. `105 < phi < 345` for the Pareto
arc `[15, 75]`.

The original code sampled the **closed endpoint 345 degrees**, which is exactly
the reflection of Pareto arm 0 at 15 degrees, so `cos(345) == cos(15)`
bit-for-bit. Arm 199 was then only *weakly* dominated
(`mu_0 - mu_199 = (0, 0.1811733)`), which zeroed two gaps and gave
`H = sum_i 1/Delta_i^2 = infinity`. **This was a boundary-sampling error, not a
deep property of the construction.**

The repair is canonical rather than tuned: the admissible interval is derived
from the Pareto arc and inset by **half the Pareto angular step**
(`p/2`, `p = 60/19 ~ 3.158 deg`). Reflections of the Pareto arms form an
arithmetic progression of step `p` from each critical boundary, so insetting by
`p/2` places the endpoint exactly midway between consecutive reflections: the
maximal distance from any reflection at the design's own angular resolution.
The new dominated arc is `[120, 343.4211]`. Centre, radius, the 20/180 split
and the Pareto arc are unchanged. Result: **H = 685,035.73**, zero ties.

`environments/hardness.py` gained `objective_value_ties`,
`minimum_objective_separation` and `assert_non_degenerate` so this bug class
cannot recur. Ties are fatal **only when they actually zero a gap**, because
EgeExp7 has 18 legitimate lattice ties yet finite hardness (7,000).

### EgeExp3 is also frozen, for platform independence

The circle construction is deterministic, but `cos`/`sin` differ by up to
**1 ULP** between platform `libm` implementations: 7 of 400 coordinates
differed by 1.1e-16 between macOS/arm64 and Linux/x86-64. That is ~13 orders of
magnitude below the smallest decision gap and cannot change a Pareto or
algorithmic decision, but it changes the byte-exact instance hash, which made
validation pass on Linux and fail on macOS. EgeExp3 is therefore frozen from
the Linux values that generated the final dataset (`%.17g` round-trips
bit-exactly), with the analytic derivation retained in source as documentation
and as the regeneration path.

### Hardness (all finite, two independent derivations agree exactly)

| Environment | K | D | H(nu) | Budget B = round(H) |
|---|---:|---:|---:|---:|
| EgeExp1 | 60 | 2 | 15,217.76 | 15,218 |
| EgeExp2 | 10 | 2 | 2,543.76 | 2,544 |
| EgeExp3 | 200 | 2 | 685,035.73 | 685,036 |
| EgeExp4 | 50 | 10 | 15,977.30 | 15,977 |
| EgeExp5 | 20 | 2 | 197,372.62 | 197,373 |
| EgeExp6 | 10 | 2 | 51,966.41 | 51,966 |
| EgeExp7 | 22 | 2 | 7,000.00 | 7,000 |
| EgeExp8 | 5 | 2 | 237,957.54 | 237,958 |

Gaps were computed twice -- by the direct case-split and by the equivalent
lemma in `TTPFTS MOMAB PSI.md` -- and agree to exactly 0.0 for every arm in
every environment. No gap was ever floored to obtain a finite answer.

> **Decision requiring confirmation.** `B = round(H(nu))` and the 101-point
> normalized evaluation grid were selected autonomously at the Gate 3/4 pauses
> because the user was unavailable. This is the transparent rounding the plan
> specifies, but EgeExp3 (685,036) and EgeExp8 (237,958) are large and remain
> **pending explicit confirmation**.

## 4. Runner correctness

- Budget `B` means exactly `B` pulls: the prior is recorded at budget 0 before
  pulling, then pulls/updates run for `t = 1..B`.
- One explicit evaluation-budget generator for all algorithms, always including
  0 and the exact maximum.
- EGE results are seeded by
  `(master seed, environment, algorithm, repetition, evaluated budget)`, so a
  result at a shared budget is identical whether produced on a dense or sparse
  grid. Anytime trajectories are seeded by
  `(master seed, environment, algorithm, repetition)` so logging stride cannot
  alter pulls.
- Atomic per-repetition shards replace shared-CSV append/resume. A repetition
  counts as complete only when its shard covers *exactly* the expected grid.
- Self-describing tidy schema storing the recommended arm set, which is what
  allows all three metrics to be recomputed independently.

## 5. Verification gates

`run_all_gates.py` runs every gate; all pass on macOS and on the HPC.

| Gate | Purpose |
|---|---|
| `ape` | C++ oracle sampling equivalence |
| `posteriors` | analytic, simulation, reset and calibration checks |
| `reward-report` | covariance 0.25 I_D configuration and samples |
| `environment` | instance definitions, hardness, degeneracy guards |
| `runner-grid-schema` | budget semantics, grid generation, schema |
| `runner-shards` | atomic shard write/resume integrity |
| `runner-pipeline` | end-to-end invariance |
| `legacy-fixes` | the five original fixes (replaces `verify_fixes.py`) |

`validate_results.py` independently recomputes all three metrics from the
stored recommendation sets and frozen Pareto sets -- a genuine cross-check,
not a re-import. `test_validate_results.py` proves it *catches* eight distinct
corruption classes, including the manifest-clobber case described below.

## 6. Execution and job accounting

Cluster: VUB AI-lab, SLURM 23.11.4, nodes 102/103 (AMD EPYC 9634, Zen 4).
Environment: Python 3.11.5 (EasyBuild), numpy 1.26.4, pandas 2.2.2,
scipy 1.13.0, numba 0.62.1. Dependencies are validated fail-fast; a scientific
job never installs packages.

| Job | Purpose | State | Elapsed |
|---|---|---|---|
| `50463_1` | pilot (EgeExp2, 4 reps) | COMPLETED | 00:00:13 |
| `50464_[0,1,3,4,5,6,7]` | final, 7 environments | COMPLETED | 00:00:38 - 00:28:12 |
| `50464_2` | final, EgeExp3 | CANCELLED (deliberate) | 06:34:34 |
| `50474_2` | EgeExp3 rerun, exclusive node | COMPLETED | 03:51:38 |
| `50475_[0-7]` | manifest regeneration via resume | COMPLETED | ~00:00:07 |

### Deviation 1: measured slowdown was 2.0x, not the assumed 1.3x

Direct measurement on an idle `node103` gave APE on EgeExp3 at
**9033.7 us/pull** versus 4492.8 locally -- a **2.01x** per-core factor, not the
assumed 1.3x. With 25 contending workers and SLURM allocating hyperthread
siblings (52 logical CPUs for `--cpus-per-task=25`), task `50464_2` would have
needed roughly 22 hours and been killed at its 12 h limit. It was therefore
cancelled deliberately and resubmitted as `50474_2` with `--cpus-per-task=50`,
`--exclusive` and `--time=24:00:00`, finishing in 3h52m.

**No data was lost or recomputed inconsistently.** Atomic shards made the
resource correction safe: the 300 already-complete EGE_SR/EGE_SH/Uniform shards
were reused unchanged and only APE and the two TTPFTS variants were recomputed.
Because seeding is by `(master seed, environment, algorithm, repetition)`,
resumed results are identical to those of an uninterrupted run -- confirmed by
job `50475`, where every repetition was correctly skipped as already complete.

### Deviation 2: a manifest clobber that made validation under-report

With one array task per environment sharing a run directory, **every task wrote
`run/manifest.json` and the last finisher clobbered the rest**. The validator
then loaded a manifest naming a single environment and reported **PASS while
checking 60,600 of 484,800 rows**. This was caught by cross-checking the
reported row count against the expected total.

Fixed at the root: manifests are now written per environment under
`run/manifests/`, the validator prefers them and unions their per-environment
sections, and **any result file not described by a manifest is a failure**, so
silent under-validation cannot recur. A regression test reproduces the clobber.

### Other robustness fixes prompted by execution

- The SLURM script defaulted `REPO_DIR` to a stale checkout, surfacing only as
  a confusing `ModuleNotFoundError`. It now validates expected files up front.
- The synced tree has no `.git`, so the job logged `Commit: n/a`.
  `hpc/sync_to_hpc.sh` refuses to sync a dirty worktree and writes
  `SOURCE_PROVENANCE.txt`; the job now **fails** rather than running from an
  unknown or dirty revision.
- `validate_results.py` hardcoded `.venv/bin/python`, so its invariance check
  failed on HPC; it now defaults to the running interpreter.

### Known limitation

Shards are written only after **all** repetitions of an algorithm complete
(`Parallel(...)` collects results first). A job killed mid-algorithm loses that
algorithm's work, though completed algorithms survive. This did not affect the
final dataset but is worth changing to per-repetition writes.

## 7. Result integrity

All checks pass over the full 484,800 rows:

```
[PASS] manifest agreement -- 8 consistent manifest(s) covering 8 environment(s)
[PASS] load merged data -- 484800 rows
[PASS] completeness and uniqueness -- 484800 expected rows present
[PASS] row values and independent metrics -- 484800 rows cross-checked
[PASS] instance agreement -- one hash per environment and matches frozen instance
[PASS] EGE grid invariance -- shared rows match
VALIDATION PASSED
```

| Environment | Budget | Rows | SHA-256 (first 16) |
|---|---:|---:|---|
| EgeExp1 | 15,218 | 60,600 | `ba841805d8369b14` |
| EgeExp2 | 2,544 | 60,600 | `1d0ad3f47b588a8b` |
| EgeExp3 | 685,036 | 60,600 | `563f62e9aa6b35f8` |
| EgeExp4 | 15,977 | 60,600 | `5b2086c5aee9bcc8` |
| EgeExp5 | 197,373 | 60,600 | `91343e4af00c80ef` |
| EgeExp6 | 51,966 | 60,600 | `e1231b0025d5fefe` |
| EgeExp7 | 7,000 | 60,600 | `0d337a29366f3741` |
| EgeExp8 | 237,958 | 60,600 | `ede3d6e60a58e8e4` |

## 8. Figures and headline result

`figures/publication/` contains the three figures and a machine-readable table
of every plotted mean, interval and sample count. Each subplot's x-values are
`arm_pull / environment_max_budget` with the endpoint asserted to be exactly 1;
there is a single centered **Normalized Budget** x-label, one **Jaccard
Metric**-style y-label, the 2x4 environment grid, a framed six-entry legend
above the grid, and lightened grid lines. Only actual recorded budgets are
plotted; sparse EGE evaluations are never interpolated into observations.
Uncertainty is Wilson intervals for the binary exact-identification metric and
bounded repetition-level intervals for Jaccard and misidentification.

Endpoint Jaccard, averaged over the eight environments:

| Algorithm | Endpoint Jaccard |
|---|---:|
| TTPFTS_UNKNOWN_VAR | 0.9346 |
| TTPFTS_KNOWN_VAR | 0.9330 |
| APE | 0.9274 |
| EGE_SR | 0.9234 |
| EGE_SH | 0.9011 |
| Uniform | 0.8445 |

Both TTPFTS variants finish ahead of APE and the EGE algorithms, with Uniform
clearly last; the ordering is consistent across metrics, and the separation is
most visible on EgeExp3, the hardest instance.

## 9. Manuscript-ready methods text

> We evaluate six algorithms -- EGE-SH, EGE-SR, Uniform, APE, and Thompson
> sampling with top-two Pareto-front selection (TTPFTS) under known and unknown
> reward variance -- on the eight EgeExp benchmark environments. Rewards are
> Gaussian with covariance 0.25 I_D. TTPFTS uses rho = 0.5 and a warm-up of
> four pulls per arm, counted within the budget; the known-variance variant
> places a flat prior on the mean with known variance 0.25, giving the
> posterior N(xbar, 0.25/n), while the unknown-variance variant uses the
> location-scale reference prior p(mu, sigma^2) ~ 1/sigma^2, giving
> t_{n-1}(xbar, sqrt(S/[n(n-1)])) with S the centred sum of squares. APE
> follows the original sampling rule, verified against the reference
> implementation on shared reward tapes, and is run in fixed-budget mode,
> recommending the empirical Pareto set at each evaluation budget.
>
> Each environment's budget is the rounded problem complexity B = round(H(nu)),
> with H(nu) = sum_i 1/Delta_i^2 computed from the frozen arm means and
> cross-checked by two independent derivations. Benchmark instances are frozen
> and checksummed so that every algorithm and repetition sees an identical
> instance. Each configuration is run for 100 repetitions with deterministic
> seed derivation from master seed 0, and metrics are recorded on a common
> 101-point normalized budget grid. We report exact identification (a Bernoulli
> success indicator), the Jaccard index between the recommended and true Pareto
> sets, and misidentification, with Wilson intervals for the binary metric and
> repetition-level intervals otherwise.

## 10. Reproducing this work

```bash
python run_all_gates.py                      # every verification gate
./hpc/sync_to_hpc.sh                         # refuses a dirty worktree
REPO_DIR=$HOME/DENV-Bandits-final \
    sbatch hpc/run_preregistered_final.slurm # 8-task array
python validate_results.py results_preregistered_final_ff29cc456351
python plot_results_local.py --figures_dir figures/publication
```

## 11. Open items

1. **Confirm the budget rule** `B = round(H(nu))`, especially EgeExp3 (685,036)
   and EgeExp8 (237,958), and the 101-point evaluation grid. Both were chosen
   autonomously at the gate pauses.
2. **Confirm the 4-pull TTPFTS warm-up** is final.
3. **Independent source cross-check.** The EGE environment definitions were
   verified against `TTPFTS MOMAB PSI.md` and the supplied supplementary
   material; no second independent published source was available locally.
4. **Shard granularity.** Consider writing shards per repetition rather than
   per algorithm, so an interrupted algorithm does not lose completed
   repetitions.
