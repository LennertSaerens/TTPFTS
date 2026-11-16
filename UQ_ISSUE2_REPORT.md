# Uncertainty Quantification Variants Report (TTPFTSAAMAS Issue #2)

## 1. Problem Diagnosis: Structural Failure Modes of Eq. (10)

The baseline uncertainty quantification metric in Section 6 / Eq. (10) of the manuscript defines uncertainty between the empirical top-two Pareto fronts $\hat{\mathcal{F}}_1$ and $\hat{\mathcal{F}}_2$ under independent Gaussian coordinate posteriors $p_a(\mathbf{x}) = \prod_{c=1}^D \mathcal{N}(x_c; \hat{\mu}_{a,c}, \sigma_{a,c}^2)$ as:

$$\mathcal{U}_\beta(\hat{\mathcal{F}}_1, \hat{\mathcal{F}}_2) = \frac{1}{|\hat{\mathcal{F}}_1||\hat{\mathcal{F}}_2|} \sum_{a_i \in \hat{\mathcal{F}}_1} \sum_{a_j \in \hat{\mathcal{F}}_2} \prod_{c=1}^D \beta_c(a_i, a_j),$$

where $\beta_c(u, v) = \sqrt{\frac{2\sigma_{u,c}\sigma_{v,c}}{\sigma_{u,c}^2 + \sigma_{v,c}^2}} \exp\left(-\frac{(\hat{\mu}_{u,c}-\hat{\mu}_{v,c})^2}{4(\sigma_{u,c}^2 + \sigma_{v,c}^2)}\right) \in (0, 1]$ is the 1D Bhattacharyya coefficient.

Theoretical inspection and controlled synthetic stress tests establish five distinct failure modes:

1. **Coordinate-tie masking (product collapse across objectives):** Multiplying $\prod_{c=1}^D \beta_c(a_i, a_j)$ treats joint 2D density overlap as a proxy for dominance ambiguity. Yet Pareto dominance between $a_i$ and $a_j$ reverses if $a_j$ overtakes $a_i$ in *any single* coordinate. In a two-arm setting with $\hat{\mu}_1 = (10\sigma, 0.01\sigma)$ and $\hat{\mu}_2 = (0, 0)$ (matched posterior $\sigma$), coordinate 1 gives $\beta_1 \approx 3.73 \times 10^{-6}$ while coordinate 2 has $\approx 0.497$ flip probability. Eq. (10) yields $\mathcal{U}_\beta = 3.73 \times 10^{-6}$ even though the exact posterior Pareto-set disagreement probability is $q_t = \Pr_{\Theta \sim \Pi_t}[\operatorname{Pareto}(\Theta) \neq \hat{\mathcal{F}}_1] = 0.49718$.
2. **Empty second front ($\hat{\mathcal{F}}_2 = \emptyset$) and within-$\hat{\mathcal{F}}_1$ ambiguity:** When all empirical means are mutually non-dominated ($\hat{\mathcal{F}}_1 = [K], \hat{\mathcal{F}}_2 = \emptyset$, e.g. `EgeExp6` where `empty_second_frac = 0.3298` under `TTPFTS_KNOWN_VAR` and `0.2924` under `TTPFTS_UNKNOWN_VAR`, or 2-arm comparisons with opposite-sign objective gaps), Eq. (10) is undefined (`0/0`) and falls back to $0.0$/unavailable. On `(10\sigma, -0.01\sigma)` vs `(0, 0)`, Eq. (10) reports `0.0` (unavailable) despite $q_t = 0.49718$; on diffuse equal-mean posteriors (`empty_F2_diffuse`, $\sigma = 2.0$), Eq. (10) reports `0.0` despite $q_t = 0.49665$. In $50,000$ random two-arm uniform instances at $n=10$ pulls, $24,897$ instances have $\hat{\mathcal{F}}_2 = \emptyset$ and $6,680$ of those return the wrong Pareto set.
3. **Quadratic front-size dilution:** Dividing by $|\hat{\mathcal{F}}_1||\hat{\mathcal{F}}_2|$ averages localized uncertain boundary pairs against all $O(|\hat{\mathcal{F}}_1||\hat{\mathcal{F}}_2|)$ well-separated pairs across distant parts of the Pareto front. Keeping one uncertain boundary pair fixed at $q_t = 0.497$ while adding $M \in \{2, 10, 50\}$ well-separated pairs drops $\mathcal{U}_\beta$ from $3.73\times 10^{-6}$ to $1.49\times 10^{-7}$ ($M=10$) and $5.96\times 10^{-9}$ ($M=50$). On `EgeExp3` ($K=200, |\mathcal{F}^*|=20$), endpoint mean $\mathcal{U}_\beta = 0.0019$ even though $53\%$ of endpoints fail exact Pareto recovery.
4. **Arms dominated below $\hat{\mathcal{F}}_2$:** Eq. (10) ignores arms in fronts $\ge 3$. An under-sampled arm slightly dominated in front 3 (`undersampled_below_F2_moderate`) gives $\mathcal{U}_\beta = 1.23 \times 10^{-4}$ while true posterior disagreement is $q_t = 0.38125$.
5. **Consequence on Environment 8 and stopping:** On `EgeExp8` ($K=5, |\mathcal{F}^*|=1$, vanishing gaps), TTPFTS frequently places a suboptimal arm with an inferior far-away objective in $\hat{\mathcal{F}}_1$ or drops noisy near-tied challengers into $\hat{\mathcal{F}}_2$, producing a **wrong-sign positive** trajectory median correlation for $\mathcal{U}_\beta$: $+0.265$ vs. Jaccard and $+0.436$ vs. Bernoulli ($-0.265$ vs. misidentification). Moreover, stopping rules $\mathcal{U}_\beta \le 0.01$ or $\le 0.0001$ stop prematurely with $98.55\%$ and $100.0\%$ (`37/37`) error on `EgeExp8`, and $50.83\%$ / $38.14\%$ error on random $K=10, d=2, \sigma=0.25$ adaptive runs.

---

## 2. Two New Uncertainty Quantification Metrics

Let $\hat{\boldsymbol{\mu}}_u \in \mathbb{R}^D$ and $\boldsymbol{\sigma}_u > 0$ be the posterior mean and coordinate standard deviation vectors of arm $u \in [K]$ under the factorized Gaussian approximation $\Theta_{u,c} \sim \mathcal{N}(\hat{\mu}_{u,c}, \sigma_{u,c}^2)$, and let $\hat{\mathcal{F}}_1$ be the empirical non-dominated front sorted weakly (`paretoset(..., distinct=False)`). Both variants evaluate to $0.0$ when $K \le 1$, return `NaN` on non-finite/non-positive inputs, handle $\hat{\mathcal{F}}_2 = \emptyset$ seamlessly, and run in $O(K^2 D)$ time without posterior sampling.

### 2.1 Variant 1: Directional Chernoff-Bhattacharyya Front-Boundary Overlap ($\mathcal{U}_{\mathrm{V1}} / \mathcal{U}_{\mathrm{BC}}^{\mathrm{dir}}$)

Variant 1 stays as close as possible to Eq. (10) by using **only** the 1D pairwise Bhattacharyya coefficients $\beta_c(u, v)$, converted to the symmetric Chernoff coordinate flip bound:
$$q_c(u, v) = \tfrac{1}{2} \beta_c(u, v)^2 \in (0, 0.5].$$
It repairs all structural aggregation flaws of Eq. (10) via two event rules and dilution-free noisy-OR aggregation:

1. **Within-$\hat{\mathcal{F}}_1$ collapse risk:** For each unordered pair $\{a, b\} \subset \hat{\mathcal{F}}_1$, $a$ comes to dominate $b$ only if $a$ holds its strictly leading coordinates ($1 - q_c$) and flips its deficit/tied coordinates ($q_c$):
   $$p^{\mathrm{BC}}_{a \succ b} = \prod_{c:\,\hat{\mu}_{a,c} \le \hat{\mu}_{b,c}} q_c(a, b) \prod_{c:\,\hat{\mu}_{a,c} > \hat{\mu}_{b,c}} \bigl(1 - q_c(a, b)\bigr), \qquad u_{\mathrm{in}}(a,b) = \min\!\bigl(1,\, p^{\mathrm{BC}}_{a\succ b} + p^{\mathrm{BC}}_{b\succ a}\bigr).$$
2. **Outside-$\hat{\mathcal{F}}_1$ challenger escape risk:** Arm $j \notin \hat{\mathcal{F}}_1$ threatens to enter $\operatorname{Pareto}(\Theta)$ if it escapes every empirical dominator $i \in D_1(j) = \{i \in \hat{\mathcal{F}}_1 : i \succ_{\hat{\boldsymbol{\mu}}} j\}$. Overcoming coordinate-tie masking, a challenger escapes a single dominator $i$ if it wins in *at least one* coordinate:
   $$u_{\mathrm{esc}}(i, j) = 1 - \prod_{c=1}^D \bigl(1 - q_c(i, j)\bigr), \qquad u_{\mathrm{out}}(j) = \min_{i \in D_1(j)} u_{\mathrm{esc}}(i, j).$$
3. **Dilution-free aggregation:**
   $$\mathcal{U}_{\mathrm{V1}} = 1 - \prod_{\{a,b\} \subset \hat{\mathcal{F}}_1} \bigl(1 - u_{\mathrm{in}}(a, b)\bigr) \prod_{j \notin \hat{\mathcal{F}}_1} \bigl(1 - u_{\mathrm{out}}(j)\bigr) \in [0, 1].$$

#### Remaining Counterexample for Pure Density Overlap
Analytical stress testing (`undersampled_below_F2_extreme_counterexample`) exposes a fundamental ceiling shared by *all* density-overlap coefficients (including $\mathcal{U}_{\mathrm{V1}}$) when posterior variances differ sharply: for a concentrated dominator $\sigma_0 = 0.0005$ and an under-sampled challenger $\sigma_2 = 0.5$ separated by $0.25$ in both coordinates, $\sqrt{2\sigma_0\sigma_2/(\sigma_0^2+\sigma_2^2)} \approx 0.0447$ forces $\beta_c \to 0.042$ so $\mathcal{U}_{\mathrm{V1}} = 0.00199$, whereas exact dominance-flip risk is $q_t = 0.7089$. Fundamentally, $\beta_c$ measures whether two posteriors have the same *scale* as well as overlapping location, whereas Pareto ordering depends only on the standardized mean difference $(\hat{\mu}_{u,c}-\hat{\mu}_{v,c})/\sqrt{\sigma_{u,c}^2+\sigma_{v,c}^2}$. This observation directly motivates Variant 2.

### 2.2 Variant 2: Decision-Based Analytic Dominance Risk ($\mathcal{U}_{\mathrm{V2}} / \mathcal{U}_{\mathrm{risk}}$)

Variant 2 drops the Bhattacharyya functional and directly approximates the **posterior Pareto-set disagreement probability**
$$q_t = \Pr_{\Theta \sim \Pi_t}\bigl[\operatorname{Pareto}(\Theta) \neq \hat{\mathcal{F}}_1\bigr]$$
analytically in $O(K^2 D)$ time via three ingredients:

1. **Exact Gaussian pairwise dominance probabilities:** Replacing density overlap with difference-variance standardization removes the determinant mismatch ceiling completely:
   $$P_{u \succ v} = \prod_{c=1}^D \Phi\!\left(\frac{\hat{\mu}_{u,c} - \hat{\mu}_{v,c}}{\sqrt{\sigma_{u,c}^2 + \sigma_{v,c}^2}}\right), \qquad p_{u,v,c} = \Phi\!\left(\frac{\hat{\mu}_{u,c} - \hat{\mu}_{v,c}}{\sqrt{\sigma_{u,c}^2 + \sigma_{v,c}^2}}\right),$$
   computed stably via `scipy.special.log_ndtr`. Notice this is **exact** for any $K=2$ Gaussian instance.
2. **Exact front-identity equivalence theorem:** $\operatorname{Pareto}(\Theta) = \hat{\mathcal{F}}_1$ holds *if and only if*:
   - Every unordered pair $\{a, b\} \subset \hat{\mathcal{F}}_1$ remains incomparable ($a \not\succ_\Theta b \land b \not\succ_\Theta a$, event risk $r_{\mathrm{in}}(a, b) = \min(1, P_{a\succ b} + P_{b\succ a})$), **and**
   - Every arm $j \notin \hat{\mathcal{F}}_1$ remains dominated by at least one arm in $\hat{\mathcal{F}}_1$. Computing survival separately on dominators and challengers prevents the double counting of a pair $\{i, j\}$ appearing as both an exit and entry event.
3. **Coordinate-aligned shared-arm variance correlation discounting:** Naïve independence over pairs over-counts risk when multiple challengers share a single noisy dominator $i$ (or when an under-sampled challenger $j$ faces multiple well-sampled dominators $i \in \hat{\mathcal{F}}_1$), whereas orthogonal threats do act nearly independently. Grouping events by their common bottleneck arm $u$ (ordered largest-to-smallest risk $m = 0, 1, \dots$), we weight each coordinate by its normalized failure probability $w_c(a, b) \propto \max(0, 1 - p_{a,b,c})$ and measure shared-variance correlation between events $p$ and $m$ on common arm $u$:
   $$\rho_{p,m\mid u} = \sum_{c=1}^D \min\!\bigl(w_c(\text{pair}_p),\, w_c(\text{pair}_m)\bigr) \frac{\sigma_{u,c}^2}{\sqrt{(\sigma_{p,c}^2 + \sigma_{u,c}^2)(\sigma_{m,c}^2 + \sigma_{u,c}^2)}}.$$
   The conditional effective exponent for event $m$ given preceding events $\{0,\dots,m-1\}$ in the group is
   $$\mathrm{eff}_m = \frac{1 - \max_{p < m} \rho_{p,m\mid u}}{1 + 0.5 \sum_{p < m} \rho_{p,m\mid u}},$$
   giving survival factor $\prod_m (1 - r_m)^{\mathrm{eff}_m}$. This preserves near-unity exponents for orthogonal coordinates ($\rho \approx 0 \Rightarrow \mathrm{eff}_m \approx 1$), discounts identical aligned correlations toward $1/(m+1)$ ($20$ aligned challengers: $\mathcal{U}_{\mathrm{V2}} = 0.0571$ vs. MC $0.0648$), and avoids over-damping diffuse tails in high dimensions ($D=10$, `EgeExp4`).

### 2.3 Positioning Relative to MC $q_t$, PSIPS, and PESMO

- **Monte Carlo posterior disagreement ($q_t$):** $\mathcal{U}_{\mathrm{V2}}$ is a deterministic, zero-variance $O(K^2 D)$ analytic surrogate for $q_t = \Pr_{\Theta\sim\Pi_t}[\operatorname{Pareto}(\Theta)\neq\hat{\mathcal{F}}_1]$, achieving Pearson $r \in [0.9836, 0.9965]$ and MAE $0.0069\text{--}0.0227$ against 512-draw MC across all 8 benchmark environments while avoiding MC sampling jitter in high-precision stopping regimes and achieving $5.7\times$ (`EgeExp4` multi-dim excluded: `44.3 vs 87.7 ms`, $6.6\times$ on `EgeExp3`: `183.8 vs 1205.9 ms`, $7.6\times$–$10\times$ on $K=10\text{--}22$) speedup at runtime.
- **PSIPS (`Kone2025ParetoSI`):** Fixed-confidence PSI algorithms such as PSIPS build frequentist confidence/empirical sub-optimality gap stopping conditions derived from PAC sample-complexity bounds. Those bounds require known sub-Gaussian noise constants and log-horizon factors that are often conservative in small-horizon adaptive runs, whereas $\mathcal{U}_{\mathrm{V2}}$ yields an interpretable probability-scale diagnostic directly computed from the Bayesian posterior moments ($\hat{\boldsymbol{\mu}}, \boldsymbol{\sigma}$).
- **PESMO (`HernandezLobato2016PESMO`):** Predictive Entropy Search targets expected information gain $\mathbb{I}[(\mathbf{x}, \mathbf{y}); \mathcal{F}^* \mid \mathcal{D}_t]$ over future evaluations via EP/MC entropy approximation to guide *arm acquisition*. By contrast, TTPFTS already performs Posterior Sampling exploration, and $\mathcal{U}_{\mathrm{V1}} / \mathcal{U}_{\mathrm{V2}}$ measure *current state Pareto-set instability* at decision/stopping time without requiring lookahead predictive entropy integrals.

---

## 3. Analytic Unit Tests & Counterexample Verification

Measured from `experiments/uq/reports/analytic_checks.json` (`test_uq_variants.py`: 14/14 assertions pass):

| Synthetic Case | $K, D$ | Eq. (10) | $\mathcal{U}_{\mathrm{V1}}$ | $\mathcal{U}_{\mathrm{V2}}$ | MC $q_t$ ($20\text{k}$ draws) | Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `two_arm_coordinate_tie_10s_0.01s` | $2,2$ | `3.73e-6` | `0.5000` | `0.4972` | `0.4977` | Fixes coordinate-tie masking |
| `resolved_dominance_10s_10s` | $2,2$ | `1.39e-11` | `1.39e-11` | `1.54e-12` | `0.0000` | Low when clearly resolved |
| `confidently_incomparable` | $2,2$ | `0.0000` (unavail) | `0.0000` | `0.0000` | `0.0000` | Low when both Pareto & separated |
| `uncertain_within_F1_empty_F2` | $2,2$ | `0.0000` (unavail) | `0.5000` | `0.4972` | `0.4972` | Catches $\hat{\mathcal{F}}_1$ instability on $\hat{\mathcal{F}}_2=\emptyset$ |
| `empty_F2_diffuse` | $2,2$ | `0.0000` (unavail) | `0.5000` | `0.4937` | `0.4967` | Distinguishes diffuse vs. |
| `empty_F2_concentrated` | $2,2$ | `0.0000` (unavail) | `0.0000` | `0.0000` | `0.0000` | concentrated all-Pareto states |
| `dilution_M2_baseline` | $2,2$ | `3.73e-6` | `0.5000` | `0.4972` | `0.4977` | Baseline before adding distant pairs |
| `dilution_M10_distant_pairs` | $10,2$ | `1.49e-7` | `0.5000` | `0.4972` | `0.4990` | Dilution invariant |
| `dilution_M50_distant_pairs` | $50,2$ | `5.96e-9` | `0.5000` | `0.4972` | `0.4962` | Dilution invariant |
| `undersampled_below_F2_moderate` | $3,2$ | `1.23e-4` | `0.1392` | `0.3807` | `0.3813` | Detects front-3 challenger |
| `undersampled_below_F2_extreme_counterexample` | $3,2$ | `0.0020` | `0.0020` | `0.7086` | `0.7089` | Exposes BC variance-mismatch limit |
| `shared_dominator_20_aligned` | $21,2$ | `0.0104` | `0.1889` | `0.0571` | `0.0648` | Shared-arm discount avoids inflation |
| `shared_dominator_orthogonal_coords` | $3,2$ | `1.83e-4` | `0.0100` | `0.7296` | `0.7330` | Preserves orthogonal threats |
| `shared_noisy_challenger_10_optimal` | $11,2$ | `0.0078` | `0.9998` | `0.9946` | `0.9962` | Catches single noisy challenger |

Across the 28-point two-arm gap sweep $(z_1 \in \{0,2,5,10\}, z_2 \in \{-10,-2,-0.01,0.01,0.5,2,10\})$, $\mathcal{U}_{\mathrm{V2}}$ matches exact analytic $q_t$ to `< 1e-12` at every point, while $\mathcal{U}_{\mathrm{V1}}$ stays strictly within a tight Chernoff upper bound ($0.0$ false negatives on uncertain configurations).

---

## 4. Eight-Environment Validation (`TTPFTS_KNOWN_VAR` & `TTPFTS_UNKNOWN_VAR`)

All `TTPFTS_KNOWN_VAR` experiments ran on HPC (`NormalPosterior`, $\sigma=0.5$, 4-pull warm-up per arm, 100 runs $\times$ 101 checkpoints up to $B = \operatorname{round}(H(\nu))$) and passed 100% identity/count/recompute validation (`merge_validate_kv.py`) against the archived `results_preregistered_final_ff29cc456351` runs. We also re-scored all 80,800 checkpoints from saved `TTPFTS_UNKNOWN_VAR` (`TPosterior`) shards after verifying the `TPosterior` sampler scale vs. posterior std contract (`n=6, df=5`, `sampler_scale = batch_scale = 0.45947`, `logged_std = reconstructed_std = 0.59317`, ratio $\sqrt{\nu/(\nu-2)} = 1.29099$).

### 4.1 Endpoint Scores, Runtime & MC Agreement (`TTPFTS_KNOWN_VAR`, $B=1.0$)

| Env | $K, D, |\mathcal{F}^*|$ | Budget | Exact Acc | Jaccard | Eq. (10) Mean | $\mathcal{U}_{\mathrm{V1}}$ Mean | $\mathcal{U}_{\mathrm{V2}}$ Mean | MC $q_t$ Mean | $\mathrm{MAE}(\mathcal{U}_{\mathrm{V2}}, q_t)$ | $r(\mathcal{U}_{\mathrm{V2}}, q_t)$ | Time Eq10 / V1 / V2 / MC512 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `EgeExp1` | $60, 2, 10$ | $15,218$ | `0.43` | `0.9316` | `0.0155` | `0.9920` | `0.8146` | `0.8266` | `0.0079` | `0.9885` | `2.71 / 0.98 / 14.43 / 116.38` |
| `EgeExp2` | $10, 2, 2$ | $2,544$ | `0.77` | `0.9200` | `0.0097` | `0.6932` | `0.4541` | `0.4703` | `0.0219` | `0.9952` | `0.55 / 0.19 / 0.47 / 3.62` |
| `EgeExp3` | $200, 2, 20$ | $685,036$ | `0.47` | `0.9620` | `0.0019` | `0.9817` | `0.8581` | `0.8737` | `0.0069` | `0.9857` | `2.50 / 6.24 / 183.76 / 1205.93` |
| `EgeExp4` | $50, 10, 20$ | $15,977$ | `0.91` | `0.9955` | `0.0000` | `0.2304` | `0.1827` | `0.1857` | `0.0185` | `0.9965` | `15.81 / 2.61 / 44.27 / 87.66` |
| `EgeExp5` | $20, 2, 4$ | $197,373$ | `0.73` | `0.9305` | `0.0126` | `0.5919` | `0.4067` | `0.4118` | `0.0200` | `0.9936` | `0.48 / 0.30 / 1.38 / 13.40` |
| `EgeExp6` | $10, 2, 10$ | $51,966$ | `0.55` | `0.9490` | `0.0312` | `0.7792` | `0.5852` | `0.5985` | `0.0227` | `0.9851` | `0.37 / 0.49 / 0.41 / 3.59` |
| `EgeExp7` | $22, 2, 8$ | $7,000$ | `0.34` | `0.8751` | `0.0529` | `0.9913` | `0.8628` | `0.8719` | `0.0094` | `0.9836` | `1.32 / 0.43 / 2.81 / 16.20` |
| `EgeExp8` | $5, 2, 1$ | $237,958$ | `0.82` | `0.9000` | `0.2880` | `0.3801` | `0.2542` | `0.2525` | `0.0158` | `0.9935` | `0.21 / 0.14 / 0.19 / 1.19` |

Note: At $B = H(\nu)$, `EgeExp1`, `EgeExp3`, and `EgeExp7` still exhibit high empirical exact-set error (`57%`, `53%`, `66%`) because multiple optimal/near-optimal boundary arms remain close relative to $\sigma=0.5$. Accordingly, $\mathcal{U}_{\mathrm{V2}}$ and MC $q_t$ remain near $0.81\text{--}0.87$, accurately reflecting unresolved exact Pareto-set uncertainty rather than falsely collapsing to $\le 0.05$ like Eq. (10).

### 4.2 Per-Run Trajectory Correlations (`TTPFTS_KNOWN_VAR` & `TTPFTS_UNKNOWN_VAR`)

Median Pearson $r$ `[Q1, Q3]` (`n_valid` trajectories out of 100; exclusions occur only when a trajectory has $<2$ distinct GT values, e.g. constant Bernoulli $=0$):

#### Known-Variance Gaussian (`TTPFTS_KNOWN_VAR`, Figure 2 & Figure 3)
| Env | GT Metric | Eq. (10) $\mathcal{U}_\beta$ | Variant 1 ($\mathcal{U}_{\mathrm{V1}}$) | Variant 2 ($\mathcal{U}_{\mathrm{V2}}$) | MC Reference ($q_t$) | `n_valid` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `EgeExp1` | Jaccard | `-0.895 [-0.918,-0.870]` | `-0.446 [-0.540,-0.377]` | `-0.638 [-0.701,-0.560]` | `-0.625 [-0.682,-0.540]` | 100 |
| `EgeExp2` | Jaccard | `-0.733 [-0.855,-0.625]` | `-0.668 [-0.753,-0.576]` | `-0.740 [-0.815,-0.641]` | `-0.734 [-0.820,-0.647]` | 100 |
| `EgeExp3` | Jaccard | `-0.927 [-0.948,-0.896]` | `-0.452 [-0.500,-0.371]` | `-0.572 [-0.608,-0.513]` | `-0.545 [-0.588,-0.489]` | 100 |
| `EgeExp4` | Jaccard | `-0.931 [-0.973,-0.859]` | `-0.407 [-0.483,-0.339]` | `-0.640 [-0.681,-0.612]` | `-0.632 [-0.675,-0.602]` | 100 |
| `EgeExp5` | Jaccard | `-0.610 [-0.748,-0.425]` | `-0.599 [-0.714,-0.419]` | `-0.574 [-0.701,-0.433]` | `-0.557 [-0.706,-0.399]` | 100 |
| `EgeExp6` | Jaccard | `-0.728 [-0.790,-0.576]` | `-0.514 [-0.618,-0.366]` | `-0.415 [-0.517,-0.232]` | `-0.414 [-0.517,-0.256]` | 100 |
| `EgeExp7` | Jaccard | `-0.850 [-0.885,-0.806]` | `-0.531 [-0.621,-0.420]` | `-0.569 [-0.670,-0.450]` | `-0.575 [-0.656,-0.475]` | 100 |
| **`EgeExp8`** | **Jaccard** | **`+0.265 [-0.103,+0.561]`** | **`-0.280 [-0.499,+0.083]`** | **`-0.352 [-0.531,+0.014]`** | **`-0.334 [-0.537,-0.005]`** | 99 |
| `EgeExp1` | Bernoulli | `-0.318 [-0.449,-0.216]` | `-0.524 [-0.685,-0.203]` | `-0.616 [-0.763,-0.195]` | `-0.615 [-0.762,-0.240]` | 73 |
| `EgeExp2` | Bernoulli | `-0.560 [-0.718,-0.371]` | `-0.622 [-0.749,-0.465]` | `-0.671 [-0.788,-0.525]` | `-0.668 [-0.792,-0.527]` | 94 |
| `EgeExp3` | Bernoulli | `-0.284 [-0.455,-0.190]` | `-0.553 [-0.699,-0.273]` | `-0.625 [-0.755,-0.280]` | `-0.628 [-0.729,-0.319]` | 68 |
| `EgeExp4` | Bernoulli | `-0.581 [-0.720,-0.419]` | `-0.590 [-0.675,-0.466]` | `-0.865 [-0.907,-0.794]` | `-0.849 [-0.889,-0.772]` | 100 |
| `EgeExp5` | Bernoulli | `-0.415 [-0.667,-0.238]` | `-0.534 [-0.697,-0.284]` | `-0.518 [-0.702,-0.234]` | `-0.521 [-0.700,-0.193]` | 96 |
| `EgeExp6` | Bernoulli | `-0.815 [-0.876,-0.663]` | `-0.384 [-0.605,+0.040]` | `-0.280 [-0.529,+0.157]` | `-0.298 [-0.534,+0.125]` | 89 |
| `EgeExp7` | Bernoulli | `-0.319 [-0.555,-0.125]` | `-0.407 [-0.621,-0.065]` | `-0.310 [-0.668,-0.044]` | `-0.329 [-0.647,-0.044]` | 57 |
| **`EgeExp8`** | **Bernoulli** | **`+0.436 [+0.070,+0.722]`** | **`-0.184 [-0.479,+0.262]`** | **`-0.311 [-0.525,+0.242]`** | **`-0.302 [-0.529,+0.234]`** | 98 |
| `EgeExp1` | Misid | `+0.908 [+0.886,+0.932]` | `+0.411 [+0.355,+0.487]` | `+0.591 [+0.521,+0.653]` | `+0.577 [+0.511,+0.633]` | 100 |
| `EgeExp4` | Misid | `+0.947 [+0.886,+0.982]` | `+0.346 [+0.303,+0.401]` | `+0.561 [+0.508,+0.606]` | `+0.552 [+0.509,+0.605]` | 100 |
| **`EgeExp8`** | **Misid** | **`-0.265 [-0.555,+0.132]`** | **`+0.280 [-0.043,+0.499]`** | **`+0.379 [+0.039,+0.541]`** | **`+0.372 [+0.069,+0.551]`** | 99 |

Notice an insightful distinction between **Bernoulli (Exact Recovery)** and **Jaccard**: Eq. (10) correlates strongly with early-budget Jaccard jumps simply because Eq. (10) decays exponentially right after warm-up (`1/sqrt(...)`) at the exact same early stage where suboptimal fronts are pruned, yet for **exact recovery (Bernoulli)** — which Variant 2 is explicitly designed to track — $\mathcal{U}_{\mathrm{V2}}$ and $\mathcal{U}_{\mathrm{V1}}$ substantially beat Eq. (10) on `EgeExp1` (`-0.616/-0.524` vs `-0.318`), `EgeExp2` (`-0.671/-0.622` vs `-0.560`), `EgeExp3` (`-0.625/-0.553` vs `-0.284`), `EgeExp4` (`-0.865/-0.590` vs `-0.581`), `EgeExp5` (`-0.518/-0.534` vs `-0.415`), and fix the wrong sign on `EgeExp8` (`-0.311/-0.184` vs `+0.436`). (Note: on `EgeExp6`, `bhattacharyya` is zero-imputed when $\hat{\mathcal{F}}_2 = \emptyset$, inducing an artificial step jump from positive to $0.0$; over strictly non-empty $\hat{\mathcal{F}}_2$ checkpoints `bhattacharyya_raw` drops to `-0.365` vs Jaccard.)

#### Unknown-Variance Re-scored (`TTPFTS_UNKNOWN_VAR`, `uv_finite_avail == 1`)
Consistent patterns hold on the saved Student-$t$ runs (`n_valid=100` for Jaccard except `98` on `EgeExp8`):
- **Jaccard medians (`Eq10 / V1 / V2`)**: `EgeExp1`: `-0.885 / -0.462 / -0.628`; `EgeExp2`: `-0.773 / -0.633 / -0.736`; `EgeExp3`: `-0.918 / -0.439 / -0.559`; `EgeExp4`: `-0.847 / -0.453 / -0.738`; `EgeExp5`: `-0.669 / -0.598 / -0.591`; `EgeExp6`: `-0.692 / -0.465 / -0.372`; `EgeExp7`: `-0.841 / -0.532 / -0.581`; **`EgeExp8`: `+0.296 / -0.239 / -0.313`**.
- **Bernoulli medians (`Eq10 / V1 / V2`)**: `EgeExp1`: `-0.341 / -0.481 / -0.545`; `EgeExp2`: `-0.635 / -0.588 / -0.687`; `EgeExp3`: `-0.332 / -0.602 / -0.675`; `EgeExp4`: `-0.709 / -0.559 / -0.839`; `EgeExp5`: `-0.478 / -0.554 / -0.521`; `EgeExp6`: `-0.804 / -0.168 / -0.083`; `EgeExp7`: `-0.332 / -0.425 / -0.299`; **`EgeExp8`: `+0.513 / -0.023 / -0.234`**.

### 4.3 Matched-Budget Exact-Error Calibration (`TTPFTS_KNOWN_VAR`)

At normalized budget $1.00$ across all $800$ trajectories (`overall_err = 0.3725`, `n=800`):
- **Eq. (10)**: `[0.00, 0.01]` contains `444` checkpoints with **`98/444 = 22.1%` wrong**; `(0.01, 0.05]` has `112/184 = 60.9%` wrong; `(0.05, 0.10]` has `71/89 = 79.8%` wrong.
- **Variant 1 ($\mathcal{U}_{\mathrm{V1}}$)**: `(0.01, 0.05]` has `0/3 = 0.0%` wrong; `(0.05, 0.10]` has `0/29 = 0.0%` wrong; `(0.10, 0.25]` has `2/70 = 2.9%` wrong; `(0.25, 0.50]` has `17/106 = 16.0%` wrong; `(0.50, 1.00]` has `279/592 = 47.1%` wrong.
- **Variant 2 ($\mathcal{U}_{\mathrm{V2}}$)**: `[0.00, 0.01]` has `0/2 = 0.0%` wrong; `(0.01, 0.05]` has `0/40 = 0.0%` wrong; `(0.05, 0.10]` has `2/30 = 6.7%` wrong; `(0.10, 0.25]` has `4/92 = 4.3%` wrong; `(0.25, 0.50]` has `37/172 = 21.5%` wrong; `(0.50, 1.00]` has `255/464 = 55.0%` wrong. At $B=0.25$ (`overall_err = 0.7500`), top-10% lowest-score accuracy is `68.0%` for $\mathcal{U}_{\mathrm{V2}}$ and `50.0%` for $\mathcal{U}_{\mathrm{V1}}$, compared to `24.0%` for Eq. (10).

---

## 5. Held-Out Random Instances & Stopping Rule Evaluation

### 5.1 Checkpoint Calibration & Matched-Coverage Accuracy (R1 Suite)

Selected checkpoints from `random_checkpoint_stopping.json`:

| Scenario (Repeats, Checkpoint Pulls) | Overall Err | Metric | `[0.00, 0.01]` Err (`w/n`) | `(0.01, 0.05]` Err (`w/n`) | Top-10% Acc | Top-25% Acc |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `uniform_K2_d2_s0.5` ($50\text{k}, n=10$) | `0.2793` | Eq. (10) | `0.2509` (`6936/27649`) | `0.1554` (`553/3558`) | `0.7270` | `0.7283` |
| `uniform_K2_d2_s0.5` ($50\text{k}, n=10$) | `0.2793` | $\mathcal{U}_{\mathrm{V1}}$ | `0.0007` (`1/1430`) | `0.0151` (`43/2854`) | `0.9876` | `0.9476` |
| `uniform_K2_d2_s0.5` ($50\text{k}, n=10$) | `0.2793` | $\mathcal{U}_{\mathrm{V2}}$ = MC | `0.0063` (`20/3200`) | `0.0455` (`245/5383`) | `0.9874` | `0.9478` |
| `normal_K2_d2_s0.5` ($50\text{k}, n=10$) | `0.2862` | Eq. (10) | `0.2552` (`7088/27776`) | `0.1545` (`508/3287`) | `0.7310` | `0.7298` |
| `normal_K2_d2_s0.5` ($50\text{k}, n=10$) | `0.2862` | $\mathcal{U}_{\mathrm{V1}}$ | `0.0022` (`3/1348`) | `0.0160` (`42/2620`) | `0.9856` | `0.9375` |
| `normal_K2_d2_s0.5` ($50\text{k}, n=10$) | `0.2862` | $\mathcal{U}_{\mathrm{V2}}$ = MC | `0.0072` (`22/3059`) | `0.0452` (`226/5000`) | `0.9856` | `0.9382` |
| `uniform_K10_d2_s0.5` ($3\text{k}, t=100$) | `0.5053` | Eq. (10) | `0.3917` (`503/1284`) | `0.5226` (`427/817`) | `0.7200` | `0.6693` |
| `uniform_K10_d2_s0.5` ($3\text{k}, t=100$) | `0.5053` | $\mathcal{U}_{\mathrm{V1}}$ | `0.0000` (`0/14`) | `0.0000` (`0/43`) | `0.9367` | `0.8307` |
| `uniform_K10_d2_s0.5` ($3\text{k}, t=100$) | `0.5053` | $\mathcal{U}_{\mathrm{V2}}$ | `0.0000` (`0/47`) | `0.0204` (`2/98`) | `0.9367` | `0.8480` |
| `uniform_K10_d2_s0.25` ($3\text{k}, t=100$) | `0.2973` | Eq. (10) | `0.2614` (`647/2475`) | `0.4258` (`132/310`) | `0.8533` | `0.8173` |
| `uniform_K10_d2_s0.25` ($3\text{k}, t=100$) | `0.2973` | $\mathcal{U}_{\mathrm{V1}}$ | `0.0000` (`0/269`) | `0.0130` (`2/154`) | `1.0000` | `0.9773` |
| `uniform_K10_d2_s0.25` ($3\text{k}, t=100$) | `0.2973` | $\mathcal{U}_{\mathrm{V2}}$ | `0.0026` (`1/379`) | `0.0321` (`8/249`) | `1.0000` | `0.9800` |
| `uniform_K10_d4_s0.5` ($3\text{k}, t=100$) | `0.6153` | Eq. (10) | `0.6074` (`1759/2896`) | `0.8350` (`86/103`) | `0.5533` | `0.5013` |
| `uniform_K10_d4_s0.5` ($3\text{k}, t=100$) | `0.6153` | $\mathcal{U}_{\mathrm{V1}}$ | `-` (`0/0`) | `0.0000` (`0/3`) | `0.7967` | `0.6667` |
| `uniform_K10_d4_s0.5` ($3\text{k}, t=100$) | `0.6153` | $\mathcal{U}_{\mathrm{V2}}$ | `0.0000` (`0/2`) | `0.0000` (`0/7`) | `0.8233` | `0.6773` |
| `uniform_K20_d2_s0.5` ($2\text{k}, t=100$) | `0.6665` | Eq. (10) | `0.5023` (`325/647`) | `0.7023` (`493/702`) | `0.5750` | `0.5260` |
| `uniform_K20_d2_s0.5` ($2\text{k}, t=100$) | `0.6665` | $\mathcal{U}_{\mathrm{V1}}$ | `-` (`0/0`) | `0.0000` (`0/3`) | `0.7950` | `0.6360` |
| `uniform_K20_d2_s0.5` ($2\text{k}, t=100$) | `0.6665` | $\mathcal{U}_{\mathrm{V2}}$ | `0.0000` (`0/1`) | `0.0000` (`0/12`) | `0.8050` | `0.6440` |

Note: Even under Eq. (10)'s legacy stringent cutoff `<= 4e-6`, error rates remain `0.2682` (`6,680/24,907`) on `uniform_K2_d2_s0.5` at $n=10$, `0.2102` (`309/1,470`) on `uniform_K10_d2_s0.25` at $t=100$, and `0.4865` (`36/74`) on `uniform_K20_d2_s0.5` at $t=100$. On `uniform_K50_d2_s0.5` at $t=100$ (overall error `0.8710`), Eq. (10) places `179` instances in `[0,0.01]` with `78.21%` wrong (`4e-6`: `60.0%` wrong), whereas neither $\mathcal{U}_{\mathrm{V1}}$ nor $\mathcal{U}_{\mathrm{V2}}$ allows false premature confidence (`0/0` below `0.05`, top-10% accuracy `44.0%` vs `21.0%` for Eq. 10).

### 5.2 First-Crossing Stopping Calibration (`uniform_K10_d2_s0.25`, `normal_K10_d2_s0.5`, and `EgeExp8`)

Conditioned on crossing threshold $\tau$ (prior to horizon $B=1,000$ in R1 or $B=237,958$ in `EgeExp8`):

| Setting | Rule | Crossed (`n/N`) | Error @ Crossing | Jaccard @ Crossing | Mean Pulls (Crossed / All) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `uniform_K10_d2_s0.25` | Eq. (10) $\le 0.01$ | `94.4%` (`2831/3000`) | `0.5083` | `0.8145` | `261.2 / 302.8` |
| `uniform_K10_d2_s0.25` | Eq. (10) $\le 4\times 10^{-6}$ | `58.5%` (`1754/3000`) | `0.3324` | `0.8800` | `500.7 / 708.1` |
| `uniform_K10_d2_s0.25` | $\mathcal{U}_{\mathrm{V1}} \le 0.10 / 0.05 / 0.01$ | `22.1% / 16.9% / 10.5%` | `0.0287 / 0.0098 / 0.0000` | `0.9892 / 0.9966 / 1.0000` | `497.8 / 527.8 / 606.4` (crossed) |
| `uniform_K10_d2_s0.25` | $\mathcal{U}_{\mathrm{V2}} \le 0.10 / 0.05 / 0.01$ | `38.7% / 27.2% / 14.8%` | **`0.0990 / 0.0466 / 0.0045`** | `0.9647 / 0.9829 / 0.9987` | `442.8 / 482.5 / 527.3` (crossed) |
| `normal_K10_d2_s0.5` | Eq. (10) $\le 0.01$ | `62.9%` (`1888/3000`) | `0.4513` | `0.8274` | `509.2 / 691.1` |
| `normal_K10_d2_s0.5` | $\mathcal{U}_{\mathrm{V1}} \le 0.05$ | `7.4%` (`221/3000`) | `0.0090` | `0.9970` | `636.9 / 973.3` |
| `normal_K10_d2_s0.5` | $\mathcal{U}_{\mathrm{V2}} \le 0.10 / 0.05 / 0.01$ | `22.9% / 15.3% / 6.7%` | **`0.0946 / 0.0522 / 0.0149`** | `0.9624 / 0.9804 / 0.9950` | `531.9 / 577.2 / 631.0` (crossed) |
| `EgeExp8` KV | Eq. (10) $\le 0.05 / 0.01 / 10^{-4}$ | `86% / 69% / 37%` | `0.8837 / 0.9855 / 1.0000` | `0.521 / 0.466 / 0.473` | `73.1k / 116.9k / 202.1k` (all) |
| `EgeExp8` KV | $\mathcal{U}_{\mathrm{V1}} \le 0.05$ | `9%` (`9/100`) | `0.0000` | `1.0000` | `231.2k` (all) |
| `EgeExp8` KV | $\mathcal{U}_{\mathrm{V2}} \le 0.10 / 0.05 / 0.01$ | `40% / 23% / 4%` | **`0.1000 / 0.0435 / 0.0000`** | `0.9375 / 0.9783 / 1.0000` | `187.2k / 214.1k / 234.6k` (all) |

Critically, $\mathcal{U}_{\mathrm{V2}} \le \alpha$ tracks the nominal risk level $\alpha \in \{0.10, 0.05, 0.01\}$ across both adaptive random instances and `EgeExp8`, whereas Eq. (10) fails catastrophically. We also document an honest caveat: under optional stopping, posterior disagreement $q_t$ depends on the correctness of the learner's empirical belief ($\Pi_t$) and becomes conservative ($0\%$ crossing within $B$) when many near-optimal boundary arms coexist relative to $\sigma$ (`EgeExp1`, `EgeExp3`, `EgeExp6`, `EgeExp7`, `uniform_K50_d2_s0.5`).

### 5.3 Held-Out Random Instance Trajectory Correlations (R2 Suite, Figure 4)

Median Pearson $r$ `[Q1, Q3]` across 100 adaptive trajectories per family (`80,800` checkpoints total):

| Family | GT Metric | Eq. (10) $\mathcal{U}_\beta$ | Variant 1 ($\mathcal{U}_{\mathrm{V1}}$) | Variant 2 ($\mathcal{U}_{\mathrm{V2}}$) | MC Reference ($q_t$) | `n_valid` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Unif_K10_d2` | Jaccard | `-0.584 [-0.778,-0.230]` | `-0.567 [-0.680,-0.400]` | `-0.583 [-0.693,-0.417]` | `-0.583 [-0.689,-0.411]` | 90 |
| `Norm_K10_d2` | Jaccard | `-0.550 [-0.828,-0.324]` | `-0.530 [-0.668,-0.377]` | `-0.578 [-0.717,-0.421]` | `-0.577 [-0.719,-0.423]` | 90 |
| `Unif_K20_d2` | Jaccard | `-0.596 [-0.793,-0.315]` | `-0.536 [-0.672,-0.389]` | `-0.593 [-0.729,-0.446]` | `-0.607 [-0.720,-0.442]` | 98 |
| `Norm_K20_d2` | Jaccard | `-0.517 [-0.701,-0.168]` | `-0.495 [-0.650,-0.291]` | `-0.572 [-0.736,-0.303]` | `-0.578 [-0.734,-0.308]` | 85 |
| `Unif_K50_d2` | Jaccard | `-0.551 [-0.761,-0.340]` | `-0.547 [-0.663,-0.394]` | `-0.595 [-0.713,-0.429]` | `-0.588 [-0.707,-0.439]` | 100 |
| `Unif_K10_d4` | Jaccard | `-0.403 [-0.577,-0.143]` | `-0.541 [-0.668,-0.386]` | `-0.575 [-0.693,-0.386]` | `-0.574 [-0.692,-0.381]` | 97 |
| `Unif_K20_d4` | Jaccard | `-0.488 [-0.625,-0.343]` | `-0.521 [-0.647,-0.357]` | `-0.582 [-0.706,-0.401]` | `-0.570 [-0.705,-0.398]` | 100 |
| `Unif_K10_d2_s05` | Jaccard | `-0.556 [-0.790,-0.229]` | `-0.545 [-0.664,-0.366]` | `-0.589 [-0.685,-0.403]` | `-0.582 [-0.691,-0.405]` | 97 |
| `Unif_K10_d4` | Bernoulli | `-0.220 [-0.438,-0.072]` | `-0.520 [-0.669,-0.310]` | `-0.565 [-0.695,-0.310]` | `-0.561 [-0.703,-0.310]` | 93 |
| `Unif_K20_d4` | Bernoulli | `-0.148 [-0.272,-0.081]` | `-0.452 [-0.640,-0.059]` | `-0.470 [-0.680,-0.127]` | `-0.483 [-0.683,-0.105]` | 87 |
| `Unif_K50_d2` | Bernoulli | `-0.253 [-0.462,-0.114]` | `-0.478 [-0.651,-0.264]` | `-0.524 [-0.654,-0.171]` | `-0.503 [-0.673,-0.184]` | 94 |

Higher dimensionality ($D=4$) and larger $K$ ($K=50$) sharply degrade Eq. (10)'s trajectory correlation (dropping to `-0.220` and `-0.148` vs Bernoulli), whereas $\mathcal{U}_{\mathrm{V1}}$ and $\mathcal{U}_{\mathrm{V2}}$ remain consistent (`-0.45` to `-0.60`) across all 8 random families and closely match the Monte Carlo reference.

---

## 6. Deliverables & Reproducibility

- **Core implementations**: `uncertainty_quantification.py` (`uncertainty_quantification_v1`, `uncertainty_quantification_v2`) and `experiments/uq/metrics.py` (`uq_directional_bc_from_moments`, `uq_dominance_risk_from_moments`, `uq_mc_disagreement_from_moments`).
- **Analytic & replay pipelines**: `experiments/uq/analytic_checks.py`, `experiments/uq_kv/{config.py,config.json,replay_kv.py,merge_validate_kv.py,random_instances.py,rescore_uv.py,analyze.py,plot.py}`.
- **Publication figures (`figures/publication/`)**:
  - `uq_variants_evolution_8envs.png` (plus `uq_v1_evolution_kv.png`, `uq_v2_evolution_kv.png`, `bhattacharyya_evolution_kv.png`)
  - `uq_v1_correlation_8envs.png` (plus `uq_v1_jaccard_correlation_kv.png`)
  - `uq_v2_correlation_8envs.png` (plus `uq_v2_jaccard_correlation_kv.png`)
  - `uq_random_instances_correlation.png`
- **Tests & verification gates**: `test_uq_variants.py`, `test_uq_replay.py`, `run_all_gates.py --only uq-variants`.
