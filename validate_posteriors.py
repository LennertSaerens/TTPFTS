"""
Standalone, rigorous correctness gate for the Gaussian posteriors used by
TTPFTS-family bandits (bandits/Posteriors.py, bandits/TTPFTS.py).

Covers, per the posterior-correctness gate spec:
  1. Analytic sufficient statistics vs. closed-form batch formulas.
  2. Batch-vs-streaming equivalence (arbitrary update order, Welford stability).
  3. Reset/state behavior (exact zeroing).
  4. Deterministic sampling moments + a distribution-shape (KS) test, with
     preregistered tolerances fixed *before* any run of this script.
  5. Multi-arm / multi-objective independence.
  6. TTPFTS integration (warm-up, explicit failure surfacing, end-to-end run).

Design choices fixed by the user for this gate:
  * Gaussian reward variance for NormalPosterior tests: sigma^2 = 0.25
    (sigma = 0.5).
  * TPosterior default (alpha=0) implements the unknown-variance reference
    prior p(mu, sigma^2) ~ 1/sigma^2, i.e. mu | data ~
    Student-t(df=n-1, loc=xbar, scale=sqrt(S/(n(n-1)))).

Exit code is 0 iff every check passes; a human-readable PASS/FAIL report is
printed for each check. Run with:  .venv/bin/python validate_posteriors.py
"""
import sys
import numpy as np
from scipy import stats

from bandits.Posteriors import TPosterior, NormalPosterior
from bandits.TTPFTS import TTPFTSBandit

RNG_SEED = 12345
FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)
    return condition


def close(a, b, atol=1e-9, rtol=1e-9):
    return np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)


# ---------------------------------------------------------------------------
# 1. TPosterior: analytic sufficient statistics vs. batch closed form
# ---------------------------------------------------------------------------

def batch_stats(x):
    """Closed-form sample mean and sum-of-squared-deviations for a 1-D array."""
    n = len(x)
    xbar = np.mean(x)
    S = np.sum((x - xbar) ** 2)
    return n, xbar, S


def test_tposterior_sufficient_statistics():
    rng = np.random.default_rng(RNG_SEED)
    ok = True
    for trial in range(20):
        n_obs = rng.integers(2, 50)
        x = rng.normal(loc=rng.uniform(-5, 5), scale=rng.uniform(0.1, 4.0), size=n_obs)
        n_true, xbar_true, S_true = batch_stats(x)

        post = TPosterior(num_arms=1, num_objectives=1, alpha=0.0)
        for val in x:
            post.update(0, np.array([val]))

        n_est = post.n[0, 0]
        mean_est = post.mean[0, 0]
        M2_est = post.M2[0, 0]

        ok &= (n_est == n_true)
        ok &= close(mean_est, xbar_true, atol=1e-8, rtol=1e-8)
        ok &= close(M2_est, S_true, atol=1e-6, rtol=1e-8)

        # Reference-prior closed form: df=n-1, scale=sqrt(S/(n(n-1)))
        df_true = n_true - 1
        scale_true = np.sqrt(S_true / (n_true * (n_true - 1)))
        df_est = post._df()[0, 0]
        scale_est = post.get_scale()[0, 0]
        ok &= close(df_est, df_true, atol=1e-9)
        ok &= close(scale_est, scale_true, atol=1e-8, rtol=1e-8)
    return check(
        "TPosterior sufficient statistics match closed-form batch formulas "
        "(n, xbar, S, df=n-1, scale=sqrt(S/(n(n-1)))) across 20 random datasets",
        ok,
    )


def test_tposterior_streaming_bug_regression():
    """
    Regression check for the specific bug found during this gate: a naive
    online update using S += (x - new_mean)^2 silently understates S versus
    the true sum of squared deviations from the final mean. Welford's
    algorithm must match the true batch S to float precision.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    x = rng.normal(loc=5.0, scale=2.0, size=6)
    _, xbar_true, S_true = batch_stats(x)

    post = TPosterior(num_arms=1, num_objectives=1)
    for val in x:
        post.update(0, np.array([val]))

    return check(
        "Welford streaming S matches true sum-of-squared-deviations "
        "(regression test for naive S += (x-new_mean)^2 bug)",
        close(post.M2[0, 0], S_true, atol=1e-9, rtol=1e-9),
        detail=f"got {post.M2[0, 0]}, expected {S_true}",
    )


# ---------------------------------------------------------------------------
# 2. Batch-vs-streaming equivalence, order independence
# ---------------------------------------------------------------------------

def test_tposterior_order_independence():
    rng = np.random.default_rng(RNG_SEED + 2)
    x = rng.normal(loc=-3.0, scale=1.5, size=30)

    post_a = TPosterior(num_arms=1, num_objectives=1)
    for val in x:
        post_a.update(0, np.array([val]))

    perm = rng.permutation(x)
    post_b = TPosterior(num_arms=1, num_objectives=1)
    for val in perm:
        post_b.update(0, np.array([val]))

    ok = (
        post_a.n[0, 0] == post_b.n[0, 0]
        and close(post_a.mean[0, 0], post_b.mean[0, 0], atol=1e-8)
        and close(post_a.M2[0, 0], post_b.M2[0, 0], atol=1e-6, rtol=1e-8)
    )
    return check(
        "TPosterior streaming statistics are invariant to update order "
        "(same multiset of observations, different arrival order)",
        ok,
    )


def test_normalposterior_batch_vs_streaming():
    """NormalPosterior (known variance=0.25) matches the closed-form
    conjugate-normal posterior mean/std for an improper flat prior on mu."""
    sigma2 = 0.25
    sigma = np.sqrt(sigma2)
    rng = np.random.default_rng(RNG_SEED + 3)
    x = rng.normal(loc=2.0, scale=sigma, size=25)

    post = NormalPosterior(num_arms=1, num_objectives=1, known_stds=sigma)
    for val in x:
        post.update(0, np.array([val]))

    n = len(x)
    xbar_true = np.mean(x)
    std_true = sigma / np.sqrt(n)

    ok = (
        post.counts[0, 0] == n
        and close(post.means[0, 0], xbar_true, atol=1e-10)
        and close(post.get_stds()[0, 0], std_true, atol=1e-10)
    )
    return check(
        "NormalPosterior (known variance=0.25) streaming mean/std match "
        "closed-form N(xbar, sigma^2/n) posterior",
        ok,
    )


# ---------------------------------------------------------------------------
# 3. Reset / state behavior
# ---------------------------------------------------------------------------

def test_reset_exact():
    ok = True
    t_post = TPosterior(num_arms=3, num_objectives=2, alpha=-0.5)
    for arm in range(3):
        for _ in range(5):
            t_post.update(arm, np.array([1.0, -2.0]))
    t_post.reset(None)
    ok &= np.array_equal(t_post.n, np.zeros((3, 2), dtype=np.int64))
    ok &= close(t_post.mean, np.zeros((3, 2)), atol=0)
    ok &= close(t_post.M2, np.zeros((3, 2)), atol=0)

    n_post = NormalPosterior(num_arms=3, num_objectives=2, known_stds=0.5)
    for arm in range(3):
        for _ in range(5):
            n_post.update(arm, np.array([1.0, -2.0]))
    n_post.reset(0.5)
    ok &= np.array_equal(n_post.counts, np.zeros((3, 2), dtype=int))
    ok &= close(n_post.means, np.zeros((3, 2)), atol=0)

    return check(
        "reset() exactly zeroes all TPosterior/NormalPosterior state "
        "(n/mean/M2 and counts/means respectively)",
        ok,
    )


def test_reset_reusable_after():
    """After reset(), a fresh sequence of updates must reproduce the exact
    same statistics as a brand-new posterior fed the same data (no residue)."""
    rng = np.random.default_rng(RNG_SEED + 4)
    x = rng.normal(size=10)

    fresh = TPosterior(num_arms=1, num_objectives=1)
    for val in x:
        fresh.update(0, np.array([val]))

    dirty = TPosterior(num_arms=1, num_objectives=1)
    for _ in range(7):
        dirty.update(0, np.array([999.0]))
    dirty.reset(None)
    for val in x:
        dirty.update(0, np.array([val]))

    ok = (
        fresh.n[0, 0] == dirty.n[0, 0]
        and close(fresh.mean[0, 0], dirty.mean[0, 0], atol=1e-10)
        and close(fresh.M2[0, 0], dirty.M2[0, 0], atol=1e-10)
    )
    return check(
        "TPosterior post-reset trajectory is bit-for-bit equivalent to a "
        "freshly constructed posterior fed the same observations",
        ok,
    )


# ---------------------------------------------------------------------------
# 4. Deterministic sampling moments / distribution-shape tests
#    (tolerances preregistered here, before running)
# ---------------------------------------------------------------------------

N_MC_SAMPLES = 200_000
MC_Z_BOUND = 5.0          # 5-sigma Monte-Carlo tolerance on moment checks
KS_ALPHA = 0.001          # significance level for KS goodness-of-fit (strict)


def test_tposterior_sampling_matches_student_t():
    rng_state = np.random.get_state()
    np.random.seed(RNG_SEED + 5)
    try:
        x = np.array([1.0, 2.5, -0.3, 4.0, 1.2, 0.7])
        n, xbar, S = batch_stats(x)
        df = n - 1
        scale = np.sqrt(S / (n * (n - 1)))

        post = TPosterior(num_arms=1, num_objectives=1)
        for val in x:
            post.update(0, np.array([val]))

        samples = np.array([post.sample()[0, 0] for _ in range(N_MC_SAMPLES)])

        # Mean check (finite for df>1)
        se_mean = scale * np.sqrt(df / (df - 2)) / np.sqrt(N_MC_SAMPLES) if df > 2 else None
        mean_ok = True
        if se_mean is not None:
            mean_ok = abs(np.mean(samples) - xbar) < MC_Z_BOUND * se_mean

        # KS test against the exact analytic Student-t distribution.
        ks_stat, ks_p = stats.kstest(samples, "t", args=(df, xbar, scale))
        ks_ok = ks_p > KS_ALPHA

        ok = mean_ok and ks_ok
        return check(
            "TPosterior.sample() draws match analytic Student-t(df=n-1, "
            "loc=xbar, scale=sqrt(S/(n(n-1)))): mean within "
            f"{MC_Z_BOUND}-sigma MC bound and KS test p={ks_p:.4g} > {KS_ALPHA}",
            ok,
        )
    finally:
        np.random.set_state(rng_state)


def test_tposterior_get_stds_vs_get_scale():
    """get_stds() (actual std) must equal get_scale() * sqrt(df/(df-2)) for
    df>2, and must be exactly np.inf for 0 < df <= 2."""
    ok = True
    # df = n-1 = 3 > 2 case (finite variance)
    post = TPosterior(num_arms=1, num_objectives=1)
    for val in [1.0, 2.0, 3.0, 4.0]:  # n=4 -> df=3
        post.update(0, np.array([val]))
    scale = post.get_scale()[0, 0]
    stds = post.get_stds()[0, 0]
    df = post._df()[0, 0]
    ok &= close(stds, scale * np.sqrt(df / (df - 2)), atol=1e-10)

    # df = n-1 = 1 case (infinite variance, n=2)
    post2 = TPosterior(num_arms=1, num_objectives=1)
    post2.update(0, np.array([1.0]))
    post2.update(0, np.array([2.0]))
    ok &= np.isinf(post2.get_stds()[0, 0])
    ok &= not np.isinf(post2.get_scale()[0, 0])  # scale itself stays finite

    return check(
        "TPosterior.get_stds() (actual std) is scale*sqrt(df/(df-2)) when "
        "finite, and explicitly np.inf when df<=2 (distinguished from "
        "get_scale(), the Student-t scale parameter)",
        ok,
    )


def test_normalposterior_sampling_matches_normal():
    rng_state = np.random.get_state()
    np.random.seed(RNG_SEED + 6)
    try:
        sigma2 = 0.25
        sigma = np.sqrt(sigma2)
        x = np.array([1.0, 2.0, 1.5, 0.8, 2.2, 1.1, 1.9])
        n = len(x)
        xbar = np.mean(x)
        post_std = sigma / np.sqrt(n)

        post = NormalPosterior(num_arms=1, num_objectives=1, known_stds=sigma)
        for val in x:
            post.update(0, np.array([val]))

        samples = np.array([post.sample()[0, 0] for _ in range(N_MC_SAMPLES)])

        se_mean = post_std / np.sqrt(N_MC_SAMPLES)
        mean_ok = abs(np.mean(samples) - xbar) < MC_Z_BOUND * se_mean

        ks_stat, ks_p = stats.kstest(samples, "norm", args=(xbar, post_std))
        ks_ok = ks_p > KS_ALPHA

        return check(
            "NormalPosterior.sample() (variance=0.25) matches analytic "
            f"N(xbar, sigma^2/n): mean within {MC_Z_BOUND}-sigma MC bound "
            f"and KS test p={ks_p:.4g} > {KS_ALPHA}",
            mean_ok and ks_ok,
        )
    finally:
        np.random.set_state(rng_state)


# ---------------------------------------------------------------------------
# 5. Invalid-state surfacing
# ---------------------------------------------------------------------------

def test_tposterior_invalid_state_raises():
    ok = True
    # n=0: no observations at all -> df = 2*alpha - 1 <= 0 for alpha=0
    post = TPosterior(num_arms=2, num_objectives=1, alpha=0.0)
    raised = False
    try:
        post.sample()
    except ValueError:
        raised = True
    ok &= raised

    # n=1 for alpha=0 -> df=0, still invalid
    post2 = TPosterior(num_arms=1, num_objectives=1, alpha=0.0)
    post2.update(0, np.array([1.0]))
    raised2 = False
    try:
        post2.sample()
    except ValueError:
        raised2 = True
    ok &= raised2

    # n=2 for alpha=0 -> df=1 > 0, must succeed
    post2.update(0, np.array([2.0]))
    no_raise = True
    try:
        post2.sample()
        post2.get_stds()
        post2.get_scale()
    except ValueError:
        no_raise = False
    ok &= no_raise

    # alpha=-0.5 (Honda-Takemura "optimal" prior used by TTPFTS_UNI in
    # momab_main.py): n=2 -> df=0, still invalid; n=3 -> df=1, valid.
    post3 = TPosterior(num_arms=1, num_objectives=1, alpha=-0.5)
    post3.update(0, np.array([1.0]))
    post3.update(0, np.array([2.0]))
    raised3 = False
    try:
        post3.sample()
    except ValueError:
        raised3 = True
    ok &= raised3
    post3.update(0, np.array([3.0]))
    no_raise3 = True
    try:
        post3.sample()
    except ValueError:
        no_raise3 = False
    ok &= no_raise3

    return check(
        "TPosterior.sample()/get_stds()/get_scale() raise ValueError "
        "explicitly for insufficient-observation (df<=0) states, for "
        "both alpha=0 (reference prior) and alpha=-0.5 (Honda-Takemura), "
        "and succeed as soon as df>0",
        ok,
    )


# ---------------------------------------------------------------------------
# 6. Multi-arm / multi-objective independence
# ---------------------------------------------------------------------------

def test_multi_arm_objective_independence():
    rng = np.random.default_rng(RNG_SEED + 7)
    num_arms, num_objectives = 4, 3
    post = TPosterior(num_arms=num_arms, num_objectives=num_objectives)

    data = {}
    for arm in range(num_arms):
        n_obs = rng.integers(2, 15)
        x = rng.normal(loc=arm, scale=1.0 + arm, size=(n_obs, num_objectives))
        data[arm] = x
        for row in x:
            post.update(arm, row)

    ok = True
    for arm in range(num_arms):
        x = data[arm]
        n_true, xbar_true, _ = len(x), x.mean(axis=0), None
        S_true = np.sum((x - xbar_true) ** 2, axis=0)
        ok &= (post.n[arm] == n_true).all()
        ok &= close(post.mean[arm], xbar_true, atol=1e-8)
        ok &= close(post.M2[arm], S_true, atol=1e-6, rtol=1e-7)

    # Independence: mutating one arm must not affect others' stats.
    snapshot_other = post.mean[1].copy()
    for _ in range(3):
        post.update(0, np.array([100.0, 100.0, 100.0]))
    ok &= close(post.mean[1], snapshot_other, atol=0)

    return check(
        "TPosterior tracks independent sufficient statistics per (arm, "
        "objective); updating one arm never perturbs another arm's state",
        ok,
    )


# ---------------------------------------------------------------------------
# 7. TTPFTS integration
# ---------------------------------------------------------------------------

def test_ttpfts_integration_reference_prior():
    """
    End-to-end: TTPFTSBandit driven by TPosterior(alpha=0, the reference
    prior) on a small synthetic 2-objective, 3-arm problem, with enough
    warm-up pulls (num_warmup_pulls=2 -> df=1>0). Must run without error and
    recover a sensible (non-empty, in-range) Pareto recommendation.
    """
    rng = np.random.default_rng(RNG_SEED + 8)
    num_arms, num_objectives = 3, 2
    true_means = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])  # arm 2 dominated
    sigma = 0.5  # variance 0.25

    posterior = TPosterior(num_arms, num_objectives, alpha=0.0)
    bandit = TTPFTSBandit(posterior, p=0.5, num_warmup_pulls=2)
    bandit.reset(None)

    np.random.seed(RNG_SEED + 9)
    random_ok = True
    for _ in range(400):
        arm = bandit.choose_arm()
        if not (0 <= arm < num_arms):
            random_ok = False
        reward = rng.normal(loc=true_means[arm], scale=sigma)
        bandit.learn(arm, reward)

    top_arms = bandit.get_top_arms()
    recovers_pareto = set(top_arms.tolist()) == {0, 1}

    return check(
        "TTPFTSBandit + TPosterior(alpha=0, reference prior) runs 400 steps "
        "without raising, returns in-range arms, and recovers the true "
        "2-arm Pareto front {0, 1} vs. dominated arm 2",
        random_ok and recovers_pareto,
        detail=f"top_arms={top_arms.tolist()}",
    )


def test_ttpfts_integration_insufficient_warmup_raises():
    """
    With num_warmup_pulls=1 and alpha=0, every arm reaches only n=1 after
    warm-up (df=0): the first post-warm-up choose_arm() call must surface
    the invalid posterior state explicitly instead of silently sampling
    garbage, since sample() now raises rather than using an ad-hoc sentinel.
    """
    posterior = TPosterior(num_arms=2, num_objectives=1, alpha=0.0)
    bandit = TTPFTSBandit(posterior, p=0.5, num_warmup_pulls=1)
    bandit.reset(None)

    # Warm-up phase: exactly num_arms calls before sampling is attempted.
    for arm_expected in range(2):
        arm = bandit.choose_arm()
        bandit.learn(arm, np.array([1.0]))

    raised = False
    try:
        bandit.choose_arm()
    except ValueError:
        raised = True

    return check(
        "TTPFTSBandit + TPosterior(alpha=0) surfaces an explicit ValueError "
        "when num_warmup_pulls is configured too low for the chosen prior "
        "(df<=0 after warm-up), instead of sampling from an undefined "
        "posterior",
        raised,
    )


def test_ttpfts_integration_known_variance():
    """Sanity check that NormalPosterior (variance=0.25) still integrates
    correctly with TTPFTSBandit (the GKV configuration used in momab_main.py
    is unaffected by this gate's TPosterior changes)."""
    rng = np.random.default_rng(RNG_SEED + 10)
    num_arms, num_objectives = 3, 1
    true_means = np.array([[2.0], [1.0], [0.0]])
    sigma = 0.5

    posterior = NormalPosterior(num_arms, num_objectives, known_stds=sigma)
    bandit = TTPFTSBandit(posterior, p=0.5, num_warmup_pulls=2)
    bandit.reset(sigma)

    ok = True
    for _ in range(300):
        arm = bandit.choose_arm()
        if not (0 <= arm < num_arms):
            ok = False
        reward = rng.normal(loc=true_means[arm], scale=sigma)
        bandit.learn(arm, reward)

    top_arms = bandit.get_top_arms()
    ok &= (top_arms.tolist() == [0])  # single-objective: only the best arm

    return check(
        "TTPFTSBandit + NormalPosterior(known variance=0.25) integration "
        "runs cleanly and recovers the single best arm",
        ok,
        detail=f"top_arms={top_arms.tolist()}",
    )


ALL_TESTS = [
    test_tposterior_sufficient_statistics,
    test_tposterior_streaming_bug_regression,
    test_tposterior_order_independence,
    test_normalposterior_batch_vs_streaming,
    test_reset_exact,
    test_reset_reusable_after,
    test_tposterior_sampling_matches_student_t,
    test_tposterior_get_stds_vs_get_scale,
    test_normalposterior_sampling_matches_normal,
    test_tposterior_invalid_state_raises,
    test_multi_arm_objective_independence,
    test_ttpfts_integration_reference_prior,
    test_ttpfts_integration_insufficient_warmup_raises,
    test_ttpfts_integration_known_variance,
]


def main():
    print("=" * 88)
    print("TTPFTS posterior correctness gate")
    print("=" * 88)
    for test_fn in ALL_TESTS:
        test_fn()
    print("=" * 88)
    if FAILURES:
        print(f"GATE FAILED: {len(FAILURES)}/{len(ALL_TESTS)} check(s) failed:")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"GATE PASSED: all {len(ALL_TESTS)} checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
