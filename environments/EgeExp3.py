import numpy as np
from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import std_from_variance

# Manuscript spec: Gaussian rewards with diagonal covariance sigma^2 I_D, sigma^2=0.25.
GAUSSIAN_VARIANCE = 0.25


CENTRE = (0.5, 0.5)
RADIUS = 0.35

NUM_PARETO_ARMS = 20
NUM_DOMINATED_ARMS = 180

# Pareto arc: 20 arms uniformly spaced on [15 deg, 75 deg].
PARETO_START_DEG = 15.0
PARETO_END_DEG = 75.0
PARETO_STEP_DEG = (PARETO_END_DEG - PARETO_START_DEG) / (NUM_PARETO_ARMS - 1)  # 60/19

# Admissible dominated-angle interval.
#
# All arms lie on a circle, so arm k has mean (cx + r cos t_k, cy + r sin t_k).
# Two arms therefore share objective 1 iff t_i = -t_j (mod 360 deg) and share
# objective 2 iff t_i = 180 deg - t_j (mod 360 deg): the objective values are
# tied exactly at the reflections of an angle about the x- and y-axis.
#
# A dominated arm at angle phi is *strictly* dominated by some Pareto arm iff
# there is a Pareto angle t with cos t > cos phi and sin t > sin phi, which
# holds iff
#     180 deg - PARETO_END_DEG  <  phi  <  360 deg - PARETO_START_DEG,
# i.e. phi lies strictly between the reflections of the two extreme Pareto
# angles. At either endpoint the arm is only *weakly* dominated: it shares one
# objective value with an extreme Pareto arm, so its gap Delta is exactly 0 and
# the problem complexity H(nu) = sum_a 1/Delta_a^2 diverges.
CRITICAL_LOW_DEG = 180.0 - PARETO_END_DEG      # 105 deg, reflection of 75 deg
CRITICAL_HIGH_DEG = 360.0 - PARETO_START_DEG   # 345 deg, reflection of 15 deg

# The original construction sampled the *closed* endpoint 345 deg, which is
# exactly CRITICAL_HIGH_DEG, creating the weakly dominated arm 199 (tied with
# Pareto arm 0 in objective 1) and hence H(nu) = infinity.
#
# Repair rule (applied once, deterministically, to both ends): inset the
# dominated arc from each critical boundary by half the Pareto angular step.
# Reflections of the Pareto arms form an arithmetic progression of step
# PARETO_STEP_DEG starting at each critical boundary, so an inset of
# PARETO_STEP_DEG / 2 places the endpoint exactly midway between the
# reflections of two consecutive Pareto arms -- the furthest any point can be
# from every Pareto reflection at the design's own angular resolution. The
# start of the arc is additionally kept at its original 120 deg, which already
# lies well inside the admissible interval.
ARC_INSET_DEG = PARETO_STEP_DEG / 2.0
DOMINATED_START_DEG = max(120.0, CRITICAL_LOW_DEG + ARC_INSET_DEG)   # 120 deg
DOMINATED_END_DEG = CRITICAL_HIGH_DEG - ARC_INSET_DEG                # 345 - 30/19 deg


def _circle_points(angles_deg: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(angles_deg)
    cx, cy = CENTRE
    return np.stack([cx + RADIUS * np.cos(radians),
                     cy + RADIUS * np.sin(radians)], axis=1)


def generate_angles() -> np.ndarray:
    """Return the 200 arm angles in degrees (20 Pareto, then 180 dominated)."""
    pareto = PARETO_START_DEG + np.arange(NUM_PARETO_ARMS) * PARETO_STEP_DEG
    dominated = DOMINATED_START_DEG + np.arange(NUM_DOMINATED_ARMS) * (
        (DOMINATED_END_DEG - DOMINATED_START_DEG) / (NUM_DOMINATED_ARMS - 1))
    return np.concatenate([pareto, dominated])


def generate_arms():
    """
    Generate the arms for the EgeExp3 environment.

    200 arms on a circle centred at (0.5, 0.5) with radius 0.35, so that all
    means lie within (0.15, 0.85). The first 20 arms (Pareto-optimal) span
    [15 deg, 75 deg]; the remaining 180 arms are uniformly spaced on
    [DOMINATED_START_DEG, DOMINATED_END_DEG], which lies strictly inside the
    admissible interval (CRITICAL_LOW_DEG, CRITICAL_HIGH_DEG) so that every
    dominated arm is strictly dominated, no two arms share an objective value,
    and H(nu) is finite.
    """
    return _circle_points(generate_angles())


class EgeExp3(BaseEnvironment):
    """
    Experiment 3: Many arms on a circle in the first quadrant.
    There are 200 arms and 2 objectives.
    Arms lie on a circle centred at (0.5, 0.5) with radius 0.35.
    The first 20 arms are Pareto-optimal (northeast arc), the remaining
    180 arms are spread over the strictly dominated portion of the circle.
    """

    def __init__(self, dist: str = "gaussian"):
        self.dist = dist
        pareto_indices = np.arange(NUM_PARETO_ARMS)  # The first 20 arms are Pareto optimal
        self._init_standard_2obj(generate_arms(), pareto_indices, dist=dist,
                                 gaussian_std=std_from_variance(GAUSSIAN_VARIANCE))
