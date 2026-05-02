import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp3 environment.

    200 arms on a circle centred at (0.5, 0.5) with radius 0.35,
    placing all means in the first quadrant within (0.15, 0.85).
    The first 20 arms (Pareto-optimal) span [π/12, 5π/12] (≈15°–75°),
    the remaining 180 arms span [2π/3, 11π/6] (≈120°–330°).
    """
    cx, cy, r = 0.5, 0.5, 0.35
    arms = []
    # Pareto arms: 20 arms spanning [15°, 75°]
    for i in range(20):
        angle = np.pi / 12 + i * (np.pi / 3) / 19
        arms.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))

    # Non-Pareto arms: 180 arms spanning [120°, 345°]
    start_angle = 2 * np.pi / 3        # 120°
    end_angle = 23 * np.pi / 12        # 345°
    for i in range(180):
        angle = start_angle + i * (end_angle - start_angle) / 179
        arms.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))

    return np.array(arms)


class EgeExp3(BaseEnvironment):
    """
    Experiment 3: Many arms on a circle in the first quadrant.
    There are 200 arms and 2 objectives.
    Arms lie on a circle centred at (0.5, 0.5) with radius 0.35.
    The first 20 arms are Pareto-optimal (northeast arc), the remaining
    180 arms are spread over the dominated portion of the circle.
    """

    def __init__(self, dist: str = "gaussian"):
        self.dist = dist
        pareto_indices = np.arange(20)  # The first 20 arms are Pareto optimal
        self._init_standard_2obj(generate_arms(), pareto_indices, dist=dist)
