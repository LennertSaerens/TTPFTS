from abc import ABC, abstractmethod

import numpy as np


class BaseMOMABAlgorithm(ABC):
    def __init__(self, num_arms: int, num_objectives: int) -> None:
        self.num_arms = num_arms
        self.num_objectives = num_objectives

    @abstractmethod
    def choose_arm(self) -> int:
        """Selects an arm to pull."""
        pass

    @abstractmethod
    def get_top_arms(self) -> np.ndarray:
        """Returns the arms considered Pareto optimal."""
        pass

    @abstractmethod
    def learn(self, arm: int, reward: np.ndarray) -> None:
        """Updates the model with the observed reward from the chosen arm."""
        pass

    @abstractmethod
    def reset(self, env_stds: np.ndarray) -> None:
        """Resets the internal state of the algorithm."""
        pass
