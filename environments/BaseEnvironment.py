from abc import ABC, abstractmethod
import numpy as np
from pymoo.indicators.hv import HV


class BaseEnvironment(ABC):
    def __init__(self, num_arms: int, num_objectives: int, pareto_indices, inverted_arms, reference_point) -> None:
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.pareto_indices = pareto_indices
        self.inverted_arms = inverted_arms
        self.reference_point = reference_point

    @abstractmethod
    def pull_arm(self, arm: int) -> np.ndarray:
        """Pulls the specified arm and returns the reward."""
        return NotImplementedError

    def bernoulli_metric(self, recommendation):
        """
        Calculate the Bernoulli metric for the specified arm.
        :param recommendation: The recommended arms.
        :return: The Bernoulli metric for the arm.
        """
        return int(set(recommendation) == set(self.pareto_indices))

    def jaccard_metric(self, recommendation):
        """
        Calculate the Jaccard similarity between the recommended arms and the pareto optimal arms.
        :param recommendation: The recommended arms.
        :return: The Jaccard similarity.
        """
        return len(set(recommendation).intersection(set(self.pareto_indices))) / len(
            set(recommendation).union(set(self.pareto_indices)))

    def hypervolume_metric(self, recommendation):
        """
        Calculate the hypervolume of the recommended arms using pymoo.
        :param recommendation: The recommended arms.
        :return: The hypervolume.
        """
        if len(recommendation) == 0:
            return 0
        F = np.array([self.inverted_arms[arm] for arm in recommendation])
        ind = HV(ref_point=self.reference_point)
        hv = ind.do(F)
        return hv

    def mis_id_metric(self, recommendation):
        """
        Calculate the average mis-identification rat over all arms.
        :param recommendation: The recommended arms.
        :return: The average mis-identification rate.
        """
        mis_identifications = 0
        for arm in np.arange(self.num_arms):
            if arm in recommendation:
                # If the arm is recommended, check if it is a Pareto optimal arm
                if arm not in self.pareto_indices:
                    mis_identifications += 1
            else:
                # If the arm is not recommended, check if it is a Pareto optimal arm
                if arm in self.pareto_indices:
                    mis_identifications += 1
        return mis_identifications / self.num_arms

    @abstractmethod
    def plot(self):
        """Plot the arms and the Pareto front."""
        return NotImplementedError

    def reset(self) -> None:
        """Resets the environment."""
        pass
