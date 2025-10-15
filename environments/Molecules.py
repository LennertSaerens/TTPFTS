import matplotlib.pyplot as plt
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, QED, MolFromSmiles
from matplotlib.patches import Ellipse

from environments.BaseEnvironment import BaseEnvironment


def calc_esol(m):
    """Estimate solubility (logS) using ESOL-like model."""
    mw = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    rb = rdMolDescriptors.CalcNumRotatableBonds(m)
    ap = rdMolDescriptors.CalcFractionCSP3(m)
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


def sweetspot_esol(esol):
    """Gaussian reward centered around a target logS value."""
    esol_target = -3.0  # Target logS value
    deviation = 1  # Acceptable deviation from the target
    return np.exp(-0.5 * ((esol - esol_target) / deviation) ** 2)


def generate_arms(num_arms):
    """
    Generate the arms for the Molecules environment.
    :param num_arms: The number of arms to generate.
    :return: A list of arms.
    """
    # Download small drug-like molecules from ChEMBL
    molecule = new_client.molecule
    chembl_records = molecule.filter(molecule_properties__aromatic_rings__gte=1)[:num_arms]

    # Extract SMILES and ChEMBL IDs
    records = []
    for r in chembl_records:
        smiles = r.get('molecule_structures', {}).get('canonical_smiles', None)
        chembl_id = r.get('molecule_chembl_id', None)
        if smiles:
            records.append((chembl_id, smiles))

    arms = []

    for chembl_id, smiles in records:
        mol = MolFromSmiles(smiles)
        if mol is None:
            continue
        qed_score = QED.qed(mol)
        logS = sweetspot_esol(calc_esol(mol))
        arms.append((qed_score, logS))

    return np.array(arms)


class Molecules(BaseEnvironment):
    """
    Bandit setting based on a simplified Molecular Design problem.
    """

    def __init__(self, num_arms=100):
        self.arms = generate_arms(num_arms)
        self.stds = [0.1, 0.1]  # Standard deviation for the normal distribution
        is_strictly_worse = np.all(self.arms[:, None, :] < self.arms[None, :, :], axis=2)
        pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
        reference_point = np.array([1.0, 1.0])
        inverted_arms = [(1 - arm[0], 1 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 2, pareto_indices, inverted_arms, reference_point)

    def pull_arm(self, arm):
        """
        Pull the specified arm and return the reward.
        :param arm: The index of the arm to pull.
        :return: The reward for the pulled arm.
        """
        mu = self.arms[arm]
        return [np.random.normal(mu[i], self.stds[i]) for i in range(2)]

    def plot(self):
        """
        Plot the arms and the Pareto front.
        """
        plt.figure(figsize=(8, 6))

        plt.scatter(*zip(*self.arms), label='Molecules')
        plt.scatter(*zip(*[self.arms[i] for i in self.pareto_indices]), color='green', label='Pareto Optimal Molecules')

        # Draw ellipses around Pareto optimal arms
        for i in self.pareto_indices:
            ellipse = Ellipse(xy=self.arms[i], width=self.stds[0], height=self.stds[1], edgecolor='green', facecolor='none', alpha=0.5)
            plt.gca().add_patch(ellipse)

        plt.xlabel('QED Score')
        plt.ylabel('Solubility Score')
        plt.title('Molecules Environment')
        plt.legend()
        plt.grid()
        plt.savefig(f'environments/plots/Molecules_{self.num_arms}.pdf', format='pdf')
        plt.show()
