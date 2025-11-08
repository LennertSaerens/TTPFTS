import array

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

from matplotlib.patches import Ellipse, Patch

# Increase the font size of the plots
plt.rcParams.update({'font.size': 17})
# Change the font to a fancy serif font for use in a latex document
plt.rcParams.update({'font.family': 'serif'})
# plt.rcParams['figure.constrained_layout.use'] = True

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray",
          "tab:olive", "tab:cyan", 'black', 'indianred', 'lightcoral', 'moccasin', 'palegoldenrod', 'lemonchiffon',
          'palegreen', 'lightcyan',
          'paleturquoise', 'darkseagreen', 'lightskyblue', "palevioletred", "pink", "lavenderblush"]


def plot_vaccination_data(df, optimal_arms, annotate=False, connect_optimal_arms=False):
    plt.rcParams.update({'font.size': 13})
    plt.rcParams.update({'font.family': 'serif'})
    # Define colors for groups of points
    colors = ['black', 'indianred', 'lightcoral', 'moccasin', 'palegoldenrod', 'lemonchiffon', 'palegreen', 'lightcyan',
              'paleturquoise', 'darkseagreen', 'lightskyblue', "palevioletred", "pink", "lavenderblush"]
    color_index = 0
    # Get the labels for the legend from the 'Description' column
    labels = df['Description'].unique()
    # Delete the NAN label
    labels = labels[~pd.isnull(labels)]
    legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]

    # Plot the first point in black
    plt.scatter(df['Medical Burden'].iloc[0], df['Monetary Cost'].iloc[0], color="black")

    # Plot the rest of the points in groups, changing colors every 4 points
    for i in range(1, len(df)):
        if i % 4 == 1:
            color_index += 1
        # annotate the points with their index if the annotate flag is set
        if annotate:
            plt.annotate(i, (df['Medical Burden'].iloc[i], df['Monetary Cost'].iloc[i]))
        plt.scatter(df['Medical Burden'].iloc[i], df['Monetary Cost'].iloc[i], color=colors[color_index])

    # Connect each optimal arm with the next one if the connect_optimal_arms flag is set
    if connect_optimal_arms:
        for i in range(len(optimal_arms) - 1):
            plt.plot([df['Medical Burden'].iloc[optimal_arms[i]], df['Medical Burden'].iloc[optimal_arms[i + 1]]],
                     [df['Monetary Cost'].iloc[optimal_arms[i]], df['Monetary Cost'].iloc[optimal_arms[i + 1]]],
                     color='black', linestyle='dotted')

    plt.xlabel('Medical Burden')
    plt.ylabel('Monetary Cost')
    plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    plt.show()


def plot_regrets(setup_dict):
    """
    Plot the evolution of the cumulative pareto regrets and the cumulative unfairness regrets for each of the different
    algorithms in the experimental setup in a single figure with two sub figures that sit side by side.
    The x-axis represents the time steps and the y-axis represents the cumulative regrets. The regrets are averaged over the experiments.
    :param setup_dict: The experimental setup dictionary.
    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for algorithm in setup_dict:
        cumulative_pareto_regrets = setup_dict[algorithm]["cumulative_pareto_regrets"]
        avg_cumulative_pareto_regrets = np.mean(cumulative_pareto_regrets, axis=0)
        std_cumulative_pareto_regrets = np.std(cumulative_pareto_regrets, axis=0)
        cumulative_unfairness_regrets = setup_dict[algorithm]["cumulative_unfairness_regrets"]
        avg_cumulative_unfairness_regrets = np.mean(cumulative_unfairness_regrets, axis=0)
        std_cumulative_unfairness_regrets = np.std(cumulative_unfairness_regrets, axis=0)
        axs[0].plot(avg_cumulative_pareto_regrets, label=f"{algorithm}")
        # Plot the 95% confidence interval for the cumulative pareto regrets
        axs[0].fill_between(range(len(avg_cumulative_pareto_regrets)),
                            avg_cumulative_pareto_regrets - 1.96 * std_cumulative_pareto_regrets / np.sqrt(
                                len(cumulative_pareto_regrets)),
                            avg_cumulative_pareto_regrets + 1.96 * std_cumulative_pareto_regrets / np.sqrt(
                                len(cumulative_pareto_regrets)),
                            alpha=0.2)
        axs[1].plot(avg_cumulative_unfairness_regrets, label=f"{algorithm}")
        # Plot the 95% confidence interval for the cumulative unfairness regrets
        axs[1].fill_between(range(len(avg_cumulative_unfairness_regrets)),
                            avg_cumulative_unfairness_regrets - 1.96 * std_cumulative_unfairness_regrets / np.sqrt(
                                len(cumulative_unfairness_regrets)),
                            avg_cumulative_unfairness_regrets + 1.96 * std_cumulative_unfairness_regrets / np.sqrt(
                                len(cumulative_unfairness_regrets)),
                            alpha=0.2)
    axs[0].set_title("Cumulative Pareto Regrets")
    axs[0].set_xlabel("Time steps")
    axs[0].set_ylabel("Cumulative Pareto Regret")
    axs[0].legend()
    axs[1].set_title("Cumulative Unfairness Regrets")
    axs[1].set_xlabel("Time steps")
    axs[1].set_ylabel("Cumulative Unfairness Regret")
    axs[1].legend()
    plt.show()


def plot_arms_pareto_front(arms, pareto_indices, plot_stds=False):
    """
    Plot the arms in the 2D objective space and highlight the Pareto front in the plot by plotting the Pareto optimal arms in a different color.
    If plot_stds is set to True, the standard deviations of the arms are also plotted as shaded ellipse around the mean.
    :param arms: The means of the arms for each objective.
    :param pareto_indices: The indices of the Pareto optimal arms.
    :param plot_stds: Whether to plot the standard deviations of the arms as well.
    :return: None
    """
    plt.scatter(arms[:, 0], arms[:, 2], color='red')
    # Annotate the arms with their index at an offset
    for i in range(len(arms)):
        plt.annotate(i, (arms[i, 0], arms[i, 2]), textcoords="offset points", xytext=(0, 5), ha='center')
    for pareto_index in pareto_indices:
        plt.scatter(arms[pareto_index, 0], arms[pareto_index, 2], color='green')
    if plot_stds:
        for arm in arms:
            ellipse = Ellipse((arm[0], arm[2]), width=arm[1], height=arm[3], alpha=0.05)
            plt.gca().add_patch(ellipse)
    plt.xlabel("Hospitalizations")
    plt.ylabel("Costs")
    plt.title("Arms in the 2D objective space")
    plt.show()


def plot_arms_PFI_setting(arms, pareto_indices, std, plot_stds=True, reference_point=None):
    """
    Create a scatter plot of the arms. Pareto optimal arms are plotted in green, others in blue. The standard deviation is plotted as an ellipse around the mean.
    :param reference_point:  The reference point for the hypervolume calculation.
    :param arms: The list of arms.
    :param pareto_indices: The indices of the Pareto optimal arms.
    :param std: The standard deviation of the arms.
    :param plot_stds: Whether to plot the standard deviations as ellipses around the means.
    :return: None
    """
    plt.scatter([arm[0] for arm in arms], [arm[1] for arm in arms], color='tab:blue')
    for pareto_index in pareto_indices:
        plt.scatter(arms[pareto_index][0], arms[pareto_index][1], color='tab:green')

    if plot_stds:
        for arm in arms:
            ellipse = Ellipse(arm, width=2 * std, height=2 * std, alpha=0.05)
            plt.gca().add_patch(ellipse)

    if reference_point is not None:
        plt.scatter(reference_point[0], reference_point[1], color='tab:orange')

    legend_patches = [Patch(color='tab:blue', label='Suboptimal arm'),
                      Patch(color='tab:green', label='Pareto optimal arm'),
                      Patch(color='tab:orange', label='Reference point')]
    plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.show()


def plot_arm_pulls(setup, optimal_arms, total_pulls, show_arm_idxs=False):
    """
    Plot the frequency of pulling each arm for each algorithm in the experimental setup. Each algorithm has its own subplot within the big plot with 4 rows and 2 columns.
    Inside each subplot, the number of times the algorithm pulled each arm is plotted as a bar for each arm. Pareto optimal arms are highlighted in a different color.
    All other arms have the same color.
    :param total_pulls: The total number of times an arm was pulled for each algorithm. Used for frequency calculation.
    :param optimal_arms: The indices of the Pareto optimal arms.
    :param setup: The experimental setup dictionary.
    :param show_arm_idxs: Whether to show the arm indices on the x-axis.
    :return: None
    """
    fig, axs = plt.subplots(4, 2, figsize=(20, 20))
    for i, algorithm in enumerate(setup):
        ax = axs[i // 2, i % 2]
        arm_pulls = setup[algorithm]["arm_pulls"]
        avg_arm_pulls = np.mean(arm_pulls, axis=0) / total_pulls
        std_arm_pulls = np.std(arm_pulls, axis=0) / total_pulls
        ax.bar(range(len(avg_arm_pulls)), avg_arm_pulls, yerr=1.96 * std_arm_pulls / np.sqrt(len(arm_pulls)))
        ax.set_title(f"{algorithm}")
        if show_arm_idxs:
            ax.set_xticks(range(len(avg_arm_pulls)))
        # Highlight the Pareto optimal arms in the plot
        for optimal_arm in optimal_arms:
            ax.get_children()[optimal_arm].set_color('green')
    # Show 'Frequency' on the y-axis of all plots in the first column
    for i in range(4):
        axs[i, 0].set_ylabel("Frequency")
    # Show 'Arm index' on the x-axis of all plots in the bottom row
    axs[3, 0].set_xlabel("Arm index")
    axs[3, 1].set_xlabel("Arm index")
    plt.show()


def plot_arm_pulls_2(setup, optimal_arms, total_pulls):
    """
    Plot the frequency of pulling each arm for each algorithm in the experimental setup. Each algorithm has its own subplot within the big plot with 2 rows and 4 columns.
    Inside each subplot, the number of times the algorithm pulled each arm is plotted as a bar for each arm. Pareto optimal arms are highlighted in a different color.
    All other arms have the same color.
    :param total_pulls: The total number of times an arm was pulled for each algorithm. Used for frequency calculation.
    :param optimal_arms: The indices of the Pareto optimal arms.
    :param setup: The experimental setup dictionary.
    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for i, algorithm in enumerate(setup):
        ax = axs[i]
        arm_pulls = setup[algorithm]["arm_pulls"]
        avg_arm_pulls = np.mean(arm_pulls, axis=0) / total_pulls
        std_arm_pulls = np.std(arm_pulls, axis=0) / total_pulls
        ax.bar(range(len(avg_arm_pulls)), avg_arm_pulls, yerr=1.96 * std_arm_pulls / np.sqrt(len(arm_pulls)))
        ax.set_title(f"{algorithm}")
        ax.set_xticks(range(len(avg_arm_pulls)))
        # Highlight the Pareto optimal arms in the plot
        for optimal_arm in optimal_arms:
            ax.get_children()[optimal_arm].set_color('green')
    # Show 'Frequency' on the y-axis of all plots
    for i in range(2):
        axs[i].set_ylabel("Frequency")
    # Show 'Arm index' on the x-axis of all plots
    for i in range(2):
        axs[i].set_xlabel("Arm index")
    plt.show()


def plot_arm_pulls_4(setup, optimal_arms, total_pulls):
    """
    Plot the frequency of pulling each arm for each algorithm in the experimental setup. Each algorithm has its own subplot within the big plot with 2 rows and 4 columns.
    Inside each subplot, the number of times the algorithm pulled each arm is plotted as a bar for each arm. Pareto optimal arms are highlighted in a different color.
    All other arms have the same color.
    :param total_pulls: The total number of times an arm was pulled for each algorithm. Used for frequency calculation.
    :param optimal_arms: The indices of the Pareto optimal arms.
    :param setup: The experimental setup dictionary.
    :return: None
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    for i, algorithm in enumerate(setup):
        ax = axs[i // 2, i % 2]
        arm_pulls = setup[algorithm]["arm_pulls"]
        avg_arm_pulls = np.mean(arm_pulls, axis=0) / total_pulls
        std_arm_pulls = np.std(arm_pulls, axis=0) / total_pulls
        ax.bar(range(len(avg_arm_pulls)), avg_arm_pulls, yerr=1.96 * std_arm_pulls / np.sqrt(len(arm_pulls)))
        ax.set_title(f"{algorithm}")
        ax.set_xticks(range(len(avg_arm_pulls)))
        # Highlight the Pareto optimal arms in the plot
        for optimal_arm in optimal_arms:
            ax.get_children()[optimal_arm].set_color('green')
    # Show 'Frequency' on the y-axis of all plots in the first column
    for i in range(2):
        axs[i, 0].set_ylabel("Frequency")
    # Show 'Arm index' on the x-axis of all plots in the second row
    for i in range(2):
        axs[1, i].set_xlabel("Arm index")
    plt.show()


def plot_arm_pulls_single(setup, algorithm_name, optimal_arms, total_pulls):
    """
    Plot the frequency of pulling each arm for a single algorithm in the experimental setup.
    The number of times the algorithm pulled each arm is plotted as a bar for each arm. Pareto optimal arms are highlighted in a different color.
    All other arms have the same color.
    :param setup: The experimental setup dictionary.
    :param algorithm_name: The name of the algorithm to plot.
    :param optimal_arms: The indices of the Pareto optimal arms.
    :param total_pulls: The total number of times an arm was pulled for the algorithm. Used for frequency calculation.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    arm_pulls = setup[algorithm_name]["arm_pulls"]
    avg_arm_pulls = np.mean(arm_pulls, axis=0) / total_pulls
    std_arm_pulls = np.std(arm_pulls, axis=0) / total_pulls
    ax.bar(range(len(avg_arm_pulls)), avg_arm_pulls, yerr=1.96 * std_arm_pulls / np.sqrt(len(arm_pulls)))
    ax.set_title(f"{algorithm_name}")
    ax.set_xticks(range(len(avg_arm_pulls)))
    # Highlight the Pareto optimal arms in the plot
    for optimal_arm in optimal_arms:
        ax.get_children()[optimal_arm].set_color('green')
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Arm index")
    plt.show()


def plot_bernoulli_metric(file, num_runs, num_arm_pulls, rolling_avg_window=1, plot_std=False, ege_sr_file=None):
    """
    Plot the evolution of the Bernoulli metric. The Bernoulli metric is the fraction of times the algorithm pulls a Pareto optimal arm.
    The x-axis represents the time steps and the y-axis represents the Bernoulli metric. The metric is averaged over the experiments.
    :param ege_sr_file: File containing the results of the EGE-SR algorithm.
    :param plot_std: Whether to plot the standard deviation of the Bernoulli metric.
    :param file: The file containing the experimental results.
    :param num_runs: The number of runs of the experiment.
    :param num_arm_pulls: The number of arm pulls in each run of the experiment.
    :param rolling_avg_window: The window size for the optional rolling average.
    :return: None
    """
    result_df = pd.read_csv(file, header=None)
    algorithm_names = result_df[0].unique()
    num_algorithms = len(algorithm_names)
    bernoulli_metrics = result_df.values[:, 3].reshape(num_algorithms, num_runs, num_arm_pulls).astype(np.float64)
    avg_bernoulli_metrics = np.mean(bernoulli_metrics, axis=1)
    std_bernoulli_metrics = np.std(bernoulli_metrics, axis=1)

    plt.figure(figsize=(8, 6))

    # algorithm_names = ["MOMAB1", "MOMAB2", "MOMAB3", "MOMAB4", "Nieuwe MOMAB"]

    for i, name in enumerate(algorithm_names):
        if rolling_avg_window > 1:
            rolling_avg = pd.Series(avg_bernoulli_metrics[i]).rolling(window=rolling_avg_window).mean()
            plt.plot(rolling_avg, label=f"{name}", color=colors[i])
            # plt.plot(avg_bernoulli_metrics[i], alpha=0.2, color=colors[i])
        else:
            plt.plot(avg_bernoulli_metrics[i], label=f"{name}")
        if plot_std:
            plt.fill_between(range(len(avg_bernoulli_metrics[i])),
                             avg_bernoulli_metrics[i] - 1.96 * std_bernoulli_metrics[i] / np.sqrt(num_runs),
                             avg_bernoulli_metrics[i] + 1.96 * std_bernoulli_metrics[i] / np.sqrt(num_runs),
                             alpha=0.2, color=colors[i])

    if ege_sr_file is not None:
        # load the ege_sr results
        ege_sr_df = pd.read_csv(ege_sr_file, header=None)
        # Extract columns 2 and 4 from the df
        ege_sr_bernoulli_timesteps = ege_sr_df.values[:, 2]
        steps_per_run = int(len(ege_sr_bernoulli_timesteps) / num_runs)
        ege_sr_bernoulli_timesteps = ege_sr_bernoulli_timesteps[:steps_per_run]
        ege_sr_bernoulli_metrics = ege_sr_df.values[:, 3].astype(np.float64).reshape(num_runs, steps_per_run).mean(axis=0)
        plt.plot(ege_sr_bernoulli_timesteps, ege_sr_bernoulli_metrics, label="EGE_SR", color="tab:gray")

    plt.ylim(0, 1)
    plt.xlabel("Arm pulls")
    plt.ylabel("Bernoulli metric")
    plt.legend()
    plt.savefig(f"{file}_bernoulli.pdf", format="pdf")
    plt.show()


def plot_jaccard_metric(file, num_runs, num_arm_pulls, rolling_avg_window=1, plot_std=False, ege_sr_file=None):
    """
    Plot the evolution of the Jaccard metric. The Jaccard metric is the Jaccard similarity between the set of Pareto optimal arms and the set of arms recommended by the algorithm.
    The x-axis represents the time steps and the y-axis represents the Jaccard metric. The metric is averaged over the experiments.
    :param ege_sr_file: File containing the results of the EGE-SR algorithm.
    :param plot_std: Whether to plot the standard deviation of the Jaccard metric.
    :param file: The file containing the experimental results.
    :param num_runs: The number of runs of the experiment.
    :param num_arm_pulls: The number of arm pulls in each run of the experiment.
    :param rolling_avg_window: The window size for the optional rolling average.
    :return: None
    """
    result_df = pd.read_csv(file, header=None)
    algorithm_names = result_df[0].unique()
    num_algorithms = len(algorithm_names)
    jaccard_metrics = result_df.values[:, 4].reshape(num_algorithms, num_runs, num_arm_pulls).astype(np.float64)
    avg_jaccard_metrics = np.mean(jaccard_metrics, axis=1)
    std_jaccard_metrics = np.std(jaccard_metrics, axis=1)

    plt.figure(figsize=(8, 6))

    for i, name in enumerate(algorithm_names):
        rolling_avg = pd.Series(avg_jaccard_metrics[i]).rolling(window=rolling_avg_window).mean()
        if rolling_avg_window > 1:
            plt.plot(rolling_avg, label=f"{name}", color=colors[i])
            plt.plot(avg_jaccard_metrics[i], alpha=0.2, color=colors[i])
        else:
            plt.plot(avg_jaccard_metrics[i], label=f"{name}")
        if plot_std:
            plt.fill_between(range(len(avg_jaccard_metrics[i])),
                             avg_jaccard_metrics[i] - 1.96 * std_jaccard_metrics[i] / np.sqrt(num_runs),
                             avg_jaccard_metrics[i] + 1.96 * std_jaccard_metrics[i] / np.sqrt(num_runs),
                             alpha=0.2, color=colors[i])

    if ege_sr_file is not None:
        # load the ege_sr results
        ege_sr_df = pd.read_csv(ege_sr_file, header=None)
        # Extract columns 2 and 4 from the df
        ege_sr_bernoulli_timesteps = ege_sr_df.values[:, 2]
        steps_per_run = int(len(ege_sr_bernoulli_timesteps) / num_runs)
        ege_sr_bernoulli_timesteps = ege_sr_bernoulli_timesteps[:steps_per_run]
        ege_sr_bernoulli_metrics = ege_sr_df.values[:, 4].astype(np.float64).reshape(num_runs, steps_per_run).mean(axis=0)
        plt.plot(ege_sr_bernoulli_timesteps, ege_sr_bernoulli_metrics, label="EGE_SR", color="tab:gray")

    plt.ylim(0, 1)
    plt.xlabel("Time steps")
    plt.ylabel("Jaccard metric")
    plt.legend(loc='lower right')
    plt.savefig(f"{file}_jaccard.pdf", format="pdf")
    plt.show()


def plot_hypervolume(file, num_runs, num_arm_pulls, y_low_lim, y_up_lim, rolling_avg_window=1, plot_std=False, baseline=None):
    """
    Plot the evolution of the hypervolume metric. The hypervolume metric is the hypervolume of the arms recommended by the algorithm.
    The x-axis represents the time steps and the y-axis represents the hypervolume metric. The metric is averaged over the experiments.
    :param y_up_lim: Upper limit of the y-axis.
    :param y_low_lim: Lower limit of the y-axis.
    :param baseline: File containing the baseline results.
    :param plot_std: Whether to plot the standard deviation of the hypervolume metric.
    :param file: The file containing the experimental results.
    :param num_runs: The number of runs of the experiment.
    :param num_arm_pulls: The number of arm pulls in each run of the experiment.
    :param rolling_avg_window: The window size for the optional rolling average.
    :return: None
    """
    result_df = pd.read_csv(file, header=None)
    algorithm_names = result_df[0].unique()
    num_algorithms = len(algorithm_names)
    hypervolumes = result_df.values[:, 5].reshape(num_algorithms, num_runs, num_arm_pulls).astype(np.float64)
    avg_hypervolumes = np.mean(hypervolumes, axis=1)
    std_hyper_volumes = np.std(hypervolumes, axis=1)

    plt.figure(figsize=(8, 6))

    for i, name in enumerate(algorithm_names):
        rolling_avg = pd.Series(avg_hypervolumes[i]).rolling(window=rolling_avg_window).mean()
        if rolling_avg_window > 1:
            plt.plot(rolling_avg, label=f"{name}", color=colors[i])
            plt.plot(avg_hypervolumes[i], alpha=0.2, color=colors[i])
        else:
            plt.plot(avg_hypervolumes[i], label=f"{name}")
        if plot_std:
            plt.fill_between(range(len(avg_hypervolumes[i])),
                             avg_hypervolumes[i] - 1.96 * std_hyper_volumes[i] / np.sqrt(num_runs),
                             avg_hypervolumes[i] + 1.96 * std_hyper_volumes[i] / np.sqrt(num_runs),
                             alpha=0.2, color=colors[i])

    if baseline is not None:
        baseline_df = pd.read_csv(baseline, header=None)
        baseline_hypervolumes = baseline_df.values[:, 5].reshape(1, num_runs, num_arm_pulls).astype(np.float64)
        avg_baseline_hypervolumes = np.mean(baseline_hypervolumes, axis=1)
        std_baseline_hypervolumes = np.std(baseline_hypervolumes, axis=1)
        rolling_avg = pd.Series(avg_baseline_hypervolumes[0]).rolling(window=rolling_avg_window).mean()
        if rolling_avg_window > 1:
            plt.plot(rolling_avg, label="Uniform Sampling", color=colors[-1])
            plt.plot(avg_baseline_hypervolumes[0], alpha=0.2, color=colors[-1])
        else:
            plt.plot(avg_baseline_hypervolumes[0], label="Uniform Sampling")
        if plot_std:
            plt.fill_between(range(len(avg_baseline_hypervolumes[0])),
                             avg_baseline_hypervolumes[0] - 1.96 * std_baseline_hypervolumes[0] / np.sqrt(num_runs),
                             avg_baseline_hypervolumes[0] + 1.96 * std_baseline_hypervolumes[0] / np.sqrt(num_runs),
                             alpha=0.2, color="tab:gray")

    plt.ylim(y_low_lim, y_up_lim)

    plt.xlabel("Time steps")
    plt.ylabel("Hypervolume metric")
    plt.legend()
    plt.savefig(f"{file}_hypervolume.pdf", format="pdf")
    plt.show()


def plot_mis_id_metric(file, num_runs, num_arm_pulls, rolling_avg_window=1, plot_std=False, ege_sr_file=None):
    """
    Plot the evolution of the mis-identification metric.
    The x-axis represents the time steps and the y-axis represents the mis-identification metric. The metric is averaged over the experiments.
    :param ege_sr_file: File containing the results of the EGE-SR algorithm.
    :param plot_std: Whether to plot the standard deviation of the mis-identification metric.
    :param file: The file containing the experimental results.
    :param num_runs: The number of runs of the experiment.
    :param num_arm_pulls: The number of arm pulls in each run of the experiment.
    :param rolling_avg_window: The window size for the optional rolling average.
    :return: None
    """
    result_df = pd.read_csv(file, header=None)
    algorithm_names = result_df[0].unique()
    num_algorithms = len(algorithm_names)
    mis_id_metrics = result_df.values[:, 5].reshape(num_algorithms, num_runs, num_arm_pulls).astype(np.float64)
    avg_mis_id_metrics = np.log10(np.mean(mis_id_metrics, axis=1))
    std_mis_id_metrics = np.std(mis_id_metrics, axis=1)

    plt.figure(figsize=(8, 6))

    for i, name in enumerate(algorithm_names):
        rolling_avg = pd.Series(avg_mis_id_metrics[i]).rolling(window=rolling_avg_window).mean()
        if rolling_avg_window > 1:
            plt.plot(rolling_avg, label=f"{name}", color=colors[i])
            plt.plot(avg_mis_id_metrics[i], alpha=0.2, color=colors[i])
        else:
            plt.plot(avg_mis_id_metrics[i], label=f"{name}")
        if plot_std:
            plt.fill_between(range(len(avg_mis_id_metrics[i])),
                             avg_mis_id_metrics[i] - 1.96 * std_mis_id_metrics[i] / np.sqrt(num_runs),
                             avg_mis_id_metrics[i] + 1.96 * std_mis_id_metrics[i] / np.sqrt(num_runs),
                             alpha=0.2, color=colors[i])

    if ege_sr_file is not None:
        # load the ege_sr results
        ege_sr_df = pd.read_csv(ege_sr_file, header=None)
        ege_sr_bernoulli_timesteps = ege_sr_df.values[:, 2]
        steps_per_run = int(len(ege_sr_bernoulli_timesteps) / num_runs)
        ege_sr_bernoulli_timesteps = ege_sr_bernoulli_timesteps[:steps_per_run]
        ege_sr_bernoulli_metrics = ege_sr_df.values[:, 5].astype(np.float64).reshape(num_runs, steps_per_run).mean(axis=0)
        plt.plot(ege_sr_bernoulli_timesteps, ege_sr_bernoulli_metrics, label="EGE_SR", color="tab:gray")

    plt.xlabel("Time steps")
    plt.ylabel("Mis-identification metric")
    plt.legend(loc='best')
    plt.savefig(f"{file}_mis_id.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_arm_pull_frequencies(file, num_runs, num_arm_pulls, optimal_arms, num_arms, algorithm, df_arm_idx):
    result_df = pd.read_csv(file, header=None)
    algorithm_results = result_df[result_df[0] == algorithm]
    arm_pulled = algorithm_results.values[:, df_arm_idx].reshape(num_runs, num_arm_pulls)
    pulls_per_arm_per_run = np.zeros((num_runs, num_arms))
    for i in range(num_runs):
        for j in range(num_arm_pulls):
            pulls_per_arm_per_run[i, arm_pulled[i, j]] += 1
    avg_pulls_per_arm = np.mean(pulls_per_arm_per_run, axis=0) / num_arm_pulls
    std_pulls_per_arm = np.std(pulls_per_arm_per_run, axis=0) / num_arm_pulls
    bars = plt.bar(range(num_arms), avg_pulls_per_arm, yerr=1.96 * std_pulls_per_arm / np.sqrt(num_runs))
    for i in optimal_arms:
        bars[i].set_color('green')
    plt.xticks(range(0, num_arms, 5))
    plt.xlabel("Arm index")
    plt.ylabel("Pull frequency")
    plt.show()


def plot_arm_rec_frequencies(file, num_runs, num_arm_pulls, optimal_arms, num_arms, algorithm):
    result_df = pd.read_csv(file, header=None, low_memory=False)
    algorithm_results = result_df[result_df[0] == algorithm]
    arms_recommended = algorithm_results.values[:, 1].reshape(num_runs, num_arm_pulls)
    recommendations_per_arm_per_run = np.zeros((num_runs, num_arms))
    for i in range(num_runs):
        for j in range(num_arm_pulls):
            rec = arms_recommended[i, j][1:-1].split()
            for arm in rec:
                recommendations_per_arm_per_run[i, int(arm)] += 1
    recommendations_per_run = np.sum(recommendations_per_arm_per_run, axis=1)
    avg_recommendations_per_run = np.mean(recommendations_per_run)
    avg_recommendations_per_arm = np.mean(recommendations_per_arm_per_run, axis=0) / avg_recommendations_per_run
    std_recommendations_per_arm = np.std(recommendations_per_arm_per_run, axis=0) / avg_recommendations_per_run
    bars = plt.bar(range(num_arms), avg_recommendations_per_arm,
                   yerr=1.96 * std_recommendations_per_arm / np.sqrt(num_runs))
    for i in optimal_arms:
        bars[i].set_color('green')
    plt.xticks(range(0, num_arms, 5))
    plt.xlabel("Arm index")
    plt.ylabel("Recommendation frequency")
    plt.show()


def plot_bernoulli_metric_coarse(file, num_runs, plot_std=False):
    """
    Plot the evolution of the Bernoulli metric with coarse time steps.
    :param plot_std: Whether to plot the standard deviation of the Bernoulli metric.
    :param file: The file containing the experimental results.
    :param num_runs: The number of runs of the experiment.
    :return: None
    """
    result_df = pd.read_csv(file, header=None)
    algorithm_names = result_df[0].unique()
    num_algorithms = len(algorithm_names)
    all_steps = result_df.values[:, 2]
    steps_per_run = int(len(all_steps) / (num_runs * num_algorithms))
    bernoulli_metrics = result_df.values[:, 3].reshape(num_algorithms, num_runs, steps_per_run).astype(np.float64)
    all_steps = all_steps[:steps_per_run]
    avg_bernoulli_metrics = np.mean(bernoulli_metrics, axis=1)
    std_bernoulli_metrics = np.std(bernoulli_metrics, axis=1)
    yerr = 1.96 * std_bernoulli_metrics / np.sqrt(num_runs)

    plt.figure(figsize=(8, 6))

    # algorithm_names = ["MOMAB1", "MOMAB2", "MOMAB3", "MOMAB4", "Nieuwe MOMAB"]

    for i, name in enumerate(algorithm_names):
        plt.errorbar(all_steps, avg_bernoulli_metrics[i], yerr[i], label=f"{name}")

    plt.ylim(0, 1)
    plt.xlabel("Arm pulls")
    plt.ylabel("Bernoulli metric")
    plt.legend()
    plt.savefig(f"{file}_bernoulli.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    # plot_bernoulli_metric_coarse("results/EGEvsTTPFTSvsPUCB1vsUniform_Coarse_EgeExp1.csv", 100, plot_std=True)
    plot_bernoulli_metric("results/TTPFTS_NIGvsT_posteriors_EgeExp1.csv", 100, 5001, plot_std=True, ege_sr_file=None, rolling_avg_window=1)
    plot_jaccard_metric("results/TTPFTS_NIGvsT_posteriors_EgeExp1.csv", 100, 5001, plot_std=True, ege_sr_file=None, rolling_avg_window=1)
    plot_mis_id_metric("results/TTPFTS_NIGvsT_posteriors_EgeExp1.csv", 100, 5001, plot_std=True, ege_sr_file=None, rolling_avg_window=1)
    # plot_arm_rec_frequencies("results/baseline_recs.csv", 100, 30_000, [0, 5, 6, 8, 14, 30, 31, 32], 53, "Uniform Sampling")
    # plot_arm_pull_frequencies("results/bandits/test2.csv", 100, 250_000, [0, 1, 2, 3], 20,
    #                           "Linear Scalarized Knowledge Gradient (objectives)", 3)
