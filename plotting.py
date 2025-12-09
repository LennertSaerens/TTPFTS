import array

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from scipy.stats import multivariate_normal
import seaborn as sns
from uncertainty_quantification import uncertainty_quantification
from matplotlib.ticker import ScalarFormatter, NullFormatter

from matplotlib.patches import Ellipse, Patch

# Increase the font size of the plots
plt.rcParams.update({'font.size': 14})
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


def plot_pfi_metric(file, num_runs, num_arm_pulls, metric, rolling_avg_window=1, plot_std=True, save_pdf=False, step=1):
    """
    The x-axis represents the arm pulls and the y-axis represents the metric. The metric is averaged over the experiments.
    :param metric: Which PFI metric to plot.
    :param save_pdf: Whether to save the plot as a pdf.
    :param plot_std: Whether to plot the standard deviation of the Bernoulli metric.
    :param file: The file containing the experimental results.
    :param num_runs: The number of runs of the experiment.
    :param num_arm_pulls: The number of arm pulls in each run of the experiment.
    :param rolling_avg_window: The window size for the optional rolling average.
    :return: None
    """
    metric_to_col = {
        "Bernoulli": 3,
        "Jaccard": 4,
        "Misidentification": 5
    }
    if metric not in metric_to_col.keys():
        raise RuntimeError(f"Metric type {metric} is not supported")
    col = metric_to_col[metric]

    result_df = pd.read_csv(file, header=None)
    algorithm_names = result_df[0].unique()
    num_algorithms = len(algorithm_names)
    pfi_metrics = result_df.values[:, col].reshape(num_algorithms, num_runs, num_arm_pulls).astype(np.float64)
    avg_pfi_metrics = np.mean(pfi_metrics, axis=1)
    std_pfi_metrics = np.var(pfi_metrics, axis=1)

    # x indices, potentially subsampled
    x = np.arange(num_arm_pulls)
    x_step = x[::step]

    plt.figure(figsize=(8, 6))

    for i, name in enumerate(algorithm_names):
        y = avg_pfi_metrics[i]

        # apply rolling average if requested
        if rolling_avg_window > 1:
            y_series = pd.Series(y).rolling(window=rolling_avg_window).mean()
            y_vals = y_series.values[::step]
        else:
            y_vals = y[::step]

        # plot mean curve, using only every `step`-th point
        plt.plot(x_step, y_vals, label=f"{name}", color=colors[i])

        if plot_std:
            ci = 1.96 * std_pfi_metrics[i] / np.sqrt(num_runs)
            ci_lower = (y - ci)[::step]
            ci_upper = (y + ci)[::step]
            plt.fill_between(
                x_step,
                ci_lower,
                ci_upper,
                alpha=0.2,
                color=colors[i]
            )

    if metric in ["Bernoulli", "Jaccard"]:
        plt.ylim(0, 1)
    if metric == "Misidentification":
        plt.yscale("log")

    plt.xlabel("Arm pulls")
    plt.ylabel(f"{metric} metric")
    plt.legend(loc="best")
    if save_pdf:
        plt.savefig(f"{file}_{metric}.pdf", format="pdf")
    plt.show()


def plot_all_pfi_metrics(file, num_runs, num_arm_pulls, rolling_avg_window=1, plot_std=True, save_pdf=False, step=1):
    for metric in ["Bernoulli", "Jaccard", "Misidentification"]:
        plot_pfi_metric(file, num_runs, num_arm_pulls, metric, rolling_avg_window, plot_std, save_pdf, step)


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


def plot_posterior_density_2d(parquet_path, num_samples=5000):
    """
    Visualize posterior distributions (2 objectives) per arm as 2D density plots.

    Assumptions:
    - Parquet file has columns: 'arm', 'means', 'stds'.
    - 'means' and 'stds' are length-2 lists/arrays: [mean_obj1, mean_obj2], [std_obj1, std_obj2].
    - Objectives are independent given the posterior, so the joint is a product of 1D normals.
    """
    df = pd.read_parquet(parquet_path)

    # Sanity check: require exactly 2 objectives
    first_means = df["means"].iloc[0]
    if len(first_means) != 2:
        raise ValueError(f"Expected exactly 2 objectives, got {len(first_means)}")

    plt.figure(figsize=(8, 6))
    cmap = sns.color_palette("husl", n_colors=df["arm"].nunique())

    for idx, (arm, row) in enumerate(df.iterrows()):
        means = np.array(row["means"], dtype=float)  # shape (2,)
        stds = np.array(row["stds"], dtype=float)  # shape (2,)

        # Sample from the 2D independent Gaussian for this arm
        samples = np.random.normal(
            loc=means,
            scale=stds,
            size=(num_samples, 2)
        )
        x = samples[:, 0]  # objective 1
        y = samples[:, 1]  # objective 2

        sns.kdeplot(
            x=x,
            y=y,
            levels=5,
            fill=True,
            alpha=0.3,
            color=cmap[idx],
            label=f"arm {int(row['arm'])}" if idx == 0 else None,  # avoid duplicate legend labels
        )

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Posterior density over two objectives per arm")
    # If you want all arms in legend, remove the conditional label logic above and call:
    # plt.legend(title="Arm")
    plt.tight_layout()
    plt.show()


def plot_posterior_density_2d_theoretical(parquet_path, grid_size=500):
    """
    Plot 2D theoretical Gaussian posterior densities per arm (2 objectives).

    Assumptions
    ----------
    - parquet has columns: 'arm', 'means', 'stds'
    - 'means' and 'stds' are length-2 lists/arrays for the two objectives
    - objectives are independent, so covariance is diag(stds**2)
    """
    df = pd.read_parquet(parquet_path)

    first_means = df["means"].iloc[0]
    if len(first_means) != 2:
        raise ValueError(f"Expected exactly 2 objectives, got {len(first_means)}")

    # Collect all means/stds to define a common plotting window
    mus = np.stack(df["means"].apply(lambda m: np.array(m, dtype=float)).values)  # shape (n_arms, 2)
    sigmas = np.stack(df["stds"].apply(lambda s: np.array(s, dtype=float)).values)  # shape (n_arms, 2)

    # 4-sigma box across all arms
    x_min = np.min(mus[:, 0] - 4 * sigmas[:, 0])
    x_max = np.max(mus[:, 0] + 4 * sigmas[:, 0])
    y_min = np.min(mus[:, 1] - 4 * sigmas[:, 1])
    y_max = np.max(mus[:, 1] + 4 * sigmas[:, 1])

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))  # shape (grid, grid, 2)

    plt.figure(figsize=(8, 6))
    cmap = sns.color_palette("husl", n_colors=df["arm"].nunique())

    for idx, row in df.iterrows():
        mean = np.array(row["means"], dtype=float)
        std = np.array(row["stds"], dtype=float)
        cov = np.diag(std ** 2)

        rv = multivariate_normal(mean=mean, cov=cov)
        Z = rv.pdf(pos)

        plt.contour(
            X,
            Y,
            Z,
            levels=5,
            colors=cmap[idx],
            alpha=0.8,
        )

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    # plt.title("Theoretical 2D Gaussian posterior densities per arm")
    plt.tight_layout()
    plt.show()


def plot_aggregate_posterior_heatmap_2d(parquet_path, grid_size=200, mode="sum"):
    """
    Plot an aggregate 2D theoretical Gaussian posterior heatmap over all arms.

    Parameters
    ----------
    parquet_path : str
        Path to parquet file with columns: 'arm', 'means', 'stds'.
    grid_size : int
        Number of grid points per axis for evaluating the density.
    mode : {"sum", "max", "mean"}
        How to aggregate per-arm densities:
        - "sum": sum of densities.
        - "max": maximum density across arms.
        - "mean": average density across arms.
    """
    df = pd.read_parquet(parquet_path)

    first_means = df["means"].iloc[0]
    if len(first_means) != 2:
        raise ValueError(f"Expected exactly 2 objectives, got {len(first_means)}")

    # Collect means/stds to define common plotting window
    mus = np.stack(df["means"].apply(lambda m: np.array(m, dtype=float)).values)  # (n_arms, 2)
    sigmas = np.stack(df["stds"].apply(lambda s: np.array(s, dtype=float)).values)  # (n_arms, 2)

    # 4-sigma box across all arms
    x_min = np.min(mus[:, 0] - 4 * sigmas[:, 0])
    x_max = np.max(mus[:, 0] + 4 * sigmas[:, 0])
    y_min = np.min(mus[:, 1] - 4 * sigmas[:, 1])
    y_max = np.max(mus[:, 1] + 4 * sigmas[:, 1])

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))  # (grid_size, grid_size, 2)

    # Evaluate density for each arm and aggregate
    agg_Z = None
    all_Z = []

    for _, row in df.iterrows():
        mean = np.array(row["means"], dtype=float)
        std = np.array(row["stds"], dtype=float)
        cov = np.diag(std ** 2)

        rv = multivariate_normal(mean=mean, cov=cov)
        Z = rv.pdf(pos)  # (grid_size, grid_size)
        all_Z.append(Z)

    all_Z = np.stack(all_Z, axis=0)  # (n_arms, grid_size, grid_size)

    if mode == "sum":
        agg_Z = np.sum(all_Z, axis=0)
    elif mode == "max":
        agg_Z = np.max(all_Z, axis=0)
    elif mode == "mean":
        agg_Z = np.mean(all_Z, axis=0)
    else:
        raise ValueError("mode must be one of {'sum', 'max', 'mean'}")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        agg_Z,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
        cmap="viridis",
    )
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(f"Aggregate posterior density over arms")
    plt.colorbar(im, label="Aggregated density")
    plt.tight_layout()
    plt.show()


def plot_uncertainty(uncertainties, timesteps, title):
    mean_uncertainties = np.mean(uncertainties, axis=0)
    std_uncertainties = np.std(uncertainties, axis=0)

    plt.figure()

    # Mean line
    sns.lineplot(x=timesteps, y=mean_uncertainties, color="C0")

    # Shaded std band
    lower = mean_uncertainties - std_uncertainties
    upper = mean_uncertainties + std_uncertainties
    plt.fill_between(timesteps, lower, upper, color="C0", alpha=0.2)

    plt.xlim(0, 5000)
    plt.title(title)
    plt.xlabel("Number of arm pulls")
    plt.ylabel("Uncertainty (Bhattacharyya sum)")
    plt.tight_layout()
    plt.show()


def plot_uncertainty_grid(env_uncertainties, timesteps):
    # 5 rows x 2 cols
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), sharex=True, sharey=False)
    axes = axes.flatten()

    for idx, (env, uncertainties) in enumerate(env_uncertainties.items()):
        ax = axes[idx]
        mean_uncertainties = np.mean(uncertainties, axis=0)
        std_uncertainties = np.std(uncertainties, axis=0)

        sns.lineplot(x=timesteps, y=mean_uncertainties, color="C0", ax=ax)
        lower = mean_uncertainties - std_uncertainties
        upper = mean_uncertainties + std_uncertainties
        ax.fill_between(timesteps, lower, upper, color="C0", alpha=0.2)

        ax.set_title(env)
        ax.set_xlim(0, 5000)

    # Remove any unused axes (in case of fewer than 10 envs)
    for j in range(len(env_uncertainties), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Uncertainty over Time", y=0.92)
    fig.text(0.5, 0.04, "Number of arm pulls", ha="center")
    fig.text(0.04, 0.5, "Uncertainty (Bhattacharyya sum)", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.94])
    plt.show()


def plot_pfi_metric_ax(
    ax,
    file,
    num_runs,
    num_arm_pulls,
    metric,
    rolling_avg_window=1,
    plot_std=True,
    step=1,
    show_legend=False,
    ylabel=None,
    title=None,
):
    metric_to_col = {
        "Bernoulli": 3,
        "Jaccard": 4,
        "Misidentification": 5
    }
    if metric not in metric_to_col:
        raise RuntimeError(f"Metric type {metric} is not supported")
    col = metric_to_col[metric]

    result_df = pd.read_csv(file, header=None)
    algorithm_names = result_df[0].unique()
    num_algorithms = len(algorithm_names)
    pfi_metrics = result_df.values[:, col].reshape(num_algorithms, num_runs, num_arm_pulls).astype(np.float64)
    avg_pfi_metrics = np.mean(pfi_metrics, axis=1)
    std_pfi_metrics = np.std(pfi_metrics, axis=1)

    x = np.arange(num_arm_pulls)
    x_step = x[::step]

    for i, name in enumerate(algorithm_names):
        y = avg_pfi_metrics[i]

        if rolling_avg_window > 1:
            y_series = pd.Series(y).rolling(window=rolling_avg_window).mean()
            y_vals = y_series.values[::step]
        else:
            y_vals = y[::step]

        ax.plot(x_step, y_vals, label=f"{name}", color=colors[i])

        if plot_std:
            ci = 1.96 * std_pfi_metrics[i] / np.sqrt(num_runs)
            ci_lower = (y - ci)[::step]
            ci_upper = (y + ci)[::step]
            ax.fill_between(
                x_step,
                ci_lower,
                ci_upper,
                alpha=0.3,
                color=colors[i]
            )

    if metric in ["Bernoulli", "Jaccard"]:
        ax.set_ylim(0, 1)
    if metric == "Misidentification":
        ax.set_yscale("log")
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(NullFormatter())

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=14)

    if show_legend:
        ax.legend(loc="best", fontsize=8)


def plot_all_pfi_metrics_grid(
    env_files,
    env_labels,
    num_runs,
    num_arm_pulls,
    rolling_avg_window=1,
    plot_std=True,
    step=1,
    figsize=(14, 24)
):
    metrics = ["Bernoulli", "Jaccard", "Misidentification"]
    n_env = len(env_files)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        nrows=n_env,
        ncols=n_metrics,
        figsize=figsize,
        sharex=True
    )

    # If only one row/col, axes may not be 2D; normalize
    if n_env == 1:
        axes = np.array([axes])
    if n_metrics == 1:
        axes = axes[:, np.newaxis]

    for row, (file, env_name) in enumerate(zip(env_files, env_labels)):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            # Y-label on first column only
            ylabel = env_name if col == 0 else None

            # Column titles on first row
            title = metric if row == 0 else None

            # Put legend only on top-right subplot (or where you prefer)
            show_legend = (row == 0 and col == n_metrics - 1)

            plot_pfi_metric_ax(
                ax=ax,
                file=file,
                num_runs=num_runs,
                num_arm_pulls=num_arm_pulls,
                metric=metric,
                rolling_avg_window=rolling_avg_window,
                plot_std=plot_std,
                step=step,
                show_legend=show_legend,
                ylabel=ylabel,
                title=title
            )

    # Common x-label on bottom row
    for col in range(n_metrics):
        axes[-1, col].set_xlabel("Arm pulls")

    plt.tight_layout()
    plt.show()


def plot_all_pfi_metrics_grid_top_legend(
    env_files,
    env_labels,
    num_runs,
    num_arm_pulls,
    rolling_avg_window=1,
    plot_std=True,
    step=1,
    figsize=(12, 24),
    save_png=False
):
    metrics = ["Bernoulli", "Jaccard", "Misidentification"]
    n_env = len(env_files)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        nrows=n_env,
        ncols=n_metrics,
        figsize=figsize,
        sharex=True
    )

    if n_env == 1:
        axes = np.array([axes])
    if n_metrics == 1:
        axes = axes[:, np.newaxis]

    # Plot, but do NOT show legends inside subplots
    for row, (file, env_name) in enumerate(zip(env_files, env_labels)):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            ylabel = env_name if col == 0 else None
            title = metric if row == 0 else None

            plot_pfi_metric_ax(
                ax=ax,
                file=file,
                num_runs=num_runs,
                num_arm_pulls=num_arm_pulls,
                metric=metric,
                rolling_avg_window=rolling_avg_window,
                plot_std=plot_std,
                step=step,
                show_legend=False,   # <-- no per‑axes legend
                ylabel=ylabel,
                title=title
            )

    # Use one axes (e.g., top-right) to grab handles/labels
    handles, labels = axes[0, -1].get_legend_handles_labels()

    # Create one horizontal legend at the top center
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(5, len(labels)),   # up to 5 algorithms per row
        fontsize=14,
        frameon=True
    )

    # Add a little extra top margin for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Common x-label on bottom row
    for col in range(n_metrics):
        axes[-1, col].set_xlabel("Arm pulls")

    if save_png:
        plt.savefig(f"/Users/lennertsaerens/Desktop/Internship/PhD/combined_pfi_plots_TTPFTSvariants_shadeCI_step{step}.png", format="png", dpi=300)

    plt.show()


if __name__ == "__main__":
    # plot_bernoulli_metric_coarse("results/EGEvsTTPFTSvsPUCB1vsUniform_Coarse_EgeExp1.csv", 100, plot_std=True)
    # plot_arm_rec_frequencies("results/baseline_recs.csv", 100, 30_000, [0, 5, 6, 8, 14, 30, 31, 32], 53, "Uniform Sampling")
    # plot_arm_pull_frequencies("results/bandits/test2.csv", 100, 250_000, [0, 1, 2, 3], 20,
    #                           "Linear Scalarized Knowledge Gradient (objectives)", 3)
    plot_all_pfi_metrics("results/EGEvsTTPFTSvsPUCB1vsUniform_EgeExp1.csv", 100, 5001)
