import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    std_pfi_metrics = np.std(pfi_metrics, axis=1)

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
        plt.plot(x_step, y_vals, label=f"{name}")

        if plot_std:
            ci = 1.96 * std_pfi_metrics[i] / np.sqrt(num_runs)
            ci_lower = (y - ci)[::step]
            ci_upper = (y + ci)[::step]
            plt.fill_between(
                x_step,
                ci_lower,
                ci_upper,
                alpha=0.2,
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


def plot_pfi_metric_ax(
        ax,
        file,
        num_runs,
        num_arm_pulls,
        num_arms,
        num_objs,
        metric,
        plot_std=True,
        step=1,
        show_legend=False,
        ylabel=None,
        title=None,
):
    df = pd.read_csv(file, header=None, names=["algorithm", "e", "t", "Bernoulli", "Jaccard", "Misclassification"],
                     low_memory=False)
    algorithms = df["algorithm"].unique()

    for i, algorithm in enumerate(algorithms):

        start = int(1.5*num_arms)

        x = np.arange(num_arm_pulls)
        x_step = x[::step]
        x_step = x_step[start:]  # The x-values for the plot start here

        df_algo = df[df["algorithm"] == algorithm]

        metric_vals = df_algo[metric].values.reshape(num_runs, num_arm_pulls)

        # Compute mean and std over runs
        mean_vals = np.mean(metric_vals, axis=0)
        std_vals = np.std(metric_vals, axis=0)

        mean_plot = mean_vals[::step]
        ci = 1.96 * std_vals / np.sqrt(num_runs)
        ci_lower = (mean_vals - ci)[::step]
        ci_upper = (mean_vals + ci)[::step]

        mean_plot = mean_plot[start:]
        ci_lower = ci_lower[start:]
        ci_upper = ci_upper[start:]

        line = None
        if algorithm == "TTPFTS_UNI":
            line, = ax.plot(x_step, mean_plot, label="TTPFTS")
        elif algorithm == "Uniform":
            line, = ax.plot(x_step, mean_plot, label="Uniform Sampling")
        else:
            line, = ax.plot(x_step, mean_plot, label=algorithm)

        # Get the color of the current line so the vertical line matches
        color = line.get_color()

        # Draw a dotted line from y=0 up to the start of the mean curve
        # x_step[0] is the x-coordinate where the curve starts
        # mean_plot[0] is the y-coordinate where the curve starts
        if start < num_arm_pulls:
            ax.vlines(x=x_step[0], ymin=0, ymax=mean_plot[0], colors=color, linestyles='dotted', alpha=0.7)

        # Plot CI band
        if plot_std:
            ax.fill_between(x_step, ci_lower, ci_upper, alpha=0.3, color=color)

    if metric in ["Bernoulli", "Jaccard"]:
        ax.set_ylim(0, 1)
    if metric == "Misclassification":
        ax.set_yscale("log")

    ax.grid(True)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=14)

    if show_legend:
        ax.legend(loc="best", fontsize=8)


def plot_all_pfi_metrics_grid_top_legend(
        env_files,
        env_labels,
        num_runs,
        num_arm_pulls,
        num_arms,
        num_objs,
        plot_std=True,
        step=1,
        figsize=(12, 24),
        save_png=False,
        save_file=None
):
    metrics = ["Bernoulli", "Jaccard", "Misclassification"]
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
        n_arms = num_arms[row]
        n_objs = num_objs[row]
        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            ylabel = env_name if col == 0 else None
            title = metric if row == 0 else None

            plot_pfi_metric_ax(
                ax=ax,
                file=file,
                num_runs=num_runs,
                num_arm_pulls=num_arm_pulls,
                num_arms=n_arms,
                num_objs=n_objs,
                metric=metric,
                plot_std=plot_std,
                step=step,
                show_legend=False,  # <-- no per‑axes legend
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
        ncol=min(5, len(labels)),  # up to 5 algorithms per row
        fontsize=14,
        frameon=True
    )

    # Add a little extra top margin for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Common x-label on bottom row
    for col in range(n_metrics):
        axes[-1, col].set_xlabel("Arm pulls")

    if save_png and save_file is not None:
        plt.savefig(save_file, format="png", dpi=300)

    plt.show()


def plot_metric_all_envs_grid_top_legend(
        env_files,
        env_labels,
        env_arms,
        env_objs,
        num_runs,
        num_arm_pulls,
        metric,
        plot_std=True,
        step=1,
        figsize=(13, 6),
        save_png=False,
        save_file=None
):
    n_env = len(env_files)
    ncols = min(4, n_env)
    nrows = int(np.ceil(n_env / ncols))

    sharey = False if metric == "Misclassification" else True

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    # Plot, but do NOT show legends inside subplots
    for i, (file, env_name) in enumerate(zip(env_files, env_labels)):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        n_arms = env_arms[i]
        n_objs = env_objs[i]

        ylabel = f"{metric} Metric" if col == 0 else None
        title = f"{env_name} (K = {n_arms}, D = {n_objs})"

        plot_pfi_metric_ax(
            ax=ax,
            file=file,
            num_runs=num_runs,
            num_arm_pulls=num_arm_pulls,
            num_arms=n_arms,
            num_objs=n_objs,
            metric=metric,
            plot_std=plot_std,
            step=step,
            show_legend=False,  # <-- no per‑axes legend
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
        ncol=min(5, len(labels)),  # up to 5 algorithms per row
        fontsize=14,
        frameon=True
    )

    # Add a little extra top margin for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.subplots_adjust(bottom=0.1)

    # Common x-label on bottom row
    for col in range(ncols):
        axes[-1, col].set_xlabel("Arm Pulls")

    if save_png and save_file is not None:
        plt.savefig(save_file, format="png", dpi=500)

    plt.show()


def plot_uncertainty_all_envs(
        env_uncertainties,
        timesteps,
        set_title=True,
        save_png=False,
        save_file=None
):
    # Single figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))

    handles = []
    labels = []

    for env, uncertainties in env_uncertainties.items():
        # uncertainties: shape (n_runs, n_timesteps)
        mean_uncertainties = np.mean(uncertainties, axis=0)
        std_uncertainties = np.std(uncertainties, axis=0)

        # extract number after 'EgeExp'
        num_label = env.replace("EgeExp", "")

        line, = ax.plot(timesteps, mean_uncertainties, label=env)
        ci = 1.96 * std_uncertainties / np.sqrt(uncertainties.shape[0])
        ci_lower = mean_uncertainties - ci
        ci_upper = mean_uncertainties + ci

        # Use same color as line but with alpha for CI
        ax.fill_between(
            timesteps,
            ci_lower,
            ci_upper,
            alpha=0.3,
            color=line.get_color()
        )

        handles.append(line)
        labels.append(num_label)

    if set_title and not save_png:
        ax.set_title("Uncertainty Across Environments", fontsize=14)
    ax.set_xlim(0, 5000)
    ax.set_xlabel("Arm Pulls")
    ax.set_ylabel("Uncertainty (Bhattacharyya Avg)")
    ax.grid(True)

    # Place legend outside on the right
    # bbox_to_anchor (x, y): x>1 pushes it outside to the right
    ax.legend(handles, labels, title="EgeExp", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    if save_png and save_file is not None:
        plt.savefig(save_file, format="png", dpi=500, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_all_pfi_metrics("results/EGEvsTTPFTSvsPUCB1vsUniform_EgeExp1.csv", 100, 5001)
