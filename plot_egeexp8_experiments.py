import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_extended():
    file_path = "results/egeexp8_extended_uq.csv"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
        
    df = pd.read_csv(file_path)
    
    import numpy as np
    # Calculate means, stds, and counts over runs
    agg_df = df.groupby('timestep').agg(['mean', 'std', 'count']).reset_index()
    ci_multiplier = 1.96 / np.sqrt(agg_df['jaccard']['count'])
    
    # Configure rcParams for paper
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14
    })
    
    # Adjusted figsize
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Left y-axis: Jaccard
    color_jaccard = 'tab:blue'
    ax1.set_xlabel('Arm Pulls')
    ax1.set_ylabel('Jaccard Metric', color=color_jaccard)
    ax1.plot(agg_df['timestep'], agg_df['jaccard']['mean'], color=color_jaccard, label='Jaccard')
    ax1.fill_between(agg_df['timestep'], 
                     agg_df['jaccard']['mean'] - ci_multiplier * agg_df['jaccard']['std'],
                     agg_df['jaccard']['mean'] + ci_multiplier * agg_df['jaccard']['std'], 
                     color=color_jaccard, alpha=0.2)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y', labelcolor=color_jaccard)
    
    # Right y-axis: Uncertainty
    ax2 = ax1.twinx()
    color_uq = 'tab:red'
    ax2.set_ylabel('Uncertainty', color=color_uq)
    ax2.plot(agg_df['timestep'], agg_df['uq']['mean'], color=color_uq, label='UQ')
    ax2.fill_between(agg_df['timestep'], 
                     agg_df['uq']['mean'] - ci_multiplier * agg_df['uq']['std'],
                     agg_df['uq']['mean'] + ci_multiplier * agg_df['uq']['std'], 
                     color=color_uq, alpha=0.2)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor=color_uq)
    
    # No title as requested
    fig.tight_layout()
    plt.savefig('figures/egeexp8_extended_uq_vs_jaccard.png', dpi=500)
    plt.close()
    
    # Reset rcParams just in case it affects other plots
    plt.rcParams.update(plt.rcParamsDefault)


def plot_noise():
    file_path = "results/egeexp8_noise_uq.csv"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
        
    df = pd.read_csv(file_path)
    
    # Group by noise scale and timestep
    agg_df = df.groupby(['noise_scale', 'timestep']).agg(['mean']).reset_index()
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Jaccard
    sns.lineplot(data=df, x='timestep', y='jaccard', hue='noise_scale', ax=ax1, palette='viridis')
    ax1.set_title('Jaccard Score vs Noise Scale')
    ax1.set_ylabel('Jaccard Score')
    
    # Plot UQ
    sns.lineplot(data=df, x='timestep', y='uq', hue='noise_scale', ax=ax2, palette='viridis')
    ax2.set_title('Uncertainty (Bhattacharyya) vs Noise Scale')
    ax2.set_ylabel('Uncertainty')
    
    plt.suptitle('EgeExp8: Impact of Noise Scaling (Variance Reduction)')
    plt.tight_layout()
    plt.savefig('figures/egeexp8_noise_scaling.png')
    plt.close()
    
    # Plot Correlation
    corr_results = []
    for scale in df['noise_scale'].unique():
        scale_df = df[df['noise_scale'] == scale]
        for run in scale_df['run'].unique():
            run_df = scale_df[scale_df['run'] == run]
            corr = run_df['jaccard'].corr(run_df['uq'])
            corr_results.append({'noise_scale': scale, 'correlation': corr})
            
    corr_df = pd.DataFrame(corr_results)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=corr_df, x='noise_scale', y='correlation', palette='viridis')
    plt.title('Pearson Correlation between Jaccard and UQ across Noise Scales')
    plt.ylabel('Correlation Coefficient')
    plt.savefig('figures/egeexp8_noise_correlation.png')
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_extended()
    plot_noise()
    print("Saved plots to figures directory.")
