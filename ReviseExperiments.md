# Instructions for revising the empirical experiments based on this repository
This Markdown file contains all the instructions for revising the experiments. All of these steps should be completed in a scientifically rigorous manner, while remaining as simple and elegant as possible. Reproducibility is a key aspect of this process, and all steps should be documented clearly.

## Current state
The current version of the manuscript shows results for the eight EgeExp environments. 
Currently, for all environments:
- The budget is set at 5000 arm pulls.
- The number of repetitions is set at 100.
- The algorithms used are: EGE-SH, EGE-SR, Uniform Sampling, TTPFTS, and APE
- The performance metrics used are: Jaccard, Bernoulli, and Misidentification.
- Gaussian rewards with diagonal covariance are used. In the manuscript we report: "We use Gaussian reward distributions with diagonal covariance matrix $\sigma^2 I_D$ with $\sigma^2=0.25$."
- For TTPFTS, we assume Gaussian likelihood with unknown mean and unknown variance. We assume weak prior information about the environment by using a uniform prior. This leads to a $t$-distributed posterior. 
- In all experiments, we set $\rho=0.5$.

This represents the current state of the experiments, The following sections will outline the envisioned revisions to the experiments.

## Mandatory checks
- The APE algorithm implemented in bandits/APE.py should be verified to be completely sure that it produces exactly the same results as the original implementation it is derived from in the "8606_Adaptive_Algorithms_for_R_Supplementary Material" directory. Extensive testing scenarios using identical random seeds should be conducted across both implementations to ensure they are equivalent. This should be concluded before any new experiment can be conducted.
- The two following claims about the reward model and bayesian inference mode should be verified, and experiments should be updated to be in line with these claims:
  - Gaussian rewards with diagonal covariance are used. In the manuscript we report: "We use Gaussian reward distributions with diagonal covariance matrix $\sigma^2 I_D$ with $\sigma^2=0.25$."
  - For TTPFTS, we assume Gaussian likelihood with unknown mean and unknown variance. We assume weak prior information about the environment by using a uniform prior. This leads to a $t$-distributed posterior.
- At this point it is still unsure which posterior model for TTPFTS I want to include in the final plots for the paper. Hence, both the previously discussed bayesian model for unknown variance, and the known-variance variant should be completely and thoroughly screened for correctness.

Subagents can screen these in parallel. Use the "interact with user" tool to discuss with me what we will do in case any problems or inconsistencies arise across these checks.

## Changes after the checks
### Different budget per environment
- Currently, the arm pull budget is set to 5000 across environments. However, some environments are more challenging or easy to learn the Pareto set. Hence, a more scientifically precise way of defining the budget for each environment is in order. One recent approach (from the EGE paper) is to set the budget for an environment equal to the hardness of that environment. This is a statistic that incorporates the number of arms and their suboptimality gaps. Exact information about these gaps and the resulting hardness metric can be found in the "'/Users/lennertsaerens/Desktop/PhD/TTPFTS/Bounds Workspace clean/Ground Truth and Knowledge/TTPFTS MOMAB PSI.md'" file. For each of the eight EGE benchmarking environments, calculate the hardness and report the change with respect to the current budget back to me. I will then decide on the final per-environment budgets for the final experiments.
### Repetitions and runtime feasibility
- The number of repetitions should remain at 100 for all environments. However, due to the new per-environment budgets, I am expecting that most of the environments will receive a larger budget. In the past I have had some trouble with the runtime of the EGE MOMAB algorithm variants where they would take a very long time to run. Since of course these specific EGE algorithms we are running for every budget between 0 and max. This is in stark contrast to the other algorithms which report their estimated pareto set after every arm pull, and hence only need to be run once per repetition. This means that for the EGE algorithms, I would like you to perform a single representative run with the maximum budget for each of the environments. This gives us an estimate of the runtime if we run it for every budget between 0 and max. Based on this, we will decide on the feasibility of running the EGE algorithms for every budget and whether we can use the "step" parameter to make this more computationally possible. Based on the decision made here, we should consider adapting the final plots to either show the evolution of the metrics over all budgets or over the subset used through the use of the "step" argument for the EGE algorithms. Runtime feasibilty should be considered over maximally parallelized repetitions both on this computer, and on the VUB HPC cluster, accessible via the HPC-EXPERIMENTS skill.
### Algorithms
- The algorithms used in the experiments should remain the same. Both variants of the EGE algorithms, the Uniform baseline, the APE algorithm, and the TTPFTS algorithm. This last one should be run both with the known-variance and unknown variance bayesian models. The rho parameter should be set at 0.5 across all experiments.
### Metrics
- The metrics used to evaluate the estimated pareto sets proposed by the algorithms should remain the same. Results should be saved and reported in such a way that they can be easily retrieved, filtered, and plotted after the experiments have concluded. Environment, algorithm, repetition, arm pull, and all three metrics should be saved during the experiments.
### Plotting the results
- Existing plotting code for plotting the algorithms' performance across the eight environments should be kept as much as possible. However, the grid lines should be more transparant. Furthermore, on the y-axis to the left of the plot "X Metric" should only be shown once instead of for both rows. On the x-axis below the plot, since the budget will differ between environments, we will use a normalized (between 0 and 1) x-axis within each of the subplots. This is achieved by dividing the current budget/arm pull by the maximum budget for that enviroment. The overall x-axis should show a single, centered label "Normalized Budget" in the middle of the four columns of the plot. The legend should remain above the plot in a separate box as is the case now.
