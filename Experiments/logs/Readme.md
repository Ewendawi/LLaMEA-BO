# Experiments

This directory contains the experimental results and algorithm implementations.

## Usage

To analyze the results:
1. Visit [IOHanalyzer](https://iohanalyzer.liacs.nl/)
2. Check the checkbox `use custom csv format`
3. Upload the CSV files
4. Use the IOHanalyzer interface to generate comparisons and visualizations

## Directory Structure

### algorithms_logs
This directory contains the algorithm files and evaluation results. 

Each algorithm has three associated files:
- `ALGORITHM_NAME.py` - The implementation code
- `ALGORITHM_NAME_prompt.md` - The prompt used to generate the algorithm
- `ALGORITHM_NAME_respond.md` - The LLM's response that led to the algorithm

For example:
- `ABETSALSDE_ARM_MBO.py`
- `ABETSALSDE_ARM_MBO_prompt.md`
- `ABETSALSDE_ARM_MBO_respond.md`

The ES search results:
- `es_100_aoc_ioh.csv` - The AOC value
- `es_100_problem_ioh.csv` - The AOC value for each problem

The BBOB benchmark results organized by problem dimension:
- `*D_ioh_loss.csv` - The benchmark results of the loss
- `*D_ioh_fx.csv` - The benchmark results of the function value
- `*D_aoc.csv` - The benchmark results of the AOC value
- `*D_mean_aoc.csv` - The benchmark results of the average AOC value for each problem
- `*D_hist.csv` - The benchmark results of `x_hist` and `y_hist` 
- `*D_mannwhitneyu_test.csv` - The Mann-Whitney U test results of the AOC values

The files with `ioh` infix can be directly imported into [IOHanalyzer](https://iohanalyzer.liacs.nl/) for visualization and analysis.

### bayesmark_logs and hpo_logs
These directories contain the results of Bayesmark an HPOBence, respectively.
For proper processing these files on iohanalyzer:
- Ensure the checkboxes `Is the data from a maximization setting?` and `Use custom csv format` are checked before clicking the `Process` button.

### atrbo_logs
This directory contains the results of the additional experiments conducted with the ATRBO algorithm.

### es_search_data
This directory stores the raw data from the Evolutionary Strategy (ES) search experiments. Each subdirectory corresponds to a specific experimental run, and its name encodes the parameters used for that run. For example:
- `ESPopulation_evol_1+1_t0.5_k60_cr0.6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0308014053/`
  - This indicates a (1+1)-ES strategy.
  - `t0.5`: Temperature parameter.
  - `k60`: Kappa parameter.
  - `cr0.6`: Crossover rate.
  - `IOHEvaluator`: The evaluator used.
  - `f2_f4_...`: The specific BBOB functions tested.
  - `dim-5`: Dimensionality of the problem.
  - `budget-100`: Evaluation budget.
  - `instances-[1]`: Problem instances.
  - `repeat-3`: Number of repetitions.
  - `0308014053`: Timestamp of the experiment.

Within each of these subdirectories, you will find files related to the individual algorithms generated and tested during that specific ES run. For each algorithm instance, there are typically three associated files:
- `[generation]-[iteration]_[ALGORITHM_NAME]_[metric_value]_prompt.md`: The prompt used to generate this specific algorithm.
- `[generation]-[iteration]_[ALGORITHM_NAME]_[metric_value]_respond.md`: The full response from the language model that produced the algorithm code.
- `[generation]-[iteration]_[ALGORITHM_NAME]_[metric_value].py`: The Python code for the generated algorithm, which is extracted from the response.

The `[generation]-[iteration]` prefix indicates the generation number and the iteration number in that ES evoluation. `[ALGORITHM_NAME]` is a descriptive name of the algorithm, and `[metric_value]` likely represents a performance score (AOC) achieved by the algorithm during its evaluation.

## Baselines

The baseline implementations used for comparison are located in `Experiments/baselines/`:

- `bo_baseline.py` - Baseline implementations and wrappers
- `vanilla_bo.py` - Vanilla Bayesian Optimization implementation


