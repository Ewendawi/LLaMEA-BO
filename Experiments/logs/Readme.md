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

The benchmark results in IOH Profiler format, organized by problem dimension:
- `5D_ioh_*.csv` - 5-dimensional benchmark results
- `10D_ioh_*.csv` - 10-dimensional benchmark results
- `20D_ioh_*.csv` - 20-dimensional benchmark results
- `40D_ioh_*.csv` - 40-dimensional benchmark results
- `*D_ioh_loss.csv` - The benchmark results of the loss
- `*D_ioh_fx.csv` - The benchmark results of the function value

These files can be directly imported into [IOHanalyzer](https://iohanalyzer.liacs.nl/) for visualization and analysis.

### bayesmark_logs and hpo_logs
These directories contain the results of Bayesmark an HPOBence, respectively.
For proper processing these files on iohanalyzer:
- Ensure the checkboxes `Is the data from a maximization setting?` and `Use custom csv format` are checked before clicking the `Process` button.

### atrbo_logs
This directory contains the results of the additional experiments conducted with the ATRBO algorithm.

### figs
This directory contains the figures generated from the experiments. 

## Baselines

The baseline implementations used for comparison are located in `Experiments/baselines/`:

- `bo_baseline.py` - Baseline implementations and wrappers
- `vanilla_bo.py` - Vanilla Bayesian Optimization implementation


