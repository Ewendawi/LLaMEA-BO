# Experiments

This directory contains the experimental results and algorithm implementations.

## Directory Structure

### Algorithm Files

Each algorithm has three associated files:
- `ALGORITHM_NAME.py` - The implementation code
- `ALGORITHM_NAME_prompt.md` - The prompt used to generate the algorithm
- `ALGORITHM_NAME_respond.md` - The LLM's response that led to the algorithm

For example:
- `ABETSALSDE_ARM_MBO.py`
- `ABETSALSDE_ARM_MBO_prompt.md`
- `ABETSALSDE_ARM_MBO_respond.md`

### IOH Benchmark Data

The directory contains benchmark results in IOH Profiler format, organized by problem dimension:
- `5D_ioh.csv` - 5-dimensional benchmark results
- `10D_ioh.csv` - 10-dimensional benchmark results
- `20D_ioh.csv` - 20-dimensional benchmark results
- `40D_ioh.csv` - 40-dimensional benchmark results

These files can be directly imported into [IOHanalyzer](https://iohanalyzer.liacs.nl/) for visualization and analysis.

## Baselines

The baseline implementations used for comparison are located in `Experiments/baselines/`:

- `bo_baseline.py` - Baseline implementations and wrappers
- `vanilla_bo.py` - Vanilla Bayesian Optimization implementation

## Usage

To analyze the results:
1. Visit [IOHanalyzer](https://iohanalyzer.liacs.nl/)
2. Check the checkbox `use custom csv format`
3. Upload the dimension-specific CSV files
4. Use the IOHanalyzer interface to generate comparisons and visualizations
