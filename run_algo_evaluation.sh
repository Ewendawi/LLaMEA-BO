#! /bin/bash


# -n the name of the algorithm
# -p the path to the algorithm
# -b indicate if the algorithm is a baseline
# -m enable bare MPI Evaluation
# -d the dimension of the problem
# -e plot the results


# Change dim to the desired dimension from [5, 10, 20, 40]
DIM=5

# Evaluate the baselines
BASELINE_keys=(
    "BLTuRBO1"
    "BLVanillaEIBO"
    "BLCMAES"
    "BLHEBO"
)
BASELINE_PATHS=(
    "Experiments/baselines/bo_baseline.py"
    "Experiments/baselines/bo_baseline.py"
    "Experiments/baselines/bo_baseline.py"
    "Experiments/baselines/bo_baseline.py"
)
for i in "${!BASELINE_keys[@]}"; do
    ALGO=${BASELINE_keys[$i]}
    ALGO_PATH=${BASELINE_PATHS[$i]}
    echo "Running $ALGO with path $ALGO_PATH"
    python run_algo_evaluation.py -n $ALGO -p $ALGO_PATH -b -d $DIM
    echo ""
done

# Evaluate the generated algorithms
generated_algorithms=(
    "AdaptiveTrustRegionOptimisticHybridBO"
    "AdaptiveEvolutionaryParetoTrustRegionBO"
    "AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE"
    "ATRBO"
    "ABETSALSDE_ARM_MBO"
)
generated_algorithms_paths=(
    "Experiments/generated_algorithms/AdaptiveTrustRegionOptimisticHybridBO.py"
    "Experiments/generated_algorithms/AdaptiveEvolutionaryParetoTrustRegionBO.py"
    "Experiments/generated_algorithms/AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE.py"
    "Experiments/generated_algorithms/ATRBO.py"
    "Experiments/generated_algorithms/ABETSALSDE_ARM_MBO.py"
)

for i in "${!generated_algorithms[@]}"; do
    ALGO=${generated_algorithms[$i]}
    ALGO_PATH=${generated_algorithms_paths[$i]}
    echo "Running $ALGO with path $ALGO_PATH"
    python run_algo_evaluation.py -n $ALGO -p $ALGO_PATH -d $DIM
    echo ""
done

# Plot the results
python run_algo_evaluation.py -e
