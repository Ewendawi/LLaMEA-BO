#! /bin/bash


# -n the name of the algorithm
# -p the path to the algorithm
# -b indicate if the algorithm is a baseline
# -m enable bare MPI Evaluation
# -e extract the results

# Evaluate the baselines
BASELINE_PATHS=(
    ["BLTuRBO1"]="Experiments/baselines/bo_baseline.py"
    ["BLVanillaEIBO"]="Experiments/baselines/bo_baseline.py"
    ["BLCMAES"]="Experiments/baselines/bo_baseline.py"
    ["BLHEBO"]="Experiments/baselines/bo_baseline.py"
)

for ALGO in "${!BASELINE_PATHS[@]}"; do
    ALGO_PATH=${BASELINE_PATHS[$ALGO]}
    echo "Running $ALGO with path $ALGO_PATH"

    python run_algo_evaluation.py -n ALGO -p ALGO_PATH -b
done 

# Evaluate the generated algorithms
GENERATED_ALGO_PATHS=(
    ["AdaptiveTrustRegionOptimisticHybridBO"]="Experiments/generated_algorithms/AdaptiveTrustRegionOptimisticHybridBO.py"
    ["AdaptiveEvolutionaryParetoTrustRegionBO"]="Experiments/generated_algorithms/AdaptiveEvolutionaryParetoTrustRegionBO.py"
    ["AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE"]="Experiments/generated_algorithms/AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE.py"
    ["ATRBO"]="Experiments/generated_algorithms/ATRBO.py"
    ["ABETSALSDE_ARM_MBO"]="Experiments/generated_algorithms/ABETSALSDE_ARM_MBO.py"
)
for ALGO in "${!GENERATED_ALGO_PATHS[@]}"; do
    ALGO_PATH=${GENERATED_ALGO_PATHS[$ALGO]}
    echo "Running $ALGO with path $ALGO_PATH"

    python run_algo_evaluation.py -n ALGO -p ALGO_PATH
done


# Extract and Plot the results
python run_algo_evaluation.py -e
