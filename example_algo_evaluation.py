import logging
import os
import pickle
from datetime import datetime
import numpy as np
from llambo.utils import setup_logger
from llambo.evaluator.ioh_evaluator import IOHEvaluator
from Experiments.plot_algo_res import extract_algo_result, plot_algo


def get_evaluator():
    budget = 100
    dim = 5
    problems = list(range(1, 25))
    problems = [4]
    instances = [[4, 5, 6]] * len(problems)
    repeat = 2
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    evaluator.ignore_over_budget = True # won't raise exception when over budget

    return evaluator

def baseline_algo_eval_param(dim, budget):
    bl_init_params = {
        "budget": budget,
        "dim": dim,
        "bounds": np.array([[-5.0] * dim, [5.0] * dim]),
        "n_init": min(2 * dim, budget // 2),
        "seed": None,
        "device": "cpu",
        # "device": "cuda",
    }
    return bl_init_params

def run_algo_eval_from_file_map(evaluator, file_map, options, is_baseline=False):
    res_list = []
    _code_map = {}
    for cls_name, file_path in file_map.items():
        if not os.path.exists(file_path):
            logging.warning("File not exist: %s", file_path)
            continue
        code = ""
        with open(file_path, "r") as f:
            code = f.read()

        _code_map[cls_name] = code

    for cls_name, code in _code_map.items():
        extra_init_params = {}
        if is_baseline:
            extra_init_params = baseline_algo_eval_param(evaluator.dim, evaluator.budget)

        res = evaluator.evaluate(
            code=code,
            cls_name=cls_name,
            cls_init_kwargs=extra_init_params,
        )

        res_list.append(res)

        if 'save_dir' in options:
            save_dir = options['save_dir']
            os.makedirs(save_dir, exist_ok=True)
            time_stamp = datetime.now().strftime("%m%d%H%M%S")
            dim = evaluator.dim
            score = res.score
            file_path = os.path.join(save_dir, f"{cls_name}_{score:.4f}_{dim}D_{time_stamp}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(res, f)

    return res_list

def run_evaluation():
    evaluator = get_evaluator()

    save_dir = 'exp_eval'

    options = {
        'save_dir': save_dir,

        # 'use_multi_process': True, # evaluate in multiple processes
        # 'max_eval_workers': 10, # number of processes

        # 'use_mpi': True, # use bare MPI for parallel evaluation
        # 'use_mpi_future': True, # use MPI for parallel evaluation with future
    }

    # the key is the name of the algorithm, the value is the path of the code file
    bl_file_map = {
        'BLTuRBO1': 'Experiments/baselines/bo_baseline.py',
        # 'BLMaternVanillaBO': 'Experiments/baselines/bo_baseline.py',
        # 'BLCMAES': 'Experiments/baselines/bo_baseline.py',
        # 'BLHEBO': 'Experiments/baselines/bo_baseline.py',
    }
    # run the baseline algorithms
    run_algo_eval_from_file_map(evaluator, bl_file_map, options, is_baseline=True)

    file_map = {
        # 'AdaptiveTrustRegionOptimisticHybridBO': 'Experiments/logs/algorithms_0319/AdaptiveTrustRegionOptimisticHybridBO.py',

        # 'AdaptiveEvolutionaryParetoTrustRegionBO': 'Experiments/logs/algorithms_0319/AdaptiveEvolutionaryParetoTrustRegionBO.py',

        # 'AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE': 'Experiments/logs/algorithms_0319/AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE.py',

        'ATRBO': 'Experiments/logs/algorithms_0319/ATRBO.py',

        # 'ABETSALSDE_ARM_MBO': 'Experiments/logs/algorithms_0319/ABETSALSDE_ARM_MBO.py',
    }
    run_algo_eval_from_file_map(evaluator, file_map, options, is_baseline=False)

    # extract results to the ioh format
    extract_algo_result(dir_path=save_dir)

    # plot the results
    plot_algo(dir_path=save_dir, fig_dir=save_dir)


if __name__ == "__main__":
    setup_logger(level=logging.INFO)

    run_evaluation()
