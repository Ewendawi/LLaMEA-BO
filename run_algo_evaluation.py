import logging
import os
import sys
import getopt
import pickle
from datetime import datetime
import numpy as np
from llamevol.utils import setup_logger
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
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

def run_evaluation(algo_name, algo_path, save_dir, is_baseline=False):
    evaluator = get_evaluator()


    options = {
        'save_dir': save_dir,
    }

    # Customize evaluation. The default is the sequential evaluation.
    # thread pool evaluation
    # evaluator.max_eval_workers = 10

    # process pool evaluation
    # evaluator.use_multi_process = True
    # evaluator.max_eval_workers = 10
    
    # bare MPI evaluation
    # evaluator.use_mpi = True

    # MPI future evaluation
    # evaluator.use_mpi_future = True

    # the key is the name of the algorithm, the value is the path of the code file
    file_map = {
        algo_name: algo_path,
    }

    run_algo_eval_from_file_map(evaluator, file_map, options, is_baseline=is_baseline)


def extract_plot_result(dir_path):
    # extract the results from the log files
    extract_algo_result(dir_path=dir_path)

    # plot the results
    plot_algo(dir_path=dir_path, fig_dir=dir_path)


if __name__ == "__main__":
    setup_logger(level=logging.INFO)

    use_mpi = False
    algo_name = None
    algo_path = None
    is_baseline = False
    is_plot = False
    save_dir = 'exp_eval'

    opts, args = getopt.getopt(sys.argv[1:], "n:p:bem", )
    for opt, arg in opts:
        if opt == "-n":
            algo_name = arg
        elif opt == "-p":
            algo_path = arg
        elif opt == "-b":
            is_baseline = True
        elif opt == "-e":
            is_plot = True
        elif opt == "-m":
            use_mpi = True

    if is_plot:
        extract_plot_result(save_dir)
    else:
        if algo_name is None or algo_path is None:
            print("Please provide the algorithm name and path with -n and -p options.")
            sys.exit(1)

        if use_mpi:
            from llamevol.evaluator.MPITaskManager import start_mpi_task_manager 

            with start_mpi_task_manager(result_recv_buffer_size=1024*1024*50) as task_manager:
                if task_manager.is_master:
                    run_evaluation(algo_name, algo_path, save_dir, is_baseline=is_baseline)
        else:
            run_evaluation(algo_name, algo_path, save_dir, is_baseline=is_baseline)

