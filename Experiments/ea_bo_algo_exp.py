import logging
import os
import sys
import getopt
import pickle
from datetime import datetime
import importlib.util
import pathlib

import numpy as np

from llamevol.utils import setup_logger
from llamevol.evaluator.bo_injector import FunctionProfiler, BOInjector
from llamevol.evaluator.ioh_evaluator import IOHEvaluator

from Experiments.plot_algo_res import plot_algo_result, plot_contour

# Utils
def get_IOHEvaluator_for_final_eval(dim=5, budget=100):
    problems = list(range(1, 25))
    instances = [[4, 5, 6]] * len(problems)
    repeat = 5
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    evaluator.timeout = None
    return evaluator

def get_IOHEvaluator_for_test(problems=[3], _instances=[1], repeat=1, budget=100, dim=5):
    instances = [_instances] * len(problems)
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    return evaluator

def dynamic_import_and_get_class(module_path, class_name):
    module_path = pathlib.Path(module_path)
    module_name = module_path.stem

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        print(f"Error: Could not find module at {module_path}")
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        class_object = getattr(module, class_name)
        return class_object
    except AttributeError:
        print(f"Error: Class '{class_name}' not found in module '{module_name}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Evaluate Algorithms

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

def _run_algrothim_eval_exp(evaluator, algo_cls, code=None, save=False, options=None):
    cls_name = algo_cls.__name__
    logging.info("Start evaluating %s on %s", cls_name, evaluator)

    _ignore_cls = False
    extra_init_params = {}
    call_params = {}
    name_suffix = ""
    injector = BOInjector()
    if options is not None:
        is_baseline = options.get("is_baseline", False)
        if is_baseline:
            extra_init_params = baseline_algo_eval_param(evaluator.dim, evaluator.budget)
            if 'device' in options:
                extra_init_params['device'] = options['device']

        if 'cls_init_kwargs' in options:
            extra_init_params.update(options['cls_init_kwargs'])

        if 'max_eval_workers' in options:
            evaluator.max_eval_workers = options['max_eval_workers']

        if 'use_multi_process' in options:
            evaluator.use_multi_process = options['use_multi_process']

        if 'use_mpi' in options:
            evaluator.use_mpi = options['use_mpi']

        if 'use_mpi_future' in options:
            evaluator.use_mpi_future = options['use_mpi_future']

        if 'ignore_cls' in options:
            _ignore_cls = options['ignore_cls']

        if 'ignore_external_metric' in options:
            injector.ignore_metric = options['ignore_external_metric']

        if 'capture_output' in options:
            call_params['capture_output'] = options['capture_output']

        if 'name_suffix' in options:
            name_suffix = options['name_suffix']
    
    if _ignore_cls:
        algo_cls = None

    res = evaluator.evaluate(
        code=code,
        cls_name=cls_name,
        cls=algo_cls,
        cls_init_kwargs=extra_init_params,
        cls_call_kwargs=call_params, 
        injector = injector
    )
    if save:
        save_dir = 'Experiments/algo_eval_res'
        if options is not None and 'save_dir' in options:
            save_dir = options['save_dir']
        dir_path = save_dir
        os.makedirs(dir_path, exist_ok=True)
        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        score = res.score
        file_path = os.path.join(dir_path, f"{cls_name}{name_suffix}_{score:.4f}_{evaluator}_{time_stamp}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(res, f)
    return res


def run_algo_eval_from_file_map(evaluator, file_map=None, cls_list=None, plot=False, save=True, options=None):
    res_list = []
    _cls_list = cls_list
    _code_list = []
    if _cls_list is None or len(_cls_list) == 0:
        _cls_list = []
        for cls_name, file_path in file_map.items():
            if not os.path.exists(file_path):
                logging.warning("File not exist: %s", file_path)
                continue
            code = ""
            with open(file_path, "r") as f:
                code = f.read()

            _cls = dynamic_import_and_get_class(file_path, cls_name)
            if _cls is None:
                continue
            _cls_list.append(_cls)
            _code_list.append(code)

    _time_profile = False
    if options is not None and 'time_profile' in options:
        _time_profile = options['time_profile']

    for i, _cls in enumerate(_cls_list):
        _wrapper = None
        if _time_profile:
            _wrapper = FunctionProfiler()
            _wrapper.name = f"{_cls.__name__}"
            _profiler = _wrapper.wrap_class(_cls)
            _cls = _profiler
        _code = _code_list[i] if i < len(_code_list) else None
        res = _run_algrothim_eval_exp(evaluator, _cls, code=_code, save=save, options=options)
        res_list.append(res)

        if _wrapper is not None:
            _wrapper.print_report()

    if plot:
        plot_algo_result(res_list)

    return res_list

def debug_algo_eval():
    problems = [2]
    # problems = list(range(1, 25))
    instances = [8]
    repeat = 3
    dim=5
    budget = 100

    evaluator = get_IOHEvaluator_for_test(problems=problems, _instances=instances, repeat=repeat, budget=budget, dim=dim)
    evaluator.ignore_over_budget = True

    file_map = {
        # 'BLTuRBO1': 'LLAMBO/Experiments/baselines/bo_baseline.py',
        # 'BLHEBO': 'LLAMBO/Experiments/baselines/bo_baseline.py',
    }

    from Experiments.baselines.bo_baseline import BLTuRBO1, BLTuRBOM, BLRandomSearch, BLSKOpt, BLMaternVanillaBO, BLScaledVanillaBO, BLCMAES, BLHEBO, BLVanillaEIBO
    from Experiments.baselines.vanilla_bo import VanillaBO

    cls_list = [
        # VanillaBO,
        # BLRandomSearch,
        # BLCMAES,
        # BLHEBO,
        # BLMaternVanillaBO,
        # BLTuRBO1,
        # BLTuRBOM,
        # BLSKOpt,
        BLVanillaEIBO,
    ]

    options = {
        # 'device': 'cuda',
        'is_baseline': True,
        # 'max_eval_workers': 4,
        # 'use_multi_process': True,
        # 'use_mpi': True,
        # 'use_mpi_future': True,
        # 'time_profile': True,
        # 'ignore_cls': True, # the module with dynamic import can't be pickled
        # 'ignore_external_metric': False,
        # 'capture_output': False,
    }

    plot = False
    res_list = run_algo_eval_from_file_map(evaluator, file_map, cls_list=cls_list, plot=plot, save=False, options=options)

    for res in res_list:
        for _, r in enumerate(res.result):
            _x_hist = r.x_hist
            if _x_hist.shape[1] == 2:
                r_id = r.id
                r_split = r_id.split("-")
                problem_id = int(r_split[0])
                instance_id = int(r_split[1])
                # repeat_id = int(r_split[2])
                title = f'{res.name} on F{problem_id} instance {instance_id}'
                plot_contour(problem_id=problem_id, instance=instance_id, title=title, points=_x_hist)

def eval_final_algo():
    dim = 10
    budget = 10 * dim + 50
    evaluator = get_IOHEvaluator_for_final_eval(dim=dim, budget=budget)
    evaluator.ignore_over_budget = True

    # problems = list(range(1, 25))
    # instances = [8]
    # repeat = 3
    # budget = 100
    # dim = 10
    # evaluator = get_IOHEvaluator_for_test(problems=problems, _instances=instances, repeat=repeat, budget=budget, dim=dim)

    options = {
        # 'device': 'cuda',
        # 'is_baseline': True,
        'save_dir': f'Experiments/final_eval_res_{dim}dim',
        # 'max_eval_workers': 10,
        # 'use_multi_process': True,
        'use_mpi': True,
        # 'use_mpi_future': True,
        'ignore_cls': True, # the module with dynamic import can't be pickled
        # 'ignore_external_metric': False,
    }
    _bl_file_map = {
        # 'BLRandomSearch': 'Experiments/baselines/bo_baseline.py',
        # 'BLTuRBO1': 'LLAMBO/Experiments/baselines/bo_baseline.py',
        # 'BLTuRBOM': 'Experiments/baselines/bo_baseline.py',
        # 'BLMaternVanillaBO': 'LLAMBO/Experiments/baselines/bo_baseline.py',
        # 'BLScaledVanillaBO': 'Experiments/baselines/bo_baseline.py',
        # 'BLSKOpt': 'Experiments/baselines/bo_baseline.py',
        # 'BLCMAES': 'LLAMBO/Experiments/baselines/bo_baseline.py',
        'BLHEBO': 'LLAMBO/Experiments/baselines/bo_baseline.py',
    }

    _file_map = {
        'AdaptiveTrustRegionOptimisticHybridBO': 'Experiments/2-33_AdaptiveTrustRegionOptimisticHybridBO_0.2043.py',

        'AdaptiveEvolutionaryParetoTrustRegionBO': 'Experiments/4-61_AdaptiveEvolutionaryParetoTrustRegionBO_0.1827.py',

        'AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE': 'Experiments/5-88_AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE_0.2138.py',

        'ATRBO': 'Experiments/0-2_ATRBO_0.0905.py',

        'ABETSALSDE_ARM_MBO': 'Experiments/6-93_ABETSALSDE_ARM_MBO_0.1813.py',
    }

    file_map = _bl_file_map
    file_map = _file_map

    run_algo_eval_from_file_map(evaluator, file_map, plot=False, save=True, options=options)

def eval_atrbo_algo():
    problems = [
        2, 4,
        6, 8,
        12, 14,
        18, 15,
        21, 23,
    ]
    problems = list(range(1, 25))
    instances = [4]
    repeat = 5
    budget = 100
    dim = 5
    evaluator = get_IOHEvaluator_for_test(problems=problems, _instances=instances, repeat=repeat, budget=budget, dim=dim)
    evaluator.ignore_over_budget = True

    default_init_params = {
        'tr_radius': 2.5,
        'rho': 0.95,
        'kappa': 2.0,
        'fix_adjust': False,
        'with_proj': True,
        'adaptive_radius': True,
        'adaptive_kappa': True,
    }

    def _copy_update_dict(d, _new_dict):
        new_dict = d.copy()
        new_dict.update(_new_dict)
        return new_dict

    exp_list = [
        # _copy_update_dict(default_init_params, {}),
        # _copy_update_dict(default_init_params, {'fix_adjust': True}),
        # _copy_update_dict(default_init_params, {'with_proj': False}),
        _copy_update_dict(default_init_params, {'fix_adjust': False, 'adaptive_radius': False}),
        _copy_update_dict(default_init_params, {'fix_adjust': False, 'adaptive_kappa': False}),
        _copy_update_dict(default_init_params, {'fix_adjust': False, 'adaptive_radius': False, 'adaptive_kappa': False}),

        # rho
        # _copy_update_dict(default_init_params, {'fix_adjust': False, 'rho': 0.8}),
        # _copy_update_dict(default_init_params, {'fix_adjust': False, 'rho': 0.65}),

        # kappa
        _copy_update_dict(default_init_params, {'fix_adjust': False, 'kappa': 4.0, 'adaptive_kappa': False, 'adaptive_radius': False}),
        # _copy_update_dict(default_init_params, {'fix_adjust': True, 'kappa': 2.0, 'adaptive_kappa': False, 'adaptive_radius': False}),
        _copy_update_dict(default_init_params, {'fix_adjust': False, 'kappa': 1.0, 'adaptive_kappa': False, 'adaptive_radius': False}),

        # tr_radius
        _copy_update_dict(default_init_params, {'fix_adjust': False, 'tr_radius': 5, 'adaptive_radius': False, 'adaptive_kappa': False}),
        # _copy_update_dict(default_init_params, {'fix_adjust': True, 'tr_radius': 2.5, 'adaptive_radius': False, 'adaptive_kappa': False}),
        _copy_update_dict(default_init_params, {'fix_adjust': False, 'tr_radius': 1, 'adaptive_radius': False, 'adaptive_kappa': False}),
    ]

    for init_params in exp_list:
        name_suffix = f"_fixed:{init_params['fix_adjust']}_proj:{init_params['with_proj']}_trr:{init_params['tr_radius']}_rho:{init_params['rho']}_k:{init_params['kappa']}_adaptr:{init_params['adaptive_radius']}_adaptk:{init_params['adaptive_kappa']}"

        options = {
            # 'device': 'cuda',
            # 'is_baseline': True,
            'save_dir': f'Experiments/atrbo_eval_res_{dim}dim',
            # 'max_eval_workers': 10,
            # 'use_multi_process': True,
            'use_mpi': True,
            # 'use_mpi_future': True,
            'ignore_cls': True, # the module with dynamic import can't be pickled
            # 'ignore_external_metric': False,
            'cls_init_kwargs': init_params, 
            'name_suffix': name_suffix,
            # 'capture_output': False,
        }

        _file_map = {
            'ATRBO': 'Experiments/0-2_ATRBO_modified.py',
        }
        logging.info("ATRBO algo eval on %s", init_params)
        run_algo_eval_from_file_map(evaluator, _file_map, plot=False, save=True, options=options)


def main():
    # setup_logger(level=logging.DEBUG)
    setup_logger(level=logging.INFO)

    # debug_algo_eval()
    # eval_final_algo()
    eval_atrbo_algo()

if __name__ == "__main__":
    use_mpi = False
    opts, args = getopt.getopt(sys.argv[1:], "m", ["mpi"])
    for opt, arg in opts:
        if opt == "-m":
            use_mpi = True
        elif opt == "--mpi":
            use_mpi = True
    

    if use_mpi:
        from llamevol.evaluator.MPITaskManager import start_mpi_task_manager 

        with start_mpi_task_manager(result_recv_buffer_size=1024*1024*50) as task_manager:
            if task_manager.is_master:
                main()
    else:
        main()
