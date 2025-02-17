import os
import random
import logging
from datetime import datetime
import importlib.util
import pathlib
import pickle
import torch
import numpy as np
from llamea import LLaMBO
from llamea.llm import LLMmanager, LLMS
from llamea.prompt_generators import PromptGenerator, BaselinePromptGenerator, TunerPromptGenerator, LightBaselinePromptGenerator, GenerationTask
from llamea.population import Population, ESPopulation, IslandESPopulation, max_divese_desc_get_parent_fn, diversity_awarness_selection_fn, desc_similarity_from_handlers, code_diff_similarity_from_handlers, family_competition_selection_fn
from llamea.evaluator.ioh_evaluator import IOHEvaluator, AbstractEvaluator
from llamea.utils import setup_logger
from llamea.individual import Individual
from llamea.evaluator.injected_critic import FunctionProfiler
from Experiments.plot import plot_algo_result, plot_contour

# Utils
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

def get_IOHEvaluator_for_evol():
    budget = 100
    dim = 5
    problems = list(range(1, 25))
    instances = [[1, 2]] * len(problems)
    repeat = 2
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    return evaluator

def get_light_IOHEvaluator_for_crossover():
    budget = 100
    dim = 5
    problems = [1]
    instances = [[1]] * len(problems)
    repeat = 1
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    return evaluator

def get_IOHEvaluator_for_final_eval():
    budget = 100
    dim = 5
    problems = list(range(1, 25))
    instances = [[4, 5, 6]] * len(problems)
    repeat = 5
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    return evaluator

def get_IOHEvaluator_for_light_evol():
    budget = 100
    dim = 5
    problems = [
        2, 4,
        6, 8,
        12, 14,
        18, 15,
        21, 23,
    ]
    instances = [[1]] * len(problems)
    repeat = 3
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    return evaluator

def get_IOHEvaluator_for_test(problems=[3], _instances=[1], repeat=1, budget=100, dim=5):
    budget = budget
    problems = problems
    instances = [_instances] * len(problems)
    repeat = repeat
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    return evaluator

def get_bo_prompt_generator():
    prompt_generator = BaselinePromptGenerator()
    prompt_generator.is_bo = True
    return prompt_generator

def get_light_Promptor_for_crossover():
    prompt_generator = LightBaselinePromptGenerator()
    prompt_generator.is_bo = True
    return prompt_generator

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

    _max_eval_workers = 0
    _use_multi_process = False
    _ignore_cls = False
    extra_init_params = {}
    if options is not None:
        is_baseline = options.get("is_baseline", False)
        if is_baseline:
            extra_init_params = baseline_algo_eval_param(evaluator.dim, evaluator.budget)
            if options is not None:
                if 'device' in options:
                    extra_init_params['device'] = options['device']

        if 'max_eval_workers' in options:
            _max_eval_workers = options['max_eval_workers']

        if 'use_multi_process' in options:
            _use_multi_process = options['use_multi_process']

        if 'ignore_cls' in options:
            _ignore_cls = options['ignore_cls']
    
    if _ignore_cls:
        algo_cls = None

    res = evaluator.evaluate(
        code=code,
        cls_name=cls_name,
        cls=algo_cls,
        cls_init_kwargs=extra_init_params,
        max_eval_workers=_max_eval_workers,
        use_multi_process=_use_multi_process
    )
    if save:
        save_dir = 'Experiments/algo_eval_res'
        if options is not None and 'save_dir' in options:
            save_dir = options['save_dir']
        dir_path = save_dir
        os.makedirs(dir_path, exist_ok=True)
        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        file_path = os.path.join(dir_path, f"{cls_name}_{evaluator}_{time_stamp}.pkl")
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

        


# EA Experiments
def _run_exp(prompt_generator:PromptGenerator,
            evaluator:AbstractEvaluator,
            llm:LLMmanager,
            population:Population,
            n_generations:int=200,
            n_population:int=30,
            gpu_name:str=None,
            max_interval:int=5,
            n_query_threads:int=0,
            n_eval_workers:int=0,
            time_out_per_eval:int=None,
            options:dict=None,
            ):
    llambo = LLaMBO()

    if options is not None:
        eval_overwrite_type = options.get("eval_overwrite_type", None)
        if eval_overwrite_type is not None:
            if eval_overwrite_type == 'test':
                _test_problems = options.get("test_eval_problems", [3])
                _test_instances = options.get("test_eval_instances", [1])
                _test_repeat = options.get("test_eval_repeat", 1)
                _test_budget = options.get("test_eval_budget", 100)
                evaluator = get_IOHEvaluator_for_test(problems=_test_problems, _instances=_test_instances, repeat=_test_repeat, budget=_test_budget)
            elif eval_overwrite_type == 'light_evol':
                evaluator = get_IOHEvaluator_for_light_evol()
            elif eval_overwrite_type == 'evol':
                evaluator = get_IOHEvaluator_for_evol()
            elif eval_overwrite_type == 'final_eval':
                evaluator = get_IOHEvaluator_for_final_eval()
        if "eval_inject_critic" in options:
            evaluator.inject_critic = options["eval_inject_critic"]

        if 'prompt_problem_desc' in options:
            prompt_generator.problem_desc = options['prompt_problem_desc']

        if "pop_load_check_point_path" in options:
            check_point_path = options["pop_load_check_point_path"]
            if os.path.exists(check_point_path):
                population = Population.load(check_point_path)
                logging.info("Load population from check point: %s", check_point_path)
            else:
                logging.warning("Check point path not exist: %s", check_point_path)
        elif "pop_warmstart_handlers" in options:
            warmstart_handlers = options["pop_warmstart_handlers"]
            warmstart_inds = []
            for _handler in warmstart_handlers:
                handler = _handler
                if isinstance(_handler, str):
                    if not os.path.exists(_handler):
                        logging.warning("Warmstart handler path not exist: %s", _handler)
                        continue
                    with open(_handler, "rb") as f:
                        handler = pickle.load(f)
                ind = Individual()
                Population.set_handler_to_individual(ind, handler)
                ind.fitness = handler.eval_result.score
                population.add_individual(ind, generation=0)
                warmstart_inds.append(ind)
            if len(warmstart_inds) > 0:
                population.select_next_generation()
                logging.info("Warmstart %d individuals", len(warmstart_inds))

        if "pop_debug_save_on_the_fly" in options:
            population.debug_save_on_the_fly = options["pop_debug_save_on_the_fly"]

        if "pop_save_check_point_interval" in options:
            population.save_per_generation = options["pop_save_check_point_interval"]

        if "pop_preorder_aware_init" in options:
            population.preorder_aware_init = options["pop_preorder_aware_init"]

        if "pop_replaceable_parent_selection" in options:
            population.replaceable_parent_selection = options["pop_replaceable_parent_selection"]
            
        if "pop_random_parent_selection" in options:
            population.random_parent_selection = options["pop_random_parent_selection"]
            
        if "pop_exclusive_operations" in options:
            population.exclusive_operations = options["pop_exclusive_operations"]

        if "pop_cross_over_rate" in options:
            population.cross_over_rate = options["pop_cross_over_rate"]

        if "pop_cr_light_eval" in options:
            population.light_cross_over_evaluator = options["pop_cr_light_eval"]

        if 'pop_cr_light_promptor' in options:
            population.light_cross_over_promptor = options['pop_cr_light_promptor']

        if "pop_parent_strategy" in options:
            population.get_parent_strategy = options["pop_parent_strategy"]

        if "pop_selection_strategy" in options:
            population.selection_strategy = options["pop_selection_strategy"]

        if "pop_save_dir" in options:
            population.save_dir = options["pop_save_dir"]

        if 'llm_mocker' in options:
            llm.mock_res_provider = options['llm_mocker']

    if population.get_current_generation() == 0:
        # population.name += f"_{llm.model_name()}_{prompt_generator}_{evaluator}"
        population.name += f"_{evaluator}"
        if torch.cuda.is_available():
            population.name += "_gpu"

    llambo.run_evolutions(llm, evaluator, prompt_generator, population,
                        n_generation=n_generations, n_population=n_population,
                        n_retry=3,
                        time_out_per_eval=time_out_per_eval,
                        n_query_threads=n_query_threads,
                        n_eval_workers=n_eval_workers,
                        gpu_name=gpu_name,
                        max_interval=max_interval
                        )

    population.save(suffix='final')

def tune_algo(file_path, cls_name, res_path, params, should_eval=False, plot=False, test_eval=False, pop_path=None):
    code = ""
    with open(file_path, "r") as f:
        code = f.read()

    tuner = TunerPromptGenerator()
    tuner_evaluator = get_IOHEvaluator_for_light_evol()
    if test_eval:
        tuner_evaluator = get_IOHEvaluator_for_test()

    params["prompt_generator"] = tuner
    params["evaluator"] = tuner_evaluator

    if should_eval:
        cls_init_kwargs = {
            'dim': 5,
            'budget': 100,
        }
        cls_init_kwargs['dim'] = params.get("dim", 5)
        cls_init_kwargs['budget'] = params.get("budget", 100)
        res = tuner_evaluator.evaluate(code=code, cls_name=cls_name, cls=None, cls_init_kwargs=cls_init_kwargs)

        # save res
        with open(res_path, "wb") as f:
            pickle.dump(res, f)

    with open(res_path, "rb") as f:
        res = pickle.load(f)

    logging.info("Results: %s", res)

    if plot:
        plot_algo_result([res])

    if pop_path is not None and os.path.exists(pop_path):
        population = Population.load(pop_path)
    else:
        population = ESPopulation(n_parent=1, n_parent_per_offspring=1, n_offspring=1)
        population.name = f"tune_{cls_name}_1+1"
        population.debug_save_on_the_fly = True

        ind = Individual()
        handler = tuner.get_response_handler()
        handler.eval_result = res
        handler.code = code
        handler.code_name = cls_name

        Population.set_handler_to_individual(ind, handler)
        ind.name = cls_name
        ind.fitness = res.score
        population.add_individual(ind, generation=0)
        population.select_next_generation()

    _run_exp(
        population=population,
        **params
    )

def run_mu_plus_lambda_exp(
                    n_parent:int=2,
                    n_offspring:int=1,
                    n_parent_per_offspring:int=2,
                    **kwargs
                    ):
    population = ESPopulation(n_parent=n_parent, n_parent_per_offspring=n_parent_per_offspring, n_offspring=n_offspring)
    population.name = f"evol_{n_parent}+{n_offspring}"

    _run_exp(
        population=population,
        **kwargs
    )

def run_island_exp(
        n_parent:int=2,
        n_offspring:int=1,
        n_parent_per_offspring:int=2,
        n_islands:int=3,
        n_warmup_generations:int=3,
        n_cambrian_generations:int=2,
        n_neogene_generations:int=2,
        **kwargs):

    population = IslandESPopulation(n_parent=n_parent,
                                    n_parent_per_offspring=n_parent_per_offspring,
                                    n_offspring=n_offspring,
                                    n_islands=n_islands,
                                    n_warmup_generations=n_warmup_generations,
                                    n_cambrian_generations=n_cambrian_generations,
                                    n_neogene_generations=n_neogene_generations
                                    )
    population.preorder_aware_init = True
    population.get_parent_strategy = max_divese_desc_get_parent_fn
    population.selection_strategy = diversity_awarness_selection_fn

    population.name = f"{n_parent}+{n_offspring}_island_{n_islands}"

    _run_exp(
        population=population,
        **kwargs
    )


def tune_vanilla_bo(params):
    file_path = "Experiments/baselines/vanilla_bo.py"
    cls_name = "VanillaBO"
    res_path = "Experiments/baselines/vanilla_bo_res.pkl"
    pop_path = None
    should_eval = False
    plot = False
    test_eval = False
    tune_algo(file_path, cls_name, res_path, params, should_eval=should_eval, plot=plot, test_eval=test_eval, pop_path=pop_path)

def debug_algo_eval():
    problems = [2]
    # problems = list(range(1, 25))
    instances = [8]
    repeat = 3
    dim=5
    budget = 100

    evaluator = get_IOHEvaluator_for_test(problems=problems, _instances=instances, repeat=repeat, budget=budget, dim=dim)
    evaluator.inject_critic = True
    evaluator.ignore_over_budget = True

    file_map = {
        # 'EnsembleLocalSearchBOv1': 'Experiments/test_cands/EnsembleLocalSearchBOv1.py',
        # 'BLTuRBO1': 'Experiments/baselines/bo_baseline.py',
        # 'AdaptiveBatchBOv7': 'Experiments/pop_40_f/ESPopulation_evol_1+1_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210043001/25-26_AdaptiveBatchBOv7_0.0748.py',

        # 'AdaptiveLocalPenaltyVarianceBOv3':'Experiments/pop_40_f/ESPopulation_evol_12+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208230817/5-40_AdaptiveLocalPenaltyVarianceBOv3_0.0482.py',
    }

    from Experiments.baselines.bo_baseline import BLTuRBO1, BLTuRBOM, BLRandomSearch, BLSKOpt, BLMaternVanillaBO, BLScaledVanillaBO, BLCMAES, BLHEBO
    from Experiments.baselines.vanilla_bo import VanillaBO
    from Experiments.test_cands.EnsembleLocalSearchBOv1 import EnsembleLocalSearchBOv1
    from Experiments.test_cands.EnsembleDeepKernelAdaptiveTSLocalSearchARDv1 import EnsembleDeepKernelAdaptiveTSLocalSearchARDv1
    from Experiments.test_cands.QMCBOv1 import GP_Matern_EI_MSL_SobolBOv1

    cls_list = [
        # VanillaBO,
        # BLRandomSearch,
        # BLCMAES,
        # BLHEBO,
        # BLMaternVanillaBO,
        # BLTuRBO1,
        # BLTuRBOM,
        # BLSKOpt,
        # EnsembleLocalSearchBOv1,
        # EnsembleDeepKernelAdaptiveTSLocalSearchARDv1,
        # GP_Matern_EI_MSL_SobolBOv1,
    ]

    options = {
        # 'device': 'cuda',
        'is_baseline': True,
        # 'max_eval_workers': 4,
        # 'use_multi_process': True,
        # 'time_profile': True,
    }

    plot = True
    res_list = run_algo_eval_from_file_map(evaluator, file_map, cls_list=cls_list, plot=plot, save=False, options=options)

    for res in res_list:
        for i, r in enumerate(res.result):
            r_id = r.id
            r_split = r_id.split("-")
            problem_id = int(r_split[0])
            instance_id = int(r_split[1])
            repeat_id = int(r_split[2])
            title = f'{res.name} on F{problem_id} instance {instance_id} repeat {repeat_id}'
            _x_hist = r.x_hist
            if _x_hist.shape[1] == 2:
                plot_contour(problem_id=problem_id, instance=instance_id, title=title, points=_x_hist)

def eval_final_algo():
    evaluator = get_IOHEvaluator_for_final_eval()
    evaluator.inject_critic = True
    evaluator.ignore_over_budget = True

    
    # problems = list(range(1, 25))
    # instances = [8]
    # repeat = 3
    # budget = 100
    # evaluator = get_IOHEvaluator_for_test(problems=problems, _instances=instances, repeat=repeat, budget=budget)

    options = {
        # 'device': 'cuda',
        'is_baseline': True,
        'save_dir': 'Experiments/final_eval_res',
        # 'max_eval_workers': 4,
        # 'use_multi_process': True,
        # 'ignore_cls': True, # sub-process can't find the module with dynamic import
    }
    _bl_file_map = {
        # 'BLRandomSearch': 'Experiments/baselines/bo_baseline.py',
        # 'BLTuRBO1': 'Experiments/baselines/bo_baseline.py',
        # 'BLTuRBOM': 'Experiments/baselines/bo_baseline.py',
        # 'BLMaternVanillaBO': 'Experiments/baselines/bo_baseline.py',
        # 'BLScaledVanillaBO': 'Experiments/baselines/bo_baseline.py',
        # 'BLSKOpt': 'Experiments/baselines/bo_baseline.py',
        # 'BLCMAES': 'Experiments/baselines/bo_baseline.py',
        'BLHEBO': 'Experiments/baselines/bo_baseline.py',
    }

    _file_map = {
        # 0.04
        # 'NoisyBanditBOv1': 'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065417/0-10_NoisyBanditBOv1_0.0434.py',
        # 'ParetoActiveBOv1': 'Experiments/pop_40_f/ESPopulation_evol_4+2_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210025714/0-3_ParetoActiveBOv1_0.0426.py',

        # 0.05
        # 'AdaptiveBatchUCBLocalSearchBOv2': 'Experiments/pop_40_f/ESPopulation_evol_4+4_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210043822/4-20_AdaptiveBatchUCBLocalSearchBOv2_0.0526.py',
        # 'AdaptiveControlVariateBOv4': 'Experiments/pop_40_f/ESPopulation_evol_8+4_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209152106/6-29_AdaptiveControlVariateBOv4_0.0595.py'

        # 0.06
        # 'AdaptiveEvoBatchHybridBOv2': 'Experiments/pop_40_f/ESPopulation_evol_12+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208224540/4-32_AdaptiveEvoBatchHybridBOv2_0.0619.py',
        # 'MultiObjectiveBOv1': 'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065417/0-19_MultiObjectiveBOv1_0.0665.py',
        # 'AdaptiveHybridBOv6':'Experiments/pop_40_f/ESPopulation_evol_1+1_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0211000947/10-11_AdaptiveHybridBOv6_0.0615.py',
        # 'AdaptiveTrustRegionDynamicAllocationBOv2': 'Experiments/pop_40_f/ESPopulation_evol_8+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065623/2-22_AdaptiveTrustRegionDynamicAllocationBOv2_0.0650.py'

        # 0.07
        # 'AdaptiveTrustRegionVarianceQuantileDEBOv2': 'Experiments/pop_40_f/ESPopulation_evol_12+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208225600/5-40_AdaptiveTrustRegionVarianceQuantileDEBOv2_0.0731.py',

        # 0.08
        # 'TrustRegionAdaptiveTempBOv2': 'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209173952/4-23_TrustRegionAdaptiveTempBOv2_0.0807.py',
        # 'BayesLocalAdaptiveAnnealBOv1': 'Experiments/pop_40_temp/ESPopulation_evol_10+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208164605/3-24_BayesLocalAdaptiveAnnealBOv1_0.0827.py',

        # 'EnsembleLocalSearchBOv1': 'Experiments/test_cands/EnsembleLocalSearchBOv1.py',
    }

    file_map = _bl_file_map

    run_algo_eval_from_file_map(evaluator, file_map, plot=False, save=True, options=options)



def show_code_similarity():
    file_paths = [
        'Experiments/temperature_res/0215081907/temperature_res.pkl',
    ]

    for file_path in file_paths:
        with open(file_path, "rb") as f:
            target = pickle.load(f)


        for temperature_res in target:
            mean_sim, sim_matrix = code_diff_similarity_from_handlers(temperature_res[0])

            print(f"Mean similarity: {mean_sim}")
            print(sim_matrix)

        pass

class temperatureRes:
    def __init__(self):
        self.parent_name = None
        self.parent_handlers = None
        self.desc_mean_sim = None
        self.desc_sim_matrix = None
        self.code_mean_sim = 0
        self.code_sim_matrix = None
        self.parent_handler = None
        self.res_list = []

def run_temperature_exp():

    # show_code_similarity()
 
    file_paths = [
        'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209171704/0-2_BOTSDynBOv1_respond.md',
        
        'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209171704/1-10_AdaEEBOv2_respond.md', 

        'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209165843/0-2_ThompsonSamplingBOv1_respond.md',
        
        
        'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-2_TrustRegionBOv1_respond.md',

        'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-3_GradientEnhancedBOv1_respond.md',
        
        'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-10_DuelingBanditBOv1_respond.md',
        
        'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-11_SurrogateModelFreeBOv1_respond.md',

        'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-10_DuelingBanditBOv1_respond.md',

        'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-9_BayesMetaLearningBOv1_respond.md',
        
        'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209061439/0-4_BayesUCBwithRBFBOv1_respond.md',
        
        'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209061439/0-5_DEwithLocalSearchBOv1_respond.md',

        'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209061439/0-9_StochasticLHSwithHistoryBOv1_respond.md',

        'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065238/0-12_VarianceReductionBOv1_respond.md',

        'Experiments/pop_40_f/ESPopulation_evol_20+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208225928/0-6_DynamicPenaltyBOv1_respond.md'
    ]

    def _get_prompt_msg(num, chunk_size, promptor, current_task):
        size = num * chunk_size
        _selected_index = np.random.choice(len(file_paths), size=size, replace=False)
        _selected_files = [file_paths[i] for i in _selected_index]

        if current_task == GenerationTask.OPTIMIZE_PERFORMANCE:
            parent_handler_list = []
            for i, file_path in enumerate(_selected_files):
                prompt = ""
                with open(file_path, "r") as f:
                    prompt = f.read()

                handler = promptor.get_response_handler() 
                handler.extract_response(prompt, current_task)
                parent_handler_list.append(handler)
            parent_handlers = [parent_handler_list[i:i+chunk_size] for i in range(0, len(parent_handler_list), chunk_size)]
        else:
            parent_handlers = [[]] * num

        messages_list = []
        for parent in parent_handlers:
            role_setting, prompt = promptor.get_prompt(
                task=current_task,
                problem_desc=None,
                candidates=parent,
                )
            session_messages = [
                {"role": "system", "content": role_setting},
                {"role": "user", "content": prompt},
            ]
            messages_list.append(session_messages)
        return messages_list, parent_handlers

    llmbo = LLaMBO()
    current_task = GenerationTask.OPTIMIZE_PERFORMANCE
    llm = get_llm()
    promptor = BaselinePromptGenerator()
    promptor.is_bo = True
    evaluator = get_IOHEvaluator_for_test(problems=[4], _instances=[1], repeat=1, budget=100)

    save_dir = 'Experiments/temperature_res'
    time_stamp = datetime.now().strftime("%m%d%H%M%S")
    save_dir = os.path.join(save_dir, time_stamp)
    os.makedirs(save_dir, exist_ok=True)

    def _save_code(file_dir, handler, prefix):
        code = handler.code
        name = handler.code_name
        file_name = f"{prefix}_{name}.py"
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, "w") as f:
            f.write(code)

        respond = handler.raw_response
        res_file_name = f"{prefix}-{name}_respond.md"
        res_file_path = os.path.join(file_dir, res_file_name)
        with open(res_file_path, "w") as f:
            f.write(respond)

     # initial
    chunk_size = 0
    # mutation
    # chunk_size = 1
    # crossover
    # chunk_size = 2

    current_task = GenerationTask.OPTIMIZE_PERFORMANCE if chunk_size > 0 else GenerationTask.INITIALIZE_SOLUTION

    temperatures = [0.0, 0.4, 0.8, 1.2, 1.6]
    temperatures = [2.0]
    params = temperatures
    param_name = "temperature"

    # top_p_list = [0.4, 0.6, 0.8, 1.0]
    # top_p_list = [0.4]
    # params = top_p_list
    # param_name = "top_p"

    # top_k_list = [4, 10, 20, 40]
    # top_k_list = [40]
    # params = top_k_list
    # param_name = "top_k"
    
    num = 1
    repeat = 2

    messages_list, parent_handlers = _get_prompt_msg(num, chunk_size, promptor, current_task)
    param_rest_map = {}
    for param in params:
        print(f"{param_name}: {param}")
        messages_res_list = []
        options = {
            'llm_params': {
                f'{param_name}': param,
            }
        }
        for i, messages in enumerate(messages_list):
            parent = parent_handlers[i]
            for j, parent_handler in enumerate(parent):
                prefix = f"{param_name}-{param}-0.{j}"
                _save_code(save_dir, parent_handler, prefix)

                print(f"Prompt {parent_handler.code_name}")
            
            res_list = []
            for j in range(repeat):
                next_handler = promptor.get_response_handler()
                llmbo.evalution_func(
                    session_messages=messages,
                    llm=llm,
                    evaluator=evaluator,
                    task=current_task,
                    retry=1,
                    response_handler=next_handler,
                    options=options
                )
                prefix = f"{param_name}-{param}-1.{j}"
                _save_code(save_dir, next_handler, prefix)
                res_list.append(next_handler)

            comp_res_list = parent + res_list

            mean_sim, sim_matrix = desc_similarity_from_handlers(comp_res_list) 
            print('Desc similarity')
            print(mean_sim)
            print(sim_matrix)

            code_mean_sim, code_sim_matrix = code_diff_similarity_from_handlers(comp_res_list)
            print('Code similarity')
            print(code_mean_sim)
            print(code_sim_matrix)

            temp_res = temperatureRes()
            temp_res.parent_name = '_'.join([handler.code_name for handler in parent])
            temp_res.parent_handlers = parent
            temp_res.res_list = res_list
            temp_res.desc_mean_sim = mean_sim
            temp_res.desc_sim_matrix = sim_matrix
            temp_res.code_mean_sim = code_mean_sim
            temp_res.code_sim_matrix = code_sim_matrix
            messages_res_list.append(temp_res)
            
        param_rest_map[param] = messages_res_list
        print("")
    file_name = f"{param_name}_res.pkl"
    with open(os.path.join(save_dir, file_name), "wb") as f:
        pickle.dump(param_rest_map, f)

def get_search_default_params():
    params = {
        # "time_out_per_eval": 60 * 20,
        "time_out_per_eval": None,

        "llm": get_llm(),
        "prompt_generator": get_bo_prompt_generator(),
        "n_generations": np.inf,
        "n_population": 40,
        "n_query_threads": 0,
        "n_eval_workers": 0,
        "gpu_name": "cuda:7",
        "max_interval": 5,
        "evaluator": get_IOHEvaluator_for_evol(),
        "options": {
            # 'pop_load_check_point_path': "Experiments/pop_temp/xxx.pkl",
            'pop_debug_save_on_the_fly': True,
            # 'pop_warmstart_handlers': [handler|handler_path],
            # 'pop_save_check_point_interval': 1,
            # 'pop_save_dir': 'Experiments/pop_temp',

            # 'pop_preorder_aware_init': True,
            # 'pop_parent_strategy': max_divese_desc_get_parent_fn,
            # 'pop_selection_strategy': diversity_awarness_selection_fn,
            # 'pop_replaceable_parent_selection': True,
            # 'pop_random_parent_selection': False,
            # 'pop_exclusive_operations': True,


            # 'eval_inject_critic': True,

            # 'eval_overwrite_type': 'test', # 'test', 'light_evol', 'evol', 'final_eval' 
            # 'test_eval_problems': [4, 10],
            # 'test_eval_instances': [1],
            # 'test_eval_repeat': 2,
            # 'test_eval_budget': 60,

            # 'llm_mocker': None,
        }
    }
    return params

def get_llm():
    # MODEL = 'deepseek/deepseek-chat'

    MODEL = 'gemini-2.0-flash-exp'
    # MODEL = 'gemini-1.5-flash'
    # MODEL = 'gemini-2.0-pro-exp'
    # MODEL = 'gemini-2.0-flash-thinking-exp'
    # MODEL = 'gemini-exp-1206'

    # MODEL = 'llama3-70b-8192'
    # MODEL = 'llama-3.3-70b-versatile'
    # MODEL = 'deepseek-r1-distill-llama-70b'
    # MODEL = 'deepseek-r1-distill-qwen-32b'
    
    # MODEL = 'o_gemini-flash-1.5-8b-exp'
    # MODEL = 'o_gemini-2.0-flash-exp'

    # MODEL = 'onehub-gemini-2.0-flash'
    # MODEL = 'onehub-gemma2-9b-it'

    llm = LLMmanager(model_key=MODEL)

    return llm

def mock_res_provider(*args, **kwargs):
    file_list = [
        'Experiments/pop_temp/ESPopulation_evol_2+2_IOHEvaluator_f4_f10_dim-5_budget-100_instances-[1]_repeat-2_0208014955/0-1_AdaGPUCBBOv2_respond.md',

        'Experiments/pop_temp/ESPopulation_evol_2+2_IOHEvaluator_f4_f10_dim-5_budget-100_instances-[1]_repeat-2_0208020129/1-4_AdaptiveBayesBOv5_respond.md',

        'Experiments/pop_temp/ESPopulation_evol_4+2_IOHEvaluator_f4_f10_dim-5_budget-100_instances-[1]_repeat-2_0208020829/1-5_BayesTrustRegionAdaptiveBOv1_respond.md',

        'Experiments/pop_temp/ESPopulation_evol_4+2_IOHEvaluator_f4_f10_dim-5_budget-100_instances-[1]_repeat-2_0208020829/0-4_BayesTrustRegionBOv1_respond.md',

        'Experiments/pop_temp/ESPopulation_evol_2+2_IOHEvaluator_f4_f10_dim-5_budget-60_instances-[1]_repeat-2_0208010238/0-2_EfficientHybridBOv1_respond.md',
    ]
    file_path = np.random.choice(file_list, size=1)[0] 
    # file_path = np.random.choice(file_list, size=1, p=[0.25, 0.25, 0.25, 0.25])[0]

    response = None
    with open(file_path, "r", encoding="utf-8") as f:
        response = f.read()
    return response

if __name__ == "__main__":
    # setup_logger(level=logging.DEBUG)
    setup_logger(level=logging.INFO)

    # run_temperature_exp()

    # debug_algo_eval()
    # eval_final_algo()

    _params = get_search_default_params()
    _new_params = {
        "n_population": 4,
        "n_query_threads": 0,

        # Choose time_out_per_eval carefully when running multiple evaluations of expriments in parallel due to OS's dispatching mechanism
        "n_eval_workers": 0,
        "time_out_per_eval": 60 * 20,

        # "gpu_name": "cuda:7",
        "gpu_name": None,

        "options": {
            'pop_debug_save_on_the_fly': True,
            # 'pop_warmstart_handlers': [],
            # 'pop_load_check_point_path': 'Experiments/pop_40_test/ESPopulation_evol_2+4_IOHEvaluator_f2_f4_f8_f14_f15_f23_dim-5_budget-100_instances-[1]_repeat-3_0216054105_b/ESPopulation_gen_checkpoint_0_0216055117.pkl',

            'pop_save_check_point_interval': 1,
            'pop_preorder_aware_init': True,
            # 'pop_parent_strategy': max_divese_desc_get_parent_fn,
            # 'pop_selection_strategy': diversity_awarness_selection_fn,
            # 'pop_selection_strategy': family_competition_selection_fn(parent_size_threshold=1, is_aggressive=True),
            'pop_save_dir': 'Experiments/pop_40_test',

            # 'pop_replaceable_parent_selection': False,
            # 'pop_random_parent_selection': True,
            # 'pop_cross_over_rate': 0.5,
            # 'pop_exclusive_operations': False,
            # 'pop_cr_light_eval': get_light_IOHEvaluator_for_crossover(),
            # 'pop_cr_light_promptor': get_light_Promptor_for_crossover(),


            'eval_inject_critic': False,
            'eval_overwrite_type': 'test', # 'test', 'light_evol', 'evol', 'final_eval' 
            # 'test_eval_problems': [4], # [4, 10],
            'test_eval_problems': [2, 4, 8, 14, 15, 23],
            'test_eval_instances': [1],
            'test_eval_repeat': 3,
            'test_eval_budget': 100,
            'prompt_problem_desc': 'one noiseless function:F2 Ellipsoid Separable Function',

            # 'llm_mocker': mock_res_provider,
        }
    }

    _params.update(_new_params)

    N_PARENT = 2
    N_OFFSPRING = 1

    # run_mu_plus_lambda_exp(
    #     n_parent=N_PARENT,
    #     n_offspring=N_OFFSPRING,
    #     **_params)
