import os
import logging
from datetime import datetime
import importlib.util
import pathlib
import pickle
import torch
import numpy as np
from llamea import LLaMBO
from llamea.llm import LLMmanager, LLMS
from llamea.prompt_generators import PromptGenerator, BaselinePromptGenerator, TunerPromptGenerator
from llamea.population import Population, ESPopulation, IslandESPopulation, max_divese_desc_get_parent_fn, diversity_awarness_selection_fn
from llamea.evaluator.ioh_evaluator import IOHEvaluator, AbstractEvaluator
from llamea.utils import setup_logger
from llamea.utils import plot_algo_result
from llamea.individual import Individual
from llamea.evaluator.injected_critic import FunctionProfiler


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

def get_IOHEvaluator_for_test(problems=[3], _instances=[1], repeat=1, budget=100):
    budget = budget
    dim = 5
    problems = problems
    instances = [_instances] * len(problems)
    repeat = repeat
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    return evaluator

def get_bo_prompt_generator():
    prompt_generator = BaselinePromptGenerator()
    prompt_generator.is_bo = True
    return prompt_generator


# Evaluate Algorithms
def baseline_algo_eval_param(dim, budget):
    bl_init_params = {
        "budget": budget,
        "dim": dim,
        "bounds": np.array([[-5.0] * 5, [5.0] * 5]),
        "n_init": min(5 * dim, budget // 2),
        "seed": None,
        "device": "cpu",
        # "device": "cuda",
    }
    return bl_init_params

def _run_algrothim_eval_exp(evaluator, algo_cls, code=None, save=False, options=None):
    cls_name = algo_cls.__name__
    logging.info("Start evaluating %s on %s", cls_name, evaluator)

    _max_eval_workers = 0
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

    res = evaluator.evaluate(
        code=code,
        cls_name=cls_name,
        cls=algo_cls,
        cls_init_kwargs=extra_init_params,
        max_eval_workers=_max_eval_workers,
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
        
        if "pop_load_check_point_path" in options:
            check_point_path = options["pop_load_check_point_path"]
            if os.path.exists(check_point_path):
                population = Population.load(check_point_path)
                logging.info("Load population from check point: %s", check_point_path)
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

def run_mu_plus_lambda_diversity_exp(
                    n_parent:int=2,
                    n_offspring:int=1,
                    n_parent_per_offspring:int=2,
                    **kwargs
                    ):
    population = ESPopulation(n_parent=n_parent, n_parent_per_offspring=n_parent_per_offspring, n_offspring=n_offspring)
    population.preorder_aware_init = True
    population.get_parent_strategy = max_divese_desc_get_parent_fn
    population.selection_strategy = diversity_awarness_selection_fn
    population.name = f"evol_{n_parent}+{n_offspring}_diversity"

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
    problems = [4, 10, 14, 21]
    # problems = list(range(1, 25))
    instances = [8]
    repeat = 3
    budget = 100
    
    evaluator = get_IOHEvaluator_for_test(problems=problems, _instances=instances, repeat=repeat, budget=budget)
    evaluator.inject_critic = True
    evaluator.ignore_over_budget = True
    
    file_map = {
        # 'EnsembleLocalSearchBOv1': 'Experiments/test_cands/EnsembleLocalSearchBOv1.py',
        # 'BLTuRBO1': 'Experiments/baselines/bo_baseline.py',
        # 'AdaptiveBatchBOv7': 'Experiments/pop_40_f/ESPopulation_evol_1+1_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210043001/25-26_AdaptiveBatchBOv7_0.0748.py',


        'AdaptiveLocalPenaltyVarianceBOv3':'Experiments/pop_40_f/ESPopulation_evol_12+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208230817/5-40_AdaptiveLocalPenaltyVarianceBOv3_0.0482.py',
    }

    from Experiments.baselines.bo_baseline import BLTuRBO1, BLTuRBOM, BLRandomSearch, BLSKOpt, BLMaternVanillaBO, BLScaledVanillaBO 
    from Experiments.baselines.vanilla_bo import VanillaBO
    from Experiments.test_cands.EnsembleLocalSearchBOv1 import EnsembleLocalSearchBOv1
    from Experiments.test_cands.EnsembleDeepKernelAdaptiveTSLocalSearchARDv1 import EnsembleDeepKernelAdaptiveTSLocalSearchARDv1
    from Experiments.test_cands.QMCBOv1 import GP_Matern_EI_MSL_SobolBOv1

    cls_list = [
        # VanillaBO,
        # BLRandomSearch,
        # BLMaternVanillaBO,
        # BLTuRBO1,
        # BLTuRBOM,
        BLSKOpt,
        # EnsembleLocalSearchBOv1,
        # EnsembleDeepKernelAdaptiveTSLocalSearchARDv1,
        # GP_Matern_EI_MSL_SobolBOv1,
    ]

    options = {
        # 'device': 'cuda',
        # 'is_baseline': True,
        # 'max_eval_workers': 4,
        # 'time_profile': True,
    }

    run_algo_eval_from_file_map(evaluator, file_map, cls_list=cls_list, plot=True, save=False, options=options)

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
        # 'is_baseline': True,
        'save_dir': 'Experiments/final_eval_res',
        # 'max_eval_workers': 0,
    }
    _bl_file_map = {
        # 'BLRandomSearch': 'Experiments/baselines/bo_baseline.py',
        # 'BLTuRBO1': 'Experiments/baselines/bo_baseline.py',
        # 'BLTuRBOM': 'Experiments/baselines/bo_baseline.py',
        'BLMaternVanillaBO': 'Experiments/baselines/bo_baseline.py',
        # 'BLScaledVanillaBO': 'Experiments/baselines/bo_baseline.py',
        # 'BLSKOpt': 'Experiments/baselines/bo_baseline.py',
    }

    _file_map = {
        # 0.04
        'NoisyBanditBOv1': 'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065417/0-10_NoisyBanditBOv1_0.0434.py',
        'ParetoActiveBOv1': 'Experiments/pop_40_f/ESPopulation_evol_4+2_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210025714/0-3_ParetoActiveBOv1_0.0426.py',

        # 0.05
        'AdaptiveBatchUCBLocalSearchBOv2': 'Experiments/pop_40_f/ESPopulation_evol_4+4_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210043822/4-20_AdaptiveBatchUCBLocalSearchBOv2_0.0526.py',
        'AdaptiveControlVariateBOv4': 'Experiments/pop_40_f/ESPopulation_evol_8+4_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209152106/6-29_AdaptiveControlVariateBOv4_0.0595.py'

        # 0.06
        # 'AdaptiveEvoBatchHybridBOv2': 'Experiments/pop_40_f/ESPopulation_evol_12+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208224540/4-32_AdaptiveEvoBatchHybridBOv2_0.0619.py',
        # 'MultiObjectiveBOv1': 'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065417/0-19_MultiObjectiveBOv1_0.0665.py',

        # 0.07
        # 'AdaptiveTrustRegionImputationDPPBOv1': 'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065238/3-40_AdaptiveTrustRegionImputationDPPBOv1_0.0769.py',
        # 'TrustRegionBOv1': 'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-2_TrustRegionBOv1_0.0720.py',

        # 0.08
        # 'TrustRegionAdaptiveTempBOv2': 'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209173952/4-23_TrustRegionAdaptiveTempBOv2_0.0807.py',
        # 'AdaptiveTrustImputationBOv2': 'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065238/2-31_AdaptiveTrustImputationBOv2_0.0806.py',
        # 'BayesLocalAdaptiveAnnealBOv1': 'Experiments/pop_40_temp/ESPopulation_evol_10+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208164605/3-24_BayesLocalAdaptiveAnnealBOv1_0.0827.py',
    }

    file_map = _file_map

    run_algo_eval_from_file_map(evaluator, file_map, plot=False, save=True, options=options)
 

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
    # MODEL = LLMS["deepseek/deepseek-chat"]
    MODEL = LLMS["gemini-2.0-flash-exp"]
    # MODEL = LLMS["gemini-1.5-flash"]
    # MODEL = LLMS["gemini-2.0-pro"]
    # MODEL = LLMS["gemini-2.0-flash-thinking"]
    # MODEL = LLMS["gemini-exp-1206"]
    # MODEL = LLMS["llama-3.1-70b-versatile"]
    # MODEL = LLMS["llama-3.3-70b-versatile"]
    # MODEL = LLMS["o_gemini-flash-1.5-8b-exp"]
    # MODEL = LLMS["o_gemini-2.0-flash-exp"]
    # MODEL = LLMS["onehub-gemini-2.0-flash"]
    # MODEL = LLMS["onehub-gemma2-9b-it"]

    llm = LLMmanager(api_key=MODEL[1], model=MODEL[0], base_url=MODEL[2], max_interval=MODEL[3])
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

    # debug_algo_eval()
    eval_final_algo()

    _params = get_search_default_params()
    _new_params = {
        "n_population": 40,
        "n_query_threads": 0,

        # Choose time_out_per_eval carefully when running multiple evaluations of expriments in parallel due to OS's dispatching mechanism
        "n_eval_workers": 0,
        "time_out_per_eval": 60 * 20,

        # "gpu_name": "cuda:7",
        "gpu_name": None,

        "options": {
            'pop_debug_save_on_the_fly': True,
            # 'pop_warmstart_handlers': [],
            # 'pop_load_check_point_path':
            'pop_save_check_point_interval': 1,
            'pop_preorder_aware_init': True,
            # 'pop_parent_strategy': max_divese_desc_get_parent_fn,
            # 'pop_selection_strategy': diversity_awarness_selection_fn,
            'pop_save_dir': 'Experiments/pop_40',


            'eval_inject_critic': False,
            'eval_overwrite_type': 'light_evol', # 'test', 'light_evol', 'evol', 'final_eval' 
            'test_eval_problems': [4], # [4, 10],
            'test_eval_instances': [1],
            'test_eval_repeat': 1,
            'test_eval_budget': 100,

            # 'llm_mocker': mock_res_provider,
        }
    }

    _params.update(_new_params)

    N_PARENT = 10
    N_PARENT_PER_OFFSPRING = 2
    N_OFFSPRING = 6

    run_mu_plus_lambda_exp(
        n_parent=N_PARENT,
        n_offspring=N_OFFSPRING,
        n_parent_per_offspring=N_PARENT_PER_OFFSPRING,
        **_params)
