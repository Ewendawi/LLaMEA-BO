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
from llamea.utils import plot_results, plot_algo_results
from llamea.individual import Individual


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
        8, 10,
        12, 14,
        18, 15,
        21, 23,
    ]
    instances = [[1]] * len(problems)
    repeat = 1
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

def _run_algrothim_eval_exp(evaluator, algo_cls, code=None, is_bl=False, save=False, options=None):
    cls_name = algo_cls.__name__
    logging.info("Start evaluating %s on %s", cls_name, evaluator)

    extra_init_params = {}
    if is_bl:
        extra_init_params = baseline_algo_eval_param(evaluator.dim, evaluator.budget)

    res = evaluator.evaluate(
        code=code,
        cls_name=cls_name,
        cls=algo_cls,
        cls_init_kwargs=extra_init_params,
    )
    if save:
        dir_path = os.path.join("Experiments", "algo_eval_res")
        os.makedirs(dir_path, exist_ok=True)
        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        file_path = os.path.join(dir_path, f"{cls_name}_{time_stamp}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(res, f)
    return res

def run_all_algo_eval_exp(plot=False):
    from Experiments.baselines.bo_baseline import BLTuRBO1, BLTuRBOM, BLRBFKernelVanillaBO, BLScaledKernelVanillaBO, BLRandomSearch, BLSKOpt
    from Experiments.test_cands.EnsembleLocalSearchBOv1 import EnsembleLocalSearchBOv1
    from Experiments.test_cands.EnsembleDeepKernelAdaptiveTSLocalSearchARDv1 import EnsembleDeepKernelAdaptiveTSLocalSearchARDv1
    from Experiments.test_cands.QMCBOv1 import GP_Matern_EI_MSL_SobolBOv1

    # evaluator = get_IOHEvaluator_for_final_eval()
    evaluator = get_IOHEvaluator_for_test()
    
    evaluator.ignore_over_budget = True
    evaluator.inject_critic = True
    options = {
    }
    save = True

    cls_list = [
        # BLRandomSearch,
        BLRBFKernelVanillaBO,
        # BLScaledKernelVanillaBO,
        # BLTuRBO1,
        # BLTuRBOM,
        # BLSKOpt,
        # EnsembleLocalSearchBOv1,
        # EnsembleDeepKernelAdaptiveTSLocalSearchARDv1,
        # GP_Matern_EI_MSL_SobolBOv1,
    ]

    res_list = []
    for _cls in cls_list:
        res = _run_algrothim_eval_exp(evaluator, _cls, save=save, options=options)
        res_list.append(res)

    if plot:
        plot_algo_results(res_list)


def run_algo_eval_from_file_map(evaluator, file_map, plot):
    res_list = []
    for cls_name, file_path in file_map.items():
        if not os.path.exists(file_path):
            logging.warning("File not exist: %s", file_path)
            continue
        code = ""
        with open(file_path, "r") as f:
            code = f.read()
    
        cls = dynamic_import_and_get_class(file_path, cls_name)
        if cls is None:
            continue
        res = _run_algrothim_eval_exp(evaluator, cls, code=code)
        res_list.append(res)
    if plot:
        plot_algo_results(res_list)
    

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

    if len(population.name) < 10:
        # population.name += f"_{llm.model_name()}_{prompt_generator}_{evaluator}"
        population.name += f"_{evaluator}"
        if torch.cuda.is_available():
            population.name += "_gpu"

    llambo.run_evolutions(llm, evaluator, prompt_generator, population,
                        n_generation=n_generations, n_population=n_population,
                        n_retry=3, sup_results=None,
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
        plot_algo_results([res])
    
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
    problems = [4,10]
    instances = [1]
    repeat = 2
    budget = 100
    
    evaluator = get_IOHEvaluator_for_test(problems=problems, _instances=instances, repeat=repeat, budget=budget)
    evaluator.inject_critic = True
    evaluator.ignore_over_budget = True
    
    file_map = {
        'EnsembleLocalSearchBOv1': 'Experiments/test_cands/EnsembleLocalSearchBOv1.py',
        'BLTuRBO1': 'Experiments/baselines/bo_baseline.py',
    }
    run_algo_eval_from_file_map(evaluator, file_map, plot=True)

def get_search_default_params():
    params = {
        # "time_out_per_eval": 60 * 20,
        "time_out_per_eval": None,

        "llm": get_llm(),
        "prompt_generator": get_bo_prompt_generator(),
        "n_generations": 200,
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
    # MODEL = LLMS["gemini-2.0-flash-exp"]
    # MODEL = LLMS["gemini-1.5-flash"]
    # MODEL = LLMS["gemini-2.0-pro"]
    MODEL = LLMS["gemini-2.0-flash-thinking"]
    # MODEL = LLMS["gemini-exp-1206"]
    # MODEL = LLMS["llama-3.1-70b-versatile"]
    # MODEL = LLMS["llama-3.3-70b-versatile"]
    # MODEL = LLMS["o_gemini-flash-1.5-8b-exp"]
    # MODEL = LLMS["o_gemini-2.0-flash-exp"]

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
    setup_logger(level=logging.INFO)

    # run_all_algo_eval_exp(plot=True)

    # debug_algo_eval()

    _params = get_search_default_params()

    _new_params = {
        "n_population": 3,
        "n_query_threads": 0,
        "n_eval_workers": 0,

        # "gpu_name": "cuda:7",
        "gpu_name": None,

        "options": {
            'pop_debug_save_on_the_fly': True,
            # 'pop_warmstart_handlers': [],
            # 'pop_load_check_point_path':
            # 'pop_save_check_point_interval': 1,
            'pop_preorder_aware_init': True,
            # 'pop_parent_strategy': max_divese_desc_get_parent_fn,
            # 'pop_selection_strategy': diversity_awarness_selection_fn,
            'pop_save_dir': 'Experiments/pop_temp1',


            'eval_inject_critic': True,
            'eval_overwrite_type': 'test', # 'test', 'light_evol', 'evol', 'final_eval' 
            'test_eval_problems': [4], # [4, 10],
            'test_eval_instances': [1],
            'test_eval_repeat': 3,
            'test_eval_budget': 60,

            # 'llm_mocker': mock_res_provider,
        }
    }

    _params.update(_new_params)

    N_PARENT = 1
    N_PARENT_PER_OFFSPRING = 1
    N_OFFSPRING = 1

    run_mu_plus_lambda_exp(
        n_parent=N_PARENT,
        n_offspring=N_OFFSPRING,
        n_parent_per_offspring=N_PARENT_PER_OFFSPRING,
        **_params)
