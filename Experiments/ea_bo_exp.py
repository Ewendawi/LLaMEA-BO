import os
import getopt
import sys
import logging
import pickle
import json
import torch
import numpy as np
from llamevol import LLaMEvol
from llamevol.llm import LLMmanager, LLMS
from llamevol.prompt_generators import PromptGenerator, BaselinePromptGenerator, TunerPromptGenerator, LightBaselinePromptGenerator, GenerationTask
from llamevol.population import Population, ESPopulation, IslandESPopulation, max_divese_desc_get_parent_fn, diversity_awarness_selection_fn
from llamevol.evaluator.ioh_evaluator import IOHEvaluator, AbstractEvaluator
from llamevol.utils import setup_logger, RenameUnpickler
from llamevol.individual import Individual

# Utils
def get_IOHEvaluator_for_final_eval():
    budget = 100
    dim = 5
    problems = list(range(1, 25))
    instances = [[4, 5, 6]] * len(problems)
    repeat = 5
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    return evaluator

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
    evaluator.ignore_capture = False
    return evaluator

def get_bo_prompt_generator():
    prompt_generator = BaselinePromptGenerator()
    prompt_generator.is_bo = True
    return prompt_generator

def get_light_Promptor_for_crossover():
    prompt_generator = LightBaselinePromptGenerator()
    prompt_generator.is_bo = True
    return prompt_generator

# EA Experiments
def _run_exp(prompt_generator:PromptGenerator,
            evaluator:AbstractEvaluator,
            llm:LLMmanager,
            population:ESPopulation,
            n_generations:int=200,
            n_population:int=30,
            gpu_name:str=None,
            max_interval:int=5,
            n_query_threads:int=0,
            options:dict=None):
    llamevol = LLaMEvol()
    evol_options = {}

    if options is not None:
        # evaluator
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

        if "eval_gpu_name" in options:
            evaluator.gpu_name = options["eval_gpu_name"]

        if "time_out_per_eval" in options:
            evaluator.timeout = options["time_out_per_eval"]
            
        if "n_eval_workers" in options:
            evaluator.max_eval_workers = options["n_eval_workers"]

        if 'use_mpi' in options:
            evaluator.use_mpi = options['use_mpi']

        if 'use_mpi_future' in options:
            evaluator.use_mpi_future = options['use_mpi_future']

        # prompt_generator
        if 'prompt_problem_desc' in options:
            prompt_generator.problem_desc = options['prompt_problem_desc']

        if 'prompt_use_mini_bo' in options and prompt_generator.is_bo:
            prompt_generator.use_mini_bo = options['prompt_use_mini_bo']

        # population
        if "pop_debug_save_on_the_fly" in options:
            population.debug_save_on_the_fly = options["pop_debug_save_on_the_fly"]

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

        if 'es_pop_is_elitism' in options:
            population.use_elitism = options['es_pop_is_elitism']

        if 'llm_params' in options:
            llm_params = options['llm_params']
            p_name = population.name
            if "temperature" in llm_params:
                p_name += f'_t{llm_params["temperature"]}'
            if "top_p" in llm_params:
                p_name += f'_p{llm_params["top_p"]}'
            if "top_k" in llm_params:
                p_name += f'_k{llm_params["top_k"]}'
            population.name = p_name
        population.name += f'_cr{population.cross_over_rate}'

        if population.get_current_generation() == 0:
            # population.name += f"_{llm.model_name()}_{prompt_generator}_{evaluator}"
            population.name += f"_{evaluator}"
            if torch.cuda.is_available():
                population.name += "_gpu"

        if "pop_load_check_point_path" in options:
            check_point_path = options["pop_load_check_point_path"]
            if os.path.exists(check_point_path):
                population = Population.load(check_point_path)
                logging.info("Load population from check point: %s", check_point_path)
                offspring = population.get_offsprings()
                gen = population.get_current_generation()
                if gen != 1 and len(offspring) < population.n_offspring:
                    population.revert_last_generation()
                    logging.info("Revert to last generation")
            else:
                logging.warning("Check point path not exist: %s", check_point_path)

        if "pop_save_dir" in options:
            population.save_dir = options["pop_save_dir"]

        if "pop_save_check_point_interval" in options:
            population.save_per_generation = options["pop_save_check_point_interval"]

        if "pop_warmstart_handlers" in options:
            warmstart_handlers = options["pop_warmstart_handlers"]
            warmstart_inds = []
            _old_save_on_the_fly = population.debug_save_on_the_fly
            population.debug_save_on_the_fly = False
            for _handler in warmstart_handlers:
                handler = _handler
                if isinstance(_handler, str):
                    if not os.path.exists(_handler):
                        logging.warning("Warmstart handler path not exist: %s", _handler)
                        continue
                    with open(_handler, "rb") as f:
                        handler = RenameUnpickler.unpickle(f)
                ind = Individual()
                Population.set_handler_to_individual(ind, handler)
                handler.eval_result.update_aoc_with_new_bound_if_needed()
                ind.fitness = handler.eval_result.score
                ind.name = handler.code_name
                ind.solution = handler.code
                ind.description = handler.desc
                ind.feedback = handler.feedback
                ind.error = handler.error
                # ind.parent_id = handler.parent_ids
                population.add_individual(ind)
                warmstart_inds.append(ind)
            if len(warmstart_inds) > 0:
                logging.info("Warmstart %d individuals", len(warmstart_inds))
                if population.get_current_generation() == 0:
                    if len(warmstart_inds) >= population.n_parent:
                        population.select_next_generation()
                else:
                    if len(warmstart_inds) > population.n_offspring:
                        population.select_next_generation()
            population.debug_save_on_the_fly = _old_save_on_the_fly

        if 'llm_mocker' in options:
            llm.mock_res_provider = options['llm_mocker']
            options.pop('llm_mocker')

        if 'llm_params' in options:
            evol_options['llm_params'] = options['llm_params']

    option_str = json.dumps(options, indent=4)
    log_str = f"""Start running evolutions: 
n_generation:{n_generations}, n_population:{n_population}
n_query_threads:{n_query_threads}, max_interval:{max_interval}
{llm.model_name()}
{prompt_generator}
{evaluator}
{population}
{option_str}
"""
    logging.info(log_str)

    llamevol.run_evolutions(llm, evaluator, prompt_generator, population,
                        n_generation=n_generations, n_population=n_population,
                        n_retry=5,
                        n_query_threads=n_query_threads,
                        max_interval=max_interval,
                        options=evol_options)

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
        res = RenameUnpickler.unpickle(f)

    logging.info("Results: %s", res)

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

def run_mu_lambda_exp(
                    n_parent:int=2,
                    n_offspring:int=1,
                    options:dict=None,
                    **kwargs
                    ):
    n_parent_per_offspring = 2
    if n_parent < 2:
        n_parent_per_offspring = 1
    population = ESPopulation(n_parent=n_parent, n_parent_per_offspring=n_parent_per_offspring, n_offspring=n_offspring)

    p_name = f"{n_parent}+{n_offspring}"
    if options is not None:
        if 'es_pop_is_elitism' in options and not options['es_pop_is_elitism']:
            p_name = f'{n_parent}-{n_offspring}'

    population.name = f"evol_{p_name}"

    _run_exp(
        population=population,
        options=options,
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

def get_llm(model_name):
    
    llm = LLMmanager(model_key=model_name)

    return llm

def mock_res_provider(*args, **kwargs):
    file_list = [
    ]
    file_path = np.random.choice(file_list, size=1)[0] 
    # file_path = np.random.choice(file_list, size=1, p=[0.25, 0.25, 0.25, 0.25])[0]

    response = None
    with open(file_path, "r", encoding="utf-8") as f:
        response = f.read()
    return response

best_warmstart_handlers = [
    'Experiments/0-2_ATRBO_0.0905_handler.pkl',
    'Experiments/2-21_SobolEIAdaptiveTrustBO_0.4360_handler.pkl',
    # 'Experiments/2-37_MEIGCBO_0.4024_handler.pkl',
    'Experiments/4-61_AdaptiveEvolutionaryParetoTrustRegionBO_0.1827_handler.pkl',
    'Experiments/5-88_AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE_0.2138_handler.pkl',
    'Experiments/6-93_ABETSALSDE_ARM_MBO_0.1813_handler.pkl',
]

def _list_handlers(dir_path=None, min_order_filter=None):
    order_filter = min_order_filter 
    handlers = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('_handler.pkl'):
                file_number = file.split('_')[0]
                numbers = file_number.split('-')
                order_num = int(numbers[1])
                if order_filter and order_num <= int(order_filter) :
                    continue

                handlers.append(os.path.join(dir_path, file))
    return handlers

def main():
    # MODEL = 'gemini-2.0-flash'
    MODEL = 'gemini-2.5-flash'

    # MODEL = 'o_qwen3-coder-free'
    # MODEL = 'o_qwen3-coder'

    # MODEL = 'o_deepseek-r1-free'
    # MODEL = 'o_deepseek-r1'

    # MODEL = 'o_gpt4o'
    # MODEL = 'gpt-4o'

    custom_opt_params = {}
    if MODEL == 'o_qwen3-coder-free' or MODEL == 'o_qwen3-coder':
        custom_opt_params = {
            # exp-3
            'pop_warmstart_handlers': _list_handlers('Experiments/pop_qwen3_coder/ESPopulation_evol_4-16_t0.5_cr0.6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0728185633', min_order_filter=36),
            'pop_load_check_point_path': 'Experiments/pop_qwen3_coder/ESPopulation_evol_4-16_t0.5_cr0.6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0728185633/ESPopulation_gen_checkpoint_2_0728205936.pkl',

            'es_pop_is_elitism': False,
            'pop_save_dir': 'Experiments/pop_qwen3_coder',
            'llm_params': {
                'temperature': 0.5,
            }
        }
    elif MODEL == 'o_deepseek-r1' or MODEL == 'o_deepseek-r1-free':
        custom_opt_params = {

            # # 1
            # 'pop_warmstart_handlers': _list_handlers('Experiments/pop_r1/ESPopulation_evol_8+16_t0.5_cr0.6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0727180912', min_order_filter=56),
            # 'pop_load_check_point_path': 'Experiments/pop_r1/ESPopulation_evol_8+16_t0.5_cr0.6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0727180912/ESPopulation_gen_checkpoint_3_0728004314.pkl',

            'es_pop_is_elitism': False,
            'pop_save_dir': 'Experiments/pop_r1',
            'llm_params': {
                'temperature': 0.5,
            }
        }
    elif MODEL == 'gemini-2.5-flash':
        custom_opt_params = {
            # 'pop_warmstart_handlers': _list_handlers('Experiments/pop_2.5/ESPopulation_evol_4-16_t0.5_k60_cr0.6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0729061917', min_order_filter=69),
            # 'pop_load_check_point_path': 'Experiments/pop_2.5/ESPopulation_evol_4-16_t0.5_k60_cr0.6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0729061917/ESPopulation_gen_checkpoint_4_0729133326.pkl',

            'pop_save_dir': 'Experiments/pop_2.5',
            'es_pop_is_elitism': False,
            'pop_preorder_aware_init': True,
            'llm_params': {
                'temperature': 0.5,
                'top_k': 60,
            }
        }
    elif MODEL == 'gemini-2.0-flash':
        custom_opt_params = {
            'pop_warmstart_handlers': _list_handlers('Experiments/pop_2.0/ESPopulation_evol_4+8_t0.5_cr0.0_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0729175814', min_order_filter=20),
            'pop_load_check_point_path': 'Experiments/pop_2.0/ESPopulation_evol_4+8_t0.5_cr0.0_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0729175814/ESPopulation_gen_checkpoint_2_0729200551.pkl',

            'pop_save_dir': 'Experiments/pop_2.0',
            'es_pop_is_elitism': True,
            'pop_preorder_aware_init': False,
            'prompt_use_mini_bo': True,  # use mini BO prompt generator
            'pop_cross_over_rate': 0.0,
            'llm_params': {
                'temperature': 0.5,
                # 'top_k': 60,
            }
        }

    elif MODEL == 'o_gpt4o' or MODEL == 'gpt-4o':
        custom_opt_params = {
            # 0
            # 'pop_warmstart_handlers': _list_handlers('Experiments/pop_gpt4o/ESPopulation_evol_4-16_t0.5_cr0.6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0727220011', min_order_filter=4),
            # 'pop_load_check_point_path': 'Experiments/pop_gpt4o/ESPopulation_evol_4-16_t0.5_cr0.6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0727220011/ESPopulation_gen_checkpoint_0_0727220226.pkl',

            'pop_save_dir': 'Experiments/pop_gpt4o',
            'es_pop_is_elitism': True,
            'llm_params': {
                'temperature': 0.5,
            }
        }


    _params = {
        "n_generations": np.inf,
        "n_population": 40,

        "n_query_threads": 0,
        "max_interval": 5,

        "llm": get_llm(MODEL),
        "prompt_generator": get_bo_prompt_generator(),
        "evaluator": get_IOHEvaluator_for_evol(),

        "options": {
            'pop_debug_save_on_the_fly': True,
            # 'pop_warmstart_handlers': [],
            # 'pop_load_check_point_path': '',

            'pop_save_check_point_interval': 1,
            'pop_preorder_aware_init': True,
            # 'pop_parent_strategy': max_divese_desc_get_parent_fn,
            # 'pop_selection_strategy': diversity_awarness_selection_fn,
            # 'pop_selection_strategy': family_competition_selection_fn(parent_size_threshold=1, is_aggressive=True),
            'pop_save_dir': 'Experiments/pop_' + MODEL,

            # 'pop_replaceable_parent_selection': False,
            # 'pop_random_parent_selection': True,
            # 'pop_cross_over_rate': 0.5,
            # 'pop_exclusive_operations': False,
            # 'pop_cr_light_eval': get_light_IOHEvaluator_for_crossover(),
            # 'pop_cr_light_promptor': get_light_Promptor_for_crossover(),
            # 'es_pop_is_elitism': False,

            "n_eval_workers": 0,
            "time_out_per_eval": 60 * 30,
            'use_mpi': True,
            # 'use_mpi_future': True,

            # 'eval_gpu_name': None,
            'eval_overwrite_type': 'light_evol', # 'test', 'light_evol', 'evol', 'final_eval' 
            'test_eval_problems': [4], # [4, 10],
            # 'test_eval_problems': [2, 4, 8, 14, 15, 23],
            'test_eval_instances': [1],
            'test_eval_repeat': 1,
            'test_eval_budget': 100,
            # 'prompt_problem_desc': 'one noiseless function:F2 Ellipsoid Separable Function',
            # 'prompt_use_mini_bo': True,  # use mini BO prompt generator

            # 'llm_mocker': mock_res_provider,
            'llm_params': {
                'temperature': 0.5,
                # 'top_p': 0.7,
                # 'top_k': 60,
            }
        }
    }

    _params['options'].update(custom_opt_params)

    N_PARENT = 1
    N_OFFSPRING = 1

    run_mu_lambda_exp(
        n_parent=N_PARENT,
        n_offspring=N_OFFSPRING,
        **_params)

if __name__ == "__main__":
    use_mpi = False
    opts, args = getopt.getopt(sys.argv[1:], "m", ["mpi"])
    for opt, arg in opts:
        if opt == "-m":
            use_mpi = True
        elif opt == "--mpi":
            use_mpi = True
    
    # setup_logger(level=logging.DEBUG)
    setup_logger(level=logging.INFO)
    
    if use_mpi:
        from llamevol.evaluator.MPITaskManager import start_mpi_task_manager 

        with start_mpi_task_manager(result_recv_buffer_size=1024*200, 
                                    task_recv_buffer_size=1024*100,
                                    use_sub_process_worker=False,
                                    ) as task_manager:
            if task_manager.is_master:
                main()
    else:
        main()