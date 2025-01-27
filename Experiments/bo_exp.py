import random
import logging
import time
import os
import tqdm
import numpy as np
from llamea import LLaMBO, LLMmanager
from llamea.individual import Individual, Population, SequencePopulation, ESPopulation, max_divese_desc_get_parent_fn, diversity_awarness_selection_fn, IslandESPopulation
from llamea.prompt_generators import PromptGenerator, BoZeroPromptGenerator, BoZeroPlusPromptGenerator, BaselinePromptGenerator
from llamea.utils import setup_logger, IndividualLogger
from llamea.evaluator import RandomBoTorchTestEvaluator, IOHEvaluator, AbstractEvaluator
from llamea.llm import LLMS

def log_aggressiveness_and_botorch(population:SequencePopulation, aggressiveness:float, use_botorch:bool):
    for individual in population.all_individuals():
        individual.add_metadata("aggressiveness", aggressiveness)
        tags = individual.metadata["tags"] if "tags" in individual.metadata else []
        if use_botorch:
            tags.append("botorch")
        tags.append(f"aggr:{aggressiveness}")
        individual.add_metadata("tags", tags)

def log_population(population:SequencePopulation, save:bool=True, dirname:str="logs_temp", filename:str="bo_exp"):
    ind_logger = IndividualLogger()
    ind_logger.file_name = filename
    ind_logger.dirname = dirname
    ind_ids = []
    for individual in population.all_individuals():
        ind_logger.log_individual(individual)
        ind_ids.append(individual.id)
    exp_name = population.name
    ind_logger.log_experiment(name=exp_name, id_list=ind_ids)

    if save:
        ind_logger.save()
        ind_logger.save_reader_format()


def run_bo_exp_code_generation(model:tuple, aggressiveness:float, use_botorch:bool, prompt_generator:PromptGenerator, n_iterations:int=1, n_generations:int=1):
    llambo = LLaMBO()

    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])

    if isinstance(prompt_generator, BoZeroPlusPromptGenerator):
        prompt_generator.use_botorch = use_botorch
        prompt_generator.aggressiveness = aggressiveness
    elif isinstance(prompt_generator, BoZeroPromptGenerator):
        prompt_generator.use_botorch = use_botorch

    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations")
    for _ in range(n_iterations):
        population = SequencePopulation()
        evaluator = RandomBoTorchTestEvaluator()

        other_results = evaluator.evaluate_others()

        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, n_retry=3, sup_results=other_results)
        log_aggressiveness_and_botorch(population, aggressiveness, use_botorch)
        progress_bar.update(1)

    log_file_name = f"bo_exp_p1_{model[0]}_{aggressiveness}_{use_botorch}"
    log_dir_name = "logs_temp"
    log_population(population, save=True, dirname=log_dir_name, filename=log_file_name)

def run_bo_exp_fix_errors(model:tuple, log_path:str, prompt_generator:PromptGenerator,n_iterations:int=1, n_generations:int=1):
    llambo = LLaMBO()

    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])

    p1_logger = IndividualLogger.load(log_path)
    failed_individuals = p1_logger.get_failed_individuals()

    n_samples = 2 * n_iterations

    error_type_group = {}
    for ind in failed_individuals:
        error_type = ind.metadata["error_type"]
        if error_type not in error_type_group:
            error_type_group[error_type] = []
        error_type_group[error_type].append(ind)

    selected_failed_individuals = []
    for error_type, individuals in error_type_group.items():
        selected_failed_individuals.append(random.choice(individuals))
        if len(selected_failed_individuals) > n_samples:
            break

    if len(selected_failed_individuals) < n_samples:
        selected_failed_individuals.extend(random.sample(failed_individuals, n_samples - len(selected_failed_individuals)))

    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations")
    for _ in range(n_iterations):
        candidate = selected_failed_individuals.pop()
        aggressiveness = candidate.metadata["aggressiveness"]
        use_botorch = "botorch" in candidate.metadata["tags"]
        problem_str = candidate.metadata["problem"]
        problem_dim = candidate.metadata["dimension"]

        if isinstance(prompt_generator, BoZeroPlusPromptGenerator):
            prompt_generator.use_botorch = use_botorch
            prompt_generator.aggressiveness = aggressiveness
        elif isinstance(prompt_generator, BoZeroPromptGenerator):
            prompt_generator.use_botorch = use_botorch

        evaluator = RandomBoTorchTestEvaluator(dim=problem_dim, obj_fn_name=problem_str)
        if evaluator.obj_fn is None:
            logging.error("Failed to load the objective function for %s with dim %s", problem_str, problem_dim)
            continue

        population = SequencePopulation()
        population.add_individual(candidate)
        population.name = f"bo_exp_p2_{candidate.metadata['error_type']}_{model[0]}_{problem_str}"

        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, n_retry=3)
        log_aggressiveness_and_botorch(population, aggressiveness, use_botorch)
        progress_bar.update(1)

    log_file_name = f"bo_exp_p2_{model[0]}"
    log_dir_name = "logs_temp"
    log_population(population, save=True, dirname=log_dir_name, filename=log_file_name)

def run_bo_exp_optimize_performance(model:tuple, log_path:str, prompt_generator:PromptGenerator, n_iterations:int=1, n_generations:int=1):
    llambo = LLaMBO()

    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])

    p_logger = IndividualLogger.load(log_path)
    successful_individuals = p_logger.get_successful_individuals()

    problem_group = {}
    for ind in successful_individuals:
        problem = ind.metadata["problem"]
        if problem not in problem_group:
            problem_group[problem] = []
        problem_group[problem].append(ind)

    selected_successful_individuals = []
    for problem, individuals in problem_group.items():
        selected_successful_individuals.append(random.choice(individuals))
        if len(selected_successful_individuals) > n_iterations:
            break

    if len(selected_successful_individuals) < n_iterations:
        selected_successful_individuals.extend(random.sample(successful_individuals, n_iterations - len(selected_successful_individuals)))

    selected_successful_individuals = random.sample(successful_individuals, n_iterations)
    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations")
    for _ in range(n_iterations):
        candidate = selected_successful_individuals.pop()

        aggressiveness = 0.5
        use_botorch = False
        problem_str = candidate.metadata["problem"]
        problem_dim = None
        for tag in candidate.metadata["tags"]:
            if tag.startswith("aggr:"):
                aggressiveness = float(tag.split(":")[1])
            elif tag == "botroch:":
                use_botorch = True
            elif tag.startswith("dim:"):
                problem_dim = int(tag.split(":")[1])

        if isinstance(prompt_generator, BoZeroPlusPromptGenerator):
            prompt_generator.aggressiveness = aggressiveness
            prompt_generator.use_botorch = use_botorch
        elif isinstance(prompt_generator, BoZeroPromptGenerator):
            prompt_generator.use_botorch = use_botorch

        evaluator = RandomBoTorchTestEvaluator(dim=problem_dim, obj_fn_name=problem_str)
        if evaluator.obj_fn is None:
            logging.error("Failed to load the objective function for %s with dim %s", problem_str, problem_dim)
            continue

        population = SequencePopulation()
        population.add_individual(candidate)
        population.name = f"bo_exp_p3_{problem_str}_{model[0]}_dim{problem_dim}"

        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, n_retry=3)
        log_aggressiveness_and_botorch(population, aggressiveness, use_botorch)
        progress_bar.update(1)

    log_file_name = f"bo_exp_p3_{model[0]}"
    log_dir_name = "logs_temp"
    log_population(population, save=True, dirname=log_dir_name, filename=log_file_name)

def test_multiple_processes():

    def mock_res_provider(*args, **kwargs):
        response = None
        with open("Experiments/bbob_test_res/successful_light_res2.md", "r") as f:
            response = f.read()
        return response
    
    llambo = LLaMBO()
    model = LLMS["deepseek/deepseek-chat"]
    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])
    llm.mock_res_provider = mock_res_provider
    prompt_generator = BoBaselinePromptGenerator()

    budget = 100
    dim = 5
    problems = list(range(1, 25))
    instances = [[1, 2, 3]] * len(problems)
    repeat = 1
    time_out = 60 * budget * dim // 100
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)

    n_generations = 1
    n_parent = 1
    n_parent_per_offspring = 1
    n_offspring = 1
    n_query_threads = n_parent

    n_eval_workers = 16
    population = ESPopulation(n_parent=n_parent, n_parent_per_offspring=n_parent_per_offspring, n_offspring=n_offspring)
    logging.info("Starting with %s processes", n_eval_workers)
    start = time.perf_counter()
    llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, n_retry=3, time_out_per_eval=time_out,
                          n_query_threads=n_query_threads, 
                          n_eval_workers=n_eval_workers
                          )
    end = time.perf_counter()
    logging.info("Time taken: %s with %s processes", end - start, n_eval_workers)

    n_eval_workers = 32
    population = ESPopulation(n_parent=n_parent, n_parent_per_offspring=n_parent_per_offspring, n_offspring=n_offspring)
    logging.info("Starting with %s processes", n_eval_workers)
    start = time.perf_counter()
    llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, n_retry=3, time_out_per_eval=time_out,
                          n_query_threads=n_query_threads,
                          n_eval_workers=n_eval_workers
                          )
    end = time.perf_counter()
    logging.info("Time taken: %s with %s processes", end - start, n_eval_workers)


def plot():
    file_paths = [
        # ("logs_bbob/bbob_exp_gemini-2.0-flash-exp_0121222958.pkl", "bo"),
        ("logs_bbob/bbob_exp_gemini-2.0-flash-exp_0124195614.pkl", "es-1+1"),
        ("logs_bbob/bbob_exp_gemini-2.0-flash-exp_0124195643.pkl", "es-1+1"),
    ]
    strategy_list = []
    for file_path, name in file_paths:
        ind_logger = IndividualLogger.load(file_path)
        ind_ids = list(ind_logger.experiment_map.values())[0]["id_list"]
        inds = [ind_logger.get_individual(ind_id) for ind_id in ind_ids]
        res_list = [ind.metadata["res_handler"].eval_result for ind in inds]
        strategy_list.append((name, res_list))

    IOHEvaluator.plot_results(results=strategy_list, other_results=None)
        

def run_exp(model:tuple, prompt_generator:PromptGenerator, 
                 n_iterations:int=1, n_generations:int=1, n_population:int=1, 
                 n_query_threads:int=0, n_eval_workers:int=0, time_out_per_eval:int=None,
                 mocker=None, get_evaluator=None, get_population=None, gpu_name:str=None,
                 ):

    llambo = LLaMBO()

    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])
    llm.mock_res_provider = mocker

    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations", position=0)
    for _ in range(n_iterations):
        population = get_population()
        evaluator = get_evaluator()

        # other_results = evaluator.evaluate_others()
        other_results = None

        llambo.run_evolutions(llm, evaluator, prompt_generator, population, 
                              n_generation=n_generations, n_population=n_population,
                              n_retry=3, sup_results=other_results,
                              time_out_per_eval=time_out_per_eval,
                              n_query_threads=n_query_threads,
                              n_eval_workers=n_eval_workers,
                              gpu_name=gpu_name,
                              max_interval=5
                              )
        progress_bar.update(1)

        population.save()

        # log_file_name = population.name
        # if isinstance(prompt_generator, BoZeroPlusPromptGenerator):
        #     aggressiveness = prompt_generator.aggressiveness
        #     log_file_name = f"bbob_exp_{model[0]}_{aggressiveness}"
        # log_dir_name = "logs_bbob"
        # log_population(population, save=True, dirname=log_dir_name, filename=log_file_name)

if __name__ == "__main__":
    # logging.info(os.environ)
    # logging.info("CPU count: %s", os.cpu_count())

    # setup_logger(level=logging.DEBUG)
    setup_logger(level=logging.INFO)

    # MODEL = LLMS["deepseek/deepseek-chat"]
    MODEL = LLMS["gemini-2.0-flash-exp"]
    # MODEL = LLMS["gemini-1.5-flash"]
    # MODEL = LLMS["gemini-exp-1206"]
    # MODEL = LLMS["llama-3.1-70b-versatile"]
    # MODEL = LLMS["llama-3.3-70b-versatile"]
    # MODEL = LLMS["o_gemini-flash-1.5-8b-exp"]
    # MODEL = LLMS["o_gemini-2.0-flash-exp"]

    prompt_generator = BaselinePromptGenerator()

    # prompt_generator.is_bo = True
    # BUDGET = 100

    BUDGET = 2000 * 5

    PROBLEMS = list(range(1, 25))
    INSTANCES = [[1, 2]] * len(PROBLEMS)
    REPEAT = 2

    # PROBLEMS = [6]
    # INSTANCES = [[1, 2]] * len(PROBLEMS)
    # REPEAT = 1

    DIM = 5

    N_INTERATIONS = 2
    N_GENERATIONS = 200
    N_POPULATION = 30

    N_PARENT = 1
    N_PARENT_PER_OFFSPRING = 1
    N_OFFSPRING = 1

    N_ISLAND = 3
    N_WARMUP_GENERATIONS = 3
    N_CAMBRIAN_GENERATIONS = 2
    N_NEOGENE_GENERATIONS = 2
    PREODER_AWARE_INIT = True
    CROSSOVER_RATE = 1.0

    N_QUERY_THREADS = 0
    N_EVAL_WORKERS = 0
    GPU_NAME = "cuda:7"

    TIME_OUT_PER_EVAL = 60 * 20
    TIME_OUT_PER_EVAL = None

    # bbob experiment
    def mock_res_provider(*args, **kwargs):
        file_list = [
            "Experiments/bbob_test_res/successful_heavy_res.md",
            "Experiments/bbob_test_res/successful_light_res.md",
            "Experiments/bbob_test_res/successful_light_res1.md",
            "Experiments/bbob_test_res/fail_excute_res.md",
            "Experiments/bbob_test_res/fail_overbudget_res.md",
        ]
        file_path = np.random.choice(file_list, size=1, p=[0.0, 0.0, 1.0, 0.0, 0.0])[0]
        file_path = "Experiments/bbob_test_res/successful_bl.md"
        response = None
        with open(file_path, "r") as f:
            response = f.read()
        return response
    mocker = None
    # mocker = mock_res_provider

    def get_evaluator():
        evaluator = IOHEvaluator(budget=BUDGET, dim=DIM, problems=PROBLEMS, instances=INSTANCES, repeat=REPEAT)
        return evaluator

    def get_population():
        # population = ESPopulation(n_parent=N_PARENT, n_parent_per_offspring=N_PARENT_PER_OFFSPRING, n_offspring=N_OFFSPRING)
        # population.name = f"bbob_1+1_{MODEL[0]}_{prompt_generator}"

        population = ESPopulation(n_parent=N_PARENT, n_parent_per_offspring=N_PARENT_PER_OFFSPRING, n_offspring=N_OFFSPRING)
        population.name = f"bbob_2+1_{MODEL[0]}_{prompt_generator}"
        
        # population = ESPopulation(n_parent=N_PARENT, n_parent_per_offspring=N_PARENT_PER_OFFSPRING, n_offspring=N_OFFSPRING)
        # population.save_per_generation = 8
        # population.preorder_aware_init = True
        # population.get_parent_strategy = max_divese_desc_get_parent_fn
        # population.selection_strategy = diversity_awarness_selection_fn
        # population.name = f"bbob_2+1+warmstart+diversity_{MODEL[0]}_{prompt_generator}"

        # population = IslandESPopulation(n_parent=N_PARENT, n_parent_per_offspring=N_PARENT_PER_OFFSPRING, n_offspring=N_OFFSPRING, n_islands=N_ISLAND, 
        #                                 preoder_aware_init=PREODER_AWARE_INIT, update_strategy=max_divese_desc_get_parent_fn, selection_strategy=diversity_awarness_selection_fn,
        #                                 crossover_rate=CROSSOVER_RATE,
        #                                 n_warmup_generations=N_WARMUP_GENERATIONS, n_cambrian_generations=N_CAMBRIAN_GENERATIONS, n_neogene_generations=N_NEOGENE_GENERATIONS)
        # population.name = f"bbob_island_{MODEL[0]}_{prompt_generator}"

        return population
    
    run_exp(MODEL, prompt_generator,
            n_iterations=N_INTERATIONS, n_generations=N_GENERATIONS, n_population=N_POPULATION,
            n_query_threads=N_QUERY_THREADS, n_eval_workers=N_EVAL_WORKERS, time_out_per_eval=TIME_OUT_PER_EVAL,
            mocker=mocker,
            get_evaluator=get_evaluator,
            get_population=get_population,
            gpu_name=GPU_NAME,
            )

    # IndividualLogger.merge_logs("logs_bbob").save_reader_format()
    # test_multiple_processes()
    # plot()