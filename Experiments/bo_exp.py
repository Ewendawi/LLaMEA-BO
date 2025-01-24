import random
import logging
import time
import os
import tqdm
import numpy as np
from llamea import LLaMBO, LLMmanager
from llamea.individual import Individual, Population, SequencePopulation, ESPopulation
from llamea.prompt_generators import PromptGenerator, BoZeroPromptGenerator, BoZeroPlusPromptGenerator, BoBaselinePromptGenerator, BaselinePromptGenerator
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
        

def run_bbob_exp(model:tuple, prompt_generator:PromptGenerator, n_iterations:int=1, n_generations:int=1, budget:int=100):

    def mock_res_provider(*args, **kwargs):
        file_list = [
            "Experiments/bbob_test_res/successful_heavy_res.md",
            "Experiments/bbob_test_res/successful_light_res.md",
            "Experiments/bbob_test_res/successful_light_res1.md",
            "Experiments/bbob_test_res/fail_excute_res.md",
            "Experiments/bbob_test_res/fail_overbudget_res.md",
        ]
        file_path = np.random.choice(file_list, size=1, p=[0.0, 1.0, 0.0, 0.0, 0.0])[0]
        file_path = "Experiments/bbob_test_res/successful_bl.md"
        response = None
        with open(file_path, "r") as f:
            response = f.read()
        return response
    
    llambo = LLaMBO()

    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])
    # llm.mock_res_provider = mock_res_provider

    dim = 5
    # time_out_per_eval = 60 * 20
    time_out_per_eval = None

    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations", position=0)
    for _ in range(n_iterations):
        n_parent = 1
        n_parent_per_offspring = 1
        n_offspring = 1
        population = ESPopulation(n_parent=n_parent, n_parent_per_offspring=n_parent_per_offspring, n_offspring=n_offspring)
        population.name = f"bbob_exp_{model[0]}_{prompt_generator.__class__.__name__}"
        population.save_per_generation = 8
        problems = list(range(1, 25))
        # problems = [6,7]
        instances = [[1, 2, 3]] * len(problems)
        repeat = 3
        evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)

        # other_results = evaluator.evaluate_others()
        other_results = None

        n_query_threads = 0
        n_eval_workers = 0
        
        llambo.run_evolutions(llm, evaluator, prompt_generator, population, 
                              n_generation=n_generations, n_retry=3, sup_results=other_results,
                              time_out_per_eval=time_out_per_eval,
                              n_query_threads=n_query_threads, 
                              n_eval_workers=n_eval_workers,
                              max_interval=5
                              )
        progress_bar.update(1)

        log_file_name = f"bbob_exp_{model[0]}"
        if isinstance(prompt_generator, BoZeroPlusPromptGenerator):
            aggressiveness = prompt_generator.aggressiveness
            log_file_name = f"bbob_exp_{model[0]}_{aggressiveness}"
        log_dir_name = "logs_bbob"
        log_population(population, save=True, dirname=log_dir_name, filename=log_file_name)

if __name__ == "__main__":
    setup_logger(level=logging.INFO)

    # logging.info(os.environ)
    # logging.info("CPU count: %s", os.cpu_count())

    # MODEL = LLMS["deepseek/deepseek-chat"]
    MODEL = LLMS["gemini-2.0-flash-exp"]
    # MODEL = LLMS["gemini-1.5-flash"]
    # MODEL = LLMS["gemini-exp-1206"]
    # MODEL = LLMS["llama-3.1-70b-versatile"]
    # MODEL = LLMS["llama-3.3-70b-versatile"]
    # MODEL = LLMS["o_gemini-flash-1.5-8b-exp"]
    # MODEL = LLMS["o_gemini-2.0-flash-exp"]

    AGGRESSIVENESS = [0.3, 0.5, 0.7, 1.0]
    USE_BOTROCH = False

    # prompt_generator = BoZeroPlusPromptGenerator()
    # prompt_generator.aggressiveness = AGGRESSIVENESS[3]
    # prompt_generator.use_botorch = USE_BOTROCH

    # prompt_generator = BoZeroPromptGenerator()
    # prompt_generator.use_botorch = USE_BOTROCH

    # prompt_generator = BoBaselinePromptGenerator()
    prompt_generator = BaselinePromptGenerator()

    N_INTERATIONS = 2
    N_GENERATIONS = 30
    # BUDGET = 100
    BUDGET = 2000 * 5

    # initial solution generation experiment
    # run_bo_exp_code_generation(MODEL, AGGRESSIVENESS, USE_BOTROCH, prompt_generator, n_interations, n_generations)


    # fix errors experiment
    # log_path = """
    # logs_temp/bo_exp_p1_o_gemini-2.0-flash-exp_1.0_False
    # """
    # run_bo_exp_fix_errors(model=MODEL, log_path=log_path, prompt_generator=prompt_generator, n_iterations=n_interations, n_generations=n_generations)

    # optimize performance experiment
    # log_path = """
    # logs_temp/bo_exp_p2_o_gemini-2.0-flash-exp
    # """
    # run_bo_exp_optimize_performance(model=MODEL, log_path=log_path, prompt_generator=prompt_generator, n_iterations=n_interations, n_generations=n_generations)

    
    # bbob experiment
    run_bbob_exp(MODEL, prompt_generator, N_INTERATIONS, N_GENERATIONS, BUDGET)

    # IndividualLogger.merge_logs("logs_bbob").save_reader_format()
    # test_multiple_processes()
    # plot()