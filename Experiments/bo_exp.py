import random
import logging
import tqdm
from llamea import LLaMBO, LLMmanager, SequencePopulation, Individual
from llamea.promptGenerator import ZeroPlusBOPromptGenerator, ZeroBOPromptGenerator, PromptGenerator
from llamea.utils import setup_logger, IndividualLogger
from llamea.evaluator import RandomBoTorchTestEvaluator, IOHEvaluator
from llamea.llm import LLMS

def log_aggressiveness_and_botorch(population:SequencePopulation, aggressiveness:float, use_botorch:bool):
    for individual in population.all_individuals():
        individual.add_metadata("aggressiveness", aggressiveness)
        tags = individual.metadata["tags"] if "tags" in individual.metadata else []
        if use_botorch:
            tags.append("botorch")
        tags.append(f"aggr:{aggressiveness}")
        individual.add_metadata("tags", tags)

def run_bo_exp_code_generation(model:tuple, aggressiveness:float, use_botorch:bool, prompt_generator:PromptGenerator, n_iterations:int=1, n_generations:int=1):
    llambo = LLaMBO()

    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])

    if isinstance(prompt_generator, ZeroPlusBOPromptGenerator):
        prompt_generator.use_botorch = use_botorch
        prompt_generator.aggressiveness = aggressiveness
    elif isinstance(prompt_generator, ZeroBOPromptGenerator):
        prompt_generator.use_botorch = use_botorch

    p1_logger = IndividualLogger()
    p1_logger.should_log_experiment = True
    p1_logger.should_log_population = True
    p1_logger.auto_save = False
    p1_logger.file_name = f"bo_exp_p1_{model[0]}_{aggressiveness}_{use_botorch}"
    # p1_logger.dirname = "logs/p1_new_prompt"
    p1_logger.dirname = "logs_temp"

    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations")
    for _ in range(n_iterations):
        population = SequencePopulation()
        evaluator = RandomBoTorchTestEvaluator()

        other_results = evaluator.evaluate_others()

        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, ind_logger=p1_logger, retry=3, verbose=2, sup_results=other_results)
        log_aggressiveness_and_botorch(population, aggressiveness, use_botorch)
        progress_bar.update(1)

    p1_logger.save()
    p1_logger.save_reader_format()

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

    p2_logger = IndividualLogger()
    p2_logger.should_log_experiment = True
    p2_logger.should_log_population = True
    p2_logger.auto_save = False
    p2_logger.file_name = f"bo_exp_p2_{model[0]}"
    # p2_logger.dirname = "logs/p2"
    p2_logger.dirname = "logs_temp"

    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations")
    for _ in range(n_iterations):
        candidate = selected_failed_individuals.pop()
        aggressiveness = candidate.metadata["aggressiveness"]
        use_botorch = "botorch" in candidate.metadata["tags"]
        problem_str = candidate.metadata["problem"]
        problem_dim = candidate.metadata["dimension"]

        if isinstance(prompt_generator, ZeroPlusBOPromptGenerator):
            prompt_generator.use_botorch = use_botorch
            prompt_generator.aggressiveness = aggressiveness
        elif isinstance(prompt_generator, ZeroBOPromptGenerator):
            prompt_generator.use_botorch = use_botorch

        evaluator = RandomBoTorchTestEvaluator(dim=problem_dim, obj_fn_name=problem_str)
        if evaluator.obj_fn is None:
            logging.error("Failed to load the objective function for %s with dim %s", problem_str, problem_dim)
            continue

        population = SequencePopulation()
        population.add_individual(candidate)
        p2_logger.log_individual(candidate)
        population.name = f"bo_exp_p2_{candidate.metadata['error_type']}_{model[0]}_{problem_str}"

        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, ind_logger=p2_logger, retry=3, verbose=2)
        log_aggressiveness_and_botorch(population, aggressiveness, use_botorch)
        progress_bar.update(1)

    p2_logger.save()
    p2_logger.save_reader_format()

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

    p3_logger = IndividualLogger()
    p3_logger.should_log_experiment = True
    p3_logger.should_log_population = True
    p3_logger.auto_save = False
    p3_logger.file_name = f"bo_exp_p3_{model[0]}"
    p3_logger.dirname = "logs_temp"

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

        if isinstance(prompt_generator, ZeroPlusBOPromptGenerator):
            prompt_generator.aggressiveness = aggressiveness
            prompt_generator.use_botorch = use_botorch
        elif isinstance(prompt_generator, ZeroBOPromptGenerator):
            prompt_generator.use_botorch = use_botorch

        evaluator = RandomBoTorchTestEvaluator(dim=problem_dim, obj_fn_name=problem_str)
        if evaluator.obj_fn is None:
            logging.error("Failed to load the objective function for %s with dim %s", problem_str, problem_dim)
            continue

        population = SequencePopulation()
        population.add_individual(candidate)
        p3_logger.log_individual(candidate)
        population.name = f"bo_exp_p3_{problem_str}_{model[0]}_dim{problem_dim}"

        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, ind_logger=p3_logger, retry=3, verbose=2)
        log_aggressiveness_and_botorch(population, aggressiveness, use_botorch)
        progress_bar.update(1)

    p3_logger.save()
    p3_logger.save_reader_format()

def run_bbob_exp(model:tuple, prompt_generator:PromptGenerator, n_iterations:int=1, n_generations:int=1):
    llambo = LLaMBO()

    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])

    p_logger = IndividualLogger()
    p_logger.should_log_experiment = True
    p_logger.should_log_population = True
    p_logger.auto_save = False
    p_logger.file_name = f"bbob_exp_{model[0]}"
    p_logger.dirname = "logs_temp"

    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations")
    for _ in range(n_iterations):
        population = SequencePopulation()
        evaluator = IOHEvaluator()

        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, ind_logger=p_logger, retry=3, verbose=2)
        progress_bar.update(1)

    p_logger.save()
    p_logger.save_reader_format()

setup_logger(level=logging.INFO)

# MODEL = LLMS["deepseek/deepseek-chat"]
# MODEL = LLMS["gemini-2.0-flash-exp"]
# MODEL = LLMS["gemini-exp-1206"]
MODEL = LLMS["llama-3.1-70b-versatile"]
# MODEL = LLMS["llama-3.3-70b-versatile"]
# MODEL = LLMS["o_gemini-flash-1.5-8b-exp"]
# MODEL = LLMS["o_gemini-2.0-flash-exp"]
# MODEL = LLMS['o_llama-3.1-405b-instruct']

AGGRESSIVENESS = 0.2
USE_BOTROCH = False

# prompt_generator = ZeroPlusBOPromptGenerator()
# prompt_generator.aggressiveness = AGGRESSIVENESS
# prompt_generator.use_botorch = USE_BOTROCH

prompt_generator = ZeroBOPromptGenerator()
prompt_generator.use_botorch = USE_BOTROCH

n_interations = 1
n_generations = 4


# initial solution generation experiment
# run_bo_exp_code_generation(MODEL, AGGRESSIVENESS, USE_BOTROCH, prompt_generator, n_interations, n_generations)


# fix errors experiment
log_path = """
logs_temp/bo_exp_p1_o_gemini-2.0-flash-exp_1.0_False
"""
# run_bo_exp_fix_errors(model=MODEL, log_path=log_path, prompt_generator=prompt_generator, n_iterations=n_interations, n_generations=n_generations)

# optimize performance experiment
log_path = """
logs_temp/bo_exp_p2_o_gemini-2.0-flash-exp
"""
# run_bo_exp_optimize_performance(model=MODEL, log_path=log_path, prompt_generator=prompt_generator, n_iterations=n_interations, n_generations=n_generations)


# bbob experiment
run_bbob_exp(MODEL, prompt_generator, n_interations, n_generations)

# IndividualLogger.merge_logs("logs_new").save_reader_format()
