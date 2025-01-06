import random
import tqdm
import logging
from llamea import LLaMBO, LLMmanager, BOPromptGenerator, GenerationTask, Individual, SequencePopulation
from llamea.utils import RandomBoTorchTestEvaluator, setup_logger, IndividualLogger
from llamea.llm import LLMS

def run_bo_exp_code_gereation(model:tuple, aggressiveness:float, use_botorch:bool):
    llambo = LLaMBO()
    
    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])
    
    prompt_generator = BOPromptGenerator()
    prompt_generator.aggressiveness = aggressiveness
    prompt_generator.use_botorch = use_botorch

    individualLogger = IndividualLogger()
    individualLogger.should_log_experiment = False
    individualLogger.should_log_population = True
    individualLogger.auto_save = False
    individualLogger.file_name = f"bo_exp_p1_{model[0]}_{aggressiveness}_{use_botroch}"


    n_iterations = 10
    n_generations = 1
    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations")
    for _ in range(n_iterations):
        population = SequencePopulation()
        evaluator = RandomBoTorchTestEvaluator()
        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, ind_logger=individualLogger, retry=3)
        progress_bar.update(1)
    
    individualLogger.save()
    individualLogger.save_reader_format()

def run_bo_exp_fix_errors(model:tuple):
    llambo = LLaMBO()
    
    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])
    
    prompt_generator = BOPromptGenerator()

    log_path = "logs/all_of_all_set_0106053339.pkl"
    p1Logger = IndividualLogger.load(log_path)
    failed_individuals = p1Logger.get_failed_individuals()

    n_iterations = 1

    error_type_group = {}
    for ind in failed_individuals:
        error_type = ind.metadata["error_type"]
        if error_type not in error_type_group:
            error_type_group[error_type] = []
        error_type_group[error_type].append(ind)
    
    selected_failed_individuals = []
    for error_type, individuals in error_type_group.items():
        selected_failed_individuals.append(random.choice(individuals))
        if len(selected_failed_individuals) > n_iterations:
            break
    
    if len(selected_failed_individuals) < n_iterations:
        selected_failed_individuals.extend(random.sample(failed_individuals, n_iterations - len(selected_failed_individuals))) 
    

    p2Logger = IndividualLogger()
    p2Logger.should_log_experiment = True
    p2Logger.should_log_population = True
    p2Logger.auto_save = False
    p2Logger.file_name = f"bo_exp_p2_{model[0]}"
    p2Logger.dirname = "logs/p2"

    n_generations = 10
    selected_failed_individuals = random.sample(failed_individuals, 10)
    progress_bar = tqdm.tqdm(range(n_iterations), desc="Iterations")
    for _ in range(n_iterations):
        candidate = selected_failed_individuals.pop()
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
        prompt_generator.aggressiveness = aggressiveness
        prompt_generator.use_botorch = use_botorch

        evaluator = RandomBoTorchTestEvaluator(dim=problem_dim, obj_fn_name=problem_str)
        if evaluator.obj_fn is None:
            logging.error(f"Failed to load the objective function for {problem_str} with dim {problem_dim}")
            continue
        
        population = SequencePopulation()
        population.add_individual(candidate)
        p2Logger.log_individual(candidate)
        population.name = f"bo_exp_p2_{candidate.metadata['error_type']}_{model[0]}_{problem_str}"

        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, ind_logger=p2Logger, retry=3)
        progress_bar.update(1)
    
    p2Logger.save()
    p2Logger.save_reader_format()

def run_bo_exp_optimize_performance(model:tuple):
    llambo = LLaMBO()
    
    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])
    
    prompt_generator = BOPromptGenerator()

    log_path = "logs/all_of_all_set_0106053339.pkl"
    pLogger = IndividualLogger.load(log_path)
    successful_individuals = pLogger.get_successful_individuals()


    problem_group = {}
    for ind in successful_individuals:
        problem = ind.metadata["problem"]
        if len(problem.split(" ")) > 1:
            continue
        if problem not in problem_group:
            problem_group[problem] = []
        problem_group[problem].append(ind)

    n_iterations = 10

    selected_successful_individuals = []
    for problem, individuals in problem_group.items():
        selected_successful_individuals.append(random.choice(individuals))
        if len(selected_successful_individuals) > n_iterations:
            break
    
    if len(selected_successful_individuals) < n_iterations:
        selected_successful_individuals.extend(random.sample(successful_individuals, n_iterations - len(selected_successful_individuals)))

    p3Logger = IndividualLogger()
    p3Logger.should_log_experiment = True
    p3Logger.should_log_population = True
    p3Logger.auto_save = False
    p3Logger.file_name = f"bo_exp_p3_{model[0]}"
    p3Logger.dirname = "logs/p3_t"

    n_generations = 3
    selected_successful_individuals = random.sample(successful_individuals, 10)
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
        prompt_generator.aggressiveness = 0.8 
        prompt_generator.use_botorch = use_botorch

        evaluator = RandomBoTorchTestEvaluator(dim=problem_dim, obj_fn_name=problem_str)
        if evaluator.obj_fn is None:
            logging.error(f"Failed to load the objective function for {problem_str} with dim {problem_dim}")
            continue

        population = SequencePopulation()
        population.add_individual(candidate)
        p3Logger.log_individual(candidate)
        population.name = f"bo_exp_p3_{problem_str}_{model[0]}_dim{problem_dim}"
        
        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=n_generations, ind_logger=p3Logger, retry=3)
        progress_bar.update(1)

    p3Logger.save()
    p3Logger.save_reader_format()


# model = LLMS["deepseek/deepseek-chat"]
# model = LLMS["gemini-1.5-flash-8b"]
# model = LLMS["gemini-2.0-flash-exp"]
model = LLMS["gemini-exp-1206"]
# model = LLMS["llama-3.1-70b-versatile"]
# model = LLMS["llama-3.3-70b-versatile"]
# model = LLMS["o_gemini-flash-1.5-8b-exp"]
# model = LLMS["o_gemini-2.0-flash-exp"]
# model = LLMS['o_llama-3.1-405b-instruct']

setup_logger(level=logging.DEBUG)

# code generation experiment
# aggressiveness = 0.4
# use_botroch = True
# run_bo_exp_code_gereation(model, aggressiveness, use_botroch)


# fix errors experiment
# run_bo_exp_fix_errors(model=model)

# optimize performance experiment
run_bo_exp_optimize_performance(model=model)

# IndividualLogger.merge_logs("logs").save_reader_format()