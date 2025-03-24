import logging
from llambo.evaluator.ioh_evaluator import IOHEvaluator
from llambo.prompt_generators import BaselinePromptGenerator
from llambo.population import ESPopulation
from llambo.llm import LLMmanager
from llambo import LLaMBO
from llambo.utils import setup_logger
from Experiments.plot_search_res import plot_search


def get_IOHEvaluator():
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

# create an prompt generator
def get_bo_prompt_generator():
    prompt_generator = BaselinePromptGenerator()
    prompt_generator.is_bo = True
    return prompt_generator

def get_es_population(es_options):
    _n_parent = es_options['n_parent']
    _n_offspring = es_options['n_offspring']
    _is_elitist = es_options['is_elitist']
    _n_parent_per_offspring = 2
    if _n_parent < 2:
        _n_parent_per_offspring = 1
    population = ESPopulation(n_parent=_n_parent, n_parent_per_offspring=_n_parent_per_offspring, n_offspring=_n_offspring, use_elitism=_is_elitist)

    population.preorder_aware_init = True # pass previous population to LLM in the first generation
    population.save_per_generation = 1 # save population every generation
    population.debug_save_on_the_fly = True # save every individual in the population
    population.save_dir = es_options["log_dir"]

    p_name = f"{_n_parent}+{_n_offspring}"
    if _is_elitist:
        p_name = f'{_n_parent}-{_n_offspring}'

    population.name = f"evol_{p_name}" # the name of the population will be used as the prefix of the log directory

    return population


def run_example():
    # create an IOHEvaluator
    evaluator = get_IOHEvaluator()

    # create a prompt generator
    prompt_generator = get_bo_prompt_generator()

    # create a LLM Manager
    model_name = 'gemini-2.0-flash'
    api_key = 'your_key'
    base_url = None # use default

    # choose the llm client
    # openai: OpenaiClient
    # google: google genai client
    # others: AISuiteClient
    client = 'google'

    llm = LLMmanager(model_name=model_name, api_key=api_key, base_url=base_url, client_str=client)

    # define ES parameters
    log_dir = 'exp_es_search'
    es_options = {
        'n_parent': 1, # number of parents
        'n_offspring': 1, # number of offspring
        'is_elitist': False, # whether to use elitist selection
        'log_dir': log_dir, # directory to save logs
    }
    # create a ES Population
    population = get_es_population(es_options)

    # run the evolution
    llambo = LLaMBO()
    n_population = 4
    llm_params = {
        'temperature': 0.5,
        'top_k': 60,
    }

    llambo.run_evolutions(llm, evaluator, prompt_generator, population,
                        n_population=n_population,
                        options={'llm_params': llm_params})

    population.save(suffix='final')

    # plot the search results
    # combine all the log files with 'final' suffix
    plot_search(log_dir)


if __name__ == '__main__':
    setup_logger(level=logging.INFO)

    run_example()
