import logging
import getopt
import sys
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
from llamevol.prompt_generators import BaselinePromptGenerator
from llamevol.population import ESPopulation
from llamevol.llm import LLMmanager
from llamevol import LLaMEvol
from llamevol.utils import setup_logger
from Experiments.plot_search_res import plot_search_result


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

    population.preorder_aware_init = True # pass the code of all solutions in the first generation to the LLM
    population.save_per_generation = 1 # save population every generation
    population.debug_save_on_the_fly = True # save every individual in the population
    population.save_dir = es_options["log_dir"]

    p_name = f"{_n_parent}+{_n_offspring}"
    if not _is_elitist:
        p_name = f'{_n_parent}-{_n_offspring}'

    population.name = f"evol_{p_name}" # the name of the population will be used as the prefix of the log directory

    return population


def run_exp(n_parent, n_offspring, is_elitist, api_key, n_population=4):
    # create an IOHEvaluator
    evaluator = get_IOHEvaluator()
    evaluator.timeout = 30 * 60 # set the timeout(seconds) for each evaluation(all tasks) 

    # create a prompt generator
    prompt_generator = get_bo_prompt_generator()

    # create a LLM Manager
    model_name = 'gemini-2.0-flash'
    base_url = None # use default

    # choose the llm client, e.g. openai, google.
    # openai: OpenaiClient; google: google genai client; others: AISuiteClient
    client = 'google'

    llm = LLMmanager(model_name=model_name, api_key=api_key, base_url=base_url, client_str=client)

    # define ES parameters
    log_dir = 'exp_es_search'
    es_options = {
        'n_parent': n_parent, # number of parents
        'n_offspring': n_offspring, # number of offspring
        'is_elitist': is_elitist, # whether to use elitist selection
        'log_dir': log_dir, # directory to save logs
    }
    # create a ES Population
    population = get_es_population(es_options)

    # run the evolution
    llamevol = LLaMEvol()
    llm_params = {
        'temperature': 0.5,
        'top_k': 60, #!!!! top_k sampling, which might not be supported by all LLMs
    }

    llamevol.run_evolutions(llm, evaluator, prompt_generator, population,
                        n_population=n_population,
                        options={'llm_params': llm_params})

    population.save(suffix='final')


if __name__ == '__main__':
    setup_logger(level=logging.INFO)

    n_parents = 1
    n_offspring = 1
    is_elitist = False
    n_population = 4
    api_key = ''
    is_ploting = False 

    opts, args = getopt.getopt(sys.argv[1:], "p:o:k:en:f", )
    for opt, arg in opts:
        if opt == "-p":
            n_parents = int(arg)
        elif opt == "-o":
            n_offspring = int(arg)
        elif opt == "-k":
            api_key = arg
        elif opt == "-e":
            is_elitist = True
        elif opt == "-n":
            n_population = int(arg)
        elif opt == "-f":
            is_ploting = True
    
    if is_ploting:
        # plot the search results: combine all the log files with 'final' suffix
        print("Plotting the search results...")
        log_dir = 'exp_es_search'
        plot_search_result(log_dir, fig_dir=log_dir)
        sys.exit(0)

    print(f"n_parents: {n_parents}, n_offspring: {n_offspring}, is_elitist: {is_elitist}, n_population: {n_population}, api_key: {api_key}")

    if api_key == '':
        print("Please provide the API key with -k option.")
        sys.exit(1)

    run_exp(n_parents, n_offspring, is_elitist, api_key, n_population=n_population)

    