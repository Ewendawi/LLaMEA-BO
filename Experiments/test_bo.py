

import logging
import random
from typing import Callable

from llamea import LLaMBO, LLMmanager, BOPromptGenerator, GenerationTask, Individual, SequencePopulation
from llamea.utils import RandomBoTorchTestEvaluator, setup_logger, IndividualLogger
from llamea.llm import LLMS


def test_extract_from_response():
    """Test the extract_from_response method."""
    test_file = "tests/test_res.txt"
    # test_file = "tests/InformationGainBO.md"
    with open(test_file, "r") as f:
        response = f.read()
    
    po = BOPromptGenerator()
    name = po.extract_from_response(response, "class_name")
    description = po.extract_from_response(response, "Description")
    solution = po.extract_from_response(response, "Code")

    if name != "AckleyBO":
        raise Exception("Name extraction failed.")
    

def test_evaluate():
    """Test the evaluate method."""
    test_file = "tests/test_res.txt"
    test_file = "tests/InformationGainBO.md"
    with open(test_file, "r") as f:
        response = f.read()

    po = BOPromptGenerator()
    code = po.extract_from_response(response, "Code")[0]
    cls_name = po.extract_from_response(response, "class_name")[0]

    evaluator = RandomBoTorchTestEvaluator()

    res = evaluator.evaluate(code, cls_name)

    print(res.feedback)
    print(res.error)

def extract_key_words():
    # model = LLMS["deepseek/deepseek-chat"]
    # model = LLMS["gemini-1.5-flash-8b"]
    model = LLMS["gemini-2.0-flash-exp"]
    # model = LLMS["llama-3.1-70b-versatile"]
    # model = LLMS["llama-3.3-70b-versatile"]
    # model = LLMS["mixtral-8x7b-32768"]
    # model = LLMS["o_gemini-flash-1.5-8b-exp"]
    # model = LLMS["o_gemini-2.0-flash-exp"]
    # model = LLMS["o_gemini-2.0-flash-thinking-exp"]
    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])

    prompt_generator = BOPromptGenerator()

    individualLogger = IndividualLogger()
    individualLogger.load()

    count = 0
    for key, value in individualLogger.individual_map.items():
        if len(value['solution']) and "tags" not in value['metadata']:
            try:
                code = value['solution']
                prompt = prompt_generator.prompt_extract_keywords_from_code(code)

                session_messages = [
                    {"role": "user", "content": prompt},
                ]
                response = llm.chat(session_messages)
                key_words = response.split(",")
                key_words = [key_word.strip() for key_word in key_words]
                if len(key_words) > 1:
                    count += 1
                    value['metadata']['tags'] = key_words
                    print(f"count: {count}") 
            except Exception as e:
                print(e)
                continue

    if len(individualLogger.individual_map):
        print("Saving individual_map")


def test_llambo(task=GenerationTask.INITIALIZE_SOLUTION):
    """Test the LLaMBO class."""
    
    llambo = LLaMBO()
    # model = LLMS["deepseek/deepseek-chat"]
    # model = LLMS["gemini-1.5-flash-8b"]
    # model = LLMS["gemini-2.0-flash-exp"]
    model = LLMS["llama-3.1-70b-versatile"]
    # model = LLMS["mixtral-8x7b-32768"]
    # model = LLMS["o_gemini-flash-1.5-8b-exp"]
    # model = LLMS["o_gemini-2.0-flash-exp"]
    # model = LLMS["o_gemini-2.0-flash-thinking-exp"]
    llm = LLMmanager(api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])
    evaluator = RandomBoTorchTestEvaluator()
    prompt_generator = BOPromptGenerator()

    individualLogger = IndividualLogger()
    preiveous_individuals = []
    if task == GenerationTask.FIX_ERRORS:
        individualLogger.load()
        preiveous_individuals = individualLogger.get_failed_individuals("ExtractionError") 
    elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
        individualLogger.load()
        preiveous_individuals = individualLogger.get_successful_individuals()
        # suffle the individuals
        random.shuffle(preiveous_individuals)

    n_iterations = 1 if task == GenerationTask.INITIALIZE_SOLUTION else len(preiveous_individuals)
    for i in range(n_iterations):
        population = SequencePopulation()
        if task == GenerationTask.FIX_ERRORS:
            individual_dict = preiveous_individuals[i]
            individual = Individual.from_dict(individual_dict)
            population.add_individual(individual)
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            individual = Individual.from_dict(preiveous_individuals[i])
            problem_str = individual.metadata["problem"]
            words = problem_str.split(' ')
            problem = words[1]
            dim = int(words[0].split('-')[0])
            evaluator = RandomBoTorchTestEvaluator(dim=dim, obj_fn_name=problem)
            population.add_individual(individual)
   
        llambo.run_evolutions(llm, evaluator, prompt_generator, population, n_generation=5, ind_logger=individualLogger, retry=3)

# test_extract_from_response()
# test_evaluate()

setup_logger(level=logging.DEBUG)
# extract_key_words()

test_llambo(task=GenerationTask.INITIALIZE_SOLUTION)

# IndividualLogger.merge_logs().save_reader_format()
# IndividualLogger.load().save_reader_format()
