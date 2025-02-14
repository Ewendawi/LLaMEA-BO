"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""
import logging
import time
import concurrent.futures
import numpy as np
import torch
from tqdm import tqdm

from .population.population import Population, PopulationQueryItem
from .individual import Individual
from .llm import LLMmanager
from .prompt_generators import PromptGenerator, GenerationTask, ResponseHandler
from .utils import IndividualLogger, NoCodeException 
from .evaluator import EvaluatorResult, AbstractEvaluator

class LLaMBOTokenLogItem:
    def __init__(self, generation:int=0):
        self.generation = generation
        self.prompt_token_count = 0
        self.response_token_count = 0
        self.query_time = 0

    def __str__(self):
        return f"Generation: {self.generation}, Prompt Tokens: {self.prompt_token_count}, Response Tokens: {self.response_token_count}, Query Time: {self.query_time}"


class LLaMBO:
    """
    A class that represents the Language Model powered Bayesian Optimization(LLaMBO).
    """
    def update_current_task(self, query_item:PopulationQueryItem = None, generation:int = 0) -> GenerationTask:
        if query_item.is_initialized or query_item.parent is None or generation == 0:
            return GenerationTask.INITIALIZE_SOLUTION
        elif len(query_item.parent) == 1:
            if query_item.parent[0].error:
                return GenerationTask.FIX_ERRORS
            else:
                return GenerationTask.OPTIMIZE_PERFORMANCE
        else:
            return GenerationTask.OPTIMIZE_PERFORMANCE

    def evalution_func(self,
                       session_messages:list[dict[str, str]],
                       llm:LLMmanager,
                       response_handler:ResponseHandler,
                       task:GenerationTask,
                       evaluator:AbstractEvaluator,
                       n_eval_workers:int=0,
                       timeout:int=1800,
                       gpu_name:str=None,
                       retry:int=3,
                       options=None,
                       ) -> ResponseHandler:
        if session_messages is None:
            return response_handler

        logging.info("Querying")
        res_content = ''
        for i_try in range(retry):
            response = llm.chat(session_messages)
            res_content = response.text
            response_handler.query_time += 1
            response_handler.prompt_token_count += response.prompt_token_count
            response_handler.response_token_count += response.response_token_count

            # Retry if no response from the model
            if res_content is None or res_content == "" :
                logging.error("No response from the model.") 
                logging.error("Retrying: %s/%s", i_try + 1, retry)
            else:
                response_handler.extract_response(res_content, task=task)
                if not response_handler.code or not response_handler.code_name:
                    logging.error("No code extracted from the model.")
                    logging.error("Retrying: %s/%s", i_try + 1, retry)
                else:
                    break
        
        response_handler.sys_prompt = session_messages[0]["content"]
        response_handler.prompt = session_messages[1]["content"]
        response_handler.llm_model = llm.model_name()
        response_handler.raw_response = res_content

        if not response_handler.code or not response_handler.code_name:
            err = NoCodeException("ExtractionError: No code extracted from the model.")
            response_handler.error = str(err)
            response_handler.error_type = err.__class__.__name__

        if response_handler.error:
            return response_handler

        # search whether the code include "cuda"
        if torch.cuda.is_available(): 
            if "cuda" not in response_handler.code:
                raise Exception("CUDA is available but the code does not use 'cuda'.")
            else:
                if gpu_name is not None and gpu_name not in response_handler.code:
                    response_handler.code = response_handler.code.replace("\"cuda\"", f"\"{gpu_name}\"")
                    logging.info("replaced 'cuda' with '%s'", gpu_name)

        res = evaluator.evaluate(code=response_handler.code, cls_name=response_handler.code_name, max_eval_workers=n_eval_workers, timeout=timeout)

        response_handler.eval_result = res

        return response_handler

    def _update_individual(self, population:Population, query_item:PopulationQueryItem, handler:ResponseHandler, current_generation:int, promptor:PromptGenerator):
        if handler.code is None or handler.code_name is None:
            return
        
        parent_ids = query_item.parent
        ind = query_item.offspring
        ind.description = getattr(handler , "desc", "")
        ind.solution = handler.code
        ind.name = handler.code_name
        ind.parent_id = parent_ids
        Population.set_handler_to_individual(ind, handler)
        if handler.error:
            ind.error = str(handler.error)
            ind.fitness = handler.eval_result.score if handler.eval_result else -np.inf
        else:
            ind.fitness = handler.eval_result.score
            ind.feedback = promptor.evaluation_feedback_prompt(handler.eval_result, None)

        tags = ind.metadata["tags"] if "tags" in ind.metadata else []
        tags.append(f"gen:{current_generation}")
        ind.add_metadata("tags", tags)

    def _update_token_log(self, token_log:list[LLaMBOTokenLogItem], handler:ResponseHandler):
        if len(token_log) == 0:
            return
        token_log[-1].query_time += handler.query_time
        token_log[-1].prompt_token_count += handler.prompt_token_count
        token_log[-1].response_token_count += handler.response_token_count

    def run_evolutions(self, llm: LLMmanager,
                       evaluator: AbstractEvaluator,
                       prompt_generator: PromptGenerator,
                       population: Population,
                       n_generation: int = np.inf,
                       n_population: int = 1,
                       n_retry: int = 3,
                       n_query_threads: int = 0,
                       n_eval_workers: int = 0,
                       max_interval: int = 0,
                       gpu_name: str = None,
                       time_out_per_eval: int = 1800):

        logging.info("==========Starting==========")
        logging.info("%s", llm.model_name())
        logging.info("%s", prompt_generator)
        logging.info("%s", evaluator)
        logging.info("%s", population)

        _token_log:list[LLaMBOTokenLogItem] = []

        last_query_time = 0
        current_generation = population.get_current_generation()
        current_population = population.get_population_size()
        while current_population < n_population and current_generation < n_generation:
            logging.info("""======Start Generation %s/%s with %s/%s Population=======""", current_generation, n_generation, current_population, n_population)

            _token_log.append(LLaMBOTokenLogItem(current_generation))
            
            _max_n_offspring = n_population - current_population
            _query_items = population.get_offspring_queryitems(max_n_offspring=_max_n_offspring)

            current_query_time = time.time()
            if current_query_time - last_query_time < max_interval:
                logging.info("Sleeping for %s seconds", max_interval - (current_query_time - last_query_time))
                time.sleep(max_interval - (current_query_time - last_query_time))
            last_query_time = time.time()

            _query_queue = []
            _query_queue.extend(_query_items)

            _round = 0
            while len(_query_queue) > 0:
                logging.info("==Round %s: %s offspring==",_round, len(_query_queue))
                params = []
                for i, query_item in enumerate(_query_queue):
                    current_task = self.update_current_task(query_item=query_item, generation=current_generation)

                    # Get prompt
                    other_results = (None, None)

                    parent_handlers = [Population.get_handler_from_individual(p) for p in query_item.parent if p is not None]
                    _promptor = population.get_promptor(query_item=query_item)
                    if _promptor is None:
                        _promptor = prompt_generator
                    else:
                        logging.info("Custom promptor: %s", _promptor)

                    _evaluator = population.get_evaluator(query_item=query_item)
                    if _evaluator is None:
                        _evaluator = evaluator
                    else:
                        logging.info("Custom evaluator: %s", _evaluator)
                    _evaluator.return_checker = _promptor.get_return_checker()

                    _problem_description = _evaluator.problem_prompt()

                    role_setting, prompt = _promptor.get_prompt(
                        task=current_task,
                        problem_desc=_problem_description,
                        candidates=parent_handlers,
                        population=population,
                        other_results=other_results)
                    session_messages = [
                        {"role": "system", "content": role_setting},
                        {"role": "user", "content": prompt},
                    ]
                    
                    next_handler = _promptor.get_response_handler()
                    kwargs = {
                        "session_messages": session_messages,
                        "llm": llm,
                        "response_handler": next_handler,
                        "task": current_task,
                        "evaluator": _evaluator,
                        "n_eval_workers": n_eval_workers,
                        "timeout": time_out_per_eval,
                        "retry": n_retry,
                        "gpu_name": gpu_name,
                    }
                    params.append((kwargs, query_item))


                _next_query_items = []
                if n_query_threads > 0:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n_query_threads) as executor:
                        futures = {}
                        for i, param in enumerate(params):
                            kwargs, _query_item = param
                            future = executor.submit(self.evalution_func, **kwargs)
                            futures[future] = _query_item

                        for future in concurrent.futures.as_completed(futures):
                            _query_item = futures[future]
                            handler = future.result()


                            self._update_individual(population, _query_item, handler, current_generation, _promptor)

                            _following_query_items = population.get_next_queryitems([_query_item])
                            if _following_query_items:
                                _next_query_items.extend(_following_query_items)
                            else:
                                population.add_individual(_query_item.offspring, generation=current_generation)
                            self._update_token_log(_token_log, handler)
                else:
                    for i, param in enumerate(params):
                        kwargs, _query_item = param
                        next_handler = self.evalution_func(**kwargs)

                        self._update_individual(population, _query_item, next_handler, current_generation, _promptor)
                        _following_query_items = population.get_next_queryitems([_query_item])
                        if _following_query_items:
                            _next_query_items.extend(_following_query_items)
                        else:
                            population.add_individual(_query_item.offspring, generation=current_generation)
                        self._update_token_log(_token_log, next_handler)

                _query_queue = _next_query_items
                _round += 1

            logging.info('%s', _token_log[-1])

            population.select_next_generation()

            best_ind = population.get_best_individual(maximize=True)
            if best_ind is not None:
                logging.info("Best Individual: %s(%.4f)\n%s", best_ind.name, best_ind.fitness, best_ind.description)

            current_generation = population.get_current_generation()
            current_population = population.get_population_size()
        
        logging.info("======Finished with %s Generations and %s Population======", current_generation, current_population)
