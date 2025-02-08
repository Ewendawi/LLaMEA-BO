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
from .utils import IndividualLogger, NoCodeException, plot_results, plot_algo_results
from .evaluator import EvaluatorResult, AbstractEvaluator

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
                       ) -> ResponseHandler:
        if session_messages is None:
            return response_handler

        logging.info("Querying the model")
        for i_try in range(retry):
            response = llm.chat(session_messages)
            # Retry if no response from the model
            if response is None or response == "":
                logging.error("No response from the model.") 
                logging.error("Retrying: %s/%s", i_try + 1, retry)
            else:
                response_handler.extract_response(response, task=task)
                if not response_handler.code or not response_handler.code_name:
                    logging.error("No code extracted from the model.")
                    logging.error("Retrying: %s/%s", i_try + 1, retry)
                else:
                    break
        
        response_handler.sys_prompt = session_messages[0]["content"]
        response_handler.prompt = session_messages[1]["content"]
        response_handler.llm_model = llm.model_name()

        if not response_handler.code or not response_handler.code_name:
            err = NoCodeException("ExtractionError: No code extracted from the model.")
            response_handler.error = str(err)
            response_handler.error_type = err.__class__.__name__

        response_handler.raw_response = response

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

    def run_evolutions(self, llm: LLMmanager,
                       evaluator: AbstractEvaluator,
                       prompt_generator: PromptGenerator,
                       population: Population,
                       sup_results: list[EvaluatorResult] = None,
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

        evaluator.return_checker = prompt_generator.get_return_checker()

        problem_description = evaluator.problem_prompt()

        evolved_sharedbroad = prompt_generator.get_prompt_sharedbroad()

        last_query_time = 0
        current_generation = population.get_current_generation()
        current_population = population.get_population_size()
        while current_population < n_population and current_generation < n_generation:
            logging.info("""======Start Generation %s/%s with %s/%s Population=======""", current_generation, n_generation, current_population, n_population)
            
            _max_n_offspring = n_population - current_population
            query_items = population.get_offspring_queryitems(max_n_offspring=_max_n_offspring)

            current_query_time = time.time()
            if current_query_time - last_query_time < max_interval:
                logging.info("Sleeping for %s seconds", max_interval - (current_query_time - last_query_time))
                time.sleep(max_interval - (current_query_time - last_query_time))
            last_query_time = time.time()

            params = []
            for i, query_item in enumerate(query_items):
                current_task = self.update_current_task(query_item=query_item, generation=current_generation)

                # Get prompt
                other_results = (None, sup_results)

                parent_handlers = [Population.get_handler_from_individual(p) for p in query_item.parent if p is not None]
                role_setting, prompt = prompt_generator.get_prompt(
                    task=current_task,
                    problem_desc=problem_description,
                    candidates=parent_handlers,
                    population=population,
                    other_results=other_results,
                    sharedborad=evolved_sharedbroad)
                session_messages = [
                    {"role": "system", "content": role_setting},
                    {"role": "user", "content": prompt},
                ]
                
                next_handler = prompt_generator.get_response_handler()
                kwargs = {
                    "session_messages": session_messages,
                    "llm": llm,
                    "response_handler": next_handler,
                    "task": current_task,
                    "evaluator": evaluator,
                    "n_eval_workers": n_eval_workers,
                    "timeout": time_out_per_eval,
                    "retry": n_retry,
                    "gpu_name": gpu_name,
                }
                params.append(kwargs)

            logging.info("Querying and Evaluating %s individuals", len(params))

            next_handlers:dict[str, ResponseHandler] = {}
            if n_query_threads > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_query_threads) as executor:
                    futures = {}
                    for i, kwargs in enumerate(params):
                        future = executor.submit(self.evalution_func, **kwargs)
                        futures[future] = query_items[i].offspring.id
                    for future in concurrent.futures.as_completed(futures):
                        ind_id = futures[future]
                        handler = future.result()
                        if handler.code and handler.code_name:
                            next_handlers[ind_id] = handler
            else:
                for i, kwargs in enumerate(params):
                    next_handler = self.evalution_func(**kwargs)
                    if next_handler.code and next_handler.code_name:
                        ind_id = query_items[i].offspring.id
                        next_handlers[ind_id] = next_handler

            query_item_map = {query_item.offspring.id: query_item for query_item in query_items}
            for ind_id, handler in next_handlers.items():
                parent_ids = [p.id for p in query_item_map[ind_id].parent if p is not None]
                
                ind = query_item_map[ind_id].offspring
                ind.description = getattr(handler , "desc", "")
                ind.solution = handler.code
                ind.name = handler.code_name
                ind.parent_id = parent_ids
                Population.set_handler_to_individual(ind, handler)
                if handler.error:
                    ind.error = str(handler.error)
                    ind.fitness = handler.eval_result.score if handler.eval_result else -np.inf
                else:
                    next_other_results = (None, sup_results)
                    ind.fitness = handler.eval_result.score
                    ind.feedback = prompt_generator.evaluation_feedback_prompt(handler.eval_result, next_other_results)

                tags = ind.metadata["tags"] if "tags" in ind.metadata else []
                tags.append(f"gen:{current_generation}")
                ind.add_metadata("tags", tags)

                population.add_individual(ind, current_generation)

            population.select_next_generation()

            prompt_generator.update_sharedbroad(evolved_sharedbroad, next_handlers, population)

            best_ind = population.get_best_individual(maximize=True)
            if best_ind is not None:
                logging.info("Best Individual: %s(%.4f)\n%s", best_ind.name, best_ind.fitness, best_ind.description)

            current_generation = population.get_current_generation()
            current_population = population.get_population_size()
        
        logging.info("======Finished with %s Generations and %s Population======", current_generation, current_population)
