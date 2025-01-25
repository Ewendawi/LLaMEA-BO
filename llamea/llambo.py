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

from .individual import Individual, Population
from .llm import LLMmanager
from .prompt_generators import PromptGenerator, GenerationTask, ResponseHandler
from .utils import IndividualLogger, NoCodeException 
from .evaluator import EvaluatorResult, AbstractEvaluator

class LLaMBO:
    """
    A class that represents the Language Model powered Bayesian Optimization(LLaMBO).
    """
    def update_current_task(self, parent:list[Individual] = None, generation:int = 0) -> GenerationTask:
        if parent is None or generation == 0:
            return GenerationTask.INITIALIZE_SOLUTION
        elif len(parent) == 1:
            if parent[0].error:
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
                       retry:int=3,
                       ) -> ResponseHandler:
        if session_messages is None:
            return response_handler

        logging.debug("Session Messages:")
        logging.debug("\n%s\n%s", session_messages[0]["content"], session_messages[1]["content"])

        for i_try in range(retry):
            response = llm.chat(session_messages)
            # Retry if no response from the model
            if response is None or response == "":
                logging.error("No response from the model.") 
                logging.error("Retrying: %s/%s", i_try + 1, retry)
            else:
                break
        # logging.debug("Response:\n%s\n", response)
        logging.info("Response:\n%s\n", response)

        if response is None or response == "":
            logging.error("No response from the model. Exiting")
            err = NoCodeException("No response from the model.")
            response_handler.error = str(err)
            response_handler.error_type = err.__class__.__name__
            return response_handler

        response_handler.extract_response(response, task=task)
        if not response_handler.code or not response_handler.code_name:
            err = NoCodeException("ExtractionError: No code extracted from the model.")
            response_handler.error = str(err)
            response_handler.error_type = err.__class__.__name__
            logging.error("No code extracted from the model.")
            return response_handler

        # search whether the code include "cuda"
        if torch.cuda.is_available() and "cuda" not in response_handler.code:
            raise Exception("CUDA is available but the code does not use 'cuda'.")

        res = evaluator.evaluate(code=response_handler.code, cls_name=response_handler.code_name, max_eval_workers=n_eval_workers, timeout=timeout)

        logging.debug("Evaluation Result: %s", res)

        response_handler.eval_result = res

        return response_handler

    def run_evolutions(self, llm: LLMmanager,
                       evaluator: AbstractEvaluator,
                       prompt_generator: PromptGenerator,
                       population: Population,
                       sup_results: list[EvaluatorResult] = None,
                       n_generation: int = 1,
                       n_retry: int = 3,
                       n_query_threads: int = 0,
                       n_eval_workers: int = 0,
                       max_interval: int = 0,
                       time_out_per_eval: int = 1800):

        logging.info("Starting LLaMBO")
        logging.info("Model: %s", llm.model_name())
        logging.info(evaluator.problem_name())

        # progress_bar = tqdm(total=n_generation, desc="Generation", position=1, leave=True)

        evaluator.return_checker = prompt_generator.get_return_checker()

        problem_description = evaluator.problem_prompt()

        evolved_sharedbroad = prompt_generator.get_prompt_sharedbroad()

        last_query_time = 0
        current_generation = 0
        while current_generation < n_generation:
            logging.info("""=====================Generation: %s=====================""", current_generation)
            
            parents = population.get_parents()
            current_query_time = time.time()
            if current_query_time - last_query_time < max_interval:
                logging.info("Sleeping for %s seconds", max_interval - (current_query_time - last_query_time))
                time.sleep(max_interval - (current_query_time - last_query_time))
            last_query_time = time.time()

            next_handlers:list[ResponseHandler] = []
            params = []
            for i, parent in enumerate(parents):
                current_task = self.update_current_task(parent=parent, generation=current_generation)

                # Get prompt
                other_results = (None, sup_results)

                parent_handlers = [Population.get_handler_from_individual(p) for p in parent if p is not None]
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
                }
                params.append(kwargs)

            logging.info("Querying and Evaluating %s individuals", len(params))

            if n_query_threads > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_query_threads) as executor:
                    futures = {executor.submit(self.evalution_func, **kwargs): kwargs for kwargs in params}
                    for future in concurrent.futures.as_completed(futures):
                        handler = future.result()
                        next_handlers.append(future.result())
            else:
                for kwargs in params:
                    next_handler = self.evalution_func(**kwargs)
                    next_handlers.append(next_handler)

            for i, handler in enumerate(next_handlers):

                parent_ids = [p.id for p in parents[i] if p is not None]
                ind = Individual(solution=handler.code, name=handler.code_name, parent_id=parent_ids, generation=current_generation)
                ind.add_metadata("res_handler", handler)
                if handler.error:
                    ind.add_metadata("error_type", handler.error_type)
                    ind.error = str(handler.error)
                    ind.fitness = handler.eval_result.score if handler.eval_result else -np.inf
                else:
                    next_other_results = (None, sup_results)
                    ind.fitness = handler.eval_result.score
                    ind.feedback = prompt_generator.evaluation_feedback_prompt(handler.eval_result, next_other_results)

                ind.add_metadata("problem", evaluator.problem_name())
                ind.add_metadata('dimension', evaluator.problem_dim())
                ind.add_metadata("role_setting", role_setting)
                ind.add_metadata("prompt", prompt)
                ind.add_metadata("model", llm.model_name())
                ind.add_metadata("raw_response", handler.raw_response)
                tags = ind.metadata["tags"] if "tags" in ind.metadata else []
                tags.append(f"gen:{current_generation}")
                tags.append(f"task:{current_task.name}")
                tags.append(f"dim:{evaluator.problem_dim()}")
                ind.add_metadata("tags", tags)

                logging.info(ind.get_summary())
                if ind.error:
                    logging.info(ind.error)
                else:
                    logging.info(ind.feedback)

                population.add_individual(ind, current_generation)

            population.select_next_generation()
            
            prompt_generator.update_sharedbroad(evolved_sharedbroad, next_handlers, population)

            best_ind = population.get_best_individual(maximize=True)
            if best_ind is not None:
                logging.info("Best Individual: %s", best_ind.get_summary())

            current_generation = population.get_current_generation()
            # progress_bar.update(1)

        # progress_bar.close()
