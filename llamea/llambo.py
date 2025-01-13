"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""
import logging
import time
import numpy as np
from tqdm import tqdm

from .individual import Individual, Population
from .llm import LLMmanager
from .promptGenerator import PromptGenerator, GenerationTask, ResponseHandler
from .utils import IndividualLogger
from .evaluator import EvaluatorResult, AbstractEvaluator 

class LLaMBO:
    """
    A class that represents the Language Model powered Bayesian Optimization(LLaMBO).
    """
    def extract_individual(self, task:GenerationTask, response:str, handler:ResponseHandler,) -> Individual:
        handler.extract_response(response, task=task)

        individual = Individual(handler.code, handler.code_name, None, None, None, None)
        individual.add_metadata("res_handler", handler)

        if not handler.code or not handler.code_name:
            individual.add_metadata("error_type", "ExtractionError")
            individual.error = f"ExtractionError: {response}"

        return individual

    def get_handler_from_individual(self, individual:Individual) -> ResponseHandler:
        if individual is None:
            return None
        return individual.metadata["res_handler"] if "res_handler" in individual.metadata else None

    def get_eval_result_from_individual(self, individual:Individual) -> EvaluatorResult:
        if individual is None:
            return None
        if individual.error is not None and individual.error != "":
            return None
        return individual.metadata["eval_result"] if "eval_result" in individual.metadata else None

    def last_successful_eval_result(self, population:Population, individual:Individual) -> EvaluatorResult:
        last_successful_candidate = population.get_last_successful_parent(individual)
        if last_successful_candidate is not None:
            return self.get_eval_result_from_individual(last_successful_candidate)
        return None

    def update_current_task(self, population:Population, previous_task:GenerationTask) -> GenerationTask:
        if population.get_population_size() == 0:
            return GenerationTask.INITIALIZE_SOLUTION
        else:
            individual = population.select_next_candidate()
            if individual.error:
                if previous_task == GenerationTask.FIX_ERRORS or previous_task == GenerationTask.FIX_ERRORS_FROM_ERROR:
                    return GenerationTask.FIX_ERRORS_FROM_ERROR
                return GenerationTask.FIX_ERRORS
            else:
                return GenerationTask.OPTIMIZE_PERFORMANCE

    def merge_evaluator_results(self, individual:Individual,
                                res:EvaluatorResult,
                                prompt_generator:PromptGenerator,
                                other_results:tuple[EvaluatorResult, list[EvaluatorResult]]=None):
        for key, value in res.metadata.items():
            individual.add_metadata(key, value)
        individual.add_metadata("error_type", res.error_type)
        individual.add_metadata("eval_result", res)
        individual.error = res.error
        sup_results = other_results[1] if other_results is not None and len(other_results) > 0 else None
        other_res = {}
        if sup_results is not None:
            for result in sup_results:
                other_res[result.name] = result

        if res.error is None or res.error == "":
            individual.feedback = prompt_generator.evaluation_feedback_prompt(res, other_results)

    def run_evolutions(self, llm: LLMmanager,
                       evaluator: AbstractEvaluator,
                       prompt_generator: PromptGenerator,
                       population: Population,
                       ind_logger: IndividualLogger = None,
                       sup_results: list[EvaluatorResult] = None,
                       n_generation: int = 1,
                       retry: int = 3,
                       max_error_in_a_row: int = 3,
                       verbose: int = 1):
        logging.info("Starting LLaMBO")
        logging.info("Model: %s", llm.model_name())
        logging.info(evaluator.problem_name())

        progress_bar = tqdm(total=n_generation)

        evaluator.return_checker = prompt_generator.get_return_checker()

        population.problem = evaluator.problem_name()
        population.model = llm.model_name()
        problem_description = evaluator.problem_prompt()

        current_task = None
        evolved_sharedbroad = prompt_generator.get_prompt_sharedbroad()

        generation = 0
        n_retry = 0
        n_eroor_in_a_row = 0
        while generation < n_generation:
            current_task = self.update_current_task(population, current_task)

            # Select candidate and get prompt
            candidate = population.select_next_candidate()
            last_res = self.last_successful_eval_result(population, candidate)
            other_results = (last_res, sup_results)

            candidate_handler = self.get_handler_from_individual(candidate) if candidate is not None else None
            candidates = [candidate_handler] if candidate_handler is not None else None
            role_setting, prompt = prompt_generator.get_prompt(
                task=current_task,
                problem_desc=problem_description,
                candidates=candidates,
                other_results=other_results,
                sharedborad=evolved_sharedbroad)
            session_messages = [
                {"role": "system", "content": role_setting},
                {"role": "user", "content": prompt},
            ]

            if verbose > 1:
                logging.info(prompt)
            response = llm.chat(session_messages)
            if verbose > 1:
                logging.info(response)

            # Retry if no response from the model
            if response is None or response == "":
                n_retry += 1
                sleep_time = 5*n_retry
                logging.error("No response from the model. Sleeping for %s seconds", sleep_time)
                time.sleep(sleep_time)
                if n_retry > retry:
                    logging.error("Max retry limit reached. Exiting")
                    break
                logging.info("Retrying: %s", n_retry)
                continue

            # Extract individual from the response
            response_handler = prompt_generator.get_response_handler()
            individual = self.extract_individual(current_task, response, response_handler)
            prompt_generator.update_sharedbroad(evolved_sharedbroad, response_handler)

            individual.parent_id = candidate.id if candidate is not None else None
            individual.generation = generation
            individual.add_metadata("problem", evaluator.problem_name())
            individual.add_metadata('dimension', evaluator.problem_dim())
            individual.add_metadata("role_setting", role_setting)
            individual.add_metadata("prompt", prompt)
            individual.add_metadata("model", llm.model_name())
            individual.add_metadata("raw_response", response)
            tags = individual.metadata["tags"] if "tags" in individual.metadata else []
            tags.append(f"gen:{generation}")
            tags.append(f"task:{current_task.name}")
            tags.append(f"dim:{evaluator.problem_dim()}")
            individual.add_metadata("tags", tags)

            if individual.error:
                # Retry if extraction error
                n_retry += 1
                logging.error("No sucessful extraction from the model. Retrying")
                if ind_logger is not None and ind_logger.log_extract_error:
                    ind_logger.add_individual(individual)
                continue
            else:
                n_retry = 0
                res = evaluator.evaluate(code=individual.solution, cls_name=individual.name)
                response_handler.eval_result = res
                next_other_results = (self.get_eval_result_from_individual(candidate), sup_results)
                self.merge_evaluator_results(individual, res, prompt_generator, next_other_results)

                population.add_individual(individual)

                if res.error is not None and res.error != "":
                    n_eroor_in_a_row += 1
                    if n_eroor_in_a_row >= max_error_in_a_row:
                        logging.error("Max error in a row reached %d. Exiting", n_eroor_in_a_row)
                        break
                else:
                    n_eroor_in_a_row = 0
                generation += 1
                progress_bar.update(1)

                logging.info(individual.feedback)
                logging.info(individual.error)

        if ind_logger is not None and ind_logger.should_log_population:
            ind_ids = []
            for individual in population.all_individuals():
                ind_logger.log_individual(individual)
                ind_ids.append(individual.id)
            if ind_logger.should_log_experiment:
                exp_name = population.name
                if exp_name is None:
                    exp_name = f"{evaluator.problem_dim()}dim_{evaluator.problem_name()}_{evaluator.eval_bugdet()}_{llm.model_name()}"
                ind_logger.log_experiment(name=exp_name, id_list=ind_ids)
            if ind_logger.auto_save:
                ind_logger.save()

        progress_bar.close()
