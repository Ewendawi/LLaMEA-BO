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
from .promptGenerator import BOPromptGenerator, GenerationTask
from .utils import AbstractEvaluator, EvaluatorResult, IndividualLogger

class LLaMBO:
    """
    A class that represents the Language Model powered Bayesian Optimization(LLaMBO).
    """
    def construct_prompt(self, task:GenerationTask, candidate:Individual,prompt_generator:BOPromptGenerator, problem_desc:str) -> tuple[str, str]:

        if task == GenerationTask.INITIALIZE_SOLUTION:
            final_prompt = ""

            task_prompt = prompt_generator.task_description(task)
            final_prompt += f"{task_prompt}\n"
            
            task_instruction_prompt = prompt_generator.task_instruction(task)
            final_prompt += f"{task_instruction_prompt}\n"

            problem_prompt = prompt_generator.problem_description(problem_desc)
            final_prompt += f"{problem_prompt}\n"
            
            code_structure_prompt = prompt_generator.code_structure()
            final_prompt += f"{code_structure_prompt}\n"
            
            response_format_prompt = prompt_generator.response_format(task=task)
            final_prompt += f"{response_format_prompt}\n"
        elif task == GenerationTask.FIX_ERRORS:
            if candidate is None:
                return "", "No candidate available"
            
            final_prompt = ""

            # problem_prompt = prompt_generator.problem_description(problem_desc)
            # final_prompt += f"{problem_prompt}\n"
            
            task_prompt = prompt_generator.task_description(task)
            final_prompt += f"{task_prompt}\n"
            
            task_instruction_prompt = prompt_generator.task_instruction(task)
            final_prompt += f"{task_instruction_prompt}\n"

            final_prompt += f"### Previous Solution\n```python\n{candidate.solution}\n```\n"
            final_prompt += f"### Previous Error\n```bash\n{candidate.error}\n```\n"
            
            response_format_prompt = prompt_generator.response_format(task)
            final_prompt += f"{response_format_prompt}\n"
        else:
            if candidate is None:
                return "", "No candidate available"
            
            final_prompt = ""

            task_prompt = prompt_generator.task_description(task)
            final_prompt += f"{task_prompt}\n"
            
            task_instruction_prompt = prompt_generator.task_instruction(task)
            final_prompt += f"{task_instruction_prompt}\n"

            problem_prompt = prompt_generator.problem_description(problem_desc)
            final_prompt += f"{problem_prompt}\n"

            final_prompt += f"### Previous Solution\n```python\n{candidate.solution}\n```\n"

            final_prompt += f"### Previous Feedback\n{candidate.feedback}\n"

            #TODO: the feeeback of the trajectory? 
            
            response_format_prompt = prompt_generator.response_format(task)
            final_prompt += f"{response_format_prompt}\n"
        
        return "", final_prompt

    def extract_individual(self, task:GenerationTask, response:str, prompt_generator:BOPromptGenerator) -> Individual:
        if task == GenerationTask.INITIALIZE_SOLUTION:
            name, name_err = prompt_generator.extract_from_response(response, "class_name")
            description, desc_err = prompt_generator.extract_from_response(response, "Description")
            solution, solution_err = prompt_generator.extract_from_response(response, "Code")
            
            individual = Individual(solution, name, description, None, None, None)

            if name_err or solution_err:
                individual.add_metadata("error_type", "ExtractionError")
                individual.error = f"ExtractionError: Name-{name_err}, Description-{desc_err}, Solution-{solution_err}"

            return individual
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            name, name_err = prompt_generator.extract_from_response(response, "class_name")
            description, desc_err = prompt_generator.extract_from_response(response, "Description")
            solution, solution_err = prompt_generator.extract_from_response(response, "Code")

            individual = Individual(solution, name, description, None, None, None)
            if name_err or solution_err:
                individual.add_metadata("error_type", "ExtractionError")
                individual.error = f"ExtractionError: Name-{name_err}, Description-{desc_err}, Solution-{solution_err}"
            return individual
        elif task == GenerationTask.FIX_ERRORS:
            name, name_err = prompt_generator.extract_from_response(response, "class_name")
            description, desc_err = prompt_generator.extract_from_response(response, "Description")
            solution, solution_err = prompt_generator.extract_from_response(response, "Code")
            
            individual = Individual(solution, name, description, None, None, None)

            if name_err or solution_err:
                individual.add_metadata("error_type", "ExtractionError")
                individual.error = f"ExtractionError: Name-{name_err}, Description-{desc_err}, Solution-{solution_err}"

            return individual

    def update_current_task(self, population:Population, curruent_task:GenerationTask) -> GenerationTask:
        if population.get_population_size() == 0:
            return GenerationTask.INITIALIZE_SOLUTION
        else:
            individual = population.select_next_generation()
            if individual.error:
                return GenerationTask.FIX_ERRORS
            else:
                return GenerationTask.OPTIMIZE_PERFORMANCE

    def merge_evaluator_results(self, individual:Individual, res:EvaluatorResult):
        individual.fitness = res.result.best_y
        for key, value in res.metadata.items():
            individual.add_metadata(key, value)
        individual.add_metadata("optimal_value", res.optimal_value)
        individual.add_metadata("error_type", res.error_type)
        individual.add_metadata("budget", res.budget)
        individual.add_metadata("captured_output", res.captured_output)
        individual.add_metadata("result_values", res.result.to_dict())
        individual.error = res.error
        if len(res.other_results) > 0:
            other_results = {}
            for result in res.other_results:
                other_results[result.name] = result.to_dict()
            individual.add_metadata("other_results", other_results)
        
        if res.error is None or res.error == "":
            feedback_prompt = "### Feedback\n"
            if res.optimal_value is not None:
                feedback_prompt += f"- Optimal Value: {res.optimal_value}\n"
            feedback_prompt += f"- Budget: {res.budget}\n"
            
            all_results = [res.result] + res.other_results
            for result in all_results:
                feedback_prompt += f"#### {result.name}\n"
                feedback_prompt += f"- best y: {result.best_y:.2f}\n"
                if result.n_initial_points > 0:
                    if result.y_best_tuple is not None:
                        feedback_prompt += f"- initial best y: {result.y_best_tuple[0]:.2f}\n"
                        feedback_prompt += f"- non-initial best y: {result.y_best_tuple[1]:.2f}\n"
                    feedback_prompt += f"- AOC for non-initial y: {result.non_init_y_aoc:.2f}\n" 
                    if result.x_mean_tuple is not None and result.x_std_tuple is not None:
                        feedback_prompt += f"- mean and std of initial x: {np.array_str(result.x_mean_tuple[0], precision=2)} , {np.array_str(result.x_std_tuple[0], precision=2)}\n"
                        feedback_prompt += f"- mean and std of non-initial x: {np.array_str(result.x_mean_tuple[1], precision=2)} , {np.array_str(result.x_std_tuple[1], precision=2)}\n" 
                    feedback_prompt += f"- mean and std of non-initial y: {result.y_mean_tuple[1]:.2f} , {result.y_std_tuple[1]:.2f}\n"
                else:
                    feedback_prompt += f"- AOC for all y: {result.y_aoc:.2f}\n"
                    if result.x_mean is not None and result.x_std is not None:
                        feedback_prompt += f"- mean and std of all x: {np.array_str(result.x_mean, precision=2)} , {np.array_str(result.x_std, precision=2)}\n"
                        feedback_prompt += f"- mean and std of all y: {result.y_mean:.2f} , {result.y_std:.2f}\n"
                
                if result.surragate_model_losses is not None:
                    feedback_prompt += f"- mean and std {result.model_loss_name} of suragate model: {np.mean(result.surragate_model_losses):.2f} , {np.std(result.surragate_model_losses):.2f}\n"

                # feedback_prompt += f"Execution Time: {result.execution_time:.4f}\n"
            feedback_prompt += """#### Note: 
- AOC(Area Over the Convergence Curve): a measure of the convergence speed of the algorithm, ranged between 0.0 and 1.0. A higher value is better.
- non-initial x: the x that are sampled during the optimization process, excluding the initial points.
- Budget: Maximum number of function evaluations allowed for the algorithm.
"""
            individual.feedback = feedback_prompt

    def run_evolutions(self, llm: LLMmanager, evaluator: AbstractEvaluator, prompt_generator: BOPromptGenerator, population: Population, n_generation: int = 1, ind_logger: IndividualLogger = None, retry: int = 3,):

        progress_bar = tqdm(total=n_generation)

        evaluator.return_checker = prompt_generator.get_return_checker()

        logging.info("Starting LLaMBO")
        logging.info(f"Model: {llm.model_name()}")
        logging.info(evaluator.problem_name())

        population.problem = evaluator.problem_name()
        population.model = llm.model_name()
        problem_description = evaluator.problem_prompt()
        
        current_task = None

        generation = 0
        n_retry = 0
        while generation < n_generation:

            current_task = self.update_current_task(population, current_task)
            candidate = population.select_next_generation()

            role_setting, prompt = self.construct_prompt(current_task, candidate, prompt_generator, problem_description)
            session_messages = [
                {"role": "system", "content": role_setting},
                {"role": "user", "content": prompt},
            ]
            logging.debug(prompt)
            response = llm.chat(session_messages)
            logging.debug(response)

            # Retry if no response from the model
            if response is None or response == "":
                n_retry += 1
                sleep_time = 5*n_retry
                logging.error(f"No response from the model. Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
                if n_retry > retry:
                    logging.error("Max retry limit reached. Exiting")
                    break
                logging.info(f"Retrying:{n_retry}")
                continue


            individual = self.extract_individual(current_task, response, prompt_generator)
            individual.parent_id = candidate.id if candidate is not None else None
            individual.generation = generation
            individual.add_metadata("problem", evaluator.problem_name()) 
            individual.add_metadata('dimention', evaluator.problem_dim())
            individual.add_metadata("role_setting", role_setting)
            individual.add_metadata("prompt", prompt)
            individual.add_metadata("model", llm.model_name())
            individual.add_metadata("raw_response", response)
            individual.add_metadata("aggresiveness", prompt_generator.aggressiveness)
            tags = individual.metadata["tags"] if "tags" in individual.metadata else []
            tags.append(f"gen:{generation}")
            tags.append(f"task:{current_task}")
            tags.append(f"aggr:{prompt_generator.aggressiveness}")
            tags.append(f"dim:{evaluator.problem_dim()}")
            if prompt_generator.use_botorch:
                tags.append("botorch")
            individual.add_metadata("tags", tags)

            if individual.error:
                # Retry if extraction error
                n_retry += 1
                logging.error("No suucessful extraction from the model. Retrying") 
                if ind_logger is not None and ind_logger.log_extract_error:
                    ind_logger.add_individual(individual)
                continue
            else:
                n_retry = 0
                res = evaluator.evaluate(code=individual.solution, cls_name=individual.name)
                self.merge_evaluator_results(individual, res)

                population.add_individual(individual)

                generation += 1
                progress_bar.update(1)

                logging.info(individual.feedback)
                logging.info(res.captured_output)
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

