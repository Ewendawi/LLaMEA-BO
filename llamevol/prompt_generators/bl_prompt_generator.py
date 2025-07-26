import re
from typing import Any
import numpy as np
import torch
from .abstract_prompt_generator import PromptGenerator, ResponseHandler, GenerationTask, EvaluatorResult
from .bo_zeroplus_prompt_generator import BOPromptGeneratorReturnChecker
from ..individual import Individual
from ..population import Population


class BaselineResponseHandler(ResponseHandler):
    def __init__(self):
        super().__init__()
        self.desc = ""
        self.reason = ""

    def __to_json__(self):
        return {
            "desc": self.desc,
            "code": self.code,
            "code_name": self.code_name,
            "raw_response": self.raw_response
        }

    def extract_response(self, response:str, task:GenerationTask):
        if not response:
            return

        self.raw_response = response
        sections = ["Description",
                    "Justification",
                    "Code"]
        for section in sections:
            if section == "Code":
                self.code, err = self.extract_from_response(response, section)
                if err:
                    self.code, _ = self.extract_from_response(response, "Code2")
                self.code_name, _ = self.extract_from_response(response, "class_name")
            elif section == "Description":
                self.desc, _ = self.extract_from_response(response, section)
            elif section == "Justification":
                self.reason, _ = self.extract_from_response(response, section)

    def extract_from_response(self, response: str, section: str, pattern=None) -> tuple[str, str]:
        error_str = ""
        res = ""
        if pattern is None:
            if section == "class_name":
                pattern = r"```(?:python)?[\s\S]*?class\s+(\w+BO\w*):"
            elif section == "Code":
                pattern = r"#\s*Code[\s\S]*```(?:python)?\s([\s\S]*?)```"
            elif section == "Code2":
                pattern = r"```(?:python)?\s([\s\S]*?)```"
            else:
                pattern = rf"#\s*{section}\s*([\s\S]*?)#\s"
                # pattern = rf"#\s*{section}\s*:\s*(.*)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            res = match.group(1)
        else:
            error_str = f"{section} not found in the response."
        return res, error_str


class BaselinePromptGenerator(PromptGenerator):

    def __init__(self):
        super().__init__()
        self.is_bo = False
        self.problem_desc = "24 noiseless functions"

    def __str__(self):
        is_bo = "BO" if self.is_bo else ""
        return f"{is_bo}BaselinePromptGenerator"

# prompt generation
    def get_prompt(self, task:GenerationTask, problem_desc:str,
                   candidates:list[BaselineResponseHandler]= None,
                   population:Population= None,
                   options:dict = None
                   ) -> tuple[str, str]:

        if task != GenerationTask.INITIALIZE_SOLUTION:
            if candidates is None or len(candidates) == 0:
                return "", ""

        role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems"

        task_prompt = self.task_description(task)

        response_format_prompt = self.response_format(task=task)

        if task == GenerationTask.INITIALIZE_SOLUTION:
            pre_solution_prompt = ""
            if len(candidates) > 0:
                n_solution = len(candidates)
                pre_solution_prompt = f"{n_solution} algorithms have been designed. The new algorithm should be as **diverse** as possible from the previous ones on every aspect.\n"
                pre_solution_prompt += "If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.\n"
                for i, candidate in enumerate(candidates):
                    candidate_prompt = self.__get_candidate_prompt(candidate)
                    pre_solution_prompt += f"## {candidate.code_name}\n{candidate_prompt}\n"

                    # pre_solution_prompt += f"- {candidate.desc}\n"
                pre_solution_prompt += "\n"

            code_structure_prompt = "A code structure guide is as follows and keep the comments from the guide when generating the code.\n" + self.code_structure()
            final_prompt = f"""{task_prompt}\n{pre_solution_prompt}\n{code_structure_prompt}\n{response_format_prompt}"""
        else:
            if len(candidates) > 1:
                crossover_operator = "Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.\n"

                selected_prompt = "The selected solutions to update are:\n"

                for candidate in candidates:
                    candidate_prompt = self.__get_candidate_prompt(candidate)
                    selected_prompt += f"## {candidate.code_name}\n{candidate_prompt}\n"

                selected_prompt += f"{crossover_operator}\n"
            else:
                candidate = candidates[0]
                candidate_prompt = self.__get_candidate_prompt(candidate)
                mutation_operator = "Refine the strategy of the selected solution to improve it."

                selected_prompt = f"""The selected solution to update is:\n{candidate_prompt}\n{mutation_operator}\n"""

            population_summary = ""
            if population is not None and population.get_population_size() > 0:
                current_population = population.get_individuals()
                population_summary = "The current population of algorithms already evaluated(name, score, runtime and description):\n"
                for ind in current_population:
                    handler = Population.get_handler_from_individual(ind)
                    if handler.eval_result is None:
                        continue
                    name = handler.code_name
                    score = handler.eval_result.score
                    runtime = handler.eval_result.total_execution_time
                    desc = handler.desc
                    population_summary += f"- {name}: {score:.4f}, {runtime:.2f} seconds, {desc}\n"

            final_prompt = f"""{task_prompt}
{population_summary}

{selected_prompt}

{response_format_prompt}
"""
        return role_prompt, final_prompt

    def __get_candidate_prompt(self, candidate:BaselineResponseHandler) -> str:
        description = candidate.desc
        solution = f"```python\n{candidate.code}\n```"
        if candidate.error:
            if candidate.error_type == "NoCodeException":
                feedback = "No code was extracted. The code should be encapsulated with ``` in your response."
            else:
                feedback = f"An error occurred : {candidate.error}"
        else:
            feedback = self.evaluation_feedback_prompt(candidate.eval_result)

        
        return f"{description}\nWith code:\n{solution}\n{feedback}\n"

    def task_description(self, task:GenerationTask) -> str:
        if self.is_bo:
            return self.__bo_task_description(task)
        return self.__task_description(task)

    def __bo_task_description(self, task):
        # lib_prompt = "As an expert of numpy, scipy, scikit-learn, you are allowed to use these libraries."
        lib_prompt = "As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries."
        if torch.cuda.is_available():
            lib_prompt = "As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries, and using GPU for acceleration is mandatory."
        # problem_desc = "one noiseless functions:f6-Attractive Sector Function"
        task_prompt = f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {self.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
{lib_prompt} Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        return task_prompt

    def __task_description(self, task:GenerationTask) -> str:
        task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
Give an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.
"""
        return task_prompt

    def response_format(self, task:GenerationTask) -> str:
        output_format_prompt = """
Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>
"""
        return output_format_prompt

    def code_structure(self):
        if self.is_bo:
            return self.__bo_code_structure()
        return self.__code_structure()

    def __code_structure(self) -> str:
        return """
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        # bounds has (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(self.bounds[0], self.bounds[1])
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```
"""

    def __bo_code_structure(self) -> str:
        return """
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
class <AlgorithmName>:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = <your_strategy>

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)

        self.n_evals += len(X)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        self._evaluate_points()
        self._update_eval_points()
        while self.n_evals < budget:
            # Optimization

            # select points by acquisition function
            self._evaluate_points()
            self._update_eval_points()

        return best_y, best_x
```
"""

    def evaluation_feedback_prompt(self, eval_res:EvaluatorResult, options:dict = None) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""

        algorithm_name = eval_res.name
        aocs = []
        grouped_aocs = []
        for _ in range(5):
            grouped_aocs.append([])
        for res in eval_res.result:
            aoc = res.log_y_aoc
            aocs.append(aoc)

            res_id = res.id
            res_split = res_id.split("-")
            problem_id = int(res_split[0])
            instance_id = int(res_split[1])
            repeat_id = int(res_split[2])
            if problem_id <= 5:
                group_idx = 0
            elif problem_id <= 9:
                group_idx = 1
            elif problem_id <= 14:
                group_idx = 2
            elif problem_id <= 19:
                group_idx = 3
            else:
                group_idx = 4
            content = {
                "problem_id": problem_id,
                "instance_id": instance_id,
                "repeat_id": repeat_id,
                "y_aoc": aoc
            }
            grouped_aocs[group_idx].append(content)

        auc_mean = np.mean(aocs)
        auc_std = np.std(aocs)

        separated_aocs = [content["y_aoc"] for content in grouped_aocs[0]]
        separated_auc = np.mean(separated_aocs) if len(separated_aocs) > 0 else 0
        
        low_mod_aocs = [content["y_aoc"] for content in grouped_aocs[1]]
        low_mod_auc = np.mean(low_mod_aocs) if len(low_mod_aocs) > 0 else 0

        high_uni_aocs = [content["y_aoc"] for content in grouped_aocs[2]]
        high_uni_auc = np.mean(high_uni_aocs) if len(high_uni_aocs) > 0 else 0

        multi_adequate_aocs = [content["y_aoc"] for content in grouped_aocs[3]]
        multi_adequate_auc = np.mean(multi_adequate_aocs) if len(multi_adequate_aocs) > 0 else 0

        multi_weak_aocs = [content["y_aoc"] for content in grouped_aocs[4]]
        multi_weak_auc = np.mean(multi_weak_aocs) if len(multi_weak_aocs) > 0 else 0

        execution_time = eval_res.total_execution_time
        time_prompt = f"took {execution_time:0.2f} seconds to run."
            
        main_aoc_prompt = f"""The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard deviation {auc_std:0.4f}.
"""
        detailed_aoc_prompt = f"""
The mean AOCC score of the algorithm {algorithm_name} on Separable functions was {separated_auc:.04f}, on functions with low or moderate conditioning {low_mod_auc:.04f}, on functions with high conditioning and unimodal {high_uni_auc:.04f}, on Multi-modal functions with adequate global structure {multi_adequate_auc:.04f}, and on Multi-modal functions with weak global structure {multi_weak_auc:.04f}
"""
        final_feedback_prompt = f"{main_aoc_prompt}\n{time_prompt}"
        return final_feedback_prompt

    
# Helper functions
    def get_response_handler(self):
        return BaselineResponseHandler()

class LightBaselinePromptGenerator(BaselinePromptGenerator):
    def evaluation_feedback_prompt(self, eval_res:EvaluatorResult, options:dict = None) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""

        algorithm_name = eval_res.name
        
        execution_time = eval_res.total_execution_time
        time_prompt = f"The algorithm {algorithm_name} took {execution_time:0.2f} seconds to run."

        return time_prompt
