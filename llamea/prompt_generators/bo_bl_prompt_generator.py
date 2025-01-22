import re
from typing import Any
import numpy as np
from .abstract_prompt_generator import PromptGenerator, ResponseHandler, GenerationTask, EvaluatorResult
from .bo_zeroplus_prompt_generator import BOPromptGeneratorReturnChecker
from ..individual import Individual, ESPopulation


class BoBaselineResponseHandler(ResponseHandler):
    def __init__(self):
        super().__init__()

        self.desc = ""
        self.code = ""
        self.code_name = ""
        self.raw_response = ""

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
                    "Code"]
        for section in sections:
            if section == "Code":
                self.code, _ = self.extract_from_response(response, section)
                self.code_name, _ = self.extract_from_response(response, "class_name")
            elif section == "Description":
                self.desc, _ = self.extract_from_response(response, section)
        
    def extract_from_response(self, response: str, section: str, pattern=None) -> tuple[str, str]:
        error_str = ""
        res = ""
        if pattern is None:
            if section == "class_name":
                pattern = r"```(?:python)?[\s\S]*?class\s+(\w+v?\d*?.?\d*?):"
            elif section == "Code":
                pattern = r"```(?:python)?\s([\s\S]*?)```"
            else:
                pattern = rf"#\s*{section}\s*:\s*(.*)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            res = match.group(1)
        else:
            error_str = f"{section} not found in the response."
        return res, error_str

class BoBaselinePromptGenerator(PromptGenerator):

# prompt generation
    def get_prompt(self, task:GenerationTask, problem_desc:str,
                   candidates:list[BoBaselineResponseHandler]= None,
                   population:ESPopulation= None,
                   other_results:tuple[EvaluatorResult,list[EvaluatorResult]]= None,
                   sharedborad:Any=None) -> tuple[str, str]:

        if task != GenerationTask.INITIALIZE_SOLUTION:
            if candidates is None or len(candidates) == 0:
                return "", ""

        role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."

        task_prompt = self.task_description(task)

        response_format_prompt = self.response_format(task=task)

        if task == GenerationTask.INITIALIZE_SOLUTION:
            code_structure_prompt = "A code structure guide is as follows:\n" + self.code_structure()
            final_prompt = f"""{task_prompt}\n{code_structure_prompt}\n{response_format_prompt}"""
        else:
            candidate = candidates[0]
            description = candidate.desc
            solution = f"```python\n{candidate.code}\n```"
            if candidate.error:
                if candidate.error_type == "NoCodeException":
                    feedback = "No code was extracted. The code should be encapsulated with ``` in your response."
                else:
                    feedback = f"An error occurred : {candidate.error}"
            else:
                feedback = self.evaluation_feedback_prompt(candidate.eval_result, other_results)

            mutation_operator = "Refine the strategy of the selected solution to improve it."

            population_summary = ""
            if isinstance(population, ESPopulation) and len(population.selected_generations) > 0:
                population_summary = "The current population of algorithms already evaluated (name, description, score) is:\n"
                last_population = population.selected_generations[-1]
                last_inds = [population.individuals[ind_id] for ind_id in last_population if ind_id is not None]
                population_summary += "\n".join([ind.get_summary() for ind in last_inds])

            final_prompt = f"""{task_prompt}
{population_summary}

The selected solution to update is:
{description}

With code:
{solution}

{feedback}

{mutation_operator}
{response_format_prompt}
"""
        return role_prompt, final_prompt

    def task_description(self, task, extra = ""):
        task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries with GPU if possible. Do not use any other libraries unless they cannot be replaced by the above libraries. Name the class based on the characteristics of the algorithm with a template '<characteristics>BOv<version>'.
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea. 
"""
        return task_prompt

    def response_format(self, task:GenerationTask, extra:str="") -> str:
        output_format_prompt = """
Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code: <code>
"""
        return output_format_prompt

    def code_structure(self, extra:str="") -> str:
        return f"""
```python
from typing import Callable
from scipy.stats import qmc # If you are using QMC sampling. Otherwise or you have a better alternative, remove this line.
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

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points) -> np.ndarray:
        # sample points
        # return array of shape (n_points, n_dims)
    
    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model

    def _acquisition_function(self, X) -> np.ndarray:
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

    def _select_next_points(self, batch_size) -> np.ndarray:
        # Implement the strategy to select the next points to evaluate
        # return array of shape (batch_size, n_dims)
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # Do not change the function signature
        # Return a tuple (best_y, best_x)
        
        n_initial_points = <your_strategy>
        rest_of_budget = budget - n_initial_points
        while rest_of_budget > 0:
           # Optimization
           
           rest_of_budget -= <the number of func being called in this iteration>
        return best_y, best_x

    # Code Implementation only contain the algorithm class. No usage examples"
    {extra}
```
"""

    def evaluation_feedback_prompt(self, eval_res:EvaluatorResult, other_results:tuple[EvaluatorResult,list[EvaluatorResult]]= None) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""

        algorithm_name = eval_res.name
        aucs = []
        grouped_aucs = []
        for i in range(5):
            grouped_aucs.append([])
        for res in eval_res.result:
            aucs.append(res.y_aoc)

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
                "y_aoc": res.y_aoc
            }
            grouped_aucs[group_idx].append(content)

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)

        separated_aucs = [content["y_aoc"] for content in grouped_aucs[0]]
        separated_auc = np.mean(separated_aucs) if len(separated_aucs) > 0 else 0
        
        low_mod_aucs = [content["y_aoc"] for content in grouped_aucs[1]]
        low_mod_auc = np.mean(low_mod_aucs) if len(low_mod_aucs) > 0 else 0

        high_uni_aucs = [content["y_aoc"] for content in grouped_aucs[2]]
        high_uni_auc = np.mean(high_uni_aucs) if len(high_uni_aucs) > 0 else 0

        multi_adequate_aucs = [content["y_aoc"] for content in grouped_aucs[3]]
        multi_adequate_auc = np.mean(multi_adequate_aucs) if len(multi_adequate_aucs) > 0 else 0

        multi_weak_aucs = [content["y_aoc"] for content in grouped_aucs[4]]
        multi_weak_auc = np.mean(multi_weak_aucs) if len(multi_weak_aucs) > 0 else 0
            
        final_feedback_prompt = f"""The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}.
The mean AOCC score of the algorithm {algorithm_name} on Separable functions was {separated_auc:.02f}, on functions with low or moderate conditioning {low_mod_auc:.02f}, on functions with high conditioning and unimodal {high_uni_auc:.02f}, on Multi-modal functions with adequate global structure {multi_adequate_auc:.02f}, and on Multi-modal functions with weak global structure {multi_weak_auc:.02f}
"""
        return final_feedback_prompt

    
# Helper functions
    def get_response_handler(self):
        return BoBaselineResponseHandler()
