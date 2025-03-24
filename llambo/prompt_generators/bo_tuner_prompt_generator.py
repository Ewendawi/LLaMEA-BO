import re
from typing import Any
import numpy as np
from .abstract_prompt_generator import PromptGenerator, ResponseHandler, GenerationTask, EvaluatorResult
from ..population import Population

class TunerResponseHandler(ResponseHandler):
    def __init__(self):
        super().__init__()

        self.reason = ""
        self.code = ""
        self.code_name = ""
        self.raw_response = ""

    def __to_json__(self):
        return {
            'reason': self.reason,
            "code": self.code,
            "code_name": self.code_name,
            "raw_response": self.raw_response
        }

    def extract_response(self, response:str, task:GenerationTask):
        if not response:
            return
        
        self.raw_response = response
        sections = [
                    "Justifications",
                    "Code"
                ]
        for section in sections:
            if section == "Code":
                self.code, _ = self.extract_from_response(response, section)
                self.code_name, _ = self.extract_from_response(response, "class_name")
            elif section == "Justifications":
                self.reason, _ = self.extract_from_response(response, section)
        
    def extract_from_response(self, response: str, section: str, pattern=None) -> tuple[str, str]:
        error_str = ""
        res = ""
        if pattern is None:
            if section == "class_name":
                pattern = r"```(?:python)?[\s\S]*?class\s+(\w+BOv?\d*?.?\d*?):"
            elif section == "Code":
                pattern = r"##\s*Code[\s\S]*```(?:python)?\s([\s\S]*?)```"
            else:
                pattern = rf"##\s*{section}\s*([\s\S]*)##\sCode"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            res = match.group(1)
        else:
            error_str = f"{section} not found in the response."
        return res, error_str

class TunerPromptGenerator(PromptGenerator):
    def __str__(self):
        return "TunerPromptGenerator"

# prompt generation
    def get_prompt(self, task:GenerationTask, problem_desc:str,
                   candidates:list[TunerResponseHandler]= None,
                   population:Population= None,
                   other_results:tuple[EvaluatorResult,list[EvaluatorResult]]= None,
                   sharedborad:Any=None) -> tuple[str, str]:

        if candidates is None or len(candidates) == 0:
            raise ValueError("No candidates provided.")

        role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."

        task_prompt = self.task_description(task)

        response_format_prompt = self.response_format(task=task)

        pre_solution_prompt = "The provided Bayesian Optimization algorithm is as follows:\n"
        for i, candidate in enumerate(candidates):
            candidate_prompt = self.__get_candidate_prompt(candidate)
            pre_solution_prompt += f"## {candidate.code_name}\n{candidate_prompt}\n"

            # pre_solution_prompt += f"- {candidate.desc}\n"
        pre_solution_prompt += "\n"

        other_solutions_prompt = ""
        _all_individuals = population.all_individuals()
        for ind in _all_individuals:
            handler = Population.get_handler_from_individual(ind)
            if handler.error is not None:
                continue

            ignored = False
            for cand in candidates:
                if cand.code_name == handler.code_name and cand.reason == handler.reason:
                    ignored = True
                    break
            if ignored:
                continue
            feedback = self.evaluation_feedback_prompt(handler.eval_result, None)
            other_solutions_prompt += f"## {handler.code_name}\n{handler.reason}\n{feedback}\n"

        if len(other_solutions_prompt) > 0:
            other_solutions_prompt = f"The Eliminated solutions are:\n{other_solutions_prompt}\nThe new solution should avoid the mistakes made by the eliminated solutions and explore new ideas.\n"
            
        final_prompt = f"""{task_prompt}

{pre_solution_prompt}

{other_solutions_prompt}

{response_format_prompt}
"""
        return role_prompt, final_prompt

    def __get_candidate_prompt(self, candidate:TunerResponseHandler) -> str:
        solution = f"```python\n{candidate.code}\n```"
        if candidate.error:
            if candidate.error_type == "NoCodeException":
                feedback = "No code was extracted. The code should be encapsulated with ``` in your response."
            else:
                feedback = f"An error occurred : {candidate.error}"
        else:
            feedback = self.evaluation_feedback_prompt(candidate.eval_result, None)

        desc = candidate.reason
        
        return f"{solution}\n{feedback}\n{desc}\n"

    def task_description(self, task:GenerationTask, extra:str="") -> str:
        lib_prompt = "As an expert of numpy, scipy, scikit-learn, torch, gpytorch and botorch, you are allowed to use these libraries."
        problem_desc = "24 noiseless functions"
        task_prompt = f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {problem_desc} in a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

Your task is to Improve the given algorithm's performance on BBOB test suite and maintain or lower its computational cost
1 Analyze the provided Bayesian Optimization algorithm and its feedback.
2 What additional information would you like to have to improve the algorithm? 
- Only propose the information that can be easily expressed in text as the prompt.
3 Identify the potential improvements 
- modify the existing components
- or apply new components 
- The structure of the code should be kept as much as possible. Be cautious about the big changes.
4 Justify your changes.
5 Describe the algorithm on one line.

The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.The func() can only be called as many times as the budget allows, not more. 

{lib_prompt} Do not use any other libraries unless they cannot be replaced by the above libraries. Name the class based on the characteristics of the algorithm with a template '<characteristics>BOv<version>'.
"""
        return task_prompt
    
    def response_format(self, task:GenerationTask, extra:str="") -> str:
        output_format_prompt = """
Give the response in the format:
## Justifications 
<Analysis of the algorithm and the feedback>
<Additional information you would like to have>
<Justifications for the changes made>
<Description of the algorithm>
## Code
<code>
"""
        return output_format_prompt

    def evaluation_feedback_prompt(self, eval_res:EvaluatorResult, other_results:tuple[EvaluatorResult,list[EvaluatorResult]]= None) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""

        def _mean_ground_content(key:str, ground:int, ground_dic) -> float:
            _ground = ground_dic[ground]
            _contents = [content[key] for content in _ground]
            return np.mean(_contents) if len(_contents) > 0 else 0

        algorithm_name = eval_res.name
        aocs = []
        exploitations = []
        grouped_aocs = []
        grouped_aocs = []
        for i in range(5):
            grouped_aocs.append([])
        for res in eval_res.result:
            aoc = res.log_y_aoc
            aocs.append(aoc)

            _exploitations = res.search_result.k_distance_exploitation_list
            _exploitations = [x for x in _exploitations if x is not None] 
            _mean_exploitation = np.nanmean(_exploitations)
            exploitations.append(_mean_exploitation)

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
                "y_aoc": aoc,
                'exploitation': _mean_exploitation,
            }
            grouped_aocs[group_idx].append(content)

        auc_mean = np.mean(aocs)
        auc_std = np.std(aocs)

        exploitations_mean = np.mean(exploitations)
        exploitations_std = np.std(exploitations)

        separated_auc = _mean_ground_content("y_aoc", 0, grouped_aocs)
        low_mod_auc = _mean_ground_content("y_aoc", 1, grouped_aocs)
        high_uni_auc = _mean_ground_content("y_aoc", 2, grouped_aocs)
        multi_adequate_auc = _mean_ground_content("y_aoc", 3, grouped_aocs)
        multi_weak_auc = _mean_ground_content("y_aoc", 4, grouped_aocs)

        separated_exploitation = _mean_ground_content("exploitation", 0, grouped_aocs)
        low_mod_exploitation = _mean_ground_content("exploitation", 1, grouped_aocs)
        high_uni_exploitation = _mean_ground_content("exploitation", 2, grouped_aocs)
        multi_adequate_exploitation = _mean_ground_content("exploitation", 3, grouped_aocs)
        multi_weak_exploitation = _mean_ground_content("exploitation", 4, grouped_aocs)


        detailed_feedback = f"""
on Separable functions {separated_auc:.04f} of AOC, {separated_exploitation:.04f} of exploitation
on functions with low or moderate conditioning {low_mod_auc:.04f} of AOC, {low_mod_exploitation:.04f} of exploitation
on functions with high conditioning and unimodal {high_uni_auc:.04f} of AOC, {high_uni_exploitation:.04f} of exploitation
on Multi-modal functions with adequate global structure {multi_adequate_auc:.04f} of AOC, {multi_adequate_exploitation:.04f} of exploitation
on Multi-modal functions with weak global structure {multi_weak_auc:.04f} of AOC, {multi_weak_exploitation:.04f} of exploitation
"""
        exploitation_prompt = f"""The average exploitation score (1.0 mean most exploitative, 0.0 mean most explorative) is {exploitations_mean:0.4f} with standard deviation {exploitations_std:0.4f}."""

        final_feedback_prompt = f"""The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard deviation {auc_std:0.4f}\n{exploitation_prompt}\n{detailed_feedback}
"""
        return final_feedback_prompt


# Helper functions
    def get_response_handler(self):
        return TunerResponseHandler()
