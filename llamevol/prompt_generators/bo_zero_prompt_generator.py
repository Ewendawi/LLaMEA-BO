import re
from typing import Any
from .abstract_prompt_generator import PromptGenerator, ResponseHandler, GenerationTask, EvaluatorResult
from .bo_zeroplus_prompt_generator import BOPromptGeneratorReturnChecker


class BoZeroResponseHandler(ResponseHandler):
    def __init__(self):
        super().__init__()

        self.desc = ""
        self.pseudocode = ""
        self.code = ""
        self.code_name = ""
        self.raw_response = ""

    def __to_json__(self):
        return {
            "desc": self.desc,
            "pseudocode": self.pseudocode,
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
        if task == GenerationTask.INITIALIZE_SOLUTION or task == GenerationTask.OPTIMIZE_PERFORMANCE:
            sections.append("Pseudocode")
        for section in sections:
            if section == "Code":
                self.code, _ = self.extract_from_response(response, section)
                self.code_name, _ = self.extract_from_response(response, "class_name")
            elif section == "Description":
                self.desc, _ = self.extract_from_response(response, section)
            elif section == "Pseudocode":
                self.pseudocode, _ = self.extract_from_response(response, section)
        
    def extract_from_response(self, response: str, section: str, pattern=None) -> tuple[str, str]:
        error_str = ""
        res = ""
        if pattern is None:
            if section == "class_name":
                pattern = r"### Code[\s\S]*?class\s+(\w+BO):"
                # pattern = r"### Code[\s\S]*?class\s+(\w+):"
            elif section == "Code":
                pattern = r"### Code[\s\S]*?```\w*\s([\s\S]*?)```[\s\S]*?### /Code"
            else:
                pattern = rf"### {section}\s*([\s\S]*?)\s*### /{section}"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            res = match.group(1)
        else:
            error_str = f"{section} not found in the response."
        return res, error_str

class BoZeroPromptGenerator(PromptGenerator):
    def __init__(self):
        super().__init__()
        self.use_botorch = False

# prompt generation
    def get_prompt(self, task:GenerationTask, problem_desc:str,
                   candidates:list[BoZeroResponseHandler]= None,
                   population= None,
                   options:dict = None
                   ) -> tuple[str, str]:

        if task != GenerationTask.INITIALIZE_SOLUTION:
            if candidates is None or len(candidates) == 0:
                return "", ""
        
        final_prompt = ""

        task_prompt = self.task_description(task)
        final_prompt += f"{task_prompt}\n"

        task_instruction_prompt = self.task_instruction(task)
        final_prompt += f"{task_instruction_prompt}\n"

        if task == GenerationTask.INITIALIZE_SOLUTION:
            problem_prompt = f"""### Problem Description\n{problem_desc}"""
            final_prompt += f"{problem_prompt}\n"

            code_structure_prompt = self.code_structure()
            final_prompt += f"{code_structure_prompt}\n"
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            candidate = candidates[0]
            final_prompt += f"### Errors\n```bash\n{candidate.eval_result.error}\n```\n"
            final_prompt += f"### Solution\n```python\n{candidate.code}\n```\n"
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            candidate = candidates[0]
            problem_prompt = f"""### Problem Description\n{problem_desc}"""
            final_prompt += f"{problem_prompt}\n"
            feedback_prompt = self.evaluation_feedback_prompt(candidate.eval_result)
            final_prompt += f"{feedback_prompt}\n"
            final_prompt += f"### Solution\n```python\n{candidate.code}\n```\n"

        response_format_prompt = self.response_format(task=task)
        final_prompt += f"{response_format_prompt}\n"

        return "", final_prompt

    def task_description(self, task:GenerationTask) -> str:
        desc = """## Task Description\n"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            desc += "You will be given minimization optimization problems. Your tasks are to analyze the problem, design a feasible Bayesian Optimization algorithm, and implement it."
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            desc += "You will be given a Bayesian Optimization solution with errors. Your task is to identify and correct the errors in the provided solution."
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            desc += "You will be given a Bayesian Optimization solution with evaluation feedback. Your task is to optimize the performance of the solution."
        return desc

    def task_instruction(self, task:GenerationTask) -> str:
        desc = """## Task Instruction\n"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            desc += "You need to act as a computer scientist and programmer independently.\n"
            desc += self.task_instruction_for_scientist(task)
            desc += self.task_instruction_for_programmer(task, self.use_botorch)
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            # desc += "You need to act as computer scientist and programmer independently.\n"
            # desc += self.task_instruction_for_scientist(task)
            desc += self.task_instruction_for_programmer(task, self.use_botorch)
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            desc += "You need to act as a computer scientist, and programmer independently.\n"
            desc += self.task_instruction_for_scientist(task)
            desc += self.task_instruction_for_programmer(task, self.use_botorch)
        return desc

    def task_instruction_for_scientist(self, task:GenerationTask) -> str:
        instruction = """\n**as a computer scientist specialized in bayesian optimization**\n"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            instruction += """1. Analyze the minimization optimization problem.
2. Design a Bayesian Optimization algorithm that addresses the challenges of the problem. Justify your choices of techniques and hyperparameters.
3. Pseudocode: Write down the key steps of your chosen Bayesian Optimization algorithm in plain pseudocode, highlighting any novel components or adaptations.
"""
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            instruction += ""
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            instruction += """1. Analyze the minimization optimization problem.
2. Analyze the solution and its evaluation feedback.
3. Optimize the solution to improve its performance.
4. Pseudocode: Write down the key changes of your chosen strategy in plain pseudocode. 
"""
        return instruction

    def task_instruction_for_programmer(self, task:GenerationTask, use_botorch:bool=False) -> str:
        instruction = """\n**as a programmer specialized in python.**\n"""
        lib_instruction = "- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n"
        if use_botorch:
            instruction = "\n**as a programmer specialized in python.**\n"
            lib_instruction = "- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, Botorch, you are allowed to use these libraries.\n"
        lib_instruction += "\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries."
        lib_instruction += "\n- Code Implementation only contain the algorithm class. No usage examples"
        doc_string_instruction = "- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters."

        if task == GenerationTask.INITIALIZE_SOLUTION:
            instruction += f"""- Name the algorithm using a descriptive name that reflects the chosen components, potentially highlighting the novel aspect of the algorithm.
{doc_string_instruction}
- Implement the algorithm in Python strictly following the provided code structure guide. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the implementation of any novel methods.
{lib_instruction}
"""
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            instruction += f"""- Identify the cause of the previous errors.
- Review all the code for potential errors. Here, only make most confident guesses.
- Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
{doc_string_instruction}
- Correct the errors based on the identified causes and proposed solutions
{lib_instruction}
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. 
"""
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            instruction += f"""- Implement the algorithm in Python strictly following the previous code structure. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the modification.
{doc_string_instruction}
{lib_instruction}
"""
        return instruction

    def __get_result_feedback(self, result:EvaluatorResult, name=None) -> str:
        if result is None or len(result.result) == 0:
            return ""
        feedback_prompt = f"#### {result.name if name is None else name}\n"
        for res in result.result:
            if res.name is not None:
                feedback_prompt += f"##### {res.name}\n"
            feedback_prompt += f"- best y: {res.best_y:.2f}\n"
            feedback_prompt += f"- AOC for all y: {res.y_aoc:.2f}\n"

        return feedback_prompt

    def evaluation_feedback_prompt(self, eval_res:EvaluatorResult, options:dict=None) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""
        final_feedback_prompt = "### Feedback\n"
        final_feedback_prompt += f"- Budget: {eval_res.result[0].budget}\n"
        if len(eval_res.result) == 1:
            if eval_res.result[0].optimal_value is not None:
                final_feedback_prompt += f"- Optimal Value: {eval_res.result[0].optimal_value}\n"
        else:
            final_feedback_prompt += "- Optimal Value\n"
            for result in eval_res.result:
                if result.optimal_value is not None:
                    final_feedback_prompt += f"- {result.name}: {result.optimal_value}\n"

        last_feedback = options.get("last_feedback", None)
        res_name = None
        last_res_name = None
        if last_feedback is not None:
            res_name = f"{eval_res.name}(After Optimization)" if last_feedback is not None else None
            last_res_name = f"{last_feedback.name}(Before Optimization)" if last_feedback is not None else None
        final_feedback_prompt += self.__get_result_feedback(eval_res, res_name)
        final_feedback_prompt += self.__get_result_feedback(last_feedback, last_res_name)

        other_res = options.get("other_res", None)
        if other_res is not None:
            for other in other_res:
                final_feedback_prompt += self.__get_result_feedback(other, f"{other.name}(Baseline)")

        final_feedback_prompt += """#### Note:
- AOC(Area Over the Convergence Curve): a measure of the convergence speed of the algorithm, ranged between 0.0 and 1.0. A higher value is better.
- Budget: Maximum number of function evaluations allowed for the algorithm.
"""
        return final_feedback_prompt

    def code_structure(self, extra:str="") -> str:
        # botorch_import = "from botorch.fit import fit_gpytorch_mll //If you are using BoTorch, otherwise remove this line" if self.use_botorch else ""
        prompt_list_tech = """# add the docstring of the class here"""
        return f"""## Code Structure Guide
```python
from typing import Callable
from scipy.stats import qmc # If you are using QMC sampling. Otherwise or you have a better alternative, remove this line.
import numpy as np
class <AlgorithmName>:
    {prompt_list_tech}
    def __init__(self):
        # Initialize optimizer settings
        # Configure acquisition function
        # Do not add any other arguments without a default value

    def _sample_points(self, n_points) -> np.ndarray:
        # sample points
        # return array of shape (n_points, n_dims)
    
    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return  the model

    def _get_model_loss(self, model, X, y) -> np.float64:
        # Calculate the loss of the model
        # return the loss of the model
    
    def _acquisition_function(self, X) -> np.ndarray:
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

    def _select_next_points(self, batch_size) -> np.ndarray:
        # Implement the strategy to select the next points to evaluate
        # return array of shape (batch_size, n_dims)
    
    def optimize(self, objective_fn:Callable[[np.ndarray], np.ndarray], bounds:np.ndarray, budget:int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, str], int]:
        # Main minimize optimization loop
        # objective_fn: Callable[[np.ndarray], np.ndarray], takes array of shape (n_points, n_dims) and returns array of shape (n_points, 1).
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        # Do not change the function signature
        # Evaluate the model using the metric you choose and record the value as model_loss after each training. the size of the model_loss should be equal to the number of iterations plus one for the fit on initial points.
        # Return a tuple (all_y, all_x, (model_losses, loss_name), n_initial_points)
        
        n_initial_points = <your_strategy>
        rest_of_budget = budget - n_initial_points
        while rest_of_budget > 0:
           # Optimization
           
           rest_of_budget -= <the number of points evaluated by objective_fn in this iteration, e.g. x.shape[0] if x is an array>

    ## You are free to add additional methods as needed and modify the existing ones except for the optimize method and __init__ method.
    ## Rename the class based on the characteristics of the algorithm as '<anyName>BO'
    {extra}
```
"""

    def response_format(self, task:GenerationTask, extra:str="") -> str:
        if task == GenerationTask.INITIALIZE_SOLUTION:
            return f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Description
- problem analysis
- the design of the algorithm
### /Description

### Pseudocode
### /Pseudocode

{extra}
### Code
```
<Algorithm Implementation> 
```
### /Code
"""
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            return f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Description
- Identified Errors
- Proposed Solutions
### /Description

{extra}

### Code
```
<Corrected Code>
```
### /Code
"""
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            # comment_prompt = "\n### Comment\n- correctness and comprehensiveness\n### /Comment\n"
            return f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Description
- problem analysis
- feedback analysis
- the design of the algorithm
### /Description

### Pseudocode
### /Pseudocode

{extra}
### Code
```
<Optimized Code>
```
### /Code
"""

# Helper functions
    def get_response_handler(self):
        return BoZeroResponseHandler()

    def get_return_checker(self) -> BOPromptGeneratorReturnChecker:
        return BOPromptGeneratorReturnChecker()
