
from typing import Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import re, random
import numpy as np
from .individual import Individual, Population

class GenerationTask(Enum):
    """Enum class for generation tasks."""
    INITIALIZE_SOLUTION = 0
    FIX_ERRORS = 1
    OPTIMIZE_PERFORMANCE = 2

class PromptGenerator(ABC):
    """Abstract base class for prompt generators."""

    @abstractmethod
    def problem_description(self, extra:str="") -> str:
        pass

    @abstractmethod
    def task_description(self, task:GenerationTask, extra:str="") -> str:
        pass

    @abstractmethod
    def task_instruction(self, task:GenerationTask, extra:str="") -> str:
        """explicit COT of the task accomplishment"""
        pass

    @abstractmethod
    def code_structure(self, extra:str="") -> str:
        pass

    def get_return_checker(self) -> callable:
        return None

    @abstractmethod
    def response_format(self, task:GenerationTask, extra:str="") -> str:
        pass

    @abstractmethod
    def extract_from_response(self, response: str, section: str, pattern=None) -> Tuple[str, str]:
        pass

class BOPromptGenerator(PromptGenerator):

    def __init__(self):
        super().__init__()
        self.aggressiveness = random.uniform(0.3, 1.0)
        self.use_botorch = False

    def surrogate_models(self) -> list[str]:
        surrogate_models = [
            # Gaussian Process Based Models
            "Standard Gaussian Process (GP)",
            "Variational Sparse GP",
            "Fully Independent Training Conditional (FITC) GP",

            "Deep Gaussian Processes (Deep GPs)",
            "Heteroscedastic Gaussian Processes",

            # Tree-Based Models
            "Random Forest (RF)",
            "Tree Parzen Estimator (TPE)",

            # Neural Network Based Models
            "Bayesian Neural Networks (BNNs)",
            "Neural Networks with Attention Mechanisms",

            # Regression-Based Models
            "Support Vector Regression (SVR)",
            "Kernelized Ridge Regression",
            "Polynomial Regression",
            "Local Polynomial Regression",
            "Radial Basis Function (RBF) Networks",

            # Other Models
            "Polynomial Chaos Expansions (PCE)",
            "Gaussian Mixture Models (GMMs)",
            "B-splines",

            # Ensemble and Hybrid Approaches
            "Ensemble of Models",
            "Hybrid Models Combining Different Surrogate Models",
        ]
        return surrogate_models

    def acquisition_functions(self) -> list[str]:
        acquisition_functions = [
            "Expected Improvement (EI)",
            "Probability of Improvement (PI)",
            "Upper Confidence Bound (UCB)",
            "Thompson Sampling",
            "Entropy Search",
            "Predictive Entropy Search",
            "Information Gain",
            "Integrated Variance Reduction",
            "Max-value Entropy Search",
            "Knowledge Gradient",
            "Probability of Improvement with Gaussian Process Upper Confidence Bound (GP-UCB)",
            "Expected Improvement with Gaussian Process Upper Confidence Bound (GP-EI)",
            "Upper Confidence Bound with Gaussian Process Expected Improvement (GP-UCB-EI)",
            "Thompson Sampling with Gaussian Process Expected Improvement (GP-TS-EI)",
            "Entropy Search with Gaussian Process Expected Improvement (GP-ES-EI)",
            "Predictive Entropy Search with Gaussian Process Expected Improvement (GP-PES-EI)",
            "Information Gain with Gaussian Process Expected Improvement (GP-IG-EI)",
            "Integrated Variance Reduction with Gaussian Process Expected Improvement (GP-IVR-EI)",
            "Max-value Entropy Search with Gaussian Process Expected Improvement (GP-MES-EI)",
            "Knowledge Gradient with Gaussian Process Expected Improvement (GP-KG-EI)",
            "Trust-region-based acquisition function",
        ]
        return acquisition_functions
    
    def initialization_strategies(self) -> list[str]:
        initialization_strategies = [
            "Uniform Sampling",
            "Latin Hypercube Sampling (LHS)",

            "Sobol Sequence",
            "Halton Sequence",
            "Faure Sequences",
            "Low-discrepancy Sampling",

            "Orthogonal Array Sampling",
            "Maximin Distance Sampling",
            "Orthogonal Latin Hypercube Sampling",
            "Quasi-Monte Carlo Sampling",

            "Domain expertise-based initialization",
        ]
        return initialization_strategies

    def other_techniques(self) -> list[str]:
        other_techniques = [
        ]
        return other_techniques

    def prompt_extract_keywords_from_code(self, code:str) -> str:
        return f"""Extract and list up to 6 key technical components from the provided Python code implementing a Bayesian optimization algorithm. 
- Focus on the core techniques and mathematical concepts used. 
- Exclude the general terms like 'BayesianOptimization', 'AcquisitionFunction', 'Minimization', etc.
- Return keywords only, separated by commas.

Code:
```python
{code}
```
    """

    def problem_description(self, extra:str="") -> str:
        return f"""## Problem Description\n{extra}"""

    def task_description(self, task:GenerationTask, extra:str="") -> str:
        desc = """## Task Description\n"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            desc += "You will be given minimization optimization problems. Your tasks are to analyze the problem, design a feasible algorithm, and implement it using Bayesian Optimization."
        elif task == GenerationTask.FIX_ERRORS:
            desc += "You will be given a Bayesian Optimization solution with errors. Your task is to identify and correct the errors in the provided solution."
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            desc += "You will be given a Bayesian Optimization solution with evaluation feedback, problem analysis, and other information. Your task is to optimize the performance of the solution."
        desc += extra
        return desc

    def task_instruction_for_mathematician(self, task:GenerationTask) -> str:
        instruction = """\n**as a mathematician speciliazed in optimization**
- Identify the key characteristics of the problelms relevant to optimization, such as its multi-modality, separability, and the location of its global minimum.
- Analyze the problem, focusing on the challenges posed by the problems for optimization algorithms. Consider aspects like local optima, ruggedness, and the search space dimensionality.
"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            instruction += ""
        elif task == GenerationTask.FIX_ERRORS:
            instruction += ""
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            instruction += ""
        return instruction

    def task_instruction_for_programmer(self, task:GenerationTask, use_botorch:bool=False) -> str:
        instruction = "\n**as a programmer specialized in python**\n"

        lib_instruction = "You are allowed to use numpy, scipy scikit-learn and Gpy."
        if use_botorch:
            lib_instruction = "You are allowed to use numpy, scipy, scikit-learn, Gpy, torch, gpytorch and botorch."
        
        if task == GenerationTask.INITIALIZE_SOLUTION:
            instruction += f"""- Name the algorithm using a descriptive name that reflects the chosen components, potentially highlighting the novel aspect of the algorithm.
- Implement the algorithm in Python strictly following the provided code structure guide. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the implementation of any novel methods.
- Code Implementation only contain the algorithm class. No usage examples
- {lib_instruction}
- Use other libraries only if they can not be repalced by the above libraries. 
"""
        elif task == GenerationTask.FIX_ERRORS:
            instruction += f"""- Identify and correct any errors in the provided Bayesian Optimization solution. Focus on ensuring that the code is correct, efficient, and follows best practices.
{lib_instruction}
- Use other libraries only if they can not be repalced by the above libraries. 
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Code Implementation only contain the algorithm class. No usage examples
- Do not change the name and the function signatures of __init__ and optimize methods.
"""
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            instruction += f"""- Implement the algorithm in Python strictly following the previous code structure. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the modification.
- Code Implementation only contain the algorithm class. No usage examples
- {lib_instruction}
- Use other libraries only if they can not be repalced by the above libraries. 
"""
        return instruction


    def task_instruction_for_scientist(self, task:GenerationTask, aggressiveness:float = None) -> str:
        instruction = """\n**as a computer scientist specialized in bayesian optimization**\n"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            if aggressiveness is None:
                aggressiveness = random.uniform(0.3, 1.0)
            else:
                aggressiveness = max(0.1, min(1.0, aggressiveness))
            instruction += f"""- Based on the problem analysis, propose and compare at least three conceptually different Bayesian Optimization strategies to achieve the best performance on the given problems.
- You can choose from less complex and more widely applicable approaches(low aggressiveness), or more advanced and specialized techniques(high aggressiveness) tailored to the specific challenges of the problem. Banlance the trade-offs between reward and risk based on AGGRESSIVENESS (0.0-1.0):{aggressiveness:.2f} 
- Review your choices and design a specific Bayesian Optimization algorithm:
- Sampling Strategy: Briefly compare popular strategies. Then, explore and justify the selection of a potentially more advanced or specialized sampling technique relevant to the problems' characteristics, such as a quasi-Monte Carlo method with desirable discrepancy properties or a sequential design strategy tailored for exploration.
- The budget will be provided as a hyperparameter. Choose a strategy to balance n_initial_points and n_iterations. The total number of evaluations should not exceed the budget.
- Surrogate Model: Briefly compare the standard Gaussian Process Regression (GPR) with common kernels. Then, investigate and justify the choice of a potentially more advanced or specialized surrogate model. Explain the potential advantages of this choice over standard GPR.
- Choose a metric to evaluate the model, e.g., negative log-likelihood, or other relevant metrics. Justify your choice.
- Acquisition Function: Briefly compare standard acquisition functions. Then, consider and justify the selection of a potentially more innovative acquisition function designed to handle multi-modality or improve exploration efficiency, such as Thompson Sampling, Information Gain-based approaches, or those incorporating risk or regret considerations. Explain the rationale behind your choice.
- Choose the promising hyperparameters for the acquisition function, surrogate model, and other components.
- Other Possible Techniques: Discuss the potential benefits of incorporating cutting-edge techniques within the Bayesian Optimization framework for this specific problem. Explain how these techniques could address the identified challenges.
- Pseudocode: Write down the detailed steps of your chosen Bayesian Optimization algorithm in plain pseudocode, highlighting any novel components or adaptations.
"""
        elif task == GenerationTask.FIX_ERRORS:
            instruction += """- Identify the cause of the errors in the provided Bayesian Optimization solution. 
- If the errors are related to the algorithm design, propose alternative strategies that could address the identified issues. The proposed strategies should be conceptually similar to the original algorithm but with modifications to fix the errors.
- If the errors are related to the implementation, leave it to the programmer to correct the code. 
"""
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            instruction += """- Review the provided Bayesian Optimization solution, feedback and other relevant information to identify potential problems of previous algorithm design. 
- You can choose from less complex and more widely applicable approaches(low aggressiveness), or more advanced and specialized techniques(high aggressiveness) tailored to the specific challenges of the problem. Banlance the trade-offs between reward and risk based on AGGRESSIVENESS (0.0-1.0):{aggressiveness:.2f} 
- Propose and justify at least three strategies, which should not result in a large number of modifications.
- Choose the most promising strategy and provide a detailed explanation of how it can improve the performance of the algorithm.
- Pseudocode: Write down the detailed steps of your chosen statregy in plain pseudocode, highlighting the changes from the original algorithm.
"""
        return instruction

    def task_instruction(self, task:GenerationTask, extra:str="") -> str:
        desc = """## Task Instruction\n"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            desc += "You need to act as a mathematician, computer scientist, and programmer independently.\n"
            desc += self.task_instruction_for_mathematician(task)
            desc += self.task_instruction_for_scientist(task, self.aggressiveness)
            desc += self.task_instruction_for_programmer(task, self.use_botorch)
        elif task == GenerationTask.FIX_ERRORS:
            # desc += "You need to act as computer scientist and programmer independently.\n"
            # desc += self.task_instruction_for_scientist(task)
            desc += self.task_instruction_for_programmer(task)
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            desc += "You need to act as a computer scientist and programmer independently.\n"
            desc += self.task_instruction_for_scientist(task, self.aggressiveness)
            desc += self.task_instruction_for_programmer(task, self.use_botorch)
        desc += extra
        return desc

    def code_structure(self, extra:str="") -> str:
# {"from botorch.fit import fit_gpytorch_mll //If you are using BoTorch, otherwise remove this line" if self.use_botorch else ""}
        return f"""## Code Structure Guide
```python
from typing import Callable
from scipy.stats import qmc # If you are using QMC sampling. Otherwise or you have a better alternative, remove this line.
import numpy as np
class <AlgorithmName>:
    def __init__(self):
        # Initialize optimizer settings
        # Configure acquisition function
        # Do not add any other arguments without a default value
        pass

    def _sample_points(self, n_points):
        # sample points
        pass
    
    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        pass
    
    def _acquisition_function(self, X):
        # Implement acquisition function 
        # Handle exploration-exploitation trade-off
        pass
    
    def optimize(self, objective_fn:Callable[[np.ndarray], np.ndarray], bounds:np.ndarray, budget:int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, str], int]:
        # Main minimize optimization loop
        # objective_fn: Callable[[np.ndarray], np.ndarray], takes array of shape (n_points, n_dims) and returns array of shape (n_points, 1)
        # bounds has shape (2,<dimemsion>), bounds[0]: lower bound, bounds[1]: upper bound
        # Do not change the function signature
        # Evaluate the model using the metric you choose and record the value as model_loss after each training. the size of the model_loss should be equal to the number of iterations plus one for the fit on initial points.
        # Return a tuple (all_y, all_x, (model_losses, loss_name), n_initial_points)
        self.n_initial_points = <your_strategy>
        self.n_iterations = budget - self.n_initial_points
        pass

    ## You are free to add additional methods as needed and modify the existing ones except for the optimize method and __init__ method.
    ## Rename the class based on the characteristics of the algorithm as '<any_name>BO'
    {extra}
```
"""

    def get_return_checker(self) -> callable:
        class BOPromptGeneratorReturnChecker:
            def __call__(self, func_return: tuple) -> str:
                # check if the return is correct
                if not isinstance(func_return, tuple) or len(func_return) != 4:
                    return "The return value should be a tuple of four elements."
                else:
                    err_str = ""    
                    all_y = func_return[0]
                    if not isinstance(all_y, np.ndarray):
                        err_str += "The first element of the return value should be a numpy array."

                    all_x = func_return[1]
                    if not isinstance(all_x, np.ndarray):
                        err_str += "The second element of the return value should be a numpy array."

                    loss_tuple = func_return[2]
                    if not isinstance(loss_tuple, tuple) or len(loss_tuple) != 2:
                        err_str += "The third element of the return value should be a tuple of two elements."
                    # else: 
                    #     model_losses = loss_tuple[0]
                    #     if not isinstance(model_losses, np.ndarray):
                    #         err_str += "The first element of the third element of the return value should be a numpy array."
                    #     loss_name = loss_tuple[1]
                    #     if not isinstance(loss_name, str):
                    #         err_str += "The second element of the third element of the return value should be a string"
                    
                    return err_str
        return BOPromptGeneratorReturnChecker()
            
    def response_format(self, task:GenerationTask, extra:str="") -> str:
        if task == GenerationTask.INITIALIZE_SOLUTION:
            return f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)
### Description
- Problem Analysis
- Algorithm Design
### /Description
{extra}
### Code
```
<Algorithm Implementation> 
```
### /Code
"""
        elif task == GenerationTask.FIX_ERRORS:
            return f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)
### Description
- Error Analysis: Identify the causes
- Error Correction
### /Description
{extra}
### Code
```
<Corrected Code>
```
### /Code
"""
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            return f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)
### Description
- Potential Problems Analysis
- Proposed Strategies
- Pseudocode
- Main Changes of the implementation
### /Description
{extra}
### Code
```
<Optimized Code>
```
### /Code
"""


    def extract_from_response(self, response: str, section: str, pattern=None) -> Tuple[str, str]:
        error_str = ""
        res = ""
        if pattern is None:
            if section == "class_name":
                # pattern = r"### Code[\s\S]*?class\s+(\w+BO):"
                pattern = r"### Code[\s\S]*?class\s+(\w+):"
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

class LlamboPromptManager:
    def __init__(self):
        pass
    
    def get_improvement_prompt(
        self,
        current_code: str,
        score: float,
        metadata: Dict[str, Any],
        error_info: Dict[str, str] = None
    ) -> str:
        """Generate a prompt for code improvement."""
        base_prompt = ""

        if error_info is None:
            base_prompt = f"""Improve the following Bayesian Optimization code based on the current performance. focus on optimizing for the specific landscape features.:

            Problem Summary:
            
            Current Performance:
            - Score: {score}
            - Metadata: {metadata}
            
            """
        else :
            base_prompt += f"""
            Please fix the following code based on the problem characteristics. 
            Critical Issues:
            - Error Type: {error_info.get('type', 'Unknown')}
            - Message: {error_info.get('message', 'No message')}
            - Traceback: {error_info.get('traceback', 'No traceback')}
            """
        
        base_prompt += f"""
        Current Code:
        {current_code}
        """

        return base_prompt
