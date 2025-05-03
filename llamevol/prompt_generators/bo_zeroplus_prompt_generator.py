
import random
import re
import numpy as np
from llamevol.evaluator import EvaluatorResult
from .abstract_prompt_generator import PromptGenerator, GenerationTask, ResponseImpReturnChecker, ResponseHandler


class BOPromptGeneratorReturnChecker(ResponseImpReturnChecker):
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

class BoZeroPlusResponseHandler(ResponseHandler):
    def __init__(self):
        super().__init__()

        self.problem_analysis = ""
        self.feedback_analysis = ""
        self.potential_techniques= ""
        self.improvement = ""
        self.proposed_strategies = ""
        self.algorithm_design = ""
        self.pseudocode = ""

        self.error_analysis = ""
        self.proposed_solutions = ""
        # Feedback for error_analysis and proposed_solutions
        self.error_feedback = ""


    def __to_json__(self):
        d = super().__to_json__()
        d["problem_analysis"] = self.problem_analysis
        d["feedback_analysis"] = self.feedback_analysis
        d["potential_techniques"] = self.potential_techniques
        d["improvement"] = self.improvement
        d["proposed_strategies"] = self.proposed_strategies
        d["algorithm_design"] = self.algorithm_design
        d["pseudocode"] = self.pseudocode
        d["error_analysis"] = self.error_analysis
        d["proposed_solutions"] = self.proposed_solutions
        d["error_feedback"] = self.error_feedback
        return d

    def extract_response(self, response:str, task:GenerationTask):
        if not response:
            return

        self.raw_response = response
        
        if task == GenerationTask.INITIALIZE_SOLUTION:
            self.__extract_for_initial_solution(response)
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            self.__extract_for_error_fixing(response)
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            self.__extract_for_improvement(response)

    def __extract_for_improvement(self, response:str):
        self.raw_response = response
        sections = ["Problem Analysis", 
                    "Feedback Analysis", 
                    "Potential Techniques", 
                    "Improvements",
                    "Proposed Strategies",
                    "Proposed Strategies", 
                    "Final Algorithm Design", 
                    "Pseudocode", 
                    "Code"]
        for section in sections:
            if section == "Code":
                self.code, _ = self.extract_from_response(response, section)
                self.code_name, _ = self.extract_from_response(response, "class_name")
            elif section == "Final Algorithm Design":
                self.algorithm_design, _ = self.extract_from_response(response, section)
            elif section == "Pseudocode":
                self.pseudocode, _ = self.extract_from_response(response, section)
            elif section == "Proposed Strategies":
                self.proposed_strategies, _ = self.extract_from_response(response, section)
            elif section == "Potential Techniques":
                self.potential_techniques, _ = self.extract_from_response(response, section)
            elif section == "Feedback Analysis":
                self.feedback_analysis, _ = self.extract_from_response(response, section)
            elif section == "Problem Analysis":
                self.problem_analysis, _ = self.extract_from_response(response, section)
            elif section == "Improvements":
                self.improvement, _ = self.extract_from_response(response, section)

    def __extract_for_error_fixing(self, response:str):
        self.raw_response = response
        sections = ["Identified Errors", "Proposed Solutions", "Code", "Previous Analysis Feedback"]
        for section in sections:
            if section == "Identified Errors":
                self.error_analysis, _ = self.extract_from_response(response, section)
            elif section == "Proposed Solutions":
                self.proposed_solutions, _ = self.extract_from_response(response, section)
            elif section == "Code":
                self.code, _ = self.extract_from_response(response, section)
                self.code_name, _ = self.extract_from_response(response, "class_name")
            elif section == "Previous Analysis Feedback":
                self.error_feedback, _ = self.extract_from_response(response, section)

    def __extract_for_initial_solution(self, response:str):
        self.raw_response = response
        sections = ["Problem Analysis", "Potential Techniques", "Proposed Strategies", "Final Algorithm Design", "Pseudocode", "Code"]
        for section in sections:
            if section == "Code":
                self.code, _ = self.extract_from_response(response, section)
                self.code_name, _ = self.extract_from_response(response, "class_name")
            elif section == "Final Algorithm Design":
                self.algorithm_design, _ = self.extract_from_response(response, section)
            elif section == "Pseudocode":
                self.pseudocode, _ = self.extract_from_response(response, section)
            elif section == "Proposed Strategies":
                self.proposed_strategies, _ = self.extract_from_response(response, section)
            elif section == "Potential Techniques":
                self.potential_techniques, _ = self.extract_from_response(response, section)
            elif section == "Problem Analysis":
                self.problem_analysis, _ = self.extract_from_response(response, section)

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

class ZeroPlusBOPromptSharedboard():
    def __init__(self):
        self.problem_analysis = []
        self.tech_base = []

    def last_problem_analysis(self) -> str:
        if self.problem_analysis:
            last = self.problem_analysis[-1]
            if len(last) > 0:
                return last
        return None

    def last_tech_base(self) -> str:
        if self.tech_base:
            last = self.tech_base[-1]
            if len(last) > 0:
                return last
        return None

class BoZeroPlusPromptGenerator(PromptGenerator):
    def __init__(self):
        super().__init__()
        self.aggressiveness = random.uniform(0.3, 1.0)
        self.use_botorch = False

        self.evolved_techs = []
        self.evolved_problem_analysis = []

# method list

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

# prompt generation

    def get_prompt(self, task:GenerationTask, problem_desc:str, 
                   candidates:list[BoZeroPlusResponseHandler]= None,
                   population=None,
                   options=None,
                   ) -> tuple[str, str]:
        if task == GenerationTask.INITIALIZE_SOLUTION:
            final_prompt = ""

            task_prompt = self.task_description(task)
            final_prompt += f"{task_prompt}\n"

            task_instruction_prompt = self.task_instruction(task)
            final_prompt += f"{task_instruction_prompt}\n"

            problem_prompt = f"""### Problem Description\n{problem_desc}"""
            final_prompt += f"{problem_prompt}\n"

            code_structure_prompt = self.code_structure()
            final_prompt += f"{code_structure_prompt}\n"

            response_format_prompt = self.response_format(task=task)
            final_prompt += f"{response_format_prompt}\n"
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            if candidates is None or len(candidates) == 0:
                return "", ""

            candidate = candidates[0]

            final_prompt = ""

            task_prompt = self.task_description(task)
            final_prompt += f"{task_prompt}\n"

            task_instruction_prompt = self.task_instruction(task)
            final_prompt += f"{task_instruction_prompt}\n"

            # if task == GenerationTask.FIX_ERRORS_FROM_ERROR and candidate.error_analysis and candidate.proposed_solutions:
            #     final_prompt = f"""### Previous Error Analysis\n{candidate.error_analysis}\n### Previous Proposed Solutions\n{candidate.proposed_solutions}\n"""
            final_prompt += f"### Errors\n```bash\n{candidate.eval_result.error}\n```\n"
            final_prompt += f"### Solution\n```python\n{candidate.code}\n```\n"

            response_format_prompt = self.response_format(task)
            final_prompt += f"{response_format_prompt}\n"
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            if candidates is None or len(candidates) == 0:
                return "", ""

            candidate = candidates[0]

            final_prompt = ""

            task_prompt = self.task_description(task)
            final_prompt += f"{task_prompt}\n"

            task_instruction_prompt = self.task_instruction(task)
            final_prompt += f"{task_instruction_prompt}\n"

            problem_prompt = f"""### Problem Description\n{problem_desc}"""
            final_prompt += f"{problem_prompt}\n"

            feedback_prompt = self.evaluation_feedback_prompt(candidate.eval_result)
            final_prompt += f"{feedback_prompt}\n"

            previous_problem_analysis = candidate.problem_analysis
            previous_proposed_techniques = candidate.proposed_strategies
            if len(previous_problem_analysis) > 0:
                final_prompt += f"""### Problem Analysis\n{previous_problem_analysis}\n"""
            if len(previous_problem_analysis) > 0:
                final_prompt += f"""### Potential Techniques\n{previous_proposed_techniques}\n"""

            final_prompt += f"### Solution\n```python\n{candidate.code}\n```\n"
            response_format_prompt = self.response_format(task)
            final_prompt += f"{response_format_prompt}\n"

        return "", final_prompt

    def task_description(self, task:GenerationTask) -> str:
        desc = """## Task Description\n"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            desc += "You will be given minimization optimization problems. Your tasks are to analyze the problem, design a feasible Bayesian Optimization algorithm, and implement it."
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            desc += "You will be given a Bayesian Optimization solution with errors. Your task is to identify and correct the errors in the provided solution."
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            desc += "You will be given a Bayesian Optimization solution with evaluation feedback, problem analysis, and other information. Your task is to optimize the performance of the solution."
        return desc

    def task_instruction(self, task:GenerationTask) -> str:
        desc = """## Task Instruction\n"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            desc += "You need to act as a mathematician, computer scientist, and programmer independently.\n"
            desc += self.task_instruction_for_mathematician(task)
            desc += self.task_instruction_for_scientist(task, self.aggressiveness)
            desc += self.task_instruction_for_programmer(task, self.use_botorch)
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            desc += "You need to act as computer scientist and programmer independently.\n"
            desc += self.task_instruction_for_scientist(task)
            desc += self.task_instruction_for_programmer(task, self.use_botorch)
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            desc += "You need to act as a mathematician, computer scientist, and programmer independently.\n"
            desc += self.task_instruction_for_mathematician(task)
            desc += self.task_instruction_for_scientist(task, self.aggressiveness)
            desc += self.task_instruction_for_programmer(task, self.use_botorch)
        return desc

    def task_instruction_for_mathematician(self, task:GenerationTask) -> str:
        instruction = """\n**as a mathematician specialized in optimization**
"""
        if task == GenerationTask.INITIALIZE_SOLUTION: 
            instruction += """- Identify the key characteristics of the problems relevant to optimization, not limited to its multi-modality, separability, and the location of its global minimum.
- Analyze the problem, focusing on the challenges posed by the problems for optimization algorithms. Consider aspects should be included but not limited to local optima, ruggedness, and the search space dimensionality.
"""
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            instruction += ""
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            instruction += """- Review the provided problem analysis on correctness and comprehensiveness.
- Propose your problem analysis. Keep it consice, clear and to the point.
"""
        return instruction

    def task_instruction_for_programmer(self, task:GenerationTask, use_botorch:bool=False) -> str:
        instruction = """\n**as a programmer specialized in python.**\n"""
        lib_instruction = "- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries."
        if use_botorch:
            instruction = "\n**as a programmer specialized in python.**\n"
            lib_instruction = "- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, Botorch, you are allowed to use these libraries."
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
            instruction += f"""1. Identify the cause of the provided errors.
2. Review the code for potential errors related to the implementation. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
4. Decide the errors which need to be fixed. justisfy your choice.
- The provided errors should be on the top of the list.
5. Correct the errors. 
{doc_string_instruction}
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

    def task_instruction_for_scientist(self, task:GenerationTask, aggressiveness:float = None) -> str:
        instruction = """\n**as a computer scientist specialized in bayesian optimization**\n"""
        if task == GenerationTask.INITIALIZE_SOLUTION:
            # ss_candidates = ", ".join(self.initialization_strategies())
            # sm_candidates = ", ".join(self.surrogate_models())
            # af_candidates = ", ".join(self.acquisition_functions())
            # bs_candidates = ", ".join(self.other_techniques())
            if aggressiveness is None:
                aggressiveness = random.uniform(0.3, 1.0)
            else:
                aggressiveness = max(0.1, min(1.0, aggressiveness))
            instruction += f"""1. Based on the problem analysis, take a brainstorming session to identify the potential techniques in Bayesian Optimization that could address the challenges of the problem. The techniques could be popularly used, state-of-the-art, or innovative but less promising. Make all techniques as diverse as possible. The techniques should include but not limited to:
- Sampling Strategies
- Surrogate Models and their corresponding metrics: the options beyond Gaussian Process are encouraged.
- Acquisition Functions
- Initailization Strategies: Choose a strategy to balance the number of initial points and the number of optimization iterations based on the provided budget.
- Other Possible Techniques: Embrace the creativity and imagination.
2. Consider the options from step 1 and propose at least **three** algorithms. Here, you should just focus on the **diversity** and **performance** of the algorithms.
3. Review your options from step 2 and design a specific Bayesian Optimization algorithm based on AGGRESSIVENESS (0.0-1.0):{aggressiveness:.2f}. Justify your choices in detail. 
- You can combine from less complex and more widely applicable techniques(low aggressiveness), or more advanced and specialized techniques(high aggressiveness) tailored to the specific challenges of the problem. 
- Be aware: AGGRESSIVENESS only affects the choice of techniques, not the implementation as a parameter.
4. Pseudocode: Write down the key steps of your chosen algorithm in plain and consise pseudocode, highlighting any novel components or adaptations.
"""
        elif task == GenerationTask.FIX_ERRORS or task == GenerationTask.FIX_ERRORS_FROM_ERROR:
            instruction += """1. Identify the cause of the provided errors.
2. Review the code for potential errors related to algorithm design. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention. 
4. Decide the errors which need to be fixed. justisfy your choice.
"""
        elif task == GenerationTask.OPTIMIZE_PERFORMANCE:
            instruction += """1. Analyze the feedback.
- What does the feedback tell you about the algorithm's performance? Compare with the baseline.
- What are the key areas for improvement?
2. Review the previous proposed techniques, take a brainstorming session about the correctness and comprehensiveness. The techniques could be popularly used, state-of-the-art, or innovative but less promising. Make all techniques as diverse as possible. 
- Correct them if you find any errors,
- Propose new ones if you find any missing. 
- Update the proposed strategies. 
3. Based on problem analysis, feedback analysis, potential techniques and the provided solution, identify the potential improvements and propose at least **three** algorithms. Here, you focus on the **diversity** and **performance** of the algorithms.
- Instead of choosing different techniques, you could modify the existing techniques by adjusting hyperparameters
4. Considering the potential improvements and the corresponding workload required to implement them, decide the final algorithm design and provide a explanation. 
6. Pseudocode: Write down the key changes of your chosen strategy in plain and concise pseudocode. 
"""
        return instruction

    def __get_result_feedback(self, eval_result:EvaluatorResult, name=None) -> str:
        if eval_result is None:
            return ""

        feedback_prompt = f"#### {eval_result.name if name is None else name}\n"

        for result in eval_result.result:
            if result.name is not None:
                feedback_prompt += f"##### {result.name}\n"

            feedback_prompt += f"- best y: {result.best_y:.2f}\n"
            if result.n_initial_points > 0:
                if result.y_best_tuple is not None:
                    feedback_prompt += f"- initial best y: {result.y_best_tuple[0]:.2f}\n"
                    feedback_prompt += f"- non-initial best y: {result.y_best_tuple[1]:.2f}\n"
                feedback_prompt += f"- AOC for non-initial y: {result.non_init_y_aoc:.2f}\n"
            else:
                feedback_prompt += f"- AOC for all y: {result.y_aoc:.2f}\n"
                if result.x_mean is not None and result.x_std is not None:
                    feedback_prompt += f"- mean and std of all x: {np.array_str(result.x_mean, precision=2)} , {np.array_str(result.x_std, precision=2)}\n"
                    feedback_prompt += f"- mean and std of all y: {result.y_mean:.2f} , {result.y_std:.2f}\n"

        # feedback_prompt += f"Execution Time: {result.execution_time:.4f}\n"

        return feedback_prompt

    def evaluation_feedback_prompt(self, eval_res:EvaluatorResult, options=None) -> str:
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

        last_feedback = None if options is None else options.get("last_feedback", None)
        res_name = None
        last_res_name = None
        if last_feedback is not None:
            res_name = f"{eval_res.name}(After Optimization)" if last_feedback is not None else None
            last_res_name = f"{last_feedback.name}(Before Optimization)" if last_feedback is not None else None
        final_feedback_prompt += self.__get_result_feedback(eval_res, res_name)
        final_feedback_prompt += self.__get_result_feedback(last_feedback, last_res_name)

        other_res = None if options is None else options.get("other_res", None)
        if other_res is not None:
            for other in other_res:
                final_feedback_prompt += self.__get_result_feedback(other, f"{other.name}(Baseline)")

        bounds_prompt = f"bounded by {np.array_str(eval_res.result[0].bounds, precision=2)}" if eval_res.result[0].bounds is not None else ""
        # bounds_prompt = ""
        final_feedback_prompt += f"""#### Note:
- AOC(Area Over the Convergence Curve): a measure of the convergence speed of the algorithm, ranged between 0.0 and 1.0. A higher value is better.
- non-initial x: the x that are sampled during the optimization process, excluding the initial points.
- Budget: The maximum number(during the whole process) of the sample points which evaluated by objective_fn.
- mean and std of x: indicate exploration and exploitation in search space {bounds_prompt}.
- mean and std of y: indicate the search efficiency. 
"""
        return final_feedback_prompt

    def code_structure(self) -> str:
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
```
"""

    def response_format(self, task:GenerationTask, extra:str="") -> str:
        if task == GenerationTask.INITIALIZE_SOLUTION:
            return f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Problem Analysis
<Mathematical Analysis>
### /Problem Analysis

### Potential Techniques
### /Potential Techniques

### Proposed Strategies
<Proposed Strategies>
### /Proposed Strategies

### Final Algorithm Design
<Algorithm Design>
### /Final Algorithm Design

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
            # previous_feedback = "\n### Previous Analysis Feedback\n- feedback analysis\n- contributions to errors\n### /Previous Analysis Feedback\n" if task == GenerationTask.FIX_ERRORS_FROM_ERROR else ""
            previous_feedback = ""
            return f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.){previous_feedback}
### Identified Errors
#### Algorithm design errors
    - <error>: cause, impact, original intention, solution, confidence level of the correct identification(0-10), should be fixed or not, reason of the choice
#### Implementation errors
    - <error>: cause, impact, original intention, solution, confidence level of the correct identification(0-10), should be fixed or not, reason of the choice
### /Identified Errors

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

### Problem Analysis
- only new problem analysis. No comment about the previous one.
### /Problem Analysis

### Feedback Analysis
### /Feedback Analysis

### Potential Techniques
### /Potential Techniques

### Improvements
### /Improvements

### Proposed Strategies
### /Proposed Strategies

### Final Algorithm Design
### /Final Algorithm Design

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

    def get_response_handler(self):
        return BoZeroPlusResponseHandler()

    def get_return_checker(self) -> BOPromptGeneratorReturnChecker:
        return BOPromptGeneratorReturnChecker()

    # def get_prompt_sharedbroad(self) -> ZeroPlusBOPromptSharedboard:
    #     return ZeroPlusBOPromptSharedboard()
