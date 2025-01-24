import os
import numpy as np
from ioh import get_problem, logger
from datetime import datetime
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA

# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "codellama:7b" # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b
time_stamp = datetime.now().strftime("%m%d%H%M%S")
experiment_name = f"pop10-50-{time_stamp}"


def evaluateBBOB(solution, explogger = None, details=False):
    auc_mean = 0
    auc_std = 0
    detailed_aucs = [0, 0, 0, 0, 0]
    code = solution.solution
    algorithm_name = solution.name
    exec(code, globals())
    
    error = ""
    
    aucs = []
    detail_aucs = []
    algorithm = None
    for dim in [5]:
        budget = 2000 * dim
        budget = 100
        l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
        for fid in np.arange(1,25):
            for iid in [1,2,3]: #, 4, 5]
                problem = get_problem(fid, iid, dim)
                problem.attach_logger(l2)

                for rep in range(3):
                    np.random.seed(rep)
                    try:
                        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
                        algorithm(problem)
                    except OverBudgetException:
                        pass

                    auc = correct_aoc(problem, l2, budget)
                    aucs.append(auc)
                    detail_aucs.append(auc)
                    l2.reset(problem)
                    problem.reset()
            if fid == 5:
                detailed_aucs[0] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 9:
                detailed_aucs[1] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 14:
                detailed_aucs[2] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 19:
                detailed_aucs[3] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 24:
                detailed_aucs[4] = np.mean(detail_aucs)
                detail_aucs = []

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    i = 0
    while os.path.exists(f"currentexp/aucs-{algorithm_name}-{i}.npy"):
        i+=1
    np.save(f"currentexp/aucs-{algorithm_name}-{i}.npy", aucs)

    feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
    if details:
        feedback = (
            f"{feedback}\nThe mean AOCC score of the algorithm {algorithm_name} on Separable functions was {detailed_aucs[0]:.02f}, "
            f"on functions with low or moderate conditioning {detailed_aucs[1]:.02f}, "
            f"on functions with high conditioning and unimodal {detailed_aucs[2]:.02f}, "
            f"on Multi-modal functions with adequate global structure {detailed_aucs[3]:.02f}, "
            f"and on Multi-modal functions with weak global structure {detailed_aucs[4]:.02f}"
        )

    print(algorithm_name, algorithm, auc_mean, auc_std)
    solution.add_metadata("aucs", aucs)
    solution.set_scores(auc_mean, feedback)

    return solution


task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries. Name the class based on the characteristics of the algorithm with a template '<characteristics>BOv<version>'.
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea. 

A code structure guide is as follows:

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
    
```
"""

for experiment_i in range(1):
    #A 1+1 strategy
    es = LLaMEA(evaluateBBOB, n_parents=1, n_offspring=1, api_key=api_key, task_prompt=task_prompt, experiment_name=experiment_name, model=ai_model, elitism=True, HPO=False, budget=20)
    print(es.run())
