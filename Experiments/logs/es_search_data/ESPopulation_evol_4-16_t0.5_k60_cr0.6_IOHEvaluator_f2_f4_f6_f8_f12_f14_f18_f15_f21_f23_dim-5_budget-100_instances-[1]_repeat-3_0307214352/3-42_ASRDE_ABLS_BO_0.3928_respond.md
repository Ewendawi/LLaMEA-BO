# Description
**ASRDE-ABLS-BO: Adaptive Success Rate DE with Adaptive Batch Size and Local Search Bayesian Optimization.** This algorithm combines the strengths of DE_ABLS_BO and ASRDE_PLS_BO. It incorporates adaptive control of DE parameters (mutation factor F and crossover rate CR) based on success rate, adaptive batch size based on GP uncertainty, and a local search step using L-BFGS-B to refine the best solutions. It uses a Gaussian Process (GP) surrogate model and the Expected Improvement (EI) acquisition function. The dynamic adjustment of F and CR enhances exploration/exploitation in the acquisition function optimization, the adaptive batch size balances exploration and exploitation based on the GP's uncertainty. The adaptation of F and CR is performed using a moving average. The local search refines the best solution.

# Justification
This algorithm combines the best features of DE_ABLS_BO and ASRDE_PLS_BO to improve performance.

1.  **Adaptive Success Rate for F and CR:** The adaptation of F and CR is crucial for balancing exploration and exploitation. Using a moving average for the success rate provides a more stable adaptation, which can prevent oscillations and lead to better convergence.
2.  **Adaptive Batch Size:** Adjusting the batch size based on the GP's uncertainty allows the algorithm to focus on regions where the model is less confident, promoting exploration when needed and exploitation when the model is more certain.
3.  **Local Search:** The local search step refines the best solutions found by DE, potentially improving the final result.
4.  **Computational Efficiency:** The adaptive batch size helps to control the number of function evaluations, while the probabilistic local search in ASRDE_PLS_BO was replaced by deterministic local search to reduce the computational cost.
5.  **Initial Sampling:** Latin Hypercube Sampling is used for initial sampling to provide better coverage of the search space.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ASRDE_ABLS_BO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * dim
        self.pop_size = 5 * dim
        self.F = 0.8
        self.CR = 0.7
        self.success_rate_F = 0.5  # Initialize success rates for F and CR
        self.success_rate_CR = 0.5
        self.learning_rate = 0.1

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.batch_size = 1

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        return ei, sigma

    def _select_next_points(self, batch_size):
        population = self._sample_points(self.pop_size)
        n_success_F = 0
        n_success_CR = 0
        
        for _ in range(20):
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                x_mutated = x_r1 + self.F * (x_r2 - x_r3)
                x_mutated = np.clip(x_mutated, self.bounds[0], self.bounds[1])

                # Crossover
                x_trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        x_trial[j] = x_mutated[j]

                # Selection
                ei_trial, _ = self._acquisition_function(x_trial.reshape(1, -1))
                ei_current, _ = self._acquisition_function(population[i].reshape(1, -1))
                
                if ei_trial[0, 0] > ei_current[0, 0]:
                    population[i] = x_trial
                    n_success_F += 1
                    n_success_CR += 1

        # Adaptive F and CR based on success rate
        self.success_rate_F = 0.9 * self.success_rate_F + 0.1 * (n_success_F / self.pop_size)
        self.success_rate_CR = 0.9 * self.success_rate_CR + 0.1 * (n_success_CR / self.pop_size)

        self.F = np.clip(self.F + self.learning_rate * (self.success_rate_F - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate_CR - 0.5), 0.1, 0.9)

        ei_values, sigma = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]

        # Local search
        def obj_func(x):
            ei, _ = self._acquisition_function(x.reshape(1, -1))
            return -ei[0,0]
        
        res = minimize(obj_func, best_x, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
        next_point = res.x.reshape(1, -1)

        return next_point

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
            
        if np.min(new_y) < self.best_y:
            self.best_y = np.min(new_y)
            self.best_x = new_X[np.argmin(new_y)]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)

            # Adaptive batch size
            _, std = self.gp.predict(self.X, return_std=True)
            mean_std = np.mean(std)
            self.batch_size = max(1, int(self.dim / (1 + mean_std * 10))) # Adjust batch size based on uncertainty
            self.batch_size = min(self.batch_size, self.budget - self.n_evals) # Ensure not exceeding budget

            next_X = self._select_next_points(self.batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ASRDE_ABLS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1659 with standard deviation 0.0999.

took 1622.24 seconds to run.