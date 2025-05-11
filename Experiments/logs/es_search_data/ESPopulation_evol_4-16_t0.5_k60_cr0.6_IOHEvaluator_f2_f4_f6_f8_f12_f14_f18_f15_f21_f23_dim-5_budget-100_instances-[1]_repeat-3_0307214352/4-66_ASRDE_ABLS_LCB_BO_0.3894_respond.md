# Description
**ASRDE-ABLS-LCB-BO: Adaptive Success-Rate DE with Adaptive Batch size, Local Search and LCB Bayesian Optimization.** This algorithm combines the strengths of ASRDE_PLS_BO_v2 and DE_ABLS_LCB_BO. It uses an adaptive success-rate based Differential Evolution (DE) strategy with adaptive F and CR parameters, similar to ASRDE_PLS_BO_v2. It also incorporates an adaptive batch size based on the GP's uncertainty, as in DE_ABLS_LCB_BO, and performs local search to refine the best solutions. The acquisition function is based on the Lower Confidence Bound (LCB) to promote exploration. Additionally, the population size of DE is adapted based on the success rate of DE iterations. The local search is enhanced by considering both the GP's uncertainty (sigma) and the LCB value.

# Justification
This algorithm aims to improve upon the previous solutions by combining their key strengths:

*   **Adaptive Success-Rate DE (ASRDE):** The adaptive F and CR parameters in DE, based on the success rate, allow for better exploration and exploitation balance.
*   **Adaptive Batch Size (ABLS):** Adjusting the batch size based on the GP's uncertainty helps to efficiently allocate function evaluations, focusing on promising regions.
*   **Lower Confidence Bound (LCB):** The LCB acquisition function promotes exploration by favoring regions with high uncertainty, which is crucial for avoiding premature convergence.
*   **Probabilistic Local Search (PLS):** Local search refines the best solution found by DE, improving the overall performance. The probability of performing local search is proportional to uncertainty * LCB value.
*   **Adaptive Population Size:** Dynamically adjusting the population size based on the success rate of the DE iterations allows the algorithm to adapt to the landscape of the optimization problem.

The combination of these techniques should result in a more robust and efficient optimization algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ASRDE_ABLS_LCB_BO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * dim
        self.pop_size = 5 * dim  # Initial population size
        self.max_pop_size = 10 * dim
        self.min_pop_size = 2 * dim
        self.F = 0.8
        self.CR = 0.7
        self.success_rate_F = 0.5  # Initialize success rates for F and CR
        self.success_rate_CR = 0.5
        self.learning_rate = 0.1
        self.F_step = 0.05
        self.CR_step = 0.05
        self.pop_size_step = 0.1
        self.exploration_weight = 2.0
        self.de_iters = 20

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

        # Lower Confidence Bound acquisition function
        lcb = mu - self.exploration_weight * sigma
        return lcb, sigma

    def _select_next_points(self, batch_size):
        population = self._sample_points(self.pop_size)
        n_success_F = 0
        n_success_CR = 0

        for _ in range(self.de_iters):
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

                if ei_trial[0, 0] < ei_current[0, 0]:
                    population[i] = x_trial
                    n_success_F += 1
                    n_success_CR += 1

        # Adaptive F and CR based on success rate
        self.success_rate_F = 0.9 * self.success_rate_F + 0.1 * (n_success_F / self.pop_size)
        self.success_rate_CR = 0.9 * self.success_rate_CR + 0.1 * (n_success_CR / self.pop_size)

        # Dynamically adjust step sizes
        self.F_step = 0.05 * (1 - abs(self.success_rate_F - 0.5) * 2)  # Smaller steps when success rate is far from 0.5
        self.CR_step = 0.05 * (1 - abs(self.success_rate_CR - 0.5) * 2)

        self.F = np.clip(self.F + self.learning_rate * (self.success_rate_F - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate_CR - 0.5), 0.1, 0.9)

        # Adaptive population size
        pop_size_change = self.pop_size_step * (0.5 - (n_success_F + n_success_CR) / (2 * self.pop_size))
        self.pop_size += int(pop_size_change * self.pop_size)
        self.pop_size = np.clip(self.pop_size, self.min_pop_size, self.max_pop_size)

        lcb_values, sigma = self._acquisition_function(population)
        best_idx = np.argmin(lcb_values)
        best_x = population[best_idx]
        best_sigma = sigma[best_idx]
        best_lcb = lcb_values[best_idx]

        # Probabilistic Local Search
        if np.random.rand() < np.clip(best_sigma * abs(best_lcb), 0.0, 1.0): # Probability proportional to uncertainty * |LCB|
            def obj_func(x):
                lcb, _ = self._acquisition_function(x.reshape(1, -1))
                return lcb[0,0]

            res = minimize(obj_func, best_x, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
            next_point = res.x.reshape(1, -1)
        else:
            next_point = best_x.reshape(1, -1)

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
 The algorithm ASRDE_ABLS_LCB_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1622 with standard deviation 0.0946.

took 600.67 seconds to run.