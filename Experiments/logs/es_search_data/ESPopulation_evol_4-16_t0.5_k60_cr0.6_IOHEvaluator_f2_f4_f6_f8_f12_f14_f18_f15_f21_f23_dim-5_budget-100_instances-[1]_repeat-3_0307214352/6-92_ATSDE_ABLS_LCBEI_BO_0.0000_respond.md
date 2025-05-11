# Description
**ATSDE_ABLS_LCBEI_BO: Adaptive Temperature and Success-Rate based DE-BO with Adaptive Batch size, Local Search with LCB and EI acquisition function.** This algorithm combines adaptive temperature and success-rate based Differential Evolution (DE) with adaptive batch size and a hybrid acquisition function using both Lower Confidence Bound (LCB) and Expected Improvement (EI). The local search is enhanced by using a combination of LCB and EI as the objective function, and its probability is dynamically adjusted based on the GP's uncertainty. The algorithm incorporates a mechanism to adapt the batch size based on the GP's uncertainty to improve sampling efficiency. The local search range is also adapted based on the GP uncertainty. Furthermore, adaptive step size for local search is incorporated based on the success of local search iterations.

# Justification
This algorithm builds upon the strengths of ATSDE_ABLSEI_BO and ATSDE_ABLSLS_BO.
1.  **Hybrid Acquisition Function (LCB and EI):** LCB encourages exploration while EI promotes exploitation. Combining both allows for a better balance, especially in the early stages of optimization. The LCB is used in the DE loop for global search, while EI is used as the target for local search.
2.  **Adaptive Local Search:** The local search probability is dynamically adjusted based on the GP's uncertainty, as in ATSDE_ABLSEI_BO. The local search range is adapted based on GP uncertainty.
3.  **Adaptive Batch Size:** The batch size is adapted based on the GP's uncertainty to improve sampling efficiency, as in both parent algorithms.
4.  **Adaptive F and CR:** The DE parameters F and CR are adapted based on the success rate of the DE iterations.
5.  **Adaptive Local Search Step Size:** The step size for the local search is dynamically adjusted based on the success rate of the local search iterations.
6.  **Temperature Annealing:** The temperature parameter in DE is annealed to control exploration.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATSDE_ABLS_LCBEI_BO:
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
        self.success_rate = 0.0
        self.learning_rate = 0.1
        self.F_step = 0.05
        self.CR_step = 0.05
        self.temperature = 1.0
        self.temperature_decay = 0.95
        self.ls_prob = 0.1
        self.batch_size = 1
        self.exploration_weight = 2.0
        self.ls_step_size = 0.1
        self.ls_success_rate = 0.0
        self.ls_learning_rate = 0.1

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function_lcb(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        lcb = mu - self.exploration_weight * sigma
        return lcb

    def _acquisition_function_ei(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        return ei

    def _select_next_points(self, batch_size):
        population = self._sample_points(self.pop_size)

        n_success = 0
        for _ in range(20):
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                x_mutated = x_r1 + self.F * (x_r2 - x_r3) * self.temperature
                x_mutated = np.clip(x_mutated, self.bounds[0], self.bounds[1])

                # Crossover
                x_trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        x_trial[j] = x_mutated[j]

                # Selection
                lcb_trial = self._acquisition_function_lcb(x_trial.reshape(1, -1))[0]
                lcb_current = self._acquisition_function_lcb(population[i].reshape(1, -1))[0]

                if lcb_trial < lcb_current:
                    population[i] = x_trial
                    n_success += 1

            self.temperature *= self.temperature_decay

        # Update F and CR adaptively
        self.success_rate = 0.9 * self.success_rate + 0.1 * (n_success / self.pop_size)
        self.F = np.clip(self.F + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)

        # Local search
        ei_values = self._acquisition_function_ei(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]

        # Probabilistic Local Search
        mu, sigma = self.gp.predict(best_x.reshape(1, -1), return_std=True)
        self.ls_prob = 0.1 + 0.9 * np.exp(-sigma)

        next_point = best_x.reshape(1, -1)
        if np.random.rand() < self.ls_prob:
            def obj_func(x):
                return -self._acquisition_function_ei(x.reshape(1, -1))[0]

            # Adjust local search range based on GP uncertainty
            search_range = min(sigma[0] * 2, (self.bounds[1][0] - self.bounds[0][0]) / 2)

            res = minimize(obj_func, best_x, bounds=[(best_x[i] - search_range, best_x[i] + search_range) for i in range(self.dim)], method='L-BFGS-B, maxiter = 5')
            
            if res.success:
                self.ls_success_rate = 0.9 * self.ls_success_rate + 0.1
            else:
                self.ls_success_rate = 0.9 * self.ls_success_rate

            self.ls_step_size = np.clip(self.ls_step_size + self.ls_learning_rate * (self.ls_success_rate - 0.5), 0.01, 0.5)

            next_point = res.x.reshape(1, -1)
            next_point = np.clip(next_point, self.bounds[0], self.bounds[1])

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

            _, std = self.gp.predict(self.X, return_std=True)
            mean_std = np.mean(std)
            self.batch_size = max(1, int(self.dim / (1 + mean_std * 10)))
            self.batch_size = min(self.batch_size, self.budget - self.n_evals)

            next_X = np.vstack([self._select_next_points(1) for _ in range(self.batch_size)])
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<ATSDE_ABLS_LCBEI_BO>", line 162, in __call__
 162->             next_X = np.vstack([self._select_next_points(1) for _ in range(self.batch_size)])
  File "<ATSDE_ABLS_LCBEI_BO>", line 162, in <listcomp>
 162->             next_X = np.vstack([self._select_next_points(1) for _ in range(self.batch_size)])
  File "<ATSDE_ABLS_LCBEI_BO>", line 118, in _select_next_points
 116 |             search_range = min(sigma[0] * 2, (self.bounds[1][0] - self.bounds[0][0]) / 2)
 117 | 
 118->             res = minimize(obj_func, best_x, bounds=[(best_x[i] - search_range, best_x[i] + search_range) for i in range(self.dim)], method='L-BFGS-B, maxiter = 5')
 119 |             
 120 |             if res.success:
ValueError: Unknown solver L-BFGS-B, maxiter = 5
