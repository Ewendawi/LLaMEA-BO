# Description
**ATSDE_APLS_EIGrad_BO: Adaptive Temperature and Success-Rate based DE-BO with Adaptive Probabilistic Local Search using Expected Improvement Gradient.** This algorithm builds upon the ATSDE_APLS_EIvar_BO framework, but instead of relying solely on EI, EI variance, and GP uncertainty to trigger local search, it incorporates the gradient of the Expected Improvement (EI) to guide the local search direction and probability. The local search probability is dynamically adjusted based on GP uncertainty and the magnitude of the EI gradient. This encourages local search in regions where the GP is uncertain and where the EI is changing rapidly, indicating a promising direction for optimization. The local search is enhanced by using L-BFGS-B with multiple restarts, each initialized by moving along the EI gradient.

# Justification
The key idea is to leverage the gradient of the Expected Improvement to make the local search more efficient and targeted.

*   **EI Gradient for Local Search:** By incorporating the EI gradient, the local search is directed toward regions where the model predicts the most significant improvement. This contrasts with the previous approaches that relied on uncertainty and EI variance, which might lead to exploration in flat regions or regions with unstable EI values.
*   **Adaptive Local Search Probability:** The probability of performing local search is modulated by both GP uncertainty (sigma) and the magnitude of the EI gradient. This ensures that local search is triggered when the model is uncertain and the EI gradient is significant, indicating a promising direction for optimization.
*   **Multiple Restarts along EI Gradient:** To enhance the local search and escape local optima, L-BFGS-B is used with multiple restarts. Each restart is initialized by perturbing the current best point along the direction of the EI gradient. This allows the local search to explore the neighborhood more effectively.
*   **Computational Efficiency:** The EI gradient calculation is efficiently implemented using numerical differentiation. The adaptive temperature and success-rate based DE component, along with the adaptive population size, ensures efficient exploration of the search space.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

class ATSDE_APLS_EIGrad_BO:
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
        self.temperature = 1.0
        self.temperature_decay = 0.95
        self.ls_restarts = 3
        self.grad_step_size = 0.01

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

    def _acquisition_function(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        return ei, sigma

    def _calculate_ei_gradient(self, x):
        def ei_func(x):
            ei, _ = self._acquisition_function(x.reshape(1, -1))
            return ei[0, 0]

        grad = approx_fprime(x, ei_func, epsilon=self.grad_step_size)
        return grad

    def _select_next_points(self, batch_size):
        population = self._sample_points(self.pop_size)
        n_success_F = 0
        n_success_CR = 0
        
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
                ei_trial, _ = self._acquisition_function(x_trial.reshape(1, -1))
                ei_current, _ = self._acquisition_function(population[i].reshape(1, -1))
                
                if ei_trial[0, 0] > ei_current[0, 0]:
                    population[i] = x_trial
                    n_success_F += 1
                    n_success_CR += 1

            self.temperature *= self.temperature_decay

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
        
        ei_values, sigma = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]
        best_sigma = sigma[best_idx]
        best_ei = ei_values[best_idx]

        # Probabilistic Local Search
        ei_grad = self._calculate_ei_gradient(best_x)
        grad_norm = np.linalg.norm(ei_grad)
        ls_prob = np.clip(best_sigma * grad_norm, 0.0, 1.0) # Probability proportional to uncertainty and EI gradient magnitude

        if np.random.rand() < ls_prob:
            def obj_func(x):
                ei, _ = self._acquisition_function(x.reshape(1, -1))
                return -ei[0,0]
            
            best_lseval = float('inf')
            best_lsx = None
            for _ in range(self.ls_restarts):
                x0 = best_x + np.random.normal(0, 0.01, self.dim) + ei_grad * self.grad_step_size  # Add small noise and move along gradient
                x0 = np.clip(x0, self.bounds[0], self.bounds[1])
                res = minimize(obj_func, x0, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
                if res.fun < best_lseval:
                    best_lseval = res.fun
                    best_lsx = res.x
            next_point = best_lsx.reshape(1, -1)
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
            next_X = self._select_next_points(1)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATSDE_APLS_EIGrad_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1545 with standard deviation 0.1092.

took 948.61 seconds to run.