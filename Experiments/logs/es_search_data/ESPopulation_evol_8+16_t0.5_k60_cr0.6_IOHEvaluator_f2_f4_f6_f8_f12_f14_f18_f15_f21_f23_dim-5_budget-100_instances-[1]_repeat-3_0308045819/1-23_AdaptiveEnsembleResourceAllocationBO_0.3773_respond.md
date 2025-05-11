# Description
**Adaptive Ensemble Resource Allocation Bayesian Optimization (AERABO):** This algorithm combines the strengths of SurrogateEnsembleBO and DynamicResourceAllocationBO. It uses an ensemble of Gaussian Process Regression (GPR) models with different kernel parameters to improve robustness and handle uncertainty. It dynamically allocates the budget between exploration and exploitation phases, similar to DRABO, but the phase switching is informed by the performance of the ensemble, not just a single GP. The acquisition function is a weighted combination of Upper Confidence Bound (UCB) and Thompson Sampling, where the weights are dynamically adjusted based on the current phase (exploration vs. exploitation). This allows for a more adaptive and efficient search strategy. The ensemble weights are also dynamically adjusted based on validation performance, similar to SEBO.

# Justification
*   **Ensemble of GPs:** Using an ensemble of GPs, as in SurrogateEnsembleBO, provides more robust uncertainty estimates and reduces the risk of relying on a single, potentially inaccurate, surrogate model.
*   **Dynamic Resource Allocation:** DynamicResourceAllocationBO's strategy of switching between exploration (Thompson Sampling) and exploitation (Expected Improvement) is beneficial for efficient budget usage. However, AERABO improves upon this by using a more sophisticated phase switching mechanism based on the ensemble's performance.
*   **Adaptive Acquisition Function:** By combining UCB (from SurrogateEnsembleBO) and Thompson Sampling (from DynamicResourceAllocationBO) into a single, adaptively weighted acquisition function, the algorithm can better balance exploration and exploitation. The weights are adjusted based on the current phase, allowing the algorithm to prioritize exploration during the initial phase and exploitation as it converges to a solution.
*   **Computational Efficiency:** The algorithm aims to maintain computational efficiency by limiting the number of surrogate models in the ensemble and using efficient acquisition function optimization techniques. The local search in `DynamicResourceAllocationBO` was computationally expensive. Instead of using local search, AERABO samples multiple candidate points and evaluates them using the acquisition function, which is computationally cheaper.
*   **Error Handling:** The algorithm builds upon the error handling strategies of DynamicResourceAllocationBO by limiting the number of L-BFGS-B iterations during acquisition function optimization to avoid exceeding the budget.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

class AdaptiveEnsembleResourceAllocationBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5*dim, self.budget//10)

        self.gp_ensemble = []
        self.ensemble_weights = []
        self.best_x = None
        self.best_y = float('inf')
        self.n_ensemble = 3

        self.exploration_phase = True
        self.improvement_threshold = 1e-3
        self.improvement_rate = 0.0
        self.last_best_y = float('inf')
        self.phase_switch_interval = 10
        self.exploration_weight = 0.5  # Initial weight for Thompson Sampling

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if not self.gp_ensemble:
            kernels = [
                ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=0.5),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
            ]
            for i in range(self.n_ensemble):
                gp = GaussianProcessRegressor(kernel=kernels[i % len(kernels)], n_restarts_optimizer=0, alpha=1e-6)
                gp.fit(X_train, y_train)
                self.gp_ensemble.append(gp)
                self.ensemble_weights.append(1.0 / self.n_ensemble)
        else:
            for gp in self.gp_ensemble:
                gp.fit(X_train, y_train)

        val_errors = []
        for gp in self.gp_ensemble:
            y_pred, _ = gp.predict(X_val, return_std=True)
            error = np.mean((y_pred - y_val.flatten())**2)
            val_errors.append(error)

        val_errors = np.array(val_errors)
        weights = np.exp(-val_errors) / np.sum(np.exp(-val_errors))
        self.ensemble_weights = weights

    def _acquisition_function(self, X):
        # Weighted combination of UCB and Thompson Sampling
        ucb = self._acquisition_function_ucb(X)
        ts = self._acquisition_function_ts(X)
        return (1 - self.exploration_weight) * ucb + self.exploration_weight * ts

    def _acquisition_function_ucb(self, X):
        # Upper Confidence Bound
        if not self.gp_ensemble:
            return np.random.normal(size=(len(X), 1))
        else:
            mu_ensemble = np.zeros((len(X), 1))
            sigma_ensemble = np.zeros((len(X), 1))

            for i, gp in enumerate(self.gp_ensemble):
                mu, sigma = gp.predict(X, return_std=True)
                mu_ensemble += self.ensemble_weights[i] * mu.reshape(-1, 1)
                sigma_ensemble += self.ensemble_weights[i] * sigma.reshape(-1, 1)

            kappa = 2.0
            ucb = mu_ensemble + kappa * sigma_ensemble
            return ucb

    def _acquisition_function_ts(self, X):
        # Thompson Sampling
        if not self.gp_ensemble:
            return np.random.normal(size=(len(X), 1))
        else:
            y_samples = np.zeros((len(X), 1))
            for i, gp in enumerate(self.gp_ensemble):
                y_samples += self.ensemble_weights[i] * gp.sample_y(X, n_samples=1)
            return y_samples

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(100*batch_size)
        acquisition_values = self._acquisition_function(candidates)
        best_indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        return candidates[best_indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _check_phase_switch(self):
        if self.best_y < self.last_best_y:
            self.improvement_rate = (self.last_best_y - self.best_y) / self.last_best_y
        else:
            self.improvement_rate = 0.0

        if self.improvement_rate < self.improvement_threshold and not self.exploration_phase:
            self.exploration_phase = True
            print("Switching to exploration phase")
            self.exploration_weight = 0.7 # Increase exploration weight
        elif self.improvement_rate >= self.improvement_threshold and self.exploration_phase:
            self.exploration_phase = False
            print("Switching to exploitation phase")
            self.exploration_weight = 0.3 # Decrease exploration weight

        self.last_best_y = self.best_y

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(2, self.dim)
        iteration = 0
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            if iteration % self.phase_switch_interval == 0:
                self._check_phase_switch()

            iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveEnsembleResourceAllocationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1524 with standard deviation 0.0993.

took 107.73 seconds to run.