# Description
**LaATRBOUE: Landscape-Aware Adaptive Trust Region Bayesian Optimization with Uncertainty-Aware Exploration and Acquisition Function Dueling**

This algorithm combines landscape analysis, adaptive trust region management, uncertainty-aware exploration, and dynamic acquisition function balancing using Thompson sampling. It leverages the strengths of ALGTRBOUE (landscape analysis and uncertainty-aware exploration) and ATRDABOUE (dynamic acquisition function balancing) within an adaptive trust region framework. Additionally, it incorporates a mechanism to adjust the exploration weight dynamically based on the landscape correlation, further refining the exploration-exploitation trade-off.

# Justification
The algorithm incorporates the following key components:

1.  **Landscape Analysis:** The landscape correlation is calculated to estimate the ruggedness of the objective function. This information is used to adjust the temperature parameter in the Expected Improvement (EI) acquisition function, encouraging more exploration in rugged landscapes and more exploitation in smoother landscapes.

2.  **Adaptive Trust Region:** A trust region is maintained and adapted based on the success rate of previous iterations. This helps to focus the search in promising regions while also allowing for exploration of new areas.

3.  **Uncertainty-Aware Exploration:** An uncertainty-aware exploration term is added to the EI acquisition function to encourage exploration in regions of high uncertainty. This helps to prevent the algorithm from getting stuck in local optima.

4.  **Dynamic Acquisition Function Balancing (Thompson Sampling):** Multiple acquisition functions (EI and UCB) are used, and their weights are dynamically adjusted using Thompson sampling. This allows the algorithm to adapt to the characteristics of the objective function and to balance exploration and exploitation more effectively.

5.  **Adaptive Exploration Weight:** The exploration weight is dynamically adjusted based on the landscape correlation. When the landscape correlation is low (i.e., the landscape is rugged), the exploration weight is increased to encourage more exploration. When the landscape correlation is high (i.e., the landscape is smooth), the exploration weight is decreased to encourage more exploitation.

The combination of these components allows the algorithm to effectively balance exploration and exploitation and to adapt to the characteristics of the objective function. The adaptive trust region helps to focus the search in promising regions, while the uncertainty-aware exploration and dynamic acquisition function balancing help to prevent the algorithm from getting stuck in local optima. The landscape analysis and adaptive exploration weight further refine the exploration-exploitation trade-off, leading to more efficient optimization.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from scipy.stats import beta


class LaATRBOUE:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.trust_region_width = 2.0  # Initial trust region width
        self.success_threshold = 0.1  # Threshold for increasing trust region
        self.best_y = np.inf  # Initialize best_y with a large value
        self.best_x = None
        self.temperature = 1.0  # Initial temperature for exploration
        self.landscape_correlation = 0.0  # Initial landscape correlation
        self.smoothness_threshold = 0.5  # Threshold for considering the landscape smooth
        self.exploration_weight = 0.1  # Initial weight for exploration term in acquisition function
        self.acq_funcs = ['ei', 'ucb']
        self.n_acq = len(self.acq_funcs)
        self.alpha = np.ones(self.n_acq)  # Initialize alpha for Beta distribution
        self.beta = np.ones(self.n_acq)  # Initialize beta for Beta distribution
        self.temperature_decay = 0.95

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Sample within the trust region
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _analyze_landscape(self):
        if self.X is None or self.y is None or len(self.X) < 2:
            return 0.0

        distances = pairwise_distances(self.X)
        value_differences = np.abs(self.y - self.y.T)

        # Flatten the matrices and remove the diagonal elements
        distances = distances.flatten()
        value_differences = value_differences.flatten()
        indices = np.arange(len(distances))
        distances = distances[indices % (len(self.X) + 1) != 0]
        value_differences = value_differences[indices % (len(self.X) + 1) != 0]

        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(distances, value_differences)
        return correlation if not np.isnan(correlation) else 0.0

    def _acquisition_function(self, X, acq_func='ei'):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if acq_func == 'ei':
            imp = self.best_y - mu
            Z = imp / (self.temperature * sigma + 1e-9)  # Adding a small constant to avoid division by zero
            ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
            ei = np.clip(ei, 0, 1e10)  # Clip EI to avoid potential NaN issues
            ei[sigma <= 1e-6] = 0.0  # avoid division by zero
            acq_values = ei + self.exploration_weight * sigma
            return acq_values

        elif acq_func == 'ucb':
            ucb = mu + 2 * sigma
            acq_values = ucb + self.exploration_weight * sigma
            return acq_values

        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

    def _select_next_points(self, batch_size):
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        acq_values = np.zeros((self.n_acq, n_candidates))
        for i, acq_func in enumerate(self.acq_funcs):
            acq_values[i, :] = self._acquisition_function(X_cand, acq_func).flatten()

        # Thompson Sampling
        sampled_values = []
        for i in range(self.n_acq):
            sampled_values.append(beta.rvs(self.alpha[i], self.beta[i], size=1)[0])

        # Select acquisition function with highest sampled value
        idx = np.argmax(sampled_values)

        top_indices = np.argsort(acq_values[idx, :])[::-1][:batch_size]
        return X_cand[top_indices], idx

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def _update_weights(self, idx, new_y):
        # Reward is based on the improvement
        reward = self.best_y - new_y.min()

        # Update alpha and beta parameters for the selected acquisition function
        if reward > 0:
            self.alpha[idx] += 1
        else:
            self.beta[idx] += 1

        # Decay temperature
        self.temperature *= self.temperature_decay

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self.model = self._fit_model(self.X, self.y)

            # Analyze the landscape
            self.landscape_correlation = self._analyze_landscape()

            # Adjust temperature based on landscape correlation
            if self.landscape_correlation > self.smoothness_threshold:
                self.temperature = max(0.1, self.temperature * 0.9)  # Reduce temperature for smoother landscapes
                self.exploration_weight = min(0.2, self.exploration_weight * 1.1) # Increase exploration if smooth
            else:
                self.temperature = min(2.0, self.temperature * 1.1)  # Increase temperature for rugged landscapes
                self.exploration_weight = max(0.01, self.exploration_weight * 0.9) # Decrease exploration if rugged

            # Select next points
            X_next, idx = self._select_next_points(batch_size)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Update acquisition function weights
            self._update_weights(idx, y_next)

            # Update best solution
            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            # Adjust trust region width
            success_ratio = (self.best_y - new_best_y) / self.best_y

            if success_ratio > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful

            self.best_y = new_best_y
            self.best_x = new_best_x

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm LaATRBOUE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1927 with standard deviation 0.1129.

took 672.62 seconds to run.