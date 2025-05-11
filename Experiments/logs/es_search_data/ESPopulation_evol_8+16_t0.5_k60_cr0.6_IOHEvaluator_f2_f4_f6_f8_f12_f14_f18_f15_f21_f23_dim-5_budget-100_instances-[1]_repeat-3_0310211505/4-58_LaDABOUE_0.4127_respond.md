# Description
**LaDABOUE: Landscape-Aware Dynamic Acquisition Balancing with Uncertainty-Aware Exploration.** This algorithm combines landscape analysis, dynamic acquisition function balancing using Thompson Sampling, uncertainty-aware exploration, and adaptive trust region management. It adapts the exploration-exploitation trade-off based on the function's landscape, dynamically selects between EI and UCB acquisition functions using Thompson Sampling, incorporates an uncertainty-aware exploration term, and adjusts the trust region based on success and model uncertainty. The landscape analysis informs the temperature parameter in Thompson Sampling, influencing the balance between acquisition functions.

# Justification
This algorithm builds upon the strengths of both ALGTRBOUE and ATRDABOUE.

*   **Landscape Awareness:** ALGTRBOUE's landscape analysis provides valuable information about the function's structure. This information is used to adjust the temperature parameter in Thompson Sampling, which controls the exploration-exploitation balance when selecting between EI and UCB. A smoother landscape favors exploitation (EI), while a rugged landscape favors exploration (UCB).
*   **Dynamic Acquisition Balancing:** ATRDABOUE's dynamic acquisition function balancing using Thompson Sampling allows the algorithm to adaptively select the most promising acquisition function (EI or UCB). This is particularly useful for different types of functions, as EI is generally better for exploitation and UCB for exploration.
*   **Uncertainty-Aware Exploration:** The uncertainty-aware exploration term in the acquisition function, inherited from both ALGTRBOUE and ATRDABOUE, encourages exploration in regions with high predictive variance. This helps to avoid premature convergence and discover promising areas of the search space.
*   **Adaptive Trust Region:** The adaptive trust region mechanism, present in both parent algorithms, helps to focus the search on promising regions while maintaining diversity. The trust region is adjusted based on the success rate and model uncertainty.
*   **Computational Efficiency:** The algorithm avoids computationally expensive gradient estimations, relying instead on landscape analysis and Thompson Sampling for dynamic exploration control.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import beta
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances


class LaDABOUE:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.trust_region_width = 2.0
        self.success_threshold = 0.1
        self.best_y = np.inf
        self.best_x = None
        self.acq_funcs = ['ei', 'ucb']
        self.n_acq = len(self.acq_funcs)
        self.weights = np.ones(self.n_acq) / self.n_acq
        self.gamma = 0.1
        self.exploration_weight = 0.1
        self.alpha = np.ones(self.n_acq)  # Initialize alpha for Beta distribution
        self.beta = np.ones(self.n_acq)   # Initialize beta for Beta distribution
        self.temperature = 1.0  # Initial temperature for Thompson Sampling
        self.temperature_decay = 0.95  # Decay rate for temperature
        self.landscape_correlation = 0.0  # Initial landscape correlation
        self.smoothness_threshold = 0.5  # Threshold for considering the landscape smooth

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
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
            best = np.min(self.y)
            imp = best - mu
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0
            acq_values = ei + self.exploration_weight * sigma
            return acq_values

        elif acq_func == 'ucb':
            ucb = mu + 2 * sigma
            acq_values = ucb + self.exploration_weight * sigma
            return acq_values

        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

    def _select_next_points(self, batch_size):
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        acq_values = np.zeros((self.n_acq, n_candidates))
        for i, acq_func in enumerate(self.acq_funcs):
            acq_values[i, :] = self._acquisition_function(X_cand, acq_func).flatten()

        # Thompson Sampling
        sampled_values = []
        for i in range(self.n_acq):
            sampled_values.append(beta.rvs(self.alpha[i], self.beta[i], size=1)[0])

        # Adjust temperature based on landscape correlation
        if self.landscape_correlation > self.smoothness_threshold:
            self.temperature = max(0.1, self.temperature * 0.9)  # Reduce temperature for smoother landscapes
        else:
            self.temperature = min(2.0, self.temperature * 1.1)  # Increase temperature for rugged landscapes

        # Select acquisition function with highest sampled value, weighted by temperature
        probabilities = np.exp(np.array(sampled_values) / self.temperature)
        probabilities /= np.sum(probabilities)  # Normalize to probabilities
        idx = np.random.choice(self.n_acq, p=probabilities)

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)

            # Analyze the landscape
            self.landscape_correlation = self._analyze_landscape()

            X_next, idx = self._select_next_points(batch_size)

            y_next = self._evaluate_points(func, X_next)

            self._update_eval_points(X_next, y_next)

            self._update_weights(idx, y_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            self.best_y = new_best_y
            self.best_x = new_best_x

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm LaDABOUE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1836 with standard deviation 0.1077.

took 961.00 seconds to run.