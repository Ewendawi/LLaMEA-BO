# Description
**AdaptiveTrustRegionOptimisticHybridV3BO**: This algorithm builds upon `AdaptiveTrustRegionOptimisticHybridBO` and `AdaptiveTrustRegionHybridV2BO` by incorporating an adaptive weighting between Expected Improvement (EI) and Upper Confidence Bound (UCB) *globally*, not just within the trust region. It also includes the refined global search strategy from `AdaptiveTrustRegionHybridV2BO`, using `scipy.optimize.minimize` to optimize the acquisition function. Furthermore, it introduces a mechanism to adapt the global search probability based on the success rate of the local search, promoting global exploration when local search stagnates. The lengthscale of the GP kernel is efficiently estimated using nearest neighbors. The trust region size is adaptively adjusted based on the success of local search.

# Justification
The key improvements are:

1.  **Adaptive EI/UCB Weighting (Globally):** Instead of a fixed 50/50 weighting, the algorithm dynamically adjusts the balance between exploration (UCB) and exploitation (EI) in the *global* acquisition function. This allows the algorithm to adapt to different stages of the optimization process, prioritizing exploration early on and exploitation later. The weighting is adjusted based on the success of recent evaluations.

2.  **Refined Global Search:** The global search is enhanced using `scipy.optimize.minimize` with the L-BFGS-B method, as in `AdaptiveTrustRegionHybridV2BO`, allowing for a more directed global exploration compared to simple random sampling. This helps the algorithm escape local optima more effectively.

3.  **Adaptive Global Search Probability:** The probability of performing a global search step is adaptively adjusted based on the success rate of the local search. If the local search is not yielding improvements, the algorithm increases the probability of performing a global search to explore new regions of the search space. This helps to balance exploration and exploitation more effectively.

4.  **Computational Efficiency:** The lengthscale estimation using nearest neighbors remains efficient. The use of `scipy.optimize.minimize` in the global search is limited to a small number of iterations to maintain computational efficiency.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class AdaptiveTrustRegionOptimisticHybridV3BO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.batch_size = min(5, self.dim)
        self.trust_region_size = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 2.0
        self.success_threshold = 0.7
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99  # Decay rate for beta
        self.global_search_prob = 0.05
        self.ei_weight = 0.5  # Initial weight for Expected Improvement
        self.ei_weight_adjust = 0.05
        self.successful_local_searches = 0
        self.local_search_exploitation = 0.8 # Initial weight for exploitation in local search
        self.local_search_exploitation_adjust = 0.05

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])

        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
            length_scale=median_distance, length_scale_bounds=(1e-3, 1e3)
        )

        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        y_best = np.min(self.y)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        ucb = mu - self.beta * sigma # minimize
        acquisition = self.ei_weight * ei + (1 - self.ei_weight) * ucb
        return acquisition

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
        return candidate_points[indices]

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

    def _local_search(self, model, center, n_points=50):
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        mu, _ = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)
        ei = self._acquisition_function(candidate_points)

        weighted_values = self.local_search_exploitation * mu + (1 - self.local_search_exploitation) * (-ei)
        best_index = np.argmin(weighted_values)
        best_point = candidate_points[best_index]

        return best_point

    def _global_search(self):
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        x0 = self._sample_points(1)[0]
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})
        return result.x

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_x = self.X[best_index]
        best_y = self.y[best_index][0]

        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)

            if np.random.rand() < self.global_search_prob:
                next_x = self._global_search()
            else:
                next_x = self._local_search(model, best_x.copy())

            next_x = np.clip(next_x, self.bounds[0], self.bounds[1])
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0]
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                self.trust_region_size *= self.trust_region_expand
                self.successful_local_searches += 1
                self.ei_weight = min(1.0, self.ei_weight + self.ei_weight_adjust)
                self.local_search_exploitation = min(1.0, self.local_search_exploitation + self.local_search_exploitation_adjust)
            else:
                self.trust_region_size *= self.trust_region_shrink
                self.successful_local_searches = 0
                self.ei_weight = max(0.0, self.ei_weight - self.ei_weight_adjust)
                self.local_search_exploitation = max(0.0, self.local_search_exploitation - self.local_search_exploitation_adjust)
                # Increase global search probability if local search stagnates
                self.global_search_prob = min(0.5, self.global_search_prob + 0.02)

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)
            self.global_search_prob = max(0.05, self.global_search_prob - 0.01) # Decay global search prob

            # Decay exploration parameter
            self.beta *= self.beta_decay

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionOptimisticHybridV3BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1658 with standard deviation 0.0963.

took 991.46 seconds to run.