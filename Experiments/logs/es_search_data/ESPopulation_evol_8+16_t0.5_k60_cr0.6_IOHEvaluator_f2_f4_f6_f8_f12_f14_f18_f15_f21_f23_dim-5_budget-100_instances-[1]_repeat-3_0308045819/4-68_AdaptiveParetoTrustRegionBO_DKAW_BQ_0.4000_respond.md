# Description
**Adaptive Pareto Trust Region BO with Dynamic Kernel, Acquisition Weighting, and Bayesian Quadrature (APTRBO-DKAW-BQ):** This algorithm enhances the APTRBO-DKAW framework by incorporating Bayesian Quadrature (BQ) to improve the estimation of Expected Improvement (EI). Instead of directly using the GP's predicted mean and variance for EI calculation, BQ is employed to approximate the integral of the improvement function over the trust region. This provides a more accurate estimate of the expected improvement, especially when the GP model is uncertain or the improvement function is highly non-linear. Additionally, the dynamic kernel selection is refined by including a Matern kernel in the kernel options, and the kernel selection process is improved by using cross-validation instead of log-marginal likelihood. Finally, the diversity metric is enhanced by considering the distance to the `k` nearest neighbors instead of just the nearest neighbor.

# Justification
- **Bayesian Quadrature for EI:** Standard EI calculation relies on the assumption that the GP's predictive distribution is accurate. However, this assumption can be violated, especially in regions with limited data or complex function behavior. BQ provides a more robust estimate of EI by integrating the improvement function over the trust region, taking into account the GP's uncertainty. This can lead to better exploration and exploitation decisions.
- **Enhanced Kernel Selection:** Including the Matern kernel allows for more flexible modeling of the objective function's smoothness. Using cross-validation provides a more reliable estimate of kernel performance compared to log-marginal likelihood, which can be prone to overfitting.
- **k-Nearest Neighbors Diversity:** Considering the distance to multiple nearest neighbors provides a more comprehensive measure of diversity compared to just the nearest neighbor. This can help to avoid premature convergence and encourage exploration of different regions of the search space.
- **Computational Efficiency:** BQ is implemented using a relatively small number of samples to minimize the computational overhead. The other enhancements are also designed to be computationally efficient, ensuring that the algorithm remains practical for black-box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, KFold
from scipy.integrate import quad

class AdaptiveParetoTrustRegionBO_DKAW_BQ:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5 * dim, self.budget // 10)
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.ei_weight = 0.5  # Initial weight for Expected Improvement
        self.diversity_weight = 0.5  # Initial weight for Diversity
        self.k_neighbors = 5 # Number of neighbors for diversity metric

    def _sample_points(self, n_points):
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        # Dynamic Kernel Selection
        kernels = [
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=0.1, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=10.0, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=0.5)
        ]
        best_gp = None
        best_cv_score = float('inf')

        kf = KFold(n_splits=min(5, X.shape[0]), shuffle=True, random_state=42) # Ensure n_splits <= n_samples

        for kernel in kernels:
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            cv_scores = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                gp.fit(X_train, y_train)
                y_pred, sigma = gp.predict(X_test, return_std=True)
                cv_scores.append(np.mean((y_pred - y_test)**2))  # Mean Squared Error

            mean_cv_score = np.mean(cv_scores)

            if mean_cv_score < best_cv_score:
                best_cv_score = mean_cv_score
                best_gp = gp

        self.gp = best_gp
        return self.gp

    def _expected_improvement(self, X):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            # Bayesian Quadrature for EI
            ei = np.zeros((len(X), 1))
            for i, x in enumerate(X):
                def integrand(y):
                    mu, sigma = self.gp.predict(y.reshape(1, -1), return_std=True)
                    sigma = np.clip(sigma, 1e-9, np.inf)
                    imp = self.best_y - mu
                    Z = imp / sigma
                    return (imp * norm.cdf(Z) + sigma * norm.pdf(Z))

                # Integrate over a small region around x (Bayesian Quadrature)
                lower_bound = np.clip(x - 0.1 * self.trust_region_radius, self.bounds[0], self.bounds[1])
                upper_bound = np.clip(x + 0.1 * self.trust_region_radius, self.bounds[0], self.bounds[1])

                # Perform integration for each dimension
                integral_result = 1.0
                for d in range(self.dim):
                    res, _ = quad(lambda y: integrand(np.array([x[j] if j != d else y for j in range(self.dim)])), lower_bound[d], upper_bound[d])
                    integral_result *= res

                ei[i, 0] = integral_result

            return ei

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = cdist(X, self.X)
            min_distances = np.sort(distances, axis=1)[:, :self.k_neighbors]  # Get k nearest neighbors
            diversity = np.mean(min_distances, axis=1) # Average distance to k nearest neighbors
            return diversity.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)

        ei = self._expected_improvement(candidates)
        diversity = self._diversity_metric(candidates)

        ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
        diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

        # Adaptive Weighting
        F = np.hstack([self.ei_weight * ei_normalized, self.diversity_weight * diversity_normalized])

        is_efficient = np.ones(F.shape[0], dtype=bool)
        for i, c in enumerate(F):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(F[is_efficient] >= c, axis=1)
                is_efficient[i] = True

        pareto_front = candidates[is_efficient]

        if self.gp is not None and len(pareto_front) > 0:
            _, sigma = self.gp.predict(pareto_front, return_std=True)
            next_point = pareto_front[np.argmax(sigma)].reshape(1, -1)
        elif len(pareto_front) > 0:
            next_point = pareto_front[np.random.choice(len(pareto_front))].reshape(1, -1)
        else:
            # If Pareto front is empty, sample randomly from trust region
            next_point = self._sample_points(1)

        return next_point

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
            self.success_ratio = 1.0
        else:
            self.success_ratio *= 0.75

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            # Increase EI weight, decrease diversity weight
            self.ei_weight = min(self.ei_weight + 0.1, 1.0)
            self.diversity_weight = max(self.diversity_weight - 0.1, 0.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
            # Decrease EI weight, increase diversity weight
            self.ei_weight = max(self.ei_weight - 0.1, 0.0)
            self.diversity_weight = min(self.diversity_weight + 0.1, 1.0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(1, self.dim)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveParetoTrustRegionBO_DKAW_BQ got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1758 with standard deviation 0.1042.

took 8101.09 seconds to run.