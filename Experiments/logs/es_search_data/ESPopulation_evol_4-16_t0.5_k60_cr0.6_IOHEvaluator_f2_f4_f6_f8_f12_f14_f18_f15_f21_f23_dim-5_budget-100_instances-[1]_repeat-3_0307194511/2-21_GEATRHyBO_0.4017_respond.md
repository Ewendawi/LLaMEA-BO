# Description
**Gradient-Enhanced Adaptive Trust Region Hybrid Bayesian Optimization (GEATRHyBO)**: This algorithm synergistically fuses the strengths of AdaptiveTrustRegionHybridBO and TREGEBO. It employs an adaptive trust region mechanism, efficient lengthscale estimation using nearest neighbors for the Gaussian Process (GP) surrogate model, and gradient-enhanced local search. Specifically, it adaptively adjusts the trust region size based on the success of local search, estimates the GP kernel lengthscale using nearest neighbors, and leverages gradient information to refine the local search within the trust region. The gradient is estimated efficiently at the current best point using finite differences based on GP predictions to avoid excessive function evaluations. Furthermore, a dynamic weighting strategy is introduced to balance exploration and exploitation by modulating the influence of the Expected Improvement (EI) acquisition function and gradient information during local search.

# Justification
The GEATRHyBO algorithm is designed to address the limitations of its predecessors by combining their strengths.

1.  **Adaptive Trust Region:** Inherited from AdaptiveTrustRegionHybridBO and TREGEBO, the adaptive trust region allows for efficient exploration and exploitation. The trust region size is adjusted based on the success of the local search, expanding when improvements are found and shrinking when no progress is made.

2.  **Efficient Lengthscale Estimation:** AdaptiveTrustRegionHybridBO's nearest neighbors approach for lengthscale estimation is computationally efficient and adapts well to the local structure of the data, improving the GP model's accuracy.

3.  **Gradient-Enhanced Local Search:** TREGEBO's gradient estimation refines the local search by incorporating gradient information. GEATRHyBO uses GP predictions to estimate the gradient, which is then used to guide the local search within the trust region. This helps to accelerate convergence and improve the quality of the solutions found. The gradient estimation step is made more efficient by using the GP model to predict function values for points used in finite difference calculations, thus avoiding additional function evaluations.

4. **Dynamic Weighting:** A dynamic weighting strategy is introduced to balance the influence of the EI acquisition function and the gradient information during local search. This allows the algorithm to adapt to different stages of the optimization process, prioritizing exploration early on and exploitation later. The weight is adjusted based on the success rate of the local search, increasing the weight of the gradient when the local search is successful and decreasing it when it is not.

By combining these features, GEATRHyBO aims to achieve a better balance between exploration and exploitation, leading to improved performance on the BBOB test suite. The algorithm is designed to be computationally efficient by minimizing the number of function evaluations required for gradient estimation and lengthscale tuning.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class GEATRHyBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.batch_size = min(5, self.dim)
        self.trust_region_size = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 2.0
        self.success_threshold = 0.7
        self.delta = 1e-3
        self.gradient_weight = 0.1 # Initial weight for gradient information
        self.gradient_weight_increase = 1.1
        self.gradient_weight_decrease = 0.9
        self.success_history = [] # Keep track of recent success

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])

        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=median_distance, length_scale_bounds=(1e-3, 1e3))

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

        return ei

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

    def _estimate_gradient(self, model, x):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta

            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
            y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

            gradient[i] = (y_plus - y_minus) / (2 * self.delta)
        return gradient

    def _local_search(self, model, center, gradient, n_points=50):
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        mu, _ = model.predict(candidate_points, return_std=True)
        ei = self._acquisition_function(candidate_points)

        # Combine GP prediction, acquisition function and gradient information
        mu = mu.reshape(-1) - self.gradient_weight * np.sum(gradient * (candidate_points - center), axis=1) + ei.flatten()

        best_index = np.argmin(mu)
        best_point = candidate_points[best_index]

        return best_point

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_x = self.X[best_index]
        best_y = self.y[best_index][0]

        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)
            gradient = self._estimate_gradient(model, best_x)

            next_x = self._local_search(model, best_x.copy(), gradient)
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1])

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0]
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                self.trust_region_size *= self.trust_region_expand
                self.success_history.append(True)
                self.gradient_weight = min(1.0, self.gradient_weight * self.gradient_weight_increase) # Increase gradient weight
            else:
                self.trust_region_size *= self.trust_region_shrink
                self.success_history.append(False)
                self.gradient_weight = max(0.0, self.gradient_weight * self.gradient_weight_decrease) # Decrease gradient weight

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Keep only the last 10 success/failure records
            self.success_history = self.success_history[-10:]

        return best_y, best_x
```
## Feedback
 The algorithm GEATRHyBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1778 with standard deviation 0.1114.

took 152.33 seconds to run.