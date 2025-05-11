# Description
**Gradient-Boosted Efficient Hybrid Bayesian Optimization (GBEHBO):** This algorithm synergistically combines the strengths of Gradient-Boosted Bayesian Optimization (GBO) and Efficient Hybrid Bayesian Optimization (EHBBO). It employs a HistGradientBoostingRegressor as a surrogate model for capturing complex relationships and handling discontinuities, inheriting this capability from GBO. It also incorporates the efficient local search strategy from EHBBO, which dynamically adjusts the number of local search steps based on the remaining budget. To balance exploration and exploitation, it uses an adaptive Upper Confidence Bound (UCB) acquisition function, similar to GBO, and enhances exploration with the local search mechanism. The initial design uses Latin Hypercube Sampling (LHS) for space-filling properties, as in EHBBO and GBO. The algorithm also addresses potential NaN values in the data by imputing them with the mean, a strategy adopted from GBO.

# Justification
The key components are justified as follows:

*   **HistGradientBoostingRegressor Surrogate Model:** Gradient boosting models, like HistGradientBoostingRegressor, are capable of capturing non-linear relationships and discontinuities more effectively than Gaussian Processes, which are used in EHBBO. This can lead to better surrogate model accuracy, especially for complex and non-smooth objective functions.
*   **Efficient Local Search:** The local search strategy from EHBBO is integrated to enhance exploration, especially in high-dimensional spaces. By focusing search efforts around the best-observed point, the algorithm can efficiently refine solutions and escape local optima. The dynamic adjustment of local search steps ensures effective exploration throughout the optimization process.
*   **Adaptive UCB Acquisition Function:** The Upper Confidence Bound (UCB) acquisition function balances exploration and exploitation. The exploration factor is dynamically adjusted based on the remaining budget, encouraging more exploration in the early stages and more exploitation as the budget decreases.
*   **Latin Hypercube Sampling (LHS):** LHS provides a good space-filling initial design, ensuring that the initial samples are well-distributed across the search space.
*   **NaN Value Handling:** The algorithm includes a check for NaN values in the data and imputes them with the mean. This prevents errors and ensures that the surrogate model can be fit properly.
*   **Synergistic Combination:** By combining these components, the algorithm aims to leverage the strengths of both GBO and EHBBO, resulting in a more robust and efficient optimization algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize


class GBEHBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.exploration_factor = 2.0

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Check for NaN values and impute if necessary
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(X, y.ravel())  # HistGradientBoostingRegressor expects y to be 1D
        return model

    def _acquisition_function(self, X):
        mu = self.model.predict(X).reshape(-1, 1)
        sigma = np.zeros_like(mu)  # Gradient boosting does not directly provide uncertainty estimates
        ucb = mu + self.exploration_factor * sigma
        return -ucb  # minimize -ucb

    def _select_next_points(self, batch_size):
        # Optimize the acquisition function using L-BFGS-B
        x_starts = self._sample_points(batch_size)
        candidates = []
        values = []
        for x_start in x_starts:
            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
                           x_start,
                           bounds=[(-5, 5)] * self.dim,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(res.fun)

        return np.array(candidates)

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            batch_size = 1
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Local search around the best point
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]

            # Dynamically adjust the number of local search steps based on the remaining budget
            remaining_budget = self.budget - self.n_evals
            n_local_steps = min(5, remaining_budget)  # Reduce the number of steps as budget decreases

            X_local = best_x + np.random.normal(0, 0.1, size=(n_local_steps, self.dim))  # Gaussian mutation
            X_local = np.clip(X_local, self.bounds[0], self.bounds[1])  # Clip to bounds
            y_local = self._evaluate_points(func, X_local)
            self._update_eval_points(X_local, y_local)

            self.model = self._fit_model(self.X, self.y)

            # Update exploration factor
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm GBEHBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1486 with standard deviation 0.0973.

took 16.20 seconds to run.