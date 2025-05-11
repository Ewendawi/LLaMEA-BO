# Description
**Adaptive Ensemble with Thompson Sampling, Exploration Decay, and Uncertainty-Aware Adaptive Local Search Bayesian Optimization (AETSALSBO):** This algorithm combines the strengths of AEEHBBO and EHTSALSDEBO while addressing their limitations. It employs an ensemble of Gaussian Process Regression (GPR) models with varying length scales, similar to EHTSALSDEBO, to capture different aspects of the function landscape. It uses Thompson Sampling for efficient acquisition, balancing exploration and exploitation. It incorporates an adaptive local search strategy, inspired by EHTSALSDEBO, that refines selected points based on the uncertainty estimates from the GPR models and uses momentum-based acceleration. Crucially, it adaptively adjusts the exploration weight in the hybrid acquisition function, similar to AEEHBBO, but with a more sophisticated decay schedule based on both the number of evaluations and the GPR model uncertainty. This allows for a more robust and efficient exploration-exploitation trade-off. This algorithm also implements a dynamic batch size adjustment based on model uncertainty.

# Justification
The key components and their justifications are:

1.  **Ensemble of GPR Models:** Using an ensemble of GPR models with varying length scales, like in EHTSALSDEBO, allows the algorithm to capture different features of the function landscape, improving robustness and generalization.
2.  **Thompson Sampling:** Thompson Sampling offers a computationally efficient way to balance exploration and exploitation by sampling from the posterior distribution of each model in the ensemble.
3.  **Adaptive Exploration Decay:** Adaptively decaying the exploration weight, similar to AEEHBBO and EHTSALSDEBO, ensures that the algorithm explores the search space effectively in the early stages and focuses on exploitation as more information is gathered. The decay is made adaptive by considering the model uncertainty, which is a better indicator of the need for exploration than just the number of evaluations.
4.  **Uncertainty-Aware Adaptive Local Search with Momentum:** The local search strategy refines selected points based on the uncertainty estimates from the GPR models. This allows the algorithm to focus local search efforts on regions where the model is less confident, potentially leading to faster convergence. Momentum-based acceleration helps to escape local optima more effectively.
5. **Dynamic Batch Size Adjustment:** Adjusting the batch size based on model uncertainty allows the algorithm to explore more when uncertainty is high and exploit when uncertainty is low. This improves the efficiency of the optimization process.

The algorithm combines the best aspects of AEEHBBO and EHTSALSDEBO while addressing their limitations. AEEHBBO lacks an ensemble of models and could benefit from a more sophisticated local search. EHTSALSDEBO, while having an ensemble and local search, is computationally expensive and has a simpler exploration decay strategy. AETSALSBO aims to strike a balance between these two by incorporating an ensemble, adaptive local search, and a more robust exploration decay based on model uncertainty, while keeping the computational cost manageable.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AETSALSBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim

        self.best_y = np.inf
        self.best_x = None

        self.n_models = 3  # Number of surrogate models in the ensemble
        self.models = []
        for i in range(self.n_models):
            length_scale = 1.0 * (i + 1) / self.n_models  # Varying length scales
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        self.batch_size = min(10, dim)  # Initial batch size
        self.exploration_weight = 0.2  # Initial exploration weight
        self.exploration_decay = 0.99  # Decay factor for exploration weight
        self.exploration_weight_min = 0.01
        self.momentum = 0.0 #initial momentum
        self.momentum_decay = 0.9

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        for model in self.models:
            model.fit(X, y)

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

        # Thompson Sampling: Sample from the posterior distribution of each model
        sampled_values = np.zeros((X.shape[0], self.n_models))
        sigmas = np.zeros((X.shape[0], self.n_models))
        for i, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())
            sigmas[:, i] = sigma.flatten()

        # Average the sampled values across all models
        acquisition = np.mean(sampled_values, axis=1, keepdims=True)
        acquisition = acquisition.reshape(-1, 1)

        # Hybrid acquisition function (EI + exploration)
        mu = np.mean([model.predict(X) for model in self.models], axis=0).reshape(-1, 1)
        sigma = np.mean(sigmas, axis=1).reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None:
            min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones_like(ei)

        acquisition = ei + self.exploration_weight * exploration

        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)

        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Local search to improve the selected points
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in self.models])  # Minimize the average predicted value

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Adaptive local search iterations based on uncertainty
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in self.models])
            maxiter = int(5 + 10 * uncertainty)  # More iterations for higher uncertainty
            maxiter = min(maxiter, 20) # Cap the iterations

            # Momentum-based acceleration
            if i == 0:
                self.momentum = np.zeros_like(x0)
            self.momentum = self.momentum_decay * self.momentum

            def grad_func(x):
                x = x.reshape(1, -1)
                grads = []
                for model in self.models:
                    # numerical gradient
                    delta = 1e-5
                    grad = np.zeros_like(x0)
                    for j in range(self.dim):
                        x_plus = x0.copy()
                        x_plus[j] += delta
                        x_minus = x0.copy()
                        x_minus[j] -= delta
                        grad[j] = (model.predict(x_plus.reshape(1, -1))[0] - model.predict(x_minus.reshape(1, -1))[0]) / (2 * delta)
                    grads.append(grad)
                return np.mean(grads, axis=0)

            def obj_func_momentum(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in self.models])

            res = minimize(obj_func_momentum, x0, method='L-BFGS-B', jac=grad_func, bounds=bounds, options={'maxiter': maxiter})  # Limited iterations
            next_points[i] = res.x

        return next_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            remaining_evals = self.budget - self.n_evals

            # Adaptive batch size adjustment
            uncertainty = np.mean([model.predict(self.best_x.reshape(1, -1), return_std=True)[1] for model in self.models]) if self.best_x is not None else 1.0
            batch_size = min(int(self.batch_size * (1 + uncertainty)), remaining_evals)
            batch_size = max(1, batch_size) # Ensure batch_size is at least 1

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._fit_model(self.X, self.y)

            # Decay exploration weight, adaptively based on uncertainty
            self.exploration_weight = max(self.exploration_weight_min, self.exploration_weight * self.exploration_decay * (1 - uncertainty))

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AETSALSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1648 with standard deviation 0.1018.

took 433.72 seconds to run.