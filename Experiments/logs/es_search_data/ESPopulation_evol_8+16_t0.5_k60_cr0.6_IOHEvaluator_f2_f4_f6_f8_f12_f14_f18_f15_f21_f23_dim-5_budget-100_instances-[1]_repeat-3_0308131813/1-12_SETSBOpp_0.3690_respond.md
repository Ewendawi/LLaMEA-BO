# Description
**SETSBO++: Surrogate Ensemble with Thompson Sampling, Adaptive Local Search, and Kernel Tuning.** This enhanced version of SETSBO incorporates several improvements. First, it introduces an adaptive local search strategy that adjusts the number of L-BFGS-B iterations based on the agreement between the surrogate models and the actual function evaluations. Second, it tunes the kernel parameters of the Gaussian Process Regression models during the optimization process using a simple grid search. Finally, it uses a Sobol sequence for initial sampling to provide better space coverage.

# Justification
1.  **Adaptive Local Search:** The original SETSBO used a fixed number of iterations for local search, which might be inefficient. The adaptive local search adjusts the number of iterations based on the uncertainty (standard deviation) predicted by the GPR models. Higher uncertainty implies that the surrogate model is less confident, warranting more local search iterations. This adaptive approach balances exploitation and exploration more effectively.

2.  **Kernel Tuning:** The performance of GPR models is highly dependent on the choice of kernel parameters. Tuning these parameters during the optimization process can significantly improve the accuracy of the surrogate models. A simple grid search is used to select the best kernel parameters based on the marginal log-likelihood.

3.  **Sobol Sequence for Initial Sampling:** Latin Hypercube Sampling (LHS) is a good space-filling design, but Sobol sequences often provide better uniformity, especially in lower dimensions. Using a Sobol sequence for initial sampling can improve the initial exploration of the search space.

4. **Computational Efficiency:** The kernel tuning is performed every few iterations and the number of local search iterations is limited to avoid excessive computational overhead. Thompson sampling is already computationally efficient.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class SETSBOpp:
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
        self.kernels = []
        for i in range(self.n_models):
            length_scale = 1.0 * (i + 1) / self.n_models  # Varying length scales
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            self.kernels.append(kernel)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        # Initial Sobol sampling
        self.sobol_engine = qmc.Sobol(d=self.dim, scramble=True)

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sample = self.sobol_engine.random(n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        for i, model in enumerate(self.models):
            model.fit(X, y)

    def _tune_kernel(self, X, y, model_index):
        # Tune the kernel parameters using a grid search
        best_kernel = self.kernels[model_index]
        best_log_likelihood = -np.inf
        
        length_scale_candidates = np.logspace(-2, 2, 5)  # Example range, adjust as needed
        amplitude_candidates = np.logspace(-2, 2, 5)

        for length_scale in length_scale_candidates:
            for amplitude in amplitude_candidates:
                kernel = C(amplitude, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
                model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                try:
                    model.fit(X, y)
                    log_likelihood = model.log_marginal_likelihood()
                    if log_likelihood > best_log_likelihood:
                        best_log_likelihood = log_likelihood
                        best_kernel = kernel
                        self.models[model_index] = model # update the model
                        self.kernels[model_index] = kernel
                except:
                    pass # handle potential errors in kernel fitting
        return best_kernel

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

        # Thompson Sampling: Sample from the posterior distribution of each model
        sampled_values = np.zeros((X.shape[0], self.n_models))
        for i, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())

        # Average the sampled values across all models
        acquisition = np.mean(sampled_values, axis=1, keepdims=True)
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

        # Local search to improve the selected points
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in self.models])  # Minimize the average predicted value

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Adaptive local search iterations based on uncertainty
            std_devs = [model.predict(x0.reshape(1, -1), return_std=True)[1] for model in self.models]
            avg_std_dev = np.mean(std_devs)
            max_iter = min(10, max(1, int(5 / (1 + avg_std_dev))))  # Adjust iterations based on uncertainty

            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})  # Limited iterations
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

        kernel_tuning_interval = 5 # Tune the kernel every 5 iterations
        iteration = 0

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.n_init, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._fit_model(self.X, self.y)

            if iteration % kernel_tuning_interval == 0:
                for i in range(self.n_models):
                    self._tune_kernel(self.X, self.y, i)
                    self._fit_model(self.X, self.y) # refit after tuning
            iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm SETSBOpp got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1442 with standard deviation 0.0986.

took 103.27 seconds to run.