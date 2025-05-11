# Description
**Ensemble Evolutionary Bayesian Optimization (EEBO):** This algorithm combines the strengths of ensemble modeling and evolutionary optimization within a Bayesian Optimization framework. It uses an ensemble of Gaussian Process Regression (GPR) models with different kernels to improve the robustness of the surrogate model. The next points are selected using a differential evolution (DE) strategy to optimize a hybrid acquisition function that balances exploration (Upper Confidence Bound - UCB) and exploitation (Expected Improvement - EI), and also incorporates a diversity term. The ensemble weights are dynamically adjusted based on their validation performance.

# Justification
This algorithm builds upon the SurrogateEnsembleBO and BayesianEvolutionaryBO by integrating their key components.

*   **Ensemble of Surrogate Models:** Using an ensemble of GPs with different kernels provides a more robust and accurate surrogate model compared to a single GP. This helps to handle the uncertainty associated with black-box functions more effectively. The dynamic weighting of the ensemble members based on validation performance allows the algorithm to prioritize models that are more accurate in the current search region.
*   **Differential Evolution for Acquisition Function Optimization:** Differential Evolution is used to efficiently optimize the acquisition function. DE is well-suited for non-convex and multi-modal optimization problems, which are common in Bayesian Optimization. This allows the algorithm to find better candidate points compared to simple random sampling or gradient-based optimization.
*   **Hybrid Acquisition Function:** Combining UCB, EI, and a diversity term in the acquisition function balances exploration and exploitation, preventing premature convergence to local optima. UCB encourages exploration of regions with high uncertainty, while EI focuses on exploiting promising regions. The diversity term ensures that the selected points are well-distributed in the search space.
*   **Computational Efficiency:** By limiting the number of DE iterations based on the remaining budget and using a relatively small population size, the algorithm maintains computational efficiency. The GP models are also updated periodically rather than at every iteration to reduce the computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution

class EnsembleEvolutionaryBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(5*dim, self.budget//10)

        # Do not add any other arguments without a default value
        self.gp_ensemble = []
        self.ensemble_weights = []
        self.best_x = None
        self.best_y = float('inf')
        self.n_ensemble = 3 # Number of surrogate models in the ensemble
        self.de_pop_size = 10 # Population size for differential evolution
        self.gp_update_interval = 5 # Update GP model every n iterations

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate models in the ensemble
        # return the models
        # Do not change the function signature

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if not self.gp_ensemble:
            # Initialize the ensemble with different kernels
            kernels = [
                ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=0.5),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
            ]
            for i in range(self.n_ensemble):
                gp = GaussianProcessRegressor(kernel=kernels[i % len(kernels)], n_restarts_optimizer=0, alpha=1e-6)
                gp.fit(X_train, y_train)
                self.gp_ensemble.append(gp)
                self.ensemble_weights.append(1.0 / self.n_ensemble)  # Initialize with equal weights
        else:
            # Update the existing models
            for gp in self.gp_ensemble:
                gp.fit(X_train, y_train)

        # Adjust weights based on validation performance
        val_errors = []
        for gp in self.gp_ensemble:
            y_pred, _ = gp.predict(X_val, return_std=True)
            error = np.mean((y_pred - y_val.flatten())**2)  # Mean Squared Error
            val_errors.append(error)

        # Convert errors to weights using softmax
        val_errors = np.array(val_errors)
        weights = np.exp(-val_errors) / np.sum(np.exp(-val_errors))
        self.ensemble_weights = weights

    def _acquisition_function(self, X):
        # Implement acquisition function: EI + UCB + Diversity
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if not self.gp_ensemble:
            return np.random.normal(size=(len(X), 1))
        else:
            mu_ensemble = np.zeros((len(X), 1))
            sigma_ensemble = np.zeros((len(X), 1))

            for i, gp in enumerate(self.gp_ensemble):
                mu, sigma = gp.predict(X, return_std=True)
                mu_ensemble += self.ensemble_weights[i] * mu.reshape(-1, 1)
                sigma_ensemble += self.ensemble_weights[i] * sigma.reshape(-1, 1)

            # EI calculation
            imp = self.best_y - mu_ensemble
            Z = imp / (sigma_ensemble + 1e-9)
            ei = imp * norm.cdf(Z) + sigma_ensemble * norm.pdf(Z)

            # UCB calculation
            kappa = 2.0  # Exploration-exploitation trade-off
            ucb = mu_ensemble + kappa * sigma_ensemble

            # Diversity term: encourage exploration
            if self.X is not None:
                distances = np.min([np.linalg.norm(x - self.X, axis=1) for x in X], axis=0)
                diversity = distances.reshape(-1,1)
            else:
                diversity = np.ones((len(X), 1)) # No diversity if no points yet

            # Combine EI, UCB and diversity
            acquisition = ei + 0.5 * ucb + 0.1 * diversity # Adjust weights as needed
            return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using differential evolution
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)

        # Define the objective function for differential evolution (negative acquisition function)
        def de_objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        # Perform differential evolution
        de_bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=max(1, self.budget//(batch_size*5)), tol=0.01, disp=False) # Adjust maxiter

        # Select the best point from differential evolution
        return result.x.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

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
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        batch_size = min(1, self.dim)
        iteration = 0
        while self.n_evals < self.budget:
            # Fit the GP model periodically
            if iteration % self.gp_update_interval == 0:
                self._fit_model(self.X, self.y)

            # Select points by acquisition function using differential evolution
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            iteration += 1

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<EnsembleEvolutionaryBO>", line 171, in __call__
 171->             next_X = self._select_next_points(batch_size)
  File "<EnsembleEvolutionaryBO>", line 122, in _select_next_points
 122->         result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=max(1, self.budget//(batch_size*5)), tol=0.01, disp=False) # Adjust maxiter
  File "<EnsembleEvolutionaryBO>", line 118, in de_objective
 118->             return -self._acquisition_function(x.reshape(1, -1))[0, 0]
  File "<EnsembleEvolutionaryBO>", line 86, in _acquisition_function
  84 | 
  85 |             for i, gp in enumerate(self.gp_ensemble):
  86->                 mu, sigma = gp.predict(X, return_std=True)
  87 |                 mu_ensemble += self.ensemble_weights[i] * mu.reshape(-1, 1)
  88 |                 sigma_ensemble += self.ensemble_weights[i] * sigma.reshape(-1, 1)
ValueError: Input X contains NaN.
GaussianProcessRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
