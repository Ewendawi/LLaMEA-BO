from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split

class ADHyTSBO:
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
        self.kde_bandwidth = 0.5 / np.sqrt(dim)  # Initial bandwidth for KDE
        self.batch_size = min(10, dim)
        self.exploration_weight = 0.1 # Initial exploration weight
        self.exploration_weight_min = 0.01 # Minimum exploration weight
        self.gpr_ensemble_size = 3

        self.models = []
        for _ in range(self.gpr_ensemble_size):
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate models
        # return the models
        # Do not change the function signature

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        for model in self.models:
            model.fit(X_train, y_train)

            # Evaluate the model on the validation set
            validation_score = model.log_marginal_likelihood(model.kernel_.theta)

            # Optionally, adjust kernel parameters based on the validation score
            # This can help to improve the model's generalization performance
            # For example, you could use a gradient-based optimization method to update the kernel parameters
            # based on the validation score.
        return self.models

    def _acquisition_function(self, X, model):
        # Implement acquisition function using Thompson Sampling
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None and len(self.X) > 0:
            min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones(X.shape[0])[:, None]

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Thompson Sampling and KDE
        # return array of shape (batch_size, n_dims)

        if self.X is None or len(self.X) < self.dim + 1:
            # Not enough data for KDE, return random samples
            return self._sample_points(batch_size)

        # Fit KDE to the evaluated points
        kde = KernelDensity(bandwidth=self.kde_bandwidth).fit(self.X)

        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)

        # Calculate KDE scores for candidate points
        kde_scores = kde.score_samples(candidate_points)

        # Select top candidate points based on KDE scores
        top_indices = np.argsort(kde_scores)[-batch_size:]
        candidate_next_points = candidate_points[top_indices]

        # Refine selection using Thompson Sampling
        next_points = []
        for _ in range(batch_size):
            # Thompson Sampling: Sample from the posterior distribution of each GPR model
            sampled_values = []
            for model in self.models:
                mu, sigma = model.predict(candidate_next_points, return_std=True)
                sampled_value = np.random.normal(mu, sigma)
                sampled_values.append(sampled_value)

            # Average the sampled values from all GPR models
            averaged_sampled_values = np.mean(sampled_values, axis=0)

            # Select the point with the highest averaged sampled value
            best_index = np.argmax(averaged_sampled_values)
            next_points.append(candidate_next_points[best_index])

        return np.array(next_points)

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

        self.models = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.batch_size, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.models = self._fit_model(self.X, self.y)

            # Adaptive adjustment of KDE bandwidth and exploration weight
            self.kde_bandwidth = max(0.1 / np.sqrt(self.dim), self.kde_bandwidth * 0.95)  # Reduce bandwidth
            self.exploration_weight = max(self.exploration_weight_min, self.exploration_weight * 0.95) # Reduce exploration

        return self.best_y, self.best_x
