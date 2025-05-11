from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

class ABTSLSAAWO_DRSFDAEIWUKA_BO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim  # Initial number of points
        self.batch_size = min(10, dim) # Initial batch size, adaptively adjusted
        self.gp = None
        self.best_x = None
        self.best_y = np.inf
        self.acquisition_weight = 2.0 # Initial acquisition weight
        self.local_search_prob = 0.9  # Probability of performing local search
        self.diversification_prob = 0.05 # Probability of performing local search around a random point
        self.radius_scale = 1.0 # Initial radius scale
        self.ei_weight = 0.2 # Initial EI weight
        self.ei_weight_initial = 0.2 # Initial EI weight
        self.ei_weight_decay = 0.98 # Decay factor for EI weight when making progress
        self.ei_weight_increase = 1.02 # Increase factor for EI weight when stagnating
        self.previous_best_y = np.inf
        self.uncertainty_scale = np.sqrt(dim) # Scale factor for uncertainty, adapted to dimension
        self.kernel_lengthscale = 1.0 # Initial kernel lengthscale
        self.kernel_lengthscale_min = 1e-2
        self.kernel_lengthscale_max = 1e2
        self.kernel_lengthscale_decay = 0.95
        self.kernel_lengthscale_increase = 1.05

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(self.kernel_lengthscale, (self.kernel_lengthscale_min, self.kernel_lengthscale_max))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.zeros((len(X), 1))  # Return zeros if the model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # UCB acquisition function
        ucb = mu + self.acquisition_weight * sigma

        # Expected Improvement acquisition function
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # Handle cases with very small sigma

        # Combine UCB and EI
        return (1 - self.ei_weight) * ucb + self.ei_weight * ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points
        candidate_points = self._sample_points(10 * batch_size)
        
        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)
        
        # Select the points with the highest acquisition function values
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
        
        return candidate_points[indices]

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
        
        # Update best solution
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _local_search(self, func, x, num_points=3):
        # Perform local search around x using the surrogate model
        # Adapt radius based on GP's uncertainty and optimization progress
        if self.gp is None:
            radius = 0.1  # Default radius if GP is not fitted
        else:
            _, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            radius = max(0.01, sigma[0]) * self.radius_scale * np.sqrt(self.dim)  # Dimensionality-aware radius scaling
        
        # Sample points around x
        X = np.random.uniform(low=np.maximum(self.bounds[0], x - radius),
                                high=np.minimum(self.bounds[1], x + radius),
                                size=(num_points, self.dim))
        
        # Predict the values using the Gaussian Process model
        if self.gp is None:
            predicted_y = np.zeros((num_points, 1))
        else:
            predicted_y, _ = self.gp.predict(X, return_std=True)
            predicted_y = predicted_y.reshape(-1, 1)
        
        # Find the best point based on the predicted value
        best_index = np.argmin(predicted_y)
        best_x_candidate = X[best_index]

        # Evaluate the best candidate point using the real function, if budget allows
        if self.n_evals < self.budget:
            best_y_candidate = self._evaluate_points(func, best_x_candidate.reshape(1, -1))[0, 0]
            return best_x_candidate, best_y_candidate
        else:
            return x, self.best_y  # Return current best if no budget left
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.previous_best_y = self.best_y
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the model
            self._fit_model(self.X, self.y)
            
            # Adapt batch size based on uncertainty
            if self.gp is not None:
                _, sigma = self.gp.predict(self.X, return_std=True)
                avg_sigma = np.mean(sigma)
                # Scale the batch size based on the average uncertainty
                self.batch_size = max(1, min(int(self.budget/10), int(10 * avg_sigma * self.uncertainty_scale))) # Adjust batch size based on uncertainty
                self.batch_size = min(self.batch_size, self.dim) # Cap by dimension
            
            # Select next points
            X_next = self._select_next_points(self.batch_size)
            
            # Evaluate points
            y_next = self._evaluate_points(func, X_next)
            
            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Local search
            if np.random.rand() < self.local_search_prob:
                # Diversification: local search around a random point
                if np.random.rand() < self.diversification_prob and self.X is not None and len(self.X) > 0:
                    # Focused diversification: select point with highest uncertainty
                    _, sigma = self.gp.predict(self.X, return_std=True)
                    random_index = np.argmax(sigma)
                    x_local_search = self.X[random_index]
                else:
                    x_local_search = self.best_x

                best_x_local, best_y_local = self._local_search(func, x_local_search)
                if best_y_local < self.best_y:
                    self.best_x = best_x_local
                    self.best_y = best_y_local
        
            # Adapt acquisition weight
            self.acquisition_weight = max(0.1, self.acquisition_weight * (0.98 + 0.02 * (self.acquisition_weight/2.0)))  # Reduce exploration over time, slower decay if initial weight is high
            self.local_search_prob = max(0.1, self.local_search_prob * 0.98)
            self.radius_scale = max(0.01, self.radius_scale * 0.98) # Reduce radius scale over time

            # Adapt EI weight
            if self.gp is not None:
                _, sigma = self.gp.predict(self.X, return_std=True)
                avg_sigma = np.mean(sigma)

                # Adjust EI weight based on progress and uncertainty
                if self.best_y < self.previous_best_y:
                    self.ei_weight = max(0.01, self.ei_weight * self.ei_weight_decay)  # Reduce EI if making progress
                else:
                    self.ei_weight = min(0.99, self.ei_weight * self.ei_weight_increase) # Increase EI if stagnating

                # Further adjust EI based on uncertainty
                self.ei_weight = np.clip(self.ei_weight + 0.1 * (avg_sigma - 0.1), 0.01, 0.99) # Increase EI if uncertainty is high

            # Adapt kernel lengthscale
            if self.best_y < self.previous_best_y:
                self.kernel_lengthscale = max(self.kernel_lengthscale_min, self.kernel_lengthscale * self.kernel_lengthscale_decay)
            else:
                self.kernel_lengthscale = min(self.kernel_lengthscale_max, self.kernel_lengthscale * self.kernel_lengthscale_increase)

            self.previous_best_y = self.best_y

            if self.n_evals >= self.budget:
                break

        return self.best_y, self.best_x
