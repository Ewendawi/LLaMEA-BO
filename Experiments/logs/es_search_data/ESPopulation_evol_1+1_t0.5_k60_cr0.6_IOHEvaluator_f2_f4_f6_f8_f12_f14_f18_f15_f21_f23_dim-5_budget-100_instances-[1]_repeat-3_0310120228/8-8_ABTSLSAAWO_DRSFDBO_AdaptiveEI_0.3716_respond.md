# Description
**ABTSLSAAWO_DRSFDBO_AdaptiveEI**: Adaptive Batch Bayesian Optimization with Thompson Sampling, Improved Local Search, Adaptive Acquisition Weighting, Dynamic Radius Scaling, Focused Diversification, and **Adaptive EI Weight**. This algorithm builds upon ABTSLSAAWO_DRSFDBO by introducing an adaptive strategy for the EI weight in the acquisition function. The EI weight is dynamically adjusted based on the optimization progress, specifically focusing on the ratio of exploration to exploitation. This is achieved by monitoring the variance predicted by the GP model. If the variance is high, indicating unexplored regions, the EI weight is increased to promote exploration. Conversely, if the variance is low, the EI weight is decreased to favor exploitation.

# Justification
The key idea is to balance exploration and exploitation more effectively. The original algorithm uses a fixed EI weight, which might not be optimal throughout the optimization process. By adaptively adjusting the EI weight based on the GP's predicted variance, the algorithm can dynamically shift its focus between exploring uncertain regions and exploiting promising areas. This adaptive strategy is expected to improve the algorithm's performance, especially in complex and high-dimensional landscapes.

The choice of using the GP's predicted variance as a proxy for exploration/exploitation is justified by the fact that high variance indicates regions where the model is uncertain, thus requiring more exploration. Conversely, low variance suggests that the model is confident in its predictions, making exploitation more appropriate.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

class ABTSLSAAWO_DRSFDBO_AdaptiveEI:
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
        self.ei_weight = 0.2 # Initial weight for Expected Improvement in acquisition function
        self.ei_weight_decay = 0.99 # Decay factor for EI weight
        self.ei_weight_min = 0.01 # Minimum value for EI weight

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
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
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the model
            self._fit_model(self.X, self.y)
            
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
        
            # Adapt batch size
            self.batch_size = max(1, min(int(self.budget/10), int(self.batch_size * 0.95))) # Reduce batch size gradually, but not too small

            # Adapt acquisition weight
            self.acquisition_weight = max(0.1, self.acquisition_weight * (0.98 + 0.02 * (self.acquisition_weight/2.0)))  # Reduce exploration over time, slower decay if initial weight is high
            self.local_search_prob = max(0.1, self.local_search_prob * 0.98)
            self.radius_scale = max(0.01, self.radius_scale * 0.98) # Reduce radius scale over time

            # Adapt EI weight
            mean_sigma = np.mean(self.gp.predict(self.X, return_std=True)[1]) if self.gp is not None and self.X is not None else 0.1
            if mean_sigma > 0.1: #if uncertainty is high
                self.ei_weight = min(0.5, self.ei_weight * 1.02) #increase EI weight
            else:
                self.ei_weight = max(self.ei_weight_min, self.ei_weight * self.ei_weight_decay) #decrease EI weight

            if self.n_evals >= self.budget:
                break

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ABTSLSAAWO_DRSFDBO_AdaptiveEI got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1490 with standard deviation 0.0987.

took 78.31 seconds to run.