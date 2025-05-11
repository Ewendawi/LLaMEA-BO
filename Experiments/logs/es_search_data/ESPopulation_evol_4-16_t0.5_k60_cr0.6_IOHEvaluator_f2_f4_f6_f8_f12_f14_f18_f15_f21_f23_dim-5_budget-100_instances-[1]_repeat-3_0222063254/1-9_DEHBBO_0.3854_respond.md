# Description
DEHBBO with Adaptive Diversity Threshold and Hall of Fame Management. This enhanced version of DEHBBO improves diversity maintenance and hall of fame usage. The diversity threshold is dynamically adjusted based on the GP's uncertainty. This makes the algorithm more adaptive to the problem landscape. Additionally, the hall of fame update strategy is refined to prioritize members that not only perform well but are also diverse, thereby encouraging exploration in promising but less-explored regions. Finally, the L-BFGS-B is run multiple times with different initial points, improving the exploitation capability.

# Justification
The key improvements focus on making the diversity mechanism more adaptive and effective.

1.  **Adaptive Diversity Threshold:** Instead of a fixed threshold, the algorithm adjusts the diversity threshold based on the GP's uncertainty (sigma). This allows for more aggressive exploration in regions where the GP is uncertain and more focused exploitation in well-modeled regions. This adaptation is crucial for balancing exploration and exploitation effectively.

2.  **Hall of Fame Distance-Value Tradeoff:** The Hall of Fame update now considers both the performance (y value) and the distances to existing members, to avoid the Hall of Fame to be filled by similar good solutions. This is done by combining the normalized distance and the normalized performance.

3.  **Multiple L-BFGS-B runs:** By running L-BFGS-B multiple times, each with a different starting point, and then selecting the best result, the algorithm is more likely to find the global optimum of the acquisition function within the defined bounds.

4.  **Minimum Hall of Fame size**: Enforcing a minimum size prevents the Hall of Fame from becoming empty when the budget is small.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler


class DEHBBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(20 * dim, self.budget // 5) # Increased initial exploration
        self.best_y = float('inf')
        self.best_x = None
        self.hall_of_fame_X = []
        self.hall_of_fame_y = []
        self.hall_of_fame_size = max(5, dim // 2)  # Hall of Fame size
        self.diversity_threshold_initial = 0.5 # Initial minimum distance for diversity
        self.diversity_threshold = self.diversity_threshold_initial

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points
        # return array of shape (n_points, n_dims)
        if center is None:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            points = sampler.random(n=n_points)
            return qmc.scale(points, self.bounds[0], self.bounds[1])
        else:
            # Sample within a ball around center with radius
            points = np.random.normal(loc=center, scale=radius/3, size=(n_points, self.dim))
            points = np.clip(points, self.bounds[0], self.bounds[1])
            return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        imp = self.best_y - mu - 1e-9  # Adding a small constant to avoid division by zero
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Diversity penalty
        if self.hall_of_fame_X:
            distances = np.array([np.linalg.norm(X - hof_x, axis=1) for hof_x in self.hall_of_fame_X]).T
            min_distances = np.min(distances, axis=1, keepdims=True)
            diversity_penalty = np.where(min_distances < self.diversity_threshold, -100, 0)  # Penalize close points
            ei += diversity_penalty
        return ei

    def _select_next_points(self, batch_size, gp):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimization of acquisition function using L-BFGS-B
        num_lbfgs_starts = 5
        x_next = []
        best_acq = float('inf')
        best_x = None

        for _ in range(num_lbfgs_starts):
            x_start = self._sample_points(1)[0]
            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1), gp),
                           x_start,
                           bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                           method='L-BFGS-B')
            if res.success:
                acq_value = -res.fun
                if acq_value < best_acq:
                    best_acq = acq_value
                    best_x = res.x
        
        if best_x is not None:
            x_next.append(best_x)

        # Random sampling around the best point
        num_random_samples = max(0, batch_size - len(x_next))  # Ensure non-negative
        if self.best_x is not None:
            random_samples = self._sample_points(num_random_samples, center=self.best_x, radius=0.5)
            x_next.extend(random_samples)
        else:
            #If best_x is none, sample randomly
            random_samples = self._sample_points(num_random_samples)
            x_next.extend(random_samples)


        return np.array(x_next)

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
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

        # Update Hall of Fame
        self._update_hall_of_fame(self.best_x, self.best_y)
    
    def _update_hall_of_fame(self, x, y):
        if not self.hall_of_fame_X:
            self.hall_of_fame_X.append(x)
            self.hall_of_fame_y.append(y)
        else:
            distances = np.array([np.linalg.norm(x - hof_x) for hof_x in self.hall_of_fame_X])
            min_distance = np.min(distances)

            # Normalize distance and performance for combined score
            normalized_distance = min_distance / (np.linalg.norm(self.bounds[1] - self.bounds[0])) #Scaled between 0 and 1
            normalized_performance = (y - np.min(self.y)) / (np.max(self.y) - np.min(self.y)) if len(self.y) > 1 else 0 #Scale between 0 and 1

            # Combine distance and performance
            combined_score = normalized_distance - normalized_performance #High distance and low performance is good

            if len(self.hall_of_fame_X) < self.hall_of_fame_size or combined_score > 0:
                if len(self.hall_of_fame_X) == self.hall_of_fame_size:
                    # Remove worst performing member
                    distances_hof = np.array([np.linalg.norm(hof_x - x) for hof_x in self.hall_of_fame_X])
                    normalized_distances_hof = distances_hof / (np.linalg.norm(self.bounds[1] - self.bounds[0]))
                    normalized_performance_hof = (np.array(self.hall_of_fame_y) - np.min(self.y)) / (np.max(self.y) - np.min(self.y)) if len(self.y) > 1 else np.zeros(len(self.hall_of_fame_y))

                    combined_scores_hof = normalized_distances_hof - normalized_performance_hof

                    worst_idx = np.argmin(combined_scores_hof)
                    self.hall_of_fame_X.pop(worst_idx)
                    self.hall_of_fame_y.pop(worst_idx)

                self.hall_of_fame_X.append(x)
                self.hall_of_fame_y.append(y)
            
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)

            # Dynamic diversity threshold
            _, sigma = gp.predict(self.X, return_std=True)
            self.diversity_threshold = self.diversity_threshold_initial * (1 - np.mean(sigma) / np.std(self.y)) if np.std(self.y) > 0 else self.diversity_threshold_initial
            self.diversity_threshold = np.clip(self.diversity_threshold, 0.1, self.diversity_threshold_initial) #Ensure it stays in reasonable bounds


            # Dynamic batch size
            remaining_evals = self.budget - self.n_evals
            batch_size = min(max(1, int(remaining_evals / (self.dim * 0.1))), 20) # Ensure at least 1 point and limit to 20

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, gp)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```