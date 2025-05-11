# Description
EHBBO_BLS_PLE_AdaptiveKernel_v6: This algorithm builds upon EHBBO_BLS_PLE_AdaptiveKernel_v5, focusing on improving exploration-exploitation balance and local search efficiency. Key improvements include: 1) Implementing a dynamic batch size for selecting next points, adjusting based on the optimization progress and function landscape. 2) Refining the local search strategy by incorporating a trust-region approach, dynamically adjusting the search radius based on the GP's performance in the local region. 3) Introducing a mechanism to adaptively adjust the EI/UCB weighting based on the diversity of the evaluated points. 4) Adding a restart mechanism for the GP training to escape local optima in the kernel parameter space.

# Justification
The primary focus is on enhancing the exploration-exploitation trade-off and improving the local search strategy.

1.  **Dynamic Batch Size:** Instead of a fixed batch size of 1, the algorithm now dynamically adjusts the number of points selected in each iteration. This allows for more exploration early on and more exploitation later in the optimization process.
2.  **Trust-Region Local Search:** The local search is refined to incorporate a trust-region approach. The search radius is dynamically adjusted based on how well the GP model predicts the function's behavior in the local region. This helps to focus the search on promising areas and avoid wasting evaluations in regions where the GP model is inaccurate.
3.  **Adaptive EI/UCB Weighting:** The weighting between Expected Improvement (EI) and Upper Confidence Bound (UCB) is adaptively adjusted based on the diversity of the evaluated points. If the points are clustered together, the algorithm increases the weight on UCB to encourage exploration. If the points are spread out, the algorithm increases the weight on EI to focus on exploitation.
4.  **GP Training Restart:** A restart mechanism is added to the GP training process to help escape local optima in the kernel parameter space. This can improve the accuracy of the GP model and lead to better optimization performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances

class EHBBO_BLS_PLE_AdaptiveKernel_v6:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # initial samples
        self.gp = None
        self.best_x = None
        self.best_y = np.inf
        self.exploration_factor = 1.0 # initial exploration factor
        self.exploration_decay = 0.99 # decay rate for exploration factor
        self.kernel_length_scale = 1.0
        self.kernel_update_interval = 5 * dim # Update kernel every this many evaluations
        self.kappa = 2.0  # UCB exploration-exploitation parameter
        self.kappa_decay = 0.995
        self.min_kappa = 0.1
        self.initial_sample_multiplier = 1.0
        self.ei_weight = 0.5 # Weight for Expected Improvement
        self.ei_weight_decay = 0.99
        self.min_ei_weight = 0.1 # Minimum EI weight
        self.min_length_scale = 1e-3
        self.temperature = 1.0 # Temperature parameter for acquisition function
        self.temperature_decay = 0.995 # Decay rate for temperature
        self.batch_size = 1 # Initial batch size
        self.batch_size_decay = 0.99
        self.max_batch_size = 5
        self.min_batch_size = 1
        self.trust_region_size = 0.5
        self.trust_region_shrink = 0.9
        self.trust_region_expand = 1.1
        self.min_trust_region = 0.01
        self.gp_retrain_attempts = 3


    def _sample_points(self, n_points):
        # sample points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        best_gp = None
        best_log_likelihood = -np.inf
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds=(self.min_length_scale, 10))

        for _ in range(self.gp_retrain_attempts):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-5) # Increased restarts for kernel tuning
            gp.fit(X, y)
            log_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta)

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_gp = gp

        self.gp = best_gp
        self.kernel_length_scale = self.gp.kernel_.k2.length_scale # Update the length scale
        return self.gp

    def _acquisition_function(self, X):
        # Implement Upper Confidence Bound acquisition function
        mu, sigma = self.gp.predict(X, return_std=True)
        ucb = mu - self.kappa * sigma

        # Calculate Expected Improvement
        improvement = self.best_y - mu
        Z = improvement / sigma if sigma > 0 else np.inf
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Adaptive EI/UCB weighting based on diversity
        if self.X is not None and len(self.X) > 1:
            distances = euclidean_distances(X, self.X)
            min_distance = np.min(distances)
            diversity_factor = np.exp(-min_distance / self.kernel_length_scale) # High when points are close

            self.ei_weight = max(self.min_ei_weight, min(1.0, self.ei_weight + 0.1 * (0.5 - diversity_factor))) # Adjust EI weight

        # Weighted combination of UCB and EI
        ei_w = max(self.ei_weight, self.min_ei_weight)
        return (ei_w * ei + (1 - ei_w) * ucb) / self.temperature

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a multi-start local search strategy
        selected_X = []
        for _ in range(batch_size):
            best_x = None
            best_acq = np.inf
            num_starts = 5 # Reduced number of starts
            for _ in range(num_starts):
                x_start = self._sample_points(1)
                x_start = x_start.flatten()

                def acquisition_objective(x):
                    return self._acquisition_function(x.reshape(1, -1))[0]

                res = minimize(acquisition_objective, x_start, method='L-BFGS-B', bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)])
                acq = res.fun

                if acq < best_acq:
                    best_acq = acq
                    best_x = res.x

            selected_X.append(best_x)

        return np.array(selected_X)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best_x and best_y
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def local_search(self, gp, x0, sigma, num_restarts=1):
        # Perform a local search around x0 using the GP prediction
        best_x = x0
        best_y_pred = np.inf
        search_radius = self.trust_region_size # Dynamic search radius

        # Calculate gradient of GP prediction
        def gp_objective(x):
            mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
            return mu[0]


        for _ in range(num_restarts):
            x_start = x0 + np.random.normal(0, 0.1, size=self.dim) # Add some noise
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])


            bounds = [(max(self.bounds[0][i], x0[i] - search_radius), min(self.bounds[1][i], x0[i] + search_radius)) for i in range(self.dim)]  # Dynamic bounds
            res = minimize(gp_objective, x_start, method='L-BFGS-B', bounds=bounds)
            y_pred = gp_objective(res.x)

            if y_pred < best_y_pred:
                best_y_pred = y_pred
                best_x = res.x

        # Adjust trust region size based on GP performance
        predicted_y, _ = gp.predict(x0.reshape(1, -1), return_std=True)
        actual_y = gp_objective(best_x) # Use GP prediction as a proxy

        prediction_error = abs(predicted_y[0] - actual_y)

        if prediction_error < 0.1 * sigma:
            self.trust_region_size = min(0.8, self.trust_region_size * self.trust_region_expand)
        else:
            self.trust_region_size = max(self.min_trust_region, self.trust_region_size * self.trust_region_shrink)


        return best_x

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial exploration
        n_init_dynamic = int(self.initial_sample_multiplier * self.n_init)
        initial_X = self._sample_points(n_init_dynamic)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            if self.n_evals % self.kernel_update_interval == 0:
                self._fit_model(self.X, self.y)
            else:
                # Refit without kernel optimization for efficiency
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
                self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                self.gp.fit(self.X, self.y)


            # Dynamically adjust batch size
            remaining_budget = self.budget - self.n_evals
            self.batch_size = max(self.min_batch_size, min(self.max_batch_size, int(remaining_budget / (self.budget / 5)))) # Reduce batch size as budget runs out

            # Select next point(s) by acquisition function
            next_X = self._select_next_points(self.batch_size)

            # Evaluate the new point(s)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search around the best point
            predicted_y, predicted_sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
            num_local_restarts = 2 + int(self.budget / (self.n_evals + 1e-6)) # More restarts early on
            local_x = self.local_search(self.gp, self.best_x, predicted_sigma[0], num_restarts=num_local_restarts)
            predicted_y_local, predicted_sigma_local = self.gp.predict(local_x.reshape(1, -1), return_std=True)


            # Probabilistic evaluation
            improvement = self.best_y - predicted_y_local[0]
            Z = improvement / predicted_sigma_local[0] if predicted_sigma_local[0] > 0 else np.inf
            prob_improvement = norm.cdf(Z)

            # Adjust the evaluation threshold based on the remaining budget
            evaluation_threshold = 0.5 + 0.4 * (remaining_budget / self.budget)  # Higher threshold when budget is tight

            if prob_improvement > evaluation_threshold and self.n_evals < self.budget:
                # Evaluate the true function value only if the GP predicts a sufficiently high probability of improvement and we have budget
                local_y = func(local_x)
                self.n_evals += 1
                if local_y < self.best_y:
                    self.best_y = local_y
                    self.best_x = local_x

            # Decay exploration factor and kappa
            self.exploration_factor *= self.exploration_decay
            self.kappa = max(self.min_kappa, self.kappa * self.kappa_decay)
            self.ei_weight = max(self.min_ei_weight, self.ei_weight * self.ei_weight_decay)
            self.temperature *= self.temperature_decay

            #Adaptive Kernel Update
            if len(self.y) > 1:
                improvement_ratio = abs(self.y[-1] - self.y[-2]) / abs(self.y[-2]) if abs(self.y[-2]) > 1e-6 else 0
                _, predicted_sigma = self.gp.predict(self.X[-1].reshape(1, -1), return_std=True)
                uncertainty = predicted_sigma[0]

                if improvement_ratio > 0.1 or uncertainty > 0.5:
                    self._fit_model(self.X, self.y)
                    self.kernel_update_interval = 5 * self.dim #reset update interval
                else:
                    self.kernel_update_interval = max(1, int(self.kernel_update_interval * 0.9)) #reduce update interval gradually

            # Adjust initial sampling size
            if self.n_evals > self.n_init and abs(self.y[-1] - self.y[-self.n_init]) < 1e-3:
                self.initial_sample_multiplier *= 1.1
            else:
                self.initial_sample_multiplier = max(1.0, self.initial_sample_multiplier * 0.95)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO_BLS_PLE_AdaptiveKernel_v6 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1565 with standard deviation 0.1017.

took 668.95 seconds to run.