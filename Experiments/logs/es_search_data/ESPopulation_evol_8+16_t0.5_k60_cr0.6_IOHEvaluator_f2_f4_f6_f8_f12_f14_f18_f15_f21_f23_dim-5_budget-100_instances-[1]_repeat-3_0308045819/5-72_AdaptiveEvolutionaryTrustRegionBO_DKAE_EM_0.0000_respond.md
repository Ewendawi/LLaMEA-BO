# Description
**AdaptiveEvolutionaryTrustRegionBO with Dynamic Kappa, Adaptive DE Iterations, Enhanced Trust Region Adaptation, and Ensemble Modeling (AETRBO-DKAE-EM):** This algorithm enhances AETRBO-DKAE by incorporating an ensemble of Gaussian Process (GP) models to improve the robustness and accuracy of the surrogate model. The ensemble consists of GPs with different kernel length scales, and their predictions are combined using a weighted average based on their performance on a validation set. The trust region adaptation strategy is also refined by considering the variance of the ensemble predictions. This allows for more reliable trust region adjustments, especially in regions where the GP models disagree. The algorithm retains the dynamic kappa adjustment, adaptive DE iterations, and dimension-aware initial trust region radius from AETRBO-DKAE.

# Justification
The AETRBO-DKAE algorithm shows promising results, but its performance can be further improved by addressing the limitations of using a single GP model. An ensemble of GPs can provide more robust uncertainty estimates and improve the accuracy of the surrogate model, especially in complex or noisy landscapes. By incorporating the variance of the ensemble predictions into the trust region adaptation, the algorithm can make more informed decisions about exploration and exploitation. The dynamic kernel selection of AdaptiveEvolutionaryParetoTrustRegionBO is replaced by a GP ensemble, which is more computationally efficient. The validation set is created by splitting the existing data, which doesn't require any additional function evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split

class AdaptiveEvolutionaryTrustRegionBO_DKAE_EM:
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
        self.gp_weights = []
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0 * np.sqrt(dim) # Initial trust region radius, scaled by dimension
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0 #ratio to track the success of trust region
        self.random_restart_prob = 0.05
        self.de_pop_size = 10  # Population size for differential evolution
        self.lcb_kappa = 2.0  # Kappa parameter for Lower Confidence Bound
        self.kappa_decay = 0.99 # Decay rate for kappa
        self.min_kappa = 0.1 # Minimum value for kappa
        self.gp_error_threshold = 0.1 # Threshold for GP error
        self.noise_estimate = 1e-4 #initial noise estimate
        self.ensemble_size = 3 # Number of GPs in the ensemble

    def _sample_points(self, n_points):
        # sample points within the trust region or randomly
        # return array of shape (n_points, n_dims)
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                # Clip to bounds
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature

        # Create validation set
        if len(X) > 5:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_val, y_train, y_val = X, X, y, y

        self.gp_ensemble = []
        self.gp_weights = []

        length_scales = [0.1, 1.0, 10.0] #Different length scales for diversity

        for length_scale in length_scales:
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X_train, y_train)
            self.gp_ensemble.append(gp)

            #Calculate validation error
            y_pred, _ = gp.predict(X_val, return_std=True)
            error = np.mean((y_pred.reshape(-1,1) - y_val)**2)
            self.gp_weights.append(error)

        #Normalize weights based on validation error
        self.gp_weights = np.array(self.gp_weights)
        self.gp_weights = np.exp(-self.gp_weights) / np.sum(np.exp(-self.gp_weights))


    def _acquisition_function(self, X):
        # Implement acquisition function: Lower Confidence Bound
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if not self.gp_ensemble:
            return np.random.normal(size=(len(X), 1))
        else:
            mu_ensemble = np.zeros((len(X), 1))
            sigma_ensemble = np.zeros((len(X), 1))

            for i, gp in enumerate(self.gp_ensemble):
                mu, sigma = gp.predict(X, return_std=True)
                mu_ensemble += self.gp_weights[i] * mu.reshape(-1, 1)
                sigma_ensemble += self.gp_weights[i] * sigma.reshape(-1, 1)

            #Ensemble variance
            ensemble_variance = np.zeros((len(X), 1))
            for i, gp in enumerate(self.gp_ensemble):
                mu, _ = gp.predict(X, return_std=True)
                ensemble_variance += self.gp_weights[i] * (mu.reshape(-1,1) - mu_ensemble)**2

            sigma_ensemble = np.sqrt(sigma_ensemble**2 + ensemble_variance)

            LCB = mu_ensemble - self.lcb_kappa * sigma_ensemble
            return LCB.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using differential evolution within the trust region
        # return array of shape (batch_size, n_dims)

        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            return self._acquisition_function(x.reshape(1, -1))[0, 0]

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        # Adjust maxiter based on remaining budget and optimization progress
        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100) #limit maxiter to prevent excessive computation

        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)

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
            self.success_ratio = 1.0 #reset success ratio
        else:
            self.success_ratio *= 0.75 #reduce success ratio if not improving

        #Update noise estimate
        self.noise_estimate = np.var(self.y)


    def _adjust_trust_region(self):
        # Adjust the trust region size based on the success
        if self.best_x is not None:
            #Calculate GP error within the trust region
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            within_tr = distances < self.trust_region_radius
            if np.any(within_tr):
                X_tr = self.X[within_tr]
                y_tr = self.y[within_tr]

                #Calculate the average GP error of the ensemble
                gp_error = 0.0
                for gp in self.gp_ensemble:
                    mu, _ = gp.predict(X_tr, return_std=True)
                    gp_error += np.mean(np.abs(mu.reshape(-1,1) - y_tr))
                gp_error /= len(self.gp_ensemble)

            else:
                gp_error = 0.0

            if self.success_ratio > 0.5 and gp_error < self.gp_error_threshold:
                self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            else:
                self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius) #shrink if no best_x

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
        batch_size = min(4, self.dim) #increase batch size to 4 or dim, whichever is smaller
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            # Adjust trust region radius
            self._adjust_trust_region()

            # Decay kappa, adjust min_kappa based on noise estimate
            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate) #dynamic min kappa

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveEvolutionaryTrustRegionBO_DKAE_EM>", line 215, in __call__
 215->             next_X = self._select_next_points(batch_size)
  File "<AdaptiveEvolutionaryTrustRegionBO_DKAE_EM>", line 134, in _select_next_points
 134->         result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)
  File "<AdaptiveEvolutionaryTrustRegionBO_DKAE_EM>", line 124, in de_objective
 124->             return self._acquisition_function(x.reshape(1, -1))[0, 0]
  File "<AdaptiveEvolutionaryTrustRegionBO_DKAE_EM>", line 101, in _acquisition_function
  99 | 
 100 |             for i, gp in enumerate(self.gp_ensemble):
 101->                 mu, sigma = gp.predict(X, return_std=True)
 102 |                 mu_ensemble += self.gp_weights[i] * mu.reshape(-1, 1)
 103 |                 sigma_ensemble += self.gp_weights[i] * sigma.reshape(-1, 1)
ValueError: Input X contains NaN.
GaussianProcessRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
