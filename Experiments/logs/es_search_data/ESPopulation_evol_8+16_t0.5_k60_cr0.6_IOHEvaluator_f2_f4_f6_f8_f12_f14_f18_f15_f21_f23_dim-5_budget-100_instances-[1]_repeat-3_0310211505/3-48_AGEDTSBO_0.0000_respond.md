# Description
**AGE-DTSBO**: Adaptive Gradient-Enhanced Dueling Thompson Sampling Bayesian Optimization. This algorithm combines the gradient-enhanced modeling of AGETRBO with the acquisition function selection strategy of ATRDABO_TS. It adaptively adjusts the trust region, incorporates gradient information into the Gaussian Process model, and uses Thompson Sampling to select between Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition functions. A key refinement is the use of gradient information to guide the initial sampling within the trust region, potentially leading to faster convergence. We also introduce a dynamic scaling factor for the UCB acquisition function to better balance exploration and exploitation.

# Justification
The algorithm leverages the strengths of both AGETRBO and ATRDABO_TS. AGETRBO's gradient enhancement can lead to a more accurate surrogate model, especially in high-dimensional spaces. ATRDABO_TS's Thompson Sampling offers a robust way to balance exploration and exploitation by dynamically selecting the most promising acquisition function.

1.  **Gradient-Enhanced Modeling**: Incorporating gradient information into the Gaussian Process model can improve the accuracy of the surrogate model, especially in the early stages of optimization. Finite differences are used to estimate the gradient.
2.  **Adaptive Trust Region**: The trust region approach helps to focus the search on promising regions of the search space and adaptively adjusts the size of the region based on the success of previous iterations.
3.  **Dueling Acquisition Functions with Thompson Sampling**: Using Thompson Sampling to select between EI and UCB allows the algorithm to dynamically adjust its exploration-exploitation trade-off.
4.  **Gradient-Guided Initial Sampling**: Using the estimated gradient to guide the initial sampling within the trust region can help to identify promising regions of the search space more quickly. This is done by biasing the sampling towards the direction of the negative gradient.
5. **Dynamic UCB Scaling**: A dynamic scaling factor for the UCB acquisition function is introduced to improve the balance between exploration and exploitation. This scaling factor is adjusted based on the success of previous iterations, increasing exploration when the algorithm is stuck in a local optimum and decreasing exploration when the algorithm is making progress.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import beta
from scipy.optimize import minimize


class AGEDTSBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.trust_region_width = 2.0
        self.success_threshold = 0.1
        self.best_y = np.inf
        self.best_x = None
        self.acq_funcs = ['ei', 'ucb']
        self.n_acq = len(self.acq_funcs)
        self.alpha = np.ones(self.n_acq)  # Success counts for Thompson Sampling
        self.beta = np.ones(self.n_acq)  # Failure counts for Thompson Sampling
        self.delta = 1e-3  # Step size for finite difference gradient estimation
        self.ucb_scaling = 1.0 # Scaling factor for UCB
        self.ucb_scaling_factor = 1.05 # Scaling factor for UCB adaptation
        self.ucb_scaling_decay = 0.95 # Decay factor for UCB adaptation

    def _sample_points(self, n_points, center=None, width=None, gradient=None):
        if center is None:
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            
            # Gradient-guided sampling
            if gradient is not None:
                # Normalize gradient
                gradient_norm = np.linalg.norm(gradient)
                if gradient_norm > 0:
                    gradient = gradient / gradient_norm
                
                # Bias sampling towards negative gradient direction
                sampler = qmc.LatinHypercube(d=self.dim)
                sample = sampler.random(n=n_points)
                scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
                
                # Adjust samples based on gradient
                for i in range(n_points):
                    scaled_sample[i] = scaled_sample[i] - 0.1 * width * gradient #Biasing factor 0.1
                    scaled_sample[i] = np.clip(scaled_sample[i], lower_bound, upper_bound) #Clip back to bounds
                
                return scaled_sample
            
            else:
                sampler = qmc.Sobol(d=self.dim)
                sample = sampler.random(n=n_points)
                scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
                return scaled_sample

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _estimate_gradient(self, func, x):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
            x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
        return gradient

    def _acquisition_function(self, X, acq_func='ei'):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if acq_func == 'ei':
            best = np.min(self.y)
            imp = best - mu
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0
            return ei

        elif acq_func == 'ucb':
            ucb = mu + self.ucb_scaling * sigma
            return ucb

        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

    def _select_next_points(self, func, batch_size):
        n_candidates = max(2000, batch_size * 200)
        gradient = self._estimate_gradient(func, self.best_x)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width, gradient=gradient)

        acq_values = np.zeros((self.n_acq, n_candidates))
        for i, acq_func in enumerate(self.acq_funcs):
            acq_values[i, :] = self._acquisition_function(X_cand, acq_func).flatten()

        # Thompson Sampling for acquisition function selection
        sampled_values = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_acq)]
        idx = np.argmax(sampled_values)

        top_indices = np.argsort(acq_values[idx, :])[::-1][:batch_size]
        return X_cand[top_indices], idx

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def _update_thompson_sampling(self, idx, new_y):
        reward = self.y.min() - new_y.min()
        if reward > 0:  # Success
            self.alpha[idx] += 1
        else:  # Failure
            self.beta[idx] += 1

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)

            X_next, idx = self._select_next_points(func, batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)
            self._update_thompson_sampling(idx, y_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
                self.ucb_scaling = max(self.ucb_scaling * self.ucb_scaling_decay, 0.1) #Decrease UCB scaling on success
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)
                self.ucb_scaling = min(self.ucb_scaling * self.ucb_scaling_factor, 5.0) #Increase UCB scaling on failure

            self.best_y = new_best_y
            self.best_x = new_best_x

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AGEDTSBO>", line 156, in __call__
 156->             X_next, idx = self._select_next_points(func, batch_size)
  File "<AGEDTSBO>", line 110, in _select_next_points
 110->         gradient = self._estimate_gradient(func, self.best_x)
  File "<AGEDTSBO>", line 82, in _estimate_gradient
  80 |             x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
  81 |             x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
  82->             gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
  83 |         return gradient
  84 | 
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
