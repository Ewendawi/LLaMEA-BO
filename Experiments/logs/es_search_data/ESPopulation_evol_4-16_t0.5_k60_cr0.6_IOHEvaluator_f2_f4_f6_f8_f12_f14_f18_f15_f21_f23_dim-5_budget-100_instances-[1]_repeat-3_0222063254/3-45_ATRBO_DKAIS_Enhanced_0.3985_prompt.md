You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ATRBO_DKAIS: 0.2030, 601.46 seconds, Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, and Stochastic Radius (ATRBO_DKAIS) builds upon ATRBO-DKAI by adding a stochastic component to the trust region radius adjustment. This aims to prevent premature convergence by occasionally expanding the trust region even when no improvement is observed. The key idea is to balance the shrinking of the trust region with occasional stochastic expansions to potentially escape local optima. The initial exploration phase is also slightly enhanced by using a Latin Hypercube sampling instead of the Sobol sequence.


- ATRBO_HKRA: 0.1921, 316.18 seconds, **Adaptive Trust Region Bayesian Optimization with Hybrid Kappa and Radius Adjustment (ATRBO-HKRA):** This algorithm combines the dynamic kappa adjustment of ATRBO_DKAI and the radius adjustment based on success/failure counts from ATRBO_DKRA. Further, it incorporates a more robust kappa adjustment by taking into account both success/failure counts and the predicted variance from the Gaussian Process. Specifically, we incorporate a hybrid kappa strategy that weights the success/failure-based adjustment and the GP variance-based adjustment. This allows for a more adaptive exploration-exploitation trade-off. Additionally, the initial kappa is made adaptive to the dimensionality of the problem.


- ATRBO_DKAICA: 0.1942, 436.52 seconds, Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration and Trust Region Center Adaptation (ATRBO_DKAICA) builds upon ATRBO-DKAI by incorporating trust region center adaptation and refining the adaptive strategy for kappa and rho. The trust region center is moved to the best point found within the trust region, rather than only updating it if the new point globally improves the best objective. This allows the trust region to more accurately focus on promising local areas. Additionally, the updates to kappa and rho are refined to provide more stable and effective adaptation, especially focusing on the GP uncertainty.


- ATRBO_DKARE: 0.1847, 324.99 seconds, Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Rho, and Enhanced Exploration (ATRBO_DKARE) builds upon ATRBO_DKAR by introducing a more robust exploration strategy and refining the adaptation of kappa and rho. Key enhancements include:
1.  **Enhanced Exploration:** Instead of relying solely on the acquisition function within the trust region, a small probability is introduced to sample globally (outside the trust region) to escape local optima.
2.  **Adaptive Kappa with Gradient Information:** Adapt `kappa` not only based on GP variance, but also incorporate gradient information to refine the exploration-exploitation trade-off. Higher predicted gradients near the current best indicate a potentially steeper slope to a better optimum, encouraging exploration in that region.
3.  **Refined Rho Adaptation:** Adjust `rho` based on both the recent success rate and the relative change in the objective function value. If the objective function is improving slowly, increase `rho` to shrink the trust region faster and focus on potentially better areas.
4.  **Dynamic Noise Handling:** Introduce a mechanism to estimate the noise level in the objective function and adjust the GP's hyperparameters accordingly. This helps to prevent overfitting and improves the accuracy of the GP model.




The selected solution to update is:
Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, and Stochastic Radius (ATRBO_DKAIS) builds upon ATRBO-DKAI by adding a stochastic component to the trust region radius adjustment. This aims to prevent premature convergence by occasionally expanding the trust region even when no improvement is observed. The key idea is to balance the shrinking of the trust region with occasional stochastic expansions to potentially escape local optima. The initial exploration phase is also slightly enhanced by using a Latin Hypercube sampling instead of the Sobol sequence.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKAIS:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = min(20 * dim, self.budget // 5) # increased samples for initial exploration
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.rho = 0.95  # Shrinking factor
        self.kappa = 2.0  # Exploration-exploitation trade-off for LCB
        self.success_history = [] # Keep track of successful moves

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)  # Scale to [-1, 1]

        # Project points to a hypersphere
        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)

        points = points * radius + center

        # Clip to the bounds
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
        best_index = np.argmin(acq_values)
        return samples[best_index]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop

        # Initial exploration: Combine uniform sampling with sampling around the best seen point
        n_uniform = self.n_init // 2
        n_around_best = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1]/2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4) # Sampling around the middle of the search space as initial guess

        initial_X = np.vstack((initial_X_uniform, initial_X_best))
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to a random initial point
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)

            # Select next point
            next_x = self._select_next_point(gp)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Adjust trust region radius and kappa
            if next_y < self.best_y:
                self.trust_region_radius /= self.rho  # Expand
                self.kappa *= self.rho * 0.9 # Reduced kappa decrease.
                self.success_history.append(True)
            else:
                self.trust_region_radius *= self.rho  # Shrink
                self.trust_region_radius *= (1 + np.random.normal(0, 0.05)) #Stochastic expansion
                self.kappa /= (self.rho*0.9) # increase kappa more when unsuccessful
                self.success_history.append(False)

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)
            
            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

        return self.best_y, self.best_x

```
The algorithm ATRBO_DKAIS got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.2030 with standard deviation 0.0934.

took 601.46 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

