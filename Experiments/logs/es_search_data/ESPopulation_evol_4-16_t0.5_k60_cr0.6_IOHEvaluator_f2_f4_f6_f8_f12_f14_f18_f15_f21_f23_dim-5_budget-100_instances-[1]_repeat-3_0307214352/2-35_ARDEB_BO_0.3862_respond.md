# Description
**Adaptive RBF-DE with Batch Bayesian Optimization (ARDEB_BO):** This algorithm synergistically integrates adaptive RBF kernel bandwidth adjustment, batch-oriented Differential Evolution (DE) for acquisition function optimization, and Bayesian Optimization (BO). It employs a Gaussian Process (GP) as the surrogate model and Expected Improvement (EI) as the acquisition function. The RBF kernel bandwidth is dynamically tuned using the median heuristic, and the DE parameters (mutation factor F and crossover rate CR) are adaptively adjusted based on the success rate of DE iterations. The batch size is also dynamically adjusted based on the GP's uncertainty. This algorithm aims to improve exploration and exploitation balance while efficiently utilizing function evaluations by selecting multiple points in each iteration.

# Justification
The ARDEB_BO algorithm builds upon the strengths of DE_BO_Adaptive and ADRBF_BO, addressing their limitations by combining adaptive RBF kernel bandwidth, adaptive DE parameters, and adaptive batch size.

1.  **Adaptive RBF Kernel Bandwidth:** ADRBF\_BO showed the benefit of adapting the RBF kernel bandwidth, which is crucial for the GP's performance. Using the median heuristic allows the algorithm to automatically adjust the kernel's length scale based on the distribution of the observed data, improving the GP's ability to model the objective function accurately.
2.  **Adaptive DE Parameters:** DE\_BO\_Adaptive demonstrated the effectiveness of adapting the mutation factor (F) and crossover rate (CR) of the DE algorithm. By adjusting these parameters based on the success rate of previous DE iterations, the algorithm can dynamically balance exploration and exploitation in the acquisition function optimization.
3.  **Adaptive Batch Size:** Inspired by DE_ABS_BO, the algorithm dynamically adjusts the batch size based on the GP's uncertainty. This allows the algorithm to explore more when the GP is uncertain and exploit when the GP is confident, leading to a more efficient use of function evaluations. The batch size is determined by the standard deviation of the GP's predictions.
4.  **Local Search Removal:** The local search step (L-BFGS-B) from DE_BO_Adaptive is removed because it is computationally expensive and may not always improve the solution significantly. The adaptive DE and RBF kernel bandwidth adjustment should provide sufficient refinement of the solution.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc  # If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ARDEB_BO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = 4 * dim
        self.pop_size = 15  # Population size for DE
        self.F = 0.8  # Mutation factor for DE
        self.CR = 0.7  # Crossover rate for DE
        self.bandwidth = 1.0  # Initial RBF bandwidth
        self.bandwidth_update_interval = 5 * dim  # Update bandwidth every this many evaluations
        self.de_success_rate = 0.0
        self.de_success_history = []
        self.learning_rate = 0.1

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.batch_size = 1
        self.min_batch_size = 1
        self.max_batch_size = 5
        self.batch_size_update_interval = 5 * dim

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(self.bandwidth, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid nan values
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Differential Evolution
        # return array of shape (batch_size, n_dims)

        # Initialize population
        population = self._sample_points(self.pop_size)
        ei_values = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_ei = ei_values[best_idx]

        successful_mutations = 0

        # DE optimization loop
        for _ in range(20):
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                x_mutated = x_r1 + self.F * (x_r2 - x_r3)
                x_mutated = np.clip(x_mutated, self.bounds[0], self.bounds[1])

                # Crossover
                x_trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        x_trial[j] = x_mutated[j]

                # Selection
                ei_trial = self._acquisition_function(x_trial.reshape(1, -1))[0, 0]
                ei_current = self._acquisition_function(population[i].reshape(1, -1))[0, 0]

                if ei_trial > ei_current:
                    population[i] = x_trial
                    successful_mutations += 1

            ei_values = self._acquisition_function(population)
            current_best_idx = np.argmax(ei_values)
            current_best_ei = ei_values[current_best_idx]
            if current_best_ei > best_ei:
                best_ei = current_best_ei
                best_idx = current_best_idx

        # Update DE parameters adaptively
        self.de_success_history.append(successful_mutations / (self.pop_size * 20))
        if len(self.de_success_history) > 5:
            self.de_success_history = self.de_success_history[-5:]
        self.de_success_rate = np.mean(self.de_success_history)

        if self.de_success_rate > 0.5:
            self.CR = min(1.0, self.CR + 0.1)
        else:
            self.F = min(1.0, self.F + 0.1)
            self.CR = max(0.1, self.CR - 0.1)

        # Select top batch_size points
        ei_values = self._acquisition_function(population)
        indices = np.argsort(ei_values.flatten())[::-1][:batch_size]
        next_points = population[indices]
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
        if np.min(new_y) < self.best_y:
            self.best_y = np.min(new_y)
            self.best_x = new_X[np.argmin(new_y)]

    def _update_bandwidth(self):
        # Update the RBF kernel bandwidth using the median heuristic
        distances = np.linalg.norm(self.X[:, None, :] - self.X[None, :, :], axis=2)
        distances = distances[np.triu_indices_from(distances, k=1)]
        if len(distances) > 0:
            self.bandwidth = np.median(distances)

    def _update_batch_size(self):
        # Update batch size based on GP's uncertainty
        _, sigma = self.gp.predict(self.X, return_std=True)
        mean_sigma = np.mean(sigma)
        # Adjust batch size based on uncertainty. Higher uncertainty = larger batch
        self.batch_size = int(np.clip(mean_sigma * 5, self.min_batch_size, self.max_batch_size))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Update batch size periodically
            if self.n_evals % self.batch_size_update_interval == 0:
                self._update_batch_size()

            # Select next points by acquisition function
            next_X = self._select_next_points(self.batch_size)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)

            # Update evaluated points
            self._update_eval_points(next_X, next_y)

            # Update bandwidth periodically
            if self.n_evals % self.bandwidth_update_interval == 0:
                self._update_bandwidth()

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ARDEB_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1595 with standard deviation 0.0995.

took 1193.61 seconds to run.