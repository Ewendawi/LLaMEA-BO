# Description
**DE-RBF-AdaptiveBO**: This algorithm synergistically integrates Differential Evolution (DE) for acquisition function optimization, adaptive RBF kernel bandwidth adjustment, and adaptive DE parameter control within a Bayesian Optimization framework. It employs a Gaussian Process (GP) surrogate model with Expected Improvement (EI) as the acquisition function. The RBF kernel bandwidth is dynamically tuned using a variant of the median heuristic that incorporates a moving average to smooth bandwidth changes. The DE parameters (mutation factor F and crossover rate CR) are adaptively adjusted based on a combination of the DE's success rate and the diversity of the DE population. A local search step using L-BFGS-B is applied to refine the best solution found by DE.

# Justification
This algorithm builds upon the strengths of both `DE_BO_Adaptive` and `ADRBF_BO` while addressing some of their limitations.

1.  **Adaptive RBF Bandwidth with Smoothing:** `ADRBF_BO`'s adaptive RBF bandwidth helps to automatically adjust the length scale of the GP kernel. The median heuristic is used to calculate the bandwidth based on the distances between data points. However, directly using the median heuristic can lead to unstable bandwidth changes. To mitigate this, we introduce a moving average to smooth the bandwidth updates, making the learning process more stable.

2.  **Adaptive DE Parameters with Diversity Consideration:** `DE_BO_Adaptive` uses a success-rate-based adaptation of DE parameters. We retain this but augment it with a diversity measure. If the population collapses (low diversity), we increase the mutation factor to promote exploration. This helps to prevent premature convergence of the DE optimizer. Diversity is measured by the average distance of each individual to the population mean.

3.  **Local Search:** As in `DE_BO_Adaptive`, a local search step using L-BFGS-B is included to refine the best solution found by DE. This can improve the final solution quality, especially in high-dimensional spaces.

4. **Batch Size:** Instead of fixing batch size to 1 as in `ADRBF_BO`, a batch size of 3 is used to parallelize the evaluation of points.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class DE_RBF_AdaptiveBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 4 * dim
        self.pop_size = 15 # Population size for DE
        self.F = 0.8 # Mutation factor for DE
        self.CR = 0.7 # Crossover rate for DE
        self.success_rate = 0.0
        self.learning_rate = 0.1
        self.diversity_threshold = 0.1

        self.bandwidth = 1.0 # Initial RBF bandwidth
        self.bandwidth_update_interval = 5 * dim
        self.bandwidth_history = [self.bandwidth]

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.batch_size = 3

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
        ei[sigma <= 1e-6] = 0.0 # avoid nan values
        return ei

    def _calculate_diversity(self, population):
        # Calculate the diversity of the population
        mean = np.mean(population, axis=0)
        distances = np.linalg.norm(population - mean, axis=1)
        diversity = np.mean(distances)
        return diversity

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Differential Evolution
        # return array of shape (batch_size, n_dims)

        # Initialize population
        population = self._sample_points(self.pop_size)
        
        # DE optimization loop
        n_success = 0
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
                    n_success += 1
        
        #Update F and CR adaptively
        diversity = self._calculate_diversity(population)
        if diversity < self.diversity_threshold:
            self.F = min(self.F + 0.2, 0.9)  # Increase exploration
        else:
            self.success_rate = 0.9 * self.success_rate + 0.1 * (n_success / self.pop_size)
            self.F = np.clip(self.F + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)
            self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)


        # Return the best points from the population
        ei_values = self._acquisition_function(population)
        
        # Local search
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]
        
        def obj_func(x):
            return -self._acquisition_function(x.reshape(1, -1))[0,0]
        
        res = minimize(obj_func, best_x, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
        next_point = res.x.reshape(1, -1)
        
        return next_point

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
        # Update the RBF kernel bandwidth using the median heuristic with moving average
        distances = np.linalg.norm(self.X[:, None, :] - self.X[None, :, :], axis=2)
        distances = distances[np.triu_indices_from(distances, k=1)]
        if len(distances) > 0:
            median_distance = np.median(distances)
            self.bandwidth_history.append(median_distance)
            if len(self.bandwidth_history) > 5:
                self.bandwidth_history = self.bandwidth_history[-5:]
            self.bandwidth = np.mean(self.bandwidth_history)

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
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

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
 The algorithm DE_RBF_AdaptiveBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1607 with standard deviation 0.1023.

took 1005.98 seconds to run.