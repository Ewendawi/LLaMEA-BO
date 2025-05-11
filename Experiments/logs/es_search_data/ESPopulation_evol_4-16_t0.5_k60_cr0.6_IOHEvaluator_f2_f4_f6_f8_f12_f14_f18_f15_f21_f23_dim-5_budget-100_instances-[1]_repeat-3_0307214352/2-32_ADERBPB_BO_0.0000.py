from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ADERBPB_BO:
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
        self.bandwidth_pop_size = 5 # Population size for bandwidths
        self.bandwidths = np.ones(self.bandwidth_pop_size) # Initial bandwidth population
        self.bandwidth_F = 0.8 # Mutation factor for bandwidth DE
        self.bandwidth_CR = 0.7 # Crossover rate for bandwidth DE
        self.archive_size = 10
        self.archive_X = []
        self.archive_y = []

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.de_success_rate = 0.0
        self.de_success_history = []
        self.F_step = 0.05 # Step size for adapting F
        self.CR_step = 0.05 # Step size for adapting CR

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y, bandwidth):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(bandwidth, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid nan values
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Differential Evolution
        # return array of shape (batch_size, n_dims)

        # Initialize population
        population = self._sample_points(self.pop_size)
        best_bandwidth = self.bandwidths[np.argmin([self._fit_model(self.X, self.y, bw).log_marginal_likelihood() for bw in self.bandwidths])]
        gp = self._fit_model(self.X, self.y, best_bandwidth)
        ei_values = self._acquisition_function(population, gp)
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
                ei_trial = self._acquisition_function(x_trial.reshape(1, -1), gp)[0, 0]
                ei_current = self._acquisition_function(population[i].reshape(1, -1), gp)[0, 0]

                if ei_trial > ei_current:
                    population[i] = x_trial
                    successful_mutations += 1

            ei_values = self._acquisition_function(population, gp)
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
            self.CR = min(1.0, self.CR + self.CR_step)
        else:
            self.F = min(1.0, self.F + self.F_step)
            self.CR = max(0.1, self.CR - self.CR_step)

        # Return the best point from the population
        next_point = population[best_idx]
        return next_point.reshape(1, -1)

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

        # Update archive
        for i in range(len(new_X)):
            x = new_X[i]
            y = new_y[i]
            if len(self.archive_X) < self.archive_size:
                self.archive_X.append(x)
                self.archive_y.append(y)
            else:
                max_y_idx = np.argmax(self.archive_y)
                if y < self.archive_y[max_y_idx]:
                    self.archive_X[max_y_idx] = x
                    self.archive_y[max_y_idx] = y

    def _update_bandwidths(self):
        # Update the RBF kernel bandwidths using Differential Evolution
        for _ in range(5):
            for i in range(self.bandwidth_pop_size):
                # Mutation
                idxs = np.random.choice(self.bandwidth_pop_size, 3, replace=False)
                b_r1, b_r2, b_r3 = self.bandwidths[idxs]
                b_mutated = b_r1 + self.bandwidth_F * (b_r2 - b_r3)
                b_mutated = np.clip(b_mutated, 1e-3, 1e3) # Keep bandwidths within reasonable bounds

                # Crossover
                b_trial = np.copy(self.bandwidths[i])
                if np.random.rand() < self.bandwidth_CR:
                    b_trial = b_mutated

                # Selection
                gp_trial = self._fit_model(self.X, self.y, b_trial)
                gp_current = self._fit_model(self.X, self.y, self.bandwidths[i])
                ll_trial = gp_trial.log_marginal_likelihood()
                ll_current = gp_current.log_marginal_likelihood()

                if ll_trial > ll_current:
                    self.bandwidths[i] = b_trial

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
            # Update bandwidths
            self._update_bandwidths()

            # Select best bandwidth
            best_bandwidth = self.bandwidths[np.argmin([self._fit_model(self.X, self.y, bw).log_marginal_likelihood() for bw in self.bandwidths])]
            # Fit the GP model
            gp = self._fit_model(self.X, self.y, best_bandwidth)
            self.gp = gp

            # Select next points by acquisition function
            next_X = self._select_next_points(1)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)

            # Update evaluated points
            self._update_eval_points(next_X, next_y)

            # Periodically re-evaluate archive points
            if self.n_evals % (5 * self.dim) == 0 and len(self.archive_X) > 0:
                archive_X = np.array(self.archive_X)
                archive_y = self._evaluate_points(func, archive_X)
                self._update_eval_points(archive_X, archive_y)
                self.archive_X = list(self.X[np.argsort(self.y.flatten())[:self.archive_size]])
                self.archive_y = list(self.y[np.argsort(self.y.flatten())[:self.archive_size]].flatten())

        return self.best_y, self.best_x
