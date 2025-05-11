from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class HTRDEBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(20 * dim, self.budget // 5)
        self.best_y = float('inf')
        self.best_x = None
        self.hall_of_fame_X = []
        self.hall_of_fame_y = []
        self.hall_of_fame_size = max(5, dim // 2)
        self.diversity_threshold = 0.5
        self.trust_region_radius = 2.5
        self.rho = 0.95
        self.hallucination_strength = 0.1  # Controls the impact of hallucinated points
        self.hallucination_points = 5 #Number of hallucinated points

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        if center is None:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            points = sampler.random(n=n_points)
            return qmc.scale(points, self.bounds[0], self.bounds[1])
        else:
            points = np.random.normal(loc=center, scale=radius / 3, size=(n_points, self.dim))
            points = np.clip(points, self.bounds[0], self.bounds[1])
            return points

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu - 1e-9
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        if self.hall_of_fame_X:
            distances = np.array([np.linalg.norm(X - hof_x, axis=1) for hof_x in self.hall_of_fame_X]).T
            min_distances = np.min(distances, axis=1, keepdims=True)
            diversity_penalty = np.where(min_distances < self.diversity_threshold, -100, 0)
            ei += diversity_penalty
        return ei

    def _select_next_points(self, batch_size, gp):
        x_starts = self._sample_points(batch_size // 2, center=self.best_x, radius=self.trust_region_radius)
        x_next = []
        for x_start in x_starts:
            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1), gp),
                           x_start,
                           bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                           method='L-BFGS-B')
            x_next.append(res.x)

        if self.best_x is not None:
            random_samples = self._sample_points(batch_size - batch_size // 2, center=self.best_x, radius=self.trust_region_radius)
            x_next.extend(random_samples)

        return np.array(x_next)

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

            if not self.hall_of_fame_X:
                self.hall_of_fame_X.append(self.best_x)
                self.hall_of_fame_y.append(self.best_y)
            else:
                distances = np.array([np.linalg.norm(self.best_x - hof_x) for hof_x in self.hall_of_fame_X])
                if np.min(distances) > self.diversity_threshold:
                    self.hall_of_fame_X.append(self.best_x)
                    self.hall_of_fame_y.append(self.best_y)
                    if len(self.hall_of_fame_X) > self.hall_of_fame_size:
                        worst_idx = np.argmax(self.hall_of_fame_y)
                        self.hall_of_fame_X.pop(worst_idx)
                        self.hall_of_fame_y.pop(worst_idx)
    
    def _hallucinate(self, gp):
        # Generate hallucinated points within the trust region
        hallucinated_X = self._sample_points(self.hallucination_points, center=self.best_x, radius=self.trust_region_radius / 2) #Focus near best

        # "Evaluate" the hallucinated points using the GP
        hallucinated_y, _ = gp.predict(hallucinated_X, return_std=True)
        hallucinated_y = hallucinated_y.reshape(-1, 1)

        # Create a temporary GP with hallucinated data
        temp_X = np.vstack((self.X, hallucinated_X))
        temp_y = np.vstack((self.y, hallucinated_y * self.hallucination_strength + self.y.mean())) # Dampen the hallucination

        temp_gp = self._fit_model(temp_X, temp_y) # Fit a new model
        return temp_gp

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            gp = self._fit_model(self.X, self.y)

            # Hallucination Step
            temp_gp = self._hallucinate(gp)

            # Dynamic batch size
            remaining_evals = self.budget - self.n_evals
            batch_size = min(max(1, int(remaining_evals / (self.dim * 0.1))), 20)

            # Select points using the hallucinated GP
            next_X = self._select_next_points(batch_size, temp_gp)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Adjust trust region
            if next_y.min() < self.best_y:
                self.trust_region_radius /= self.rho
            else:
                self.trust_region_radius *= self.rho
            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)


        return self.best_y, self.best_x
