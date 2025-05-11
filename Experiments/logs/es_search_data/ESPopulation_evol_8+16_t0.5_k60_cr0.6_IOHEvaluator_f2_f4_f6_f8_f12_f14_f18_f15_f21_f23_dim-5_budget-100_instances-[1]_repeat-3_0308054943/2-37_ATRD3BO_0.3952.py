from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, RBF
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances

class ATRD3BO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.best_x = None
        self.best_y = float('inf')
        self.acquisition_functions = ['ei', 'pi', 'ucb']
        self.ucb_kappa = 2.0 # Initial value
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0
        self.kde = None
        self.distribution_weight = 0.1
        self.diversity_weight = 0.1


        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, method='sobol', trust_region_center=None):
        # sample points
        # return array of shape (n_points, n_dims)
        if method == 'sobol':
            sampler = qmc.Sobol(d=self.dim, seed=42)
            sample = sampler.random(n=n_points)
            scaled_samples = qmc.scale(sample, self.bounds[0], self.bounds[1])
            return scaled_samples
        elif method == 'kde':
            if self.kde is None:
                return self._sample_points(n_points, method='sobol')
            else:
                # Sample from KDE
                samples = self.kde.sample(n_points)
                samples = np.clip(samples, self.bounds[0], self.bounds[1])
                return samples
        elif method == 'trust_region':
            sobol_engine = qmc.Sobol(d=self.dim, seed=42)
            samples = sobol_engine.random(n=n_points)

            # Scale and shift samples to be within the trust region
            scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)

            # Clip samples to stay within the problem bounds
            scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])
            return scaled_samples
        else:
            raise ValueError("Invalid sampling method.")

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model, acq_type='ei'):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma
        
        if acq_type == 'ei':
            return ei
        elif acq_type == 'pi':
            pi = norm.cdf(Z)
            return pi
        elif acq_type == 'ucb':
            ucb = mu + self.ucb_kappa * sigma
            return ucb
        else:
            raise ValueError("Invalid acquisition function type.")
    
    def _combined_acquisition(self, X, model):
        # Combine EI, PI, UCB, diversity, and distribution
        ei = self._acquisition_function(X, model, acq_type='ei')
        pi = self._acquisition_function(X, model, acq_type='pi')
        ucb = self._acquisition_function(X, model, acq_type='ucb')
        
        # Normalize acquisition functions
        ei_norm = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-9)
        pi_norm = (pi - np.min(pi)) / (np.max(pi) - np.min(pi) + 1e-9)
        ucb_norm = (ucb - np.min(ucb)) / (np.max(ucb) - np.min(ucb) + 1e-9)

        # Diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            diversity = self.diversity_weight * min_distances
            diversity_norm = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity) + 1e-9)
        else:
            diversity_norm = np.zeros_like(ei)

        # Distribution matching term
        if self.kde is not None:
            log_likelihood = self.kde.score_samples(X).reshape(-1, 1)
            distribution = self.distribution_weight * np.exp(log_likelihood) # Use exp to avoid negative values
            distribution_norm = (distribution - np.min(distribution)) / (np.max(distribution) - np.min(distribution) + 1e-9)
        else:
            distribution_norm = np.zeros_like(ei)
        
        # Combine normalized acquisition functions and terms
        combined = ei_norm + pi_norm + ucb_norm + diversity_norm + distribution_norm
        return combined

    def _is_pareto_efficient(self, points):
        """
        Find the pareto-efficient points
        :param points: An n by m matrix of points
        :return: A boolean array of length n, True for pareto-efficient points, False otherwise
        """
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(points[is_efficient] > c, axis=1) | (points[is_efficient] == c).all(axis=1)
                is_efficient[i] = True  # Keep current point
        return is_efficient

    def _select_next_points(self, batch_size, model, trust_region_center):
        # Select the next points to evaluate
        # Sample within the trust region using Sobol sequence
        candidate_points = self._sample_points(100 * batch_size, method='trust_region', trust_region_center=trust_region_center)

        # Calculate acquisition function values
        acquisition_values = self._combined_acquisition(candidate_points, model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]

        return selected_points

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
        
        # Update KDE
        if self.X is not None:
            # Identify promising regions (e.g., top 20% of evaluated points)
            threshold = np.percentile(self.y, 20)
            promising_points = self.X[(self.y <= threshold).flatten()]

            if len(promising_points) > self.dim + 1:  # Ensure enough points for KDE
                self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(promising_points)
            else:
                self.kde = None

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init, method='sobol')
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        trust_region_center = self.X[np.argmin(self.y)] # Initialize trust region center

        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.ucb_kappa = min(self.ucb_kappa * 1.1, self.ucb_kappa_max) # Increase kappa for exploration
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion
                self.ucb_kappa = max(self.ucb_kappa * 0.9, self.ucb_kappa_min) # Decrease kappa for exploitation

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
