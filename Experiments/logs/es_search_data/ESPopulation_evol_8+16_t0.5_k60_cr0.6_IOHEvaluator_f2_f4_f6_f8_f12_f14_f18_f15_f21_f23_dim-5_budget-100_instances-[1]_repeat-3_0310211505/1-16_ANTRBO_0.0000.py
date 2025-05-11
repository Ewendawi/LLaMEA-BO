from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import torch
import torch.nn as nn
import torch.optim as optim

class ANTRBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10*dim, self.budget//5)
        self.trust_region_width = 2.0
        self.success_threshold = 0.1
        self.best_y = np.inf
        self.nn_reinit_freq = 10
        self.nn_hidden_size = 32
        self.nn_lr = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the neural network
        self.nn = nn.Sequential(
            nn.Linear(self.dim, self.nn_hidden_size),
            nn.ReLU(),
            nn.Linear(self.nn_hidden_size, self.dim),
            nn.Tanh()  # Output between -1 and 1
        ).to(self.device)

        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.nn_lr)

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        return ei

    def _generate_candidate_points(self, n_candidates, center, width):
        # Generate candidate points using the neural network within the trust region
        z = torch.randn(n_candidates, self.dim).to(self.device)

        with torch.no_grad():
            X_cand_scaled = self.nn(z)

            # Clip the output to ensure it's within [-1, 1]
            X_cand_scaled = torch.clamp(X_cand_scaled, -1.0, 1.0).cpu().numpy()

        # Scale the points to the trust region
        lower_bound = np.maximum(self.bounds[0], center - width / 2)
        upper_bound = np.minimum(self.bounds[1], center + width / 2)
        X_cand = qmc.scale(X_cand_scaled, lower_bound, upper_bound)
        return X_cand

    def _select_next_points(self, batch_size):
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._generate_candidate_points(n_candidates, self.best_x, self.trust_region_width)

        acq_values = self._acquisition_function(X_cand)
        top_indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_cand[top_indices]

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

    def _reinitialize_nn(self):
        for layer in self.nn:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.zero_()
        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.nn_lr)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        iter_count = 0
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)

            X_next = self._select_next_points(batch_size)

            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            self.best_y = new_best_y
            self.best_x = new_best_x

            if iter_count % self.nn_reinit_freq == 0:
                self._reinitialize_nn()

            iter_count += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
