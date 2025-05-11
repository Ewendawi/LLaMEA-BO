from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel
from scipy.optimize import minimize, Bounds, approx_fprime
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import torch
import gpytorch

class SpectralMixtureKernel(Kernel):
    def __init__(self, num_mixtures=4, mixture_means=None, mixture_scales=None, amplitude=None, active_dims=None, batch_shape=torch.Size([])):
        super().__init__(active_dims=active_dims, batch_shape=batch_shape)
        self.num_mixtures = num_mixtures

        if mixture_means is None:
            self.mixture_means = torch.nn.Parameter(torch.randn(num_mixtures, dtype=torch.float64))
        else:
            self.mixture_means = torch.nn.Parameter(torch.tensor(mixture_means, dtype=torch.float64))

        if mixture_scales is None:
            self.mixture_scales = torch.nn.Parameter(torch.randn(num_mixtures, dtype=torch.float64).abs())
        else:
            self.mixture_scales = torch.nn.Parameter(torch.tensor(mixture_scales, dtype=torch.float64))

        if amplitude is None:
            self.amplitude = torch.nn.Parameter(torch.ones(num_mixtures, dtype=torch.float64))
        else:
            self.amplitude = torch.nn.Parameter(torch.tensor(amplitude, dtype=torch.float64))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise NotImplementedError("last_dim_is_batch is not implemented for SpectralMixtureKernel")

        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., N1, N2, D)
        rbf_arg = diff.mul(self.mixture_scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).pow(2).mul(-2 * np.pi**2)
        rbf = self.amplitude.unsqueeze(-1).unsqueeze(-1) * torch.exp(rbf_arg).sum(dim=-4)

        cos_arg = diff.mul(self.mixture_means.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).mul(2 * np.pi)
        cos = torch.cos(cos_arg.sum(dim=-4))

        return rbf * cos

class AEBBO_DKAB_EnhancedPlusV2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # number of initial samples

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.acq_strategy = "EI+UCB"
        self.initial_exploration_weight = 0.2
        self.exploration_weight = self.initial_exploration_weight
        self.batch_size = 2
        self.kernel_length_scale = 1.0
        self.local_search_restarts = 3
        self.length_scale_weight = 0.5 # Weight for previous length scale in adaptation
        self.gradient_local_search = True # Flag to use gradient-based local search
        self.num_mixtures = 4 # Number of mixtures for SpectralMixtureKernel

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        if self.X is None or len(self.X) < self.dim + 1:
            sampler = qmc.LatinHypercube(d=self.dim)
            samples = sampler.random(n=n_points)
            return qmc.scale(samples, self.bounds[0], self.bounds[1])
        else:
            try:
                kde = gaussian_kde(self.X.T)
                samples = kde.resample(n_points)
                samples = np.clip(samples.T, self.bounds[0], self.bounds[1])
                return samples
            except np.linalg.LinAlgError:
                sampler = qmc.LatinHypercube(d=self.dim)
                samples = sampler.random(n=n_points)
                return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature

        # Convert data to tensors
        X_tensor = torch.tensor(X, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.float64).flatten()

        # Define the spectral mixture kernel
        kernel = SpectralMixtureKernel(num_mixtures=self.num_mixtures)

        # Define the Gaussian process model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.GammaPrior(1.1, 0.1))
        model = ExactGPModel(X_tensor, y_tensor, likelihood, kernel)

        # Train the model
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iter = 50
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        self.gp = model
        self.likelihood = likelihood
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.acq_strategy == "EI":
            return self._expected_improvement(X)
        elif self.acq_strategy == "UCB":
            return self._upper_confidence_bound(X)
        elif self.acq_strategy == "EI+UCB":
            ei = self._expected_improvement(X)
            ucb = self._upper_confidence_bound(X)
            return ei + self.exploration_weight * ucb
        else:
            raise ValueError("Invalid acquisition function strategy.")

    def _expected_improvement(self, X):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X_tensor = torch.tensor(X, dtype=torch.float64)
            observed_pred = self.likelihood(self.gp(X_tensor))
            mu = observed_pred.mean.cpu().numpy()
            sigma = observed_pred.stddev.cpu().numpy()

        sigma = np.maximum(sigma, 1e-6)  # avoid division by zero
        gamma = (self.best_y - mu) / (sigma + 1e-9) # avoid division by zero
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _upper_confidence_bound(self, X):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X_tensor = torch.tensor(X, dtype=torch.float64)
            observed_pred = self.likelihood(self.gp(X_tensor))
            mu = observed_pred.mean.cpu().numpy()
            sigma = observed_pred.stddev.cpu().numpy()
        return mu.reshape(-1, 1) + 2 * sigma.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        selected_points = []
        candidates = self._sample_points(100 * self.dim) # Generate a larger candidate set

        for _ in range(batch_size):
            acq_values = self._acquisition_function(candidates)
            best_index = np.argmax(acq_values)
            selected_point = candidates[best_index]

            # Local search to refine the selected point
            def obj(x):
                return -self._acquisition_function(x.reshape(1, -1))[0, 0]  # Negate for minimization

            bounds = Bounds(self.bounds[0], self.bounds[1])
            best_x = None
            best_obj = float('inf')
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                X_tensor = torch.tensor(selected_point.reshape(1, -1), dtype=torch.float64)
                observed_pred = self.likelihood(self.gp(X_tensor))
                mu = observed_pred.mean.cpu().numpy()
                sigma = observed_pred.stddev.cpu().numpy()

            # Adaptive local search restarts
            adaptive_restarts = int(self.local_search_restarts * (1 - self.n_evals / self.budget)) + 1
            if self.gradient_local_search and sigma[0] < 0.1: # High confidence, use gradient-based method
                def obj_grad(x):
                     return approx_fprime(x, lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0], epsilon=1e-6)
                res = minimize(obj, selected_point, method='L-BFGS-B', jac=obj_grad, bounds=bounds)
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
            else: # Use SLSQP
                for _ in range(adaptive_restarts):
                    res = minimize(obj, selected_point, method='SLSQP', bounds=bounds)
                    if res.fun < best_obj:
                        best_obj = res.fun
                        best_x = res.x
                    selected_point = self._sample_points(1).flatten()


            if best_x is not None:
                selected_point = best_x

            selected_points.append(selected_point)

            # Remove the selected point from the candidates to avoid duplicates in the batch
            candidates = np.delete(candidates, best_index, axis=0)

        return np.array(selected_points)

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
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

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

            # Dynamic batch size adjustment
            mean_sigma = np.mean(self.gp.likelihood(self.gp(torch.tensor(self.X, dtype=torch.float64))).stddev.cpu().numpy())
            self.batch_size = max(1, min(self.batch_size + (1 if mean_sigma > 0.5 else -1), 5))
            next_X = self._select_next_points(min(self.batch_size, self.budget - self.n_evals))

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Adaptive acquisition balancing (Thompson Sampling inspired)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mean_sigma = np.mean(self.gp.likelihood(self.gp(torch.tensor(self.X, dtype=torch.float64))).stddev.cpu().numpy())
            range_y = np.max(self.y) - np.min(self.y) if len(self.y) > 1 and np.max(self.y) != np.min(self.y) else 1.0
            self.exploration_weight = self.initial_exploration_weight * (mean_sigma / range_y) * (1 - self.n_evals / self.budget)
            self.exploration_weight = np.clip(self.exploration_weight, 0.01, self.initial_exploration_weight)

        return self.best_y, self.best_x

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
