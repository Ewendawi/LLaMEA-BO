from collections.abc import Callable
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior, GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


class VanillaBO:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray = None, n_init: int = None, seed: int = None, device: str = "cpu", surrogate_model: str = "RBFKernel"):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds
        if bounds is None:
            self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.n_init = n_init
        if n_init is None:
            self.n_init = dim + 1
        self.seed = seed
        self.device = device
        self.surrogate_model = surrogate_model
        self.X = None  
        self.y = None  

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if "cuda" in self.device:
                torch.cuda.manual_seed(seed)

    def _sample_points(self, n_points: int) -> np.ndarray:
        samples = draw_sobol_samples(bounds=torch.tensor(self.bounds, dtype=torch.float64, device=self.device),
                                     n=n_points, q=1).squeeze(1).cpu().numpy()
        return samples

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        train_X = torch.tensor(X, dtype=torch.float64, device=self.device)
        train_y = torch.tensor(y, dtype=torch.float64, device=self.device)

        # Configure the surrogate model based on the user's choice
        if self.surrogate_model == "RBFKernel":
            likelihood = GaussianLikelihood(
                noise_prior=LogNormalPrior(0.0, 1.0),
                noise_constraint=GreaterThan(1e-4)
            ).to(self.device)
            covar_module = RBFKernel(
                ard_num_dims=self.dim,
                lengthscale_prior=LogNormalPrior(0.0, 1.0),
                lengthscale_constraint=GreaterThan(1e-4)
            ).to(self.device)
            model = SingleTaskGP(train_X, train_y, covar_module=covar_module, likelihood=likelihood).to(self.device)
            model.likelihood.noise_covar.initialize(noise=1e-4)

        elif self.surrogate_model == "ScaleKernel":
            likelihood = GaussianLikelihood(
                noise_prior=LogNormalPrior(0.0, 1.0),
                noise_constraint=GreaterThan(1e-4)
            ).to(self.device)
            base_kernel = RBFKernel(
                lengthscale_prior=LogNormalPrior(0.0, 1.0),
                lengthscale_constraint=GreaterThan(1e-4)
            ).to(self.device)
            covar_module = ScaleKernel(
                base_kernel=base_kernel,
                outputscale_prior=GammaPrior(2.0, 0.15),
                outputscale_constraint=GreaterThan(1e-4)
            ).to(self.device)
            model = SingleTaskGP(train_X, train_y, covar_module=covar_module, likelihood=likelihood).to(self.device)
            model.likelihood.noise_covar.initialize(noise=1e-4)

        else:
            raise ValueError(f"Unknown surrogate model: {self.surrogate_model}")

        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(self.device)
        fit_gpytorch_mll(mll)
        return model

    def _select_next_points(self, model, batch_size: int) -> np.ndarray:
        bounds = torch.tensor(self.bounds, dtype=torch.float64, device=self.device)
        candidate, _ = optimize_acqf(
            acq_function=qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=torch.tensor(self.X, dtype=torch.float64, device=self.device),
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])).to(self.device),
                prune_baseline=True
            ),
            bounds=bounds,
            q=batch_size,
            num_restarts=4,
            raw_samples=512,
            options={
                "nonnegative": False,
                "sample_around_best": True,
                "sample_around_best_sigma": 0.1,
                "maxiter": 300,
                "batch_limit": 64,
            }
        )
        return candidate.detach().cpu().numpy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        n_initial_points = min(self.n_init, self.budget)
        self.X = self._sample_points(n_initial_points)
        self.y = np.array([func(x) for x in self.X]).reshape(-1, 1)

        rest_of_budget = self.budget - n_initial_points
        best_y = self.y.min()
        best_x = self.X[self.y.argmin()]

        while rest_of_budget > 0:
            batch_size = min(4, rest_of_budget)
            model = self._fit_model(self.X, self.y)

            # Select the next points to evaluate
            next_points = self._select_next_points(model, batch_size)
            next_evaluations = np.array([func(x) for x in next_points]).reshape(-1, 1)

            self.X = np.vstack([self.X, next_points])
            self.y = np.vstack([self.y, next_evaluations])

            if next_evaluations.min() < best_y:
                best_y = next_evaluations.min()
                best_x = next_points[next_evaluations.argmin()]

            rest_of_budget -= batch_size

        return best_y, best_x