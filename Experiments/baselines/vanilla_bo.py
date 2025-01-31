from collections.abc import Callable
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement 
from botorch.optim.optimize import optimize_acqf
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch 
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms import Standardize, Normalize
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior, GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
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

    def _sample_points(self, n_points: int) -> torch.Tensor:
        samples = draw_sobol_samples(bounds=torch.tensor(self.bounds, dtype=torch.float64, device=self.device),
                                     n=n_points, q=1).squeeze(1)
        return samples

    def _fit_model(self, X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
        train_X = X.to(self.device)
        train_y = y.to(self.device)

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
            model = SingleTaskGP(train_X, train_y, covar_module=covar_module, 
                                 input_transform=Normalize(self.dim), outcome_transform=Standardize(1),
                                 likelihood=likelihood).to(self.device)
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
            model = SingleTaskGP(train_X, train_y, covar_module=covar_module,   
                                 input_transform=Normalize(self.dim), outcome_transform=Standardize(1),
                                 likelihood=likelihood).to(self.device)
            model.likelihood.noise_covar.initialize(noise=1e-4)

        else:
            raise ValueError(f"Unknown surrogate model: {self.surrogate_model}")

        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(self.device)
        fit_gpytorch_mll_torch(
            mll,
            step_limit=100,
        )
        # fit_gpytorch_mll(
        #     mll,
        # )
        return model

    def _select_next_points(self, model, batch_size: int) -> torch.Tensor:
        bounds = torch.tensor(self.bounds, dtype=torch.float64, device=self.device)
        try:
            candidate, _ = optimize_acqf(
                acq_function=qLogNoisyExpectedImprovement(
                    model=model,
                    # X_baseline=self.X.clone(),
                    X_baseline = torch.unique(self.X.clone(), dim=0),
                    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])).to(self.device),
                    prune_baseline=True,
                ),
                bounds=bounds,
                q=batch_size,
                num_restarts=40,
                raw_samples=512,
                options={
                    "nonnegative": False,
                    "sample_around_best": True,
                    "sample_around_best_sigma": 0.1,
                    "maxiter": 200,
                    "batch_limit": 64,
                    # "disp": True, # Verbose output
                }
            )
        except Exception:
            candidate = self._sample_points(batch_size)
        return candidate.detach()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        n_initial_points = min(self.n_init, self.budget)
        self.X = self._sample_points(n_initial_points)
        self.y = torch.tensor([func(x) for x in self.X.cpu().numpy()], dtype=torch.float64, device=self.device).reshape(-1, 1)

        rest_of_budget = self.budget - n_initial_points
        best_y = self.y.min()
        best_x = self.X[self.y.argmin()]

        while rest_of_budget > 0:
            batch_size = min(4, rest_of_budget)
            model = self._fit_model(self.X, self.y)

            # Select the next points to evaluate
            next_points = self._select_next_points(model, batch_size)
            next_evaluations = torch.tensor([func(x) for x in next_points.cpu().numpy()], dtype=torch.float64, device=self.device).reshape(-1, 1)

            self.X = torch.vstack([self.X, next_points])
            self.y = torch.vstack([self.y, next_evaluations])

            if next_evaluations.min() < best_y:
                best_y = next_evaluations.min()
                best_x = next_points[next_evaluations.argmin()]

            rest_of_budget -= batch_size

        return best_y.cpu().numpy().item(), best_x.cpu().numpy().flatten()