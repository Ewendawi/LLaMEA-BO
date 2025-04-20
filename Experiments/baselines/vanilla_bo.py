from collections.abc import Callable
import numpy as np
import torch
from scipy.stats import qmc
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, LogExpectedImprovement, qExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms import Standardize, Normalize
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import LogNormalPrior, GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine


class VanillaUCB:
    def __init__(self,
                    adaptive_beta: str = 'ei',
                    initial_ucb_beta: float = 2.0,
                    min_beta: float = 0.1,
                    max_beta: float = 4.0,
                    beta_momentum: float = 0.7,
                    beta_decay_rate: float = 0.9,
                    device: str = "cpu",
                ):
        self.device = device

        # adaptive_beta: None, ei, linear, exponential
        self.adaptive_beta = adaptive_beta
        
        self.initial_ucb_beta = initial_ucb_beta
        self.ucb_beta = self.initial_ucb_beta
        self.min_beta = min_beta
        self.max_beta = max_beta

        self.ei_std_ratio_avg = 0.0
        self.beta_momentum = beta_momentum

        self.beta_decay_rate = beta_decay_rate

        self._acqf = None

    def update_acqf(self, model, sampler, y_hist, X_hist, n_evals, budget):
        if self.adaptive_beta == 'ei':
            # Calculate EI for beta adjustment
            EI = LogExpectedImprovement(model, best_f=y_hist.max())
            with torch.no_grad():
                ei_values = EI(X_hist.unsqueeze(1))

            # Get posterior at current points
            posterior = model.posterior(X_hist)
            variance = posterior.variance
            std = torch.sqrt(variance)

            # Calculate the ratio of EI to uncertainty (std)
            ei_std_ratio = (ei_values.squeeze() / (std + 1e-9)).clamp(0, 20)

            # Update moving average of EI/std ratio with momentum
            self.ei_std_ratio_avg = self.beta_momentum * self.ei_std_ratio_avg + (1 - self.beta_momentum) * ei_std_ratio.mean().item()

            # Adjust beta based on the EI/std ratio using a sigmoid function
            sigmoid_scale = (1 - n_evals / budget)
            self.ucb_beta = self.min_beta + (self.max_beta - self.min_beta) * (1 / (1 + np.exp(-self.ei_std_ratio_avg * sigmoid_scale)))
        elif self.adaptive_beta == "linear":
            # linear decay:
            scale = 1 - n_evals / budget
            self.ucb_beta = self.initial_ucb_beta - (self.initial_ucb_beta - self.min_beta) * scale
        elif self.adaptive_beta == "exponential":
            # exponential decay
            self.ucb_beta = self.initial_ucb_beta * (self.beta_decay_rate ** (n_evals / budget))

        self.ucb_beta = np.clip(self.ucb_beta, self.min_beta, self.max_beta)

        self._acqf = qUpperConfidenceBound(
            model=model,
            beta=self.ucb_beta,
            sampler=sampler,
        )
        return self
    
    def __call__(self, x):
        return self._acqf(x)

class VanillaBO:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray = None, n_init: int = None, seed: int = None, device: str = "cpu", surrogate_model: str = "MaternKernel", acqf_name: str = "log_noisy_ei", init_sampler: str = "sobol"):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds
        if bounds is None:
            self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.n_init = n_init
        if n_init is None:
            self.n_init = int(dim * 2.5) + 5
        self.seed = seed
        self.device = device
        self.surrogate_model = surrogate_model
        self.acqf_name = acqf_name
        self.init_sampler = init_sampler

        self.X = None
        self.y = None
        self.n_evals = 0

        self._acqf = None

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if "cuda" in self.device:
                torch.cuda.manual_seed(seed)

    def is_maximization(self) -> bool:
        return True

    def _sample_points(self, n_points: int) -> torch.Tensor:
        if self.init_sampler == "sobol":
            # samples = draw_sobol_samples(bounds=torch.tensor(self.bounds, dtype=torch.float64, device=self.device), n=n_points, q=1).squeeze(1)
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            samples = sampler.random(n_points)
            samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
            samples = torch.tensor(samples, dtype=torch.float64, device=self.device)
            return samples
        elif self.init_sampler == "lhs":
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n_points)
            samples = qmc.scale(sample, self.bounds[0], self.bounds[1])
            samples = torch.tensor(samples, dtype=torch.float64, device=self.device)
        elif self.init_sampler == "halton":
            sampler = qmc.Halton(d=self.dim)
            samples = sampler.random(n_points)
            samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
            samples = torch.tensor(samples, dtype=torch.float64, device=self.device)
        elif self.init_sampler == "uniform":
            samples = torch.rand(n_points, self.dim, device=self.device) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        elif self.init_sampler == "normal":
            samples = torch.randn(n_points, self.dim, device=self.device) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return samples

    def _fit_model(self, X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
        train_X = X.to(self.device)
        train_y = y.to(self.device)

        if self.surrogate_model == "RBFKernel":
            covar_module = RBFKernel(
                ard_num_dims=self.dim,
                lengthscale_prior=LogNormalPrior(1.0, 1.0),
                lengthscale_constraint=GreaterThan(1e-4)
            ).to(self.device)
        elif self.surrogate_model == "MaternKernel":
            covar_module = MaternKernel(
                nu=2.5,
                ard_num_dims=self.dim,
                lengthscale_prior=GammaPrior(3.0, 6.0),
                lengthscale_constraint=Interval(1e-4, 5.0)
            ).to(self.device)
        elif self.surrogate_model == "ScaleKernel":
            base_kernel = RBFKernel(
                lengthscale_prior=LogNormalPrior(1.0, 1.0),
                lengthscale_constraint=GreaterThan(1e-4)
            ).to(self.device)
            covar_module = ScaleKernel(
                base_kernel=base_kernel,
                outputscale_prior=GammaPrior(2.0, 0.5),
                outputscale_constraint=GreaterThan(1e-4)
            ).to(self.device)
        else:
            raise ValueError(f"Unknown surrogate model: {self.surrogate_model}")

        likelihood = GaussianLikelihood(
            noise_prior=LogNormalPrior(-3.0, 1.0),
            noise_constraint=GreaterThan(1e-5)
        ).to(self.device)
        model = SingleTaskGP(train_X, train_y,
                             covar_module=covar_module,
                             input_transform=Normalize(self.dim),
                             outcome_transform=Standardize(1),
                             likelihood=likelihood
                             ).to(self.device)
        # model.likelihood.noise_covar.initialize(noise=1e-3)

        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(self.device)
        # fit_gpytorch_mll_torch(
        #     mll,
        #     step_limit=100,
        # )
        fit_gpytorch_mll(
            mll,
        )
        return model

    def _select_next_points(self, model, batch_size: int) -> torch.Tensor:
        bounds = torch.tensor(self.bounds, dtype=torch.float64, device=self.device)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=self.seed)

        if self.acqf_name == "log_noisy_ei":
            acqf_constructor = qLogNoisyExpectedImprovement
            acqf_kwargs = {
                "model": model,
                "X_baseline": self.X.clone(),
                "sampler": sampler,
                "prune_baseline": True,
            }
        elif self.acqf_name == "log_ei":
            acqf_constructor = qLogExpectedImprovement
            acqf_kwargs = {
                "model": model,
                "best_f": self.y.max(),
                "sampler": sampler,
            }
        elif self.acqf_name == "ucb":
            if self._acqf is None:
                self._acqf = VanillaUCB(adaptive_beta='ei', initial_ucb_beta=2.0, min_beta=0.1, max_beta=2.0, beta_momentum=0.7, beta_decay_rate=0.9, device=self.device)
            acqf_constructor = self._acqf.update_acqf
            acqf_kwargs = {
                "model": model,
                "sampler": sampler,
                "y_hist": self.y,
                "X_hist": self.X,
                "n_evals": self.n_evals,
                "budget": self.budget,
            }
        else:
            raise ValueError(f"Unknown acquisition function: {self.acqf_name}")

        try:
            # options include 2 parts
            # gen initalizer: gen_batch_initial_conditions(default)
            # generator: gen_candidates_scipy(default)

            # Dynamic local search range
            remaining_budget_ratio = max(0.1, 1 - self.n_evals / self.budget)
            dynamic_sigma = 0.05 + 0.1 * remaining_budget_ratio

            # Dynamically adjust num_restarts based on remaining budget
            num_restarts = min(5, int(5 * remaining_budget_ratio))

            raw_samples = 512 * (self.dim // 2)

            candidate, _ = optimize_acqf(
                acq_function=acqf_constructor(**acqf_kwargs),
                bounds=bounds,
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={
                    "nonnegative": False,
                    "sample_around_best": True,
                    "sample_around_best_sigma": dynamic_sigma,

                    "maxiter": 100,
                    "batch_limit": 64,
                    # "initial_conditions": initial_conditions,
                    # "disp": True, # Verbose output
                }
            )
        except Exception:
            candidate = self._sample_points(batch_size)
        return candidate.detach()

    def _update_eval_points(self, new_X: torch.Tensor, new_y: torch.Tensor) -> None:
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = torch.vstack([self.X, new_X])
            self.y = torch.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        n_initial_points = min(self.n_init, self.budget)
        X = self._sample_points(n_initial_points)
        y = torch.tensor([func(x) for x in X.cpu().numpy()], dtype=torch.float64, device=self.device).reshape(-1, 1)
        self.n_evals = n_initial_points
        self._update_eval_points(X, y)
        best_y = self.y.max()
        best_x = self.X[self.y.argmax()]

        while self.n_evals < self.budget:
            batch_size = min(4, self.budget - self.n_evals)
            model = self._fit_model(self.X, self.y)

            # Select the next points to evaluate
            next_points = self._select_next_points(model, batch_size)
            next_evaluations = torch.tensor([func(x) for x in next_points.cpu().numpy()], dtype=torch.float64, device=self.device).reshape(-1, 1)

            if next_evaluations.max() > best_y:
                best_y = next_evaluations.max()
                best_x = next_points[next_evaluations.argmax()]

            self.n_evals += batch_size
            self._update_eval_points(next_points, next_evaluations)

        return best_y.cpu().numpy().item(), best_x.cpu().numpy().flatten()



class VanillaEIBO:
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds
        if bounds is None:
            self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.n_init = n_init
        if n_init is None:
            self.n_init = int(dim * 2.5) + 5
        self.seed = seed
        self.device = device

        self.n_evals = 0

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if "cuda" in self.device:
                torch.cuda.manual_seed(seed)

    def is_maximization(self) -> bool:
        return True

    def _sample_points(self, n_points: int, seed=0) -> torch.Tensor:
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_points).to(dtype=torch.float64, device=self.device)
        return X_init


    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        X_ei = self._sample_points(self.n_init)
        Y_ei = torch.tensor(
            [func(x) for x in X_ei.cpu().numpy()], dtype=torch.float64, device=self.device
        ).unsqueeze(-1)
        self.n_evals = self.n_init

        batch_size = 4
        NUM_RESTARTS = 10 
        RAW_SAMPLES = 512 

        while self.n_evals < self.budget:
            train_Y = (Y_ei - Y_ei.mean()) / Y_ei.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            model = SingleTaskGP(X_ei, train_Y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Create a batch
            ei = qExpectedImprovement(model, train_Y.max())
            candidate, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack(
                    [
                        torch.zeros(self.dim, dtype=torch.float64, device=self.device),
                        torch.ones(self.dim, dtype=torch.float64, device=self.device),
                    ]
                ),
                q=batch_size,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
            Y_next = torch.tensor(
                [func(x) for x in candidate.cpu().numpy()], dtype=torch.float64, device=self.device
            ).unsqueeze(-1)

            # Append data
            X_ei = torch.cat((X_ei, candidate), axis=0)
            Y_ei = torch.cat((Y_ei, Y_next), axis=0)

            self.n_evals += batch_size
        
        best_y = Y_ei.max()
        best_x = X_ei[Y_ei.argmax()]
        return best_y.cpu().numpy().item(), best_x.cpu().numpy().flatten()