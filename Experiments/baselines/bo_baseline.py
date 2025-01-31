import math
import numpy as np
import torch
from .vanilla_bo import VanillaBO
from .TuRBO.turbo import Turbo1, TurboM

class BaselineSearch:
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds
        if bounds is None:
            self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.n_init = n_init
        if n_init is None:
            self.n_init = dim * 5
        self.seed = seed
        self.device = device

class BLRandomSearch(BaselineSearch):
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, **kwargs):
        super().__init__(budget, dim, bounds, n_init, seed, **kwargs)
        if self.seed is not None:
            np.random.seed(seed)

    def __call__(self, func):
        f_opt = np.Inf
        x_opt = None
        
        for i in range(self.budget):
            x = np.random.uniform(self.bounds[0], self.bounds[1])
            
            f = func(x)
            if f < f_opt:
                f_opt = f
                x_opt = x
            
        return f_opt, x_opt


class BLTuRBO1(BaselineSearch):
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        super().__init__(budget, dim, bounds, n_init, seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def __call__(self, func):
        turbo = Turbo1(
            f=func, 
            lb=self.bounds[0], 
            ub=self.bounds[1], 
            n_init=self.n_init, 
            max_evals=self.budget,
            batch_size=5,
            verbose=False,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            device=self.device,
            dtype="float64",
            )
        turbo.optimize()

        best_idx = np.argmin(turbo.fX)
        f_opt = turbo.fX[best_idx]
        x_opt = turbo.X[best_idx]

        return f_opt, x_opt


class BLTuRBOM(BaselineSearch):
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        super().__init__(budget, dim, bounds, n_init, seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def __call__(self, func):
        tr = int(max(self.dim/2, 2))
        n_init = math.floor(self.n_init/tr)
        
        turbo = TurboM(
            f=func, 
            lb=self.bounds[0], 
            ub=self.bounds[1], 
            n_init=n_init, 
            max_evals=self.budget,
            n_trust_regions=tr,
            batch_size=5,
            verbose=False,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            device=self.device,
            dtype="float64",
            )
        turbo.optimize()

        best_idx = np.argmin(turbo.fX)
        f_opt = turbo.fX[best_idx]
        x_opt = turbo.X[best_idx]

        return f_opt, x_opt

class BLRBFKernelVanillaBO(BaselineSearch):
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        super().__init__(budget, dim, bounds, n_init, seed)
        self.bo = VanillaBO(
            surrogate_model="RBFKernel",
            budget=budget,
            dim=dim,
            bounds=bounds,
            n_init=n_init,
            seed=seed,
            device=device
        )

    def __call__(self, func):
        f_opt, x_opt = self.bo(func)
        return f_opt, x_opt

class BLScaledKernelVanillaBO(BaselineSearch):
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        super().__init__(budget, dim, bounds, n_init, seed)
        self.bo = VanillaBO(
            surrogate_model="ScaleKernel",
            budget=budget,
            dim=dim,
            bounds=bounds,
            n_init=n_init,
            seed=seed,
            device=device
        )

    def __call__(self, func):
        f_opt, x_opt = self.bo(func)
        return f_opt, x_opt