import math
import numpy as np
import torch
from skopt import gp_minimize
from Experiments.baselines.TuRBO.turbo import Turbo1, TurboM
from Experiments.baselines.vanilla_bo import VanillaBO, VanillaEIBO

class BLRandomSearch:
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, **kwargs):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds if bounds is not None else np.array([[-5]*dim, [5]*dim])
        if seed is not None:
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

class BLCMAES:
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, **kwargs):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds if bounds is not None else np.array([[-5]*dim, [5]*dim])
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, func):
        import cma
        options = cma.CMAOptions()
        options.set("bounds", [self.bounds[0], self.bounds[1]])
        options.set("maxfevals", self.budget)
        if self.seed is not None:
            options.set("seed", self.seed)
        # options.set("tolfun", 1e-6)
        # options.set("tolfunhist", 1e-6)
        # options.set("tolx", 1e-6)
        # options.set("tolupsigma", 1e-6)
        
        x0 = np.random.uniform(self.bounds[0], self.bounds[1])
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=1, inopts=options)
        es.optimize(func, iterations=self.budget)
        f_opt = es.result[1]
        x_opt = es.result[0]

        return f_opt, x_opt

class BLHEBO:
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, **kwargs):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds if bounds is not None else np.array([[-5]*dim, [5]*dim])
        self.n_init = n_init if n_init is not None else 2*dim
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, func):
        from hebo.optimizers.hebo import HEBO
        from hebo.design_space.design_space import DesignSpace

        critic = getattr(self, "_injected_critic", None)
        if critic is not None:
            critic.n_init = self.n_init
        X_hist = None
        y_hist = None

        _space = []
        for i in range(self.dim):
            _space.append({"name": f"x{i}", "type": "num", "lb": self.bounds[0][i], "ub": self.bounds[1][i]})
        space = DesignSpace().parse(_space)
        opt = HEBO(space,
                   rand_sample=self.n_init,
                   scramble_seed=self.seed)

        batch_size = 1
        n_evals = 0
        while n_evals < self.budget:
            _bs = min(batch_size, self.budget - n_evals)
            
            x_df = opt.suggest(n_suggestions=_bs)
            x = x_df.to_numpy()
            fx = func(x)
            opt.observe(x_df, fx)

            n_evals += _bs
            if critic is not None:
                if X_hist is None:
                    X_hist = x
                    y_hist = fx
                else:
                    X_hist = np.vstack((X_hist, x))
                    y_hist = np.vstack((y_hist, fx))
                critic.update_after_eval(X_hist, y_hist, x, fx, n_evals)


        f_opt = opt.best_y
        x_opt = opt.best_x

        return f_opt, x_opt

class BLSKOpt:
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, **kwargs):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds if bounds is not None else np.array([[-5]*dim, [5]*dim])
        self.n_init = n_init if n_init is not None else 2*dim
        self.seed = seed
        if self.seed is not None:
            np.random.seed(seed)

    def __call__(self, func):
        res = gp_minimize(
            func=func,
            acq_func="EI",
            n_initial_points=self.n_init,
            # initial_point_generator="sobol",
            dimensions=self.bounds.T,
            n_calls=self.budget,
            # acq_optimizer="sampling",
            n_points=1000,
            n_restarts_optimizer=5,
            random_state=self.seed)

        critic = getattr(self, "_injected_critic", None)
        if critic is not None:
            critic.n_init = self.n_init
            start_index = self.n_init
            next_index = start_index + 1
            model_index = 0
            while next_index <= self.budget:
                X = np.array(res.x_iters[:start_index]) if start_index > 0 else None
                y = np.array(res.func_vals[:start_index]) if start_index > 0 else None

                # model = res.models[model_index]
                # critic.update_after_model_fit(model, start_index, X, y)

                next_X = np.array(res.x_iters[start_index:next_index])
                next_y = np.array(res.func_vals[start_index:next_index])
                critic.update_after_eval(X, y, next_X, next_y, next_index) 

                start_index = next_index
                next_index += 1
                model_index += 1

        f_opt = res.fun
        x_opt = res.x

        return f_opt, x_opt

class BLTuRBO1:
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds if bounds is not None else np.array([[-5]*dim, [5]*dim])
        self.n_init = n_init if n_init is not None else 2*dim
        self.seed = seed
        self.device = device
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def __call__(self, func):
        critic = None
        if hasattr(self, "_injected_critic"):
            critic = self._injected_critic
            critic.n_init = self.n_init
        
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
            critic = critic
            )
        turbo.optimize()

        best_idx = np.argmin(turbo.fX)
        f_opt = turbo.fX[best_idx]
        x_opt = turbo.X[best_idx]

        return f_opt, x_opt


class BLTuRBOM:
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds if bounds is not None else np.array([[-5]*dim, [5]*dim])
        self.n_init = n_init if n_init is not None else 2*dim
        self.seed = seed
        self.device = device
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def __call__(self, func):
        critic = None
        if hasattr(self, "_injected_critic"):
            critic = self._injected_critic
            critic.n_init = self.n_init

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
            critic=critic,
            )
        turbo.optimize()

        best_idx = np.argmin(turbo.fX)
        f_opt = turbo.fX[best_idx]
        x_opt = turbo.X[best_idx]

        return f_opt, x_opt

class BLMaternVanillaBO(VanillaBO):
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        super().__init__(budget=budget, dim=dim, bounds=bounds, n_init=n_init, seed=seed, device=device, surrogate_model="MaternKernel")


class BLScaledVanillaBO(VanillaBO):
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        super().__init__(budget=budget, dim=dim, bounds=bounds, n_init=n_init, seed=seed, device=device, surrogate_model="ScaleKernel")

class BLVanillaEIBO(VanillaEIBO):
    def __init__(self, budget:int, dim:int, bounds:np.ndarray=None, n_init:int=None, seed:int=None, device:str="cpu"):
        super().__init__(budget=budget, dim=dim, bounds=bounds, n_init=n_init, seed=seed, device=device)