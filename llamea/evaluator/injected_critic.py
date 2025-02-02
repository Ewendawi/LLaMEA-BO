import numpy as np
import torch
import gpytorch
import sklearn
from sklearn.metrics import r2_score
from scipy.stats import qmc
from .evaluator_result import EvaluatorCoverageResult

def critic_wrapper(func):
    import functools
    functools.wraps(func)
    def injected_wrapper(self, *args, **kwargs):
        _injected_critic = None
        if hasattr(self, "_injected_critic"):
            _injected_critic = self._injected_critic

        if _injected_critic is not None:
            if func.__name__ == "_update_sample_points":
                next_X, next_y = args
                _injected_critic.update_after_eval(self.X, self.y, next_X, next_y)

        res = func(self, *args, **kwargs)

        if _injected_critic is not None:
            if func.__name__ == "_fit_model":
                new_X = args[0]
                _injected_critic.update_after_model_fit(res, new_X)
        return res
    return injected_wrapper

class AlgorithmCritic:
    # - r2 from surrogate model
    # - uncertainty from surrogate model of the same samples
    def __init__(self, dim:int, bounds:np.ndarray):
        self.dim = dim
        self.bounds = bounds
        
        self.n_test_x = 1000
        self.test_x = self._sample_points(self.n_test_x)
        self.test_y = None

        self.r_2_list = []
        self.uncertainty_list = []

        self.convergae_result = EvaluatorCoverageResult()

    def _sample_points(self, n_points) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def update_test_y(self, func):
        self.test_y = func.stateless_call(self.test_x)

    def update_after_model_fit(self, model, new_X):
        self._update_r2(model, new_X=new_X)
        self._update_uncertainty(model, new_X=new_X)

    def update_after_eval(self, x, y, next_x, next_y):
        self.convergae_result.update_next_grid_coverage(X=x, next_X=next_x, bounds=self.bounds)
        self.convergae_result.update_next_dbscan_coverage(X=x, next_X=next_x, bounds=self.bounds)
        self.convergae_result.update_next_exploitation(X=x, next_X=next_x, fX=y)

    def _update_r2(self, model, new_X):
        def _get_model_r2(model, x, y):
            _r_squared = 0.0
            if isinstance(model, sklearn.base.RegressorMixin):
                _r_squared = model.score(x, y)
            elif isinstance(model, gpytorch.models.GP):
                model.eval()
                model.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    tensor_x = torch.tensor(x, dtype=torch.float32)
                    posterior = model.likelihood(model(tensor_x))
                    mean = posterior.mean.detach().numpy()
                    _r_squared = r2_score(y, mean)
            return _r_squared

        r_squared = 0.0
        if isinstance(model, list):
            r_squared_list = [_get_model_r2(m, self.test_x, self.test_y) for m in model]
            r_squared = np.mean(r_squared_list)
        else:
            r_squared = _get_model_r2(model, self.test_x, self.test_y)

        if len(self.r_2_list) > 0:
            n_new_points = new_X.shape[0] - len(self.r_2_list)
            self.r_2_list.extend([r_squared] * n_new_points)
        else:
            n_new_points = new_X.shape[0]
            self.r_2_list.extend([r_squared] * n_new_points)

    def _update_uncertainty(self, model, new_X):
        def _get_model_uncertainty(model, x):
            _uncertainty = 0.0
            if isinstance(model, gpytorch.models.GP):
                model.eval()
                model.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    x = torch.tensor(x, dtype=torch.float32)
                    posterior = model.likelihood(model(x))
                    _uncertainty = posterior.variance.cpu().numpy()
            elif isinstance(model, sklearn.gaussian_process.GaussianProcessRegressor):
                _, _uncertainty = model.predict(x, return_std=True)
            return _uncertainty

        uncertainty = 0.0
        if isinstance(model, list):
            uncertainty_list = [_get_model_uncertainty(m, self.test_x) for m in model]
            uncertainty = np.mean(uncertainty_list)
        else:
            uncertainty = _get_model_uncertainty(model, self.test_x)

        mean_uncertainty = np.mean(uncertainty)

        if len(self.uncertainty_list) > 0:
            n_new_points = new_X.shape[0] - len(self.uncertainty_list)
            self.uncertainty_list.extend([mean_uncertainty] * n_new_points)
        else:
            n_new_points = new_X.shape[0]
            self.uncertainty_list.extend([mean_uncertainty] * n_new_points)
