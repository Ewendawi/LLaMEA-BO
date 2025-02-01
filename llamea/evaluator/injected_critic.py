import numpy as np
import torch
import gpytorch
import sklearn
from sklearn.metrics import r2_score
from scipy.stats import qmc

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

    def _sample_points(self, n_points) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _update_test_y(self, func):
        self.test_y = func(self.test_x)

    def update(self, model):
        self._update_r2(model)
        self._update_uncertainty(model)

    def _update_r2(self, model):
        def _get_model_r2(model, x):
            _r_squared = 0.0
            if isinstance(model, sklearn.base.RegressorMixin):
                _r_squared = model.score(x, self.test_y)
            elif isinstance(model, gpytorch.models.GP):
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    posterior = model.likelihood(model(x))
                    mean = posterior.mean
                    _r_squared = r2_score(self.test_y, mean)
            return _r_squared

        r_squared = 0.0
        if isinstance(model, list):
            r_squared_list = [_get_model_r2(m, self.test_x) for m in model]
            r_squared = np.mean(r_squared_list)
        else:
            r_squared = _get_model_r2(model, self.test_x)

        self.r_2_list.append(r_squared)

    def _update_uncertainty(self, model):
        def _get_model_uncertainty(model, x):
            _uncertainty = 0.0
            if isinstance(model, gpytorch.models.GP):
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    posterior = model.likelihood(model(x))
                    _uncertainty = posterior.variance
            elif isinstance(model, sklearn.gaussian_process.GaussianProcessRegressor):
                _, _uncertainty = model.predict(x, return_std=True)
            return _uncertainty

        uncertainty = 0.0
        if isinstance(model, list):
            uncertainty_list = [_get_model_uncertainty(m, self.test_x) for m in model]
            uncertainty = np.mean(uncertainty_list)
        else:
            uncertainty = _get_model_uncertainty(model, self.test_x)

        self.uncertainty_list.append(uncertainty)