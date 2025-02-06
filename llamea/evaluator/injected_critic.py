import numpy as np
import torch
import gpytorch
import botorch
import sklearn
from sklearn.metrics import r2_score
from scipy.stats import qmc
from .evaluator_result import EvaluatorSearchResult

def critic_wrapper(func):
    import functools
    functools.wraps(func)

    def to_numpy_if_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x
    
    def injected_wrapper(self, *args, **kwargs):
        _injected_critic = None
        if hasattr(self, "_injected_critic"):
            _injected_critic = self._injected_critic
            if _injected_critic.n_init == 0 and hasattr(self, "n_init"):
                _injected_critic.n_init = self.n_init

        if _injected_critic is not None:
            if func.__name__ == "_update_sample_points":
                n_evals = None
                if hasattr(self, "n_evals"):
                    n_evals = self.n_evals
                next_X, next_y = args
                next_X = to_numpy_if_tensor(next_X)
                next_y = to_numpy_if_tensor(next_y)
                X = to_numpy_if_tensor(self.X)
                y = to_numpy_if_tensor(self.y)
                _injected_critic.update_after_eval(X, y, next_X, next_y, n_evals)

        res = func(self, *args, **kwargs)

        if _injected_critic is not None:
            if func.__name__ == "_fit_model":
                new_X = args[0]
                new_y = args[1]
                new_X = to_numpy_if_tensor(new_X)
                new_y = to_numpy_if_tensor(new_y)
                n_evals = len(new_X)
                model = res
                if hasattr(self, "n_evals"):
                    n_evals = self.n_evals
                if isinstance(model, tuple):
                    model = model[0]
                _injected_critic.update_after_model_fit(model, n_evals, new_X, new_y)
        return res
    return injected_wrapper

def set_inject_maximize(cls_instance, maximize):
    if cls_instance is None:
        return
    if hasattr(cls_instance, "_inject_maximize"):
        cls_instance._inject_maximize = maximize
    else:
        setattr(cls_instance, "_inject_maximize", maximize)

def get_inject_maximize(cls_instance):
    if cls_instance is None:
        return False
    if hasattr(cls_instance, "_is_maximization"):
        return cls_instance._is_maximization()
    if hasattr(cls_instance, "_inject_maximize"):
        return cls_instance._inject_maximize
    return False

class AlgorithmCritic:
    # - r2 from surrogate model
    # - uncertainty from surrogate model of the same samples
    def __init__(self, dim:int, bounds:np.ndarray, optimal_value=None, critic_y_range=None):
        self.dim = dim
        self.bounds = bounds
        self.n_init = 0
        self.maximize = False
        
        self.n_test_x = 1000
        self.test_x = self._sample_points(self.n_test_x)
        self.test_y = None

        self.r_2_list = []
        self.r_2_list_on_train = []
        self.uncertainty_list = []
        self.uncertainty_list_on_train = []

        self.search_result = EvaluatorSearchResult()
        self.search_result.init_acq_score(optimal_value=optimal_value, y_range=critic_y_range)

        self.temp_r2_list = []
        self.temp_r2_list_on_train = []
        self.temp_uncertainty_list = []
        self.temp_uncertainty_list_on_train = []

    def _sample_points(self, n_points) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def update_test_y(self, func):
        self.test_y = func.stateless_call(self.test_x)

    def update_after_eval(self, x, y, next_x, next_y, n_evals):
        # inverse the y to treat the problem as minimization
        if self.maximize:
            y = -y if y is not None else None
            next_y = -next_y if next_y is not None else None
        
        self.search_result.update_next_grid_coverage(X=x, next_X=next_x, bounds=self.bounds, n_evals=n_evals)
        self.search_result.update_next_dbscan_coverage(X=x, next_X=next_x, bounds=self.bounds, n_evals=n_evals)
        self.search_result.update_next_exploitation(X=x, next_X=next_x, fX=y, next_fX=next_y, n_evals=n_evals)
        self.search_result.update_next_acq_score(fX=y, next_fX=next_y, n_evals=n_evals)

# model related
    def update_after_model_fit_temp(self, model, new_X, new_y):
        r2 = self._get_model_r2(model)
        self.temp_r2_list.append(r2)
        r2_on_train = self._get_model_r2(model, new_X, new_y)
        self.temp_r2_list_on_train.append(r2_on_train)
        
        uncertainty = self._get_model_uncertainty(model)
        self.temp_uncertainty_list.append(uncertainty)
        uncertainty_on_train = self._get_model_uncertainty(model, new_X)
        self.temp_uncertainty_list_on_train.append(uncertainty_on_train)

    def update_after_model_fit_with_temp(self, n_evals):
        mean_r2 = np.mean(self.temp_r2_list)
        self._update_r2(mean_r2, n_evals, self.r_2_list)
        self.temp_r2_list = []

        mean_r2_on_train = np.mean(self.temp_r2_list_on_train)
        self._update_r2(mean_r2_on_train, n_evals, self.r_2_list_on_train)
        self.temp_r2_list_on_train = []
        
        mean_uncertainty = np.mean(self.temp_uncertainty_list)
        self._update_uncertainty(mean_uncertainty, n_evals, self.uncertainty_list)
        self.temp_uncertainty_list = []

        mean_uncertainty_on_train = np.mean(self.temp_uncertainty_list_on_train)
        self._update_uncertainty(mean_uncertainty_on_train, n_evals, self.uncertainty_list_on_train)
        self.temp_uncertainty_list_on_train = []

    def update_after_model_fit(self, model, n_evals, new_X, new_y):
        r_squared_on_train = self._get_model_r2(model, new_X, new_y)
        self._update_r2(r_squared_on_train, n_evals, self.r_2_list_on_train)
        r_squared = self._get_model_r2(model)
        self._update_r2(r_squared, n_evals, self.r_2_list)

        uncertainty = self._get_model_uncertainty(model)
        self._update_uncertainty(uncertainty, n_evals, self.uncertainty_list)
        uncertainty_on_train = self._get_model_uncertainty(model, new_X)
        self._update_uncertainty(uncertainty_on_train, n_evals, self.uncertainty_list_on_train)

# r squared 
    def _get_model_r2(self, model, new_X=None, new_y=None):
        def _get_single_model_r2(model, x, y):
            _r_squared = 0.0
            if isinstance(model, sklearn.base.RegressorMixin):
                _r_squared = model.score(x, y)
            # elif isinstance(model, botorch.models.SingleTaskGP):
            #     with torch.no_grad():
            #         tensor_x = torch.tensor(x, dtype=torch.float32)
            #         mean = model.posterior(tensor_x).mean.detach().numpy()
            #         _r_squared = r2_score(y, mean)
            elif isinstance(model, gpytorch.models.GP):
                model.eval()
                model.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    tensor_x = torch.tensor(x, dtype=torch.float32)
                    posterior = model.likelihood(model(tensor_x))
                    mean = posterior.mean.detach().numpy()
                    _r_squared = r2_score(y, mean)
            return _r_squared

        eval_x = new_X
        eval_y = new_y
        if eval_x is None or eval_y is None:
            eval_x = self.test_x
            eval_y = self.test_y 

        r_squared = 0.0
        if isinstance(model, list):
            r_squared_list = [_get_single_model_r2(m, eval_x, eval_y) for m in model]
            r_squared = np.mean(r_squared_list)
        else:
            r_squared = _get_single_model_r2(model, eval_x, eval_y)
        return r_squared

    def _update_r2(self, r_squared, n_evals, target_list):
        if len(target_list) > 0:
            n_new_points = n_evals - len(target_list)
        else:
            n_new_points = n_evals

        n_fill = n_new_points - 1
        target_list.extend([np.nan] * n_fill)
        target_list.append(r_squared)

# uncertainty
    def _get_model_uncertainty(self, model, new_X=None):
        def _get_single_model_uncertainty(model, x):
            _uncertainty = 0.0
            # if isinstance(model, botorch.models.SingleTaskGP):
            #     with torch.no_grad():
            #         x = torch.tensor(x, dtype=torch.float32)
            #         _uncertainty = model.posterior(x).variance.detach().numpy()
            if isinstance(model, gpytorch.models.GP):
                model.eval()
                model.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    x = torch.tensor(x, dtype=torch.float32)
                    posterior = model.likelihood(model(x))
                    _variance = posterior.variance.cpu().numpy()
                    _uncertainty = np.sqrt(_variance)
            elif isinstance(model, sklearn.gaussian_process.GaussianProcessRegressor):
                _, _uncertainty = model.predict(x, return_std=True)
            mean_uncertainty = np.mean(_uncertainty)
            return mean_uncertainty
        
        eval_x = new_X
        if eval_x is None:
            eval_x = self.test_x
        
        uncertainty = 0.0
        if isinstance(model, list):
            uncertainty_list = [_get_single_model_uncertainty(m, eval_x) for m in model]
            uncertainty = np.mean(uncertainty_list)
        else:
            uncertainty = _get_single_model_uncertainty(model, eval_x)
        return uncertainty

    def _update_uncertainty(self, uncertainty, n_evals, target_list=None):
        if len(target_list) > 0:
            n_new_points = n_evals - len(target_list)
        else:
            n_new_points = n_evals
        n_fill = n_new_points - 1
        target_list.extend([np.nan] * n_fill)
        target_list.append(uncertainty)
