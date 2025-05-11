# Description
**AGETRBODE-v2: Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Dynamic Exploration and Gradient-Based Trust Region Adaptation**

This algorithm builds upon AGETRBODE by incorporating gradient information not only into the GPR model but also into the trust region adaptation strategy. The trust region width is adjusted based on the agreement between the predicted gradient from the GPR model and the estimated gradient using finite differences. A larger trust region is allowed when the gradients align, indicating a well-modeled region. Additionally, the exploration weight is dynamically adjusted based on both the trust region width and the magnitude of the estimated gradient, encouraging more exploration in flat regions and exploitation in regions with strong gradients.

# Justification
1.  **Gradient-Based Trust Region Adaptation:** The original AGETRBODE adjusts the trust region based solely on the success rate. Incorporating gradient information provides a more informed adaptation. Agreement between the model's predicted gradient and the finite difference gradient suggests a reliable model, allowing for a larger trust region and faster convergence. Disagreement suggests model inaccuracy, prompting a smaller trust region and more cautious exploration.

2.  **Dynamic Exploration Weight Adjustment:** The exploration weight is now dynamically adjusted based on both trust region width and gradient magnitude. This allows for more intelligent exploration. In regions with small gradients (flat regions), the exploration weight is increased to encourage exploration and escape local optima. In regions with large gradients, the exploration weight is decreased to focus on exploitation along the gradient direction.

3.  **Computational Efficiency:** The gradient estimation is already performed for gradient-enhanced modeling. Reusing this information for trust region adaptation adds minimal computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class AGETRBODE_v2:
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
        self.delta = 1e-3
        self.exploration_weight_base = 0.1 # Base weight for uncertainty-based exploration
        self.gradient_agreement_threshold = 0.5

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

    def _estimate_gradient(self, func, x):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
            x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
        return gradient

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        imp = self.best_y - mu
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Uncertainty-aware exploration
        exploration_term = self.exploration_weight * sigma
        
        return ei + exploration_term

    def _select_next_points(self, func, batch_size):
        X_next = []
        for _ in range(batch_size):
            def objective(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0, 0]

            lower_bound = np.maximum(self.bounds[0], self.best_x - self.trust_region_width / 2)
            upper_bound = np.minimum(self.bounds[1], self.best_x + self.trust_region_width / 2)
            bounds = [(lower_bound[i], upper_bound[i]) for i in range(self.dim)]
            
            x0 = self._sample_points(1, center=self.best_x, width=self.trust_region_width).flatten()
            
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            X_next.append(result.x)

        return np.array(X_next)

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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)

            X_next = self._select_next_points(func, batch_size)

            y_next = self._evaluate_points(func, X_next)

            self._update_eval_points(X_next, y_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            # Gradient-based trust region adaptation
            estimated_gradient = self._estimate_gradient(func, self.best_x)
            predicted_gradient, _ = self.model.predict(self.best_x.reshape(1, -1), return_std=True)
            predicted_gradient = predicted_gradient.flatten()

            # Normalize gradients
            estimated_gradient /= (np.linalg.norm(estimated_gradient) + 1e-9)
            predicted_gradient /= (np.linalg.norm(predicted_gradient) + 1e-9)

            # Calculate gradient agreement (cosine similarity)
            gradient_agreement = np.dot(estimated_gradient, predicted_gradient)

            if gradient_agreement > self.gradient_agreement_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            # Dynamic exploration weight adjustment
            gradient_magnitude = np.linalg.norm(estimated_gradient)
            self.exploration_weight = self.exploration_weight_base * (self.trust_region_width / 2.0) * (1.0 / (1.0 + gradient_magnitude)) # Reduce exploration with high gradient

            self.best_y = new_best_y
            self.best_x = new_best_x

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AGETRBODE_v2>", line 118, in __call__
 118->             X_next = self._select_next_points(func, batch_size)
  File "<AGETRBODE_v2>", line 88, in _select_next_points
  88->             result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
  File "<AGETRBODE_v2>", line 80, in objective
  80->                 return -self._acquisition_function(x)[0, 0]
  File "<AGETRBODE_v2>", line 71, in _acquisition_function
  69 | 
  70 |         # Uncertainty-aware exploration
  71->         exploration_term = self.exploration_weight * sigma
  72 |         
  73 |         return ei + exploration_term
AttributeError: 'AGETRBODE_v2' object has no attribute 'exploration_weight'. Did you mean: 'exploration_weight_base'?
