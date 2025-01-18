### Description
The problem involves minimizing two functions from the BBOB test suite, F24-LunacekBiRastrigin and F19-GriewankRosenbrock, in a 5-dimensional space with bounds [-5.0, 5.0] for each dimension. 
To address this problem, we can design a Bayesian Optimization algorithm that utilizes a Gaussian Process (GP) as the surrogate model and the Expected Improvement (EI) acquisition function to select the next points to evaluate. 
The GP will be used to model the objective function, and the EI acquisition function will be used to balance exploration and exploitation. 
We will also use a QMC sampling method to initialize the GP model.

### /Description

### Pseudocode
1. Initialize the GP model using QMC sampling and the objective function evaluations.
2. Fit the GP model to the initial data.
3. Calculate the acquisition function values for a set of candidate points.
4. Select the next point to evaluate based on the acquisition function values.
5. Evaluate the objective function at the selected point.
6. Update the GP model with the new data.
7. Repeat steps 3-6 until the budget is exhausted.
8. Return the best point found and its corresponding objective function value.

### /Pseudocode


### Code
```python
from typing import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

class LunacekBiRastriginGriewankRosenbrockBO:
    """
    This class implements a Bayesian Optimization algorithm using a Gaussian Process as the surrogate model 
    and the Expected Improvement acquisition function to select the next points to evaluate.
    
    Techniques used:
    - Gaussian Process (GP) as the surrogate model
    - Expected Improvement (EI) acquisition function
    - QMC sampling for initializing the GP model
    - Matern kernel for the GP model
    
    Parameters:
    - kernel: The kernel used for the GP model (default: ConstantKernel(1.0) * Matern(nu=2.5))
    - alpha: The alpha value used for the GP model (default: 1e-10)
    - n_initial_points: The number of initial points used for the QMC sampling (default: 10)
    """

    def __init__(self):
        # Initialize optimizer settings
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        self.alpha = 1e-10
        self.n_initial_points = 10

    def _sample_points(self, n_points) -> np.ndarray:
        # sample points using QMC sampling
        sampler = qmc.Sobol(d=5, scramble=True)
        points = sampler.random(n=n_points)
        return 10 * points - 5  # scale to the bounds [-5, 5]

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        gp.fit(X, y)
        return gp

    def _get_model_loss(self, model, X, y) -> np.float64:
        # Calculate the loss of the model
        y_pred = model.predict(X)
        loss = np.mean((y_pred - y) ** 2)
        return loss

    def _acquisition_function(self, X, gp, y_best) -> np.ndarray:
        # Implement acquisition function (Expected Improvement)
        y_mean, y_std = gp.predict(X, return_std=True)
        z = (y_mean - y_best) / y_std
        ei = (y_mean - y_best) * (1 - np.exp(-z)) + y_std * z * np.exp(-z)
        return ei

    def _select_next_points(self, batch_size, X, gp, y_best) -> np.ndarray:
        # Implement the strategy to select the next points to evaluate
        candidate_points = self._sample_points(100)
        ei_values = self._acquisition_function(candidate_points, gp, y_best)
        indices = np.argsort(ei_values)[-batch_size:]
        return candidate_points[indices]

    def optimize(self, objective_fn: Callable[[np.ndarray], np.ndarray], bounds: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, str], int]:
        # Main minimize optimization loop
        n_initial_points = self.n_initial_points
        X = self._sample_points(n_initial_points)
        y = objective_fn(X)
        gp = self._fit_model(X, y)
        y_best = np.min(y)
        model_losses = [self._get_model_loss(gp, X, y)]
        rest_of_budget = budget - n_initial_points
        all_x = X
        all_y = y
        while rest_of_budget > 0:
            # Optimization
            batch_size = min(rest_of_budget, 10)  # batch size
            next_points = self._select_next_points(batch_size, all_x, gp, y_best)
            next_y = objective_fn(next_points)
            all_x = np.vstack((all_x, next_points))
            all_y = np.vstack((all_y, next_y))
            gp = self._fit_model(all_x, all_y)
            y_best = np.min(all_y)
            model_losses.append(self._get_model_loss(gp, all_x, all_y))
            rest_of_budget -= batch_size
        return all_y, all_x, (np.array(model_losses), 'Mean Squared Error'), n_initial_points

```
