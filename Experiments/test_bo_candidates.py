from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import numpy as np
from llamea.utils import random_search, RandomBoTorchTestEvaluator

# mypy: ignore-errors

class AckleyBO:
    def __init__(self, n_initial_points=5, acquisition_function="EI"):
        """Initialize the Bayesian Optimization class.

        Args:
            n_initial_points (int): Number of initial random points.
            acquisition_function (str): Type of acquisition function to use ('EI').
        """
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.model = GaussianProcessRegressor(kernel=Matern(nu=2.5, length_scale_bounds=(1e-18, 1e6)), alpha=5e-3, normalize_y=True)
        self.X_samples = None
        self.y_samples = None

    def _sample_points(self, n_points, bounds):
        """Sample points using Latin Hypercube Sampling."""
        lower_bounds, upper_bounds = bounds
        dim = len(lower_bounds)
        
        # Generate LHS samples in the unit hypercube [0, 1]^dim
        unit_samples = lhs(dim, samples=n_points)
        
        # Scale the samples to the given bounds
        points = lower_bounds + (upper_bounds - lower_bounds) * unit_samples
        return points

    def _fit_model(self, X, y):
        """Fit the Gaussian Process Regressor model."""
        self.model.fit(X, y)

    def _expected_improvement(self, X, model, y_best):
        """Calculate the Expected Improvement."""
        mu, sigma = model.predict(X, return_std=True)
        imp = mu - y_best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    def _acquisition_function(self, X):
        """Implement the acquisition function."""
        if self.acquisition_function == "EI":
            y_best = np.max(self.y_samples) if self.y_samples is not None and len(self.y_samples) > 0 else -np.inf
            return self._expected_improvement(X, self.model, y_best)
        else:
            raise NotImplementedError(f"Acquisition function '{self.acquisition_function}' not implemented.")

    def optimize(self, objective_fn, bounds: list[tuple[float, float]], budgets: int):
        """Main optimization loop."""
        n_dim = len(bounds)

        # Initial sampling
        initial_X = self._sample_points(self.n_initial_points, bounds)
        initial_y = np.array([objective_fn(x) for x in initial_X])
        self.X_samples = initial_X
        self.y_samples = initial_y

        n_iterations = budgets - self.n_initial_points

        for _ in range(n_iterations):
            # Fit the model
            self._fit_model(self.X_samples, self.y_samples)

            # Generate candidate points (can be more sophisticated)
            candidate_samples = self._sample_points(100, bounds)

            # Calculate acquisition function
            acquisition_values = self._acquisition_function(candidate_samples)

            # Select the next point
            next_point = candidate_samples[np.argmax(acquisition_values)]

            # Evaluate the objective function
            next_value = objective_fn(next_point)

            # Add to samples
            self.X_samples = np.vstack((self.X_samples, next_point))
            self.y_samples = np.append(self.y_samples, next_value)

        best_index = np.argmin(self.y_samples)
        best_value = self.y_samples[best_index]
        best_params = self.X_samples[best_index]

        return best_value, best_params

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm

class BayesianOptimization:
    def __init__(self):
        # Initialize optimizer settings
        self.kernel = ConstantKernel() * Matern(nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)

    def _sample_points(self, n_points, bounds):
        # Sample points using Latin Hypercube Sampling
        import pyDOE
        samples = pyDOE.lhs(2, n_points, criterion='maximin')
        scaled_samples = bounds[0] + (bounds[1] - bounds[0]) * samples
        return scaled_samples

    def _fit_model(self, X, y):
        # Fit and tune GP surrogate model
        self.gpr.fit(X, y)

    def _acquisition_function(self, X, y_best):
        # Implement Expected Improvement acquisition function
        mean, std = self.gpr.predict(X, return_std=True)
        z = (mean - y_best) / std
        ei = (mean - y_best) * norm.cdf(z) + std * norm.pdf(z)
        return ei

    def optimize(self, objective_fn, bounds, budgets):
        # Main minimize optimization loop
        n_initial_points = 10
        X_init = self._sample_points(n_initial_points, bounds)
        y_init = np.array([objective_fn(x) for x in X_init])
        self._fit_model(X_init, y_init)
        y_best = np.min(y_init)
        x_best = X_init[np.argmin(y_init)]

        for _ in range(budgets - n_initial_points):
            # Generate candidate points
            candidate_points = self._sample_points(100, bounds)
            # Evaluate acquisition function for candidate points
            ei_values = self._acquisition_function(candidate_points, y_best)
            # Select point with highest acquisition value
            next_point = candidate_points[np.argmax(ei_values)]
            # Evaluate objective function at selected point
            next_y = objective_fn(next_point)
            # Update surrogate model
            self._fit_model(np.vstack((X_init, next_point)), np.hstack((y_init, next_y)))
            # Update best point and value
            if next_y < y_best:
                y_best = next_y
                x_best = next_point
            X_init = np.vstack((X_init, next_point))
            y_init = np.hstack((y_init, next_y))

        return x_best, y_best

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, DotProduct
from scipy.optimize import minimize

class EnsembleGP_ThompsonSampling:
    def __init__(self):
        self.kernels = [
            RBF(length_scale=1.0),
            Matern(length_scale=1.0, nu=2.5),
            RBF(length_scale=1.0) + ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * DotProduct(sigma_0=1.0, sigma_0_bounds="fixed")
        ]
        self.gps = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42) for kernel in self.kernels]
        self.n_dim = None
        self.bounds = None
    
    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.n_dim, scramble=False)
        points = sampler.random(n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])
    
    def _fit_model(self, X, y):
        for gp in self.gps:
            gp.fit(X, y)
    
    def _thompson_sampling(self, X):
        sampled_functions = []
        for gp in self.gps:
            y_sampled = gp.sample_y(X.reshape(-1,self.n_dim), n_samples=1, random_state=np.random.randint(0, 100000)).flatten()
            sampled_functions.append(y_sampled)
        
        y_sampled_ensemble = np.mean(np.array(sampled_functions),axis=0)
        
        
        
        return X[np.argmin(y_sampled_ensemble)]
    
    def optimize(self, objective_fn, bounds:tuple[list[float],list[float]], budget:int):
        self.bounds = bounds
        self.n_dim = len(bounds[0])
        n_initial_points = 5 * self.n_dim
        X_init = self._sample_points(n_initial_points)
        y_init = np.array([objective_fn(x) for x in X_init])
        X = X_init
        y = y_init
        
        best_value = np.min(y)
        best_params = X[np.argmin(y)]
        
        for i in range(budgets):
            self._fit_model(X, y)
            X_next = self._thompson_sampling(self._sample_points(1000))
            y_next = objective_fn(X_next)
            X = np.vstack((X, X_next))
            y = np.append(y, y_next)
            
            if y_next < best_value:
                best_value = y_next
                best_params = X_next
            
        return best_value, best_params

        
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AckleyBO2:
    def __init__(self, n_init_points=5, xi=0.01):
        # Initialize optimizer settings
        self.n_init_points = n_init_points
        self.xi = xi
        # Configure acquisition function
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.X = None
        self.y = None
        self.best_y = float('inf')
        self.best_x = None

    def _sample_points(self, n_points, bounds):
        # sample points
        points = []
        for _ in range(n_points):
          point = [np.random.uniform(low, high) for low, high in bounds]
          points.append(point)
        return np.array(points)
    
    def _fit_model(self, X, y):
        # Fit/update surrogate model 
        self.gp.fit(X, y)
    
    def _acquisition_function(self, X):
        # Implement acquisition function (e.g., EI, UCB)
        # Handle exploration-exploitation trade-off
        mu, sigma = self.gp.predict(X, return_std=True)
        
        imp = mu - self.best_y - self.xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        return ei

    def _propose_next_point(self, bounds):
        # Find the next point to evaluate by maximizing the acquisition function
        def min_obj(X):
          return -self._acquisition_function(X.reshape(1, -1))[0]

        starting_points = self._sample_points(100, bounds)
        min_val = float('inf')
        min_x = None
        for start_point in starting_points:
          res = minimize(min_obj, start_point, bounds=bounds, method="L-BFGS-B")
          if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
        
        return min_x
    
    def optimize(self, objective_fn, bounds:list[tuple[float, float]], budget:int):
        # Main optimization loop
        # decide initial points and n_iterations based on budgets
        # Track progress and convergence
        # Return best value and best params
        
        # Initialize:
        init_X = self._sample_points(self.n_init_points, bounds)
        init_y = np.array([objective_fn(x) for x in init_X])
        
        self.X = init_X
        self.y = init_y
        self.best_y = np.min(init_y)
        self.best_x = init_X[np.argmin(init_y)]
        
        self._fit_model(self.X, self.y)
        
        # Main optimization loop (for 'budget' iterations):
        for _ in range(budget):
            next_x = self._propose_next_point(bounds)
            next_y = objective_fn(next_x)
            
            self.X = np.vstack((self.X, next_x))
            self.y = np.append(self.y, next_y)
            
            if next_y < self.best_y:
                self.best_y = next_y
                self.best_x = next_x

            self._fit_model(self.X, self.y)
            
        return self.best_y, self.best_x

import numpy as np
from scipy.stats import qmc
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class DeepEnsembleQMCBO:
    def __init__(self, n_ensemble_members=5, n_hidden_units=50, n_candidate_samples=100):
        self.n_ensemble_members = n_ensemble_members
        self.n_hidden_units = n_hidden_units
        self.n_candidate_samples = n_candidate_samples
        self.ensemble = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_train = None
        self.y_train = None

    def _sample_points(self, n_points, bounds):
        sampler = qmc.Sobol(d=len(bounds[0]), scramble=True)
        points = sampler.random(n_points)
        return qmc.scale(points, bounds[0], bounds[1])

    def _fit_model(self, X, y):
        self.scaler_X.fit(X)
        self.scaler_y.fit(y.reshape(-1, 1))
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).ravel()

        self.ensemble = [MLPRegressor(hidden_layer_sizes=(self.n_hidden_units,), 
                                      activation='relu', 
                                      solver='adam', 
                                      max_iter=200, 
                                      random_state=i, 
                                      early_stopping=True) 
                         for i in range(self.n_ensemble_members)]
        for model in self.ensemble:
            model.fit(X_scaled, y_scaled)

    def _ThompsonSampling(self, bounds):
        candidate_points = self._sample_points(self.n_candidate_samples, bounds)
        
        if not self.ensemble:
            return candidate_points[np.random.choice(len(candidate_points))]

        ensemble_predictions = np.array([model.predict(self.scaler_X.transform(candidate_points)) for model in self.ensemble])
        
        # Sample one prediction for each candidate from each ensemble member
        sampled_predictions = ensemble_predictions[np.arange(self.n_ensemble_members), :, np.random.randint(self.n_candidate_samples, size=self.n_ensemble_members)]

        # For each candidate point, calculate the average sampled prediction
        avg_predictions = np.mean(sampled_predictions, axis=0)

        # Select the candidate with the minimum average prediction (for minimization)
        best_index = np.argmin(avg_predictions)
        return candidate_points[best_index]


    def optimize(self, objective_fn, bounds: tuple[list[float], list[float]], budget: int) -> tuple[float, list[float]]:
        n_dim = len(bounds[0])
        n_initial_points = min(2 * n_dim + 1, budget // 5)  # Heuristic for initial points
        n_initial_points = 30
        n_iterations = budget - n_initial_points

        # Initial sampling
        initial_X = self._sample_points(n_initial_points, bounds)
        initial_y = np.array([objective_fn(x) for x in initial_X])

        self.X_train = initial_X
        self.y_train = initial_y

        best_value = np.min(self.y_train)
        best_params = self.X_train[np.argmin(self.y_train)]

        for _ in range(n_iterations):
            self._fit_model(self.X_train, self.y_train)
            
            # Thompson Sampling to get the next point
            next_point = self._ThompsonSampling(bounds)
            next_point_value = objective_fn(next_point)

            self.X_train = np.vstack((self.X_train, next_point))
            self.y_train = np.append(self.y_train, next_point_value)

            if next_point_value < best_value:
                best_value = next_point_value
                best_params = next_point

        return best_value, best_params.tolist()

from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class AdvancedDropWaveBO:
    def __init__(self):
        # Initialize optimizer settings
        self.acquisition_function = 'EI'
        self.surrogate_model = GaussianProcessRegressor(kernel=Matern())
        self.n_initial_points = 10
        self.model_losses = []
        self.loss_name = 'Mean Squared Error'

    def _sample_points(self, n_points):
        # Use quasi-Monte Carlo for initial sampling
        sampler = qmc.Halton(d=2, scramble=False)
        sample = sampler.random(n=n_points)
        # Scale to bounds
        lower_bound, upper_bound = -5.12, 5.12
        return lower_bound + (upper_bound - lower_bound) * sample

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        self.surrogate_model.fit(X, y)
        # Evaluate model loss (MSE for GPR)
        y_pred = self.surrogate_model.predict(X)
        loss = np.mean((y_pred - y) ** 2)
        self.model_losses.append(loss)

    def _acquisition_function(self, X):
        # Implement Expected Improvement acquisition function
        y_pred, std_pred = self.surrogate_model.predict(X, return_std=True)
        best_y = self.surrogate_model.y_train_.min()
        improvement = best_y - y_pred
        z = improvement / std_pred
        ei = improvement * (1 - np.exp(-z)) + std_pred * z * np.exp(-z)
        return ei

    def optimize(self, objective_fn, bounds: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, str], int]:
        # Main minimize optimization loop
        lower_bound, upper_bound = bounds
        n_initial_points = min(budget, self.n_initial_points)
        n_iterations = budget - n_initial_points
        X_init = self._sample_points(n_initial_points)
        y_init = objective_fn(X_init)
        self._fit_model(X_init, y_init)
        X_all, y_all = X_init, y_init
        for _ in range(n_iterations):
            # Sample new point based on acquisition function
            new_X = self._sample_point_acquisition()
            new_y = objective_fn(new_X)
            X_all = np.vstack((X_all, new_X))
            y_all = np.vstack((y_all, new_y))
            self._fit_model(X_all, y_all)
        return y_all, X_all, (np.array(self.model_losses), self.loss_name), n_initial_points

    def _sample_point_acquisition(self):
        # Sample a new point based on the acquisition function
        # For simplicity, we use a grid search over the bounds
        grid_size = 100
        x_values = np.linspace(-5.12, 5.12, grid_size)
        x_grid, y_grid = np.meshgrid(x_values, x_values)
        points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        acquisition_values = self._acquisition_function(points)
        best_index = np.argmax(acquisition_values)
        return points[best_index].reshape(1, -1)

from typing import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class RosenbrockBO:
    def __init__(self):
        # Initialize optimizer settings
        self.n_initial_points = 10
        self.kernel = Matern(nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)

    def _sample_points(self, n_points, bounds):
        # Sample points using QMC
        sampler = qmc.Sobol(d=bounds.shape[0], scramble=True)
        points = sampler.random(n=n_points)
        # Scale points to be within bounds
        scaled_points = bounds[:, 0] + points * (bounds[:, 1] - bounds[:, 0])
        return scaled_points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        self.gpr.fit(X, y)

    def _acquisition_function(self, X, i):
        # Implement UCB acquisition function
        mu, sigma = self.gpr.predict(X, return_std=True)
        beta = 2 * np.log(i + 1)
        ucb = mu + beta * sigma
        return ucb

    def optimize(self, objective_fn: Callable[[np.ndarray], np.ndarray], bounds: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, str], int]:
        # Main minimize optimization loop
        self.n_iterations = budget - self.n_initial_points
        X_initial = self._sample_points(self.n_initial_points, bounds)
        y_initial = objective_fn(X_initial)
        all_x = X_initial
        all_y = y_initial
        model_losses = []
        for i in range(self.n_iterations):
            self._fit_model(all_x, all_y)
            ucb = self._acquisition_function(all_x, i)
            new_x = all_x[np.argmax(ucb)]
            new_y = objective_fn(new_x.reshape(1, -1))
            all_x = np.vstack((all_x, new_x))
            all_y = np.vstack((all_y, new_y))
            model_loss = -self.gpr.log_marginal_likelihood(self.gpr.kernel_.theta)
            model_losses.append(model_loss)
        return all_y, all_x, (np.array(model_losses), 'Negative Log Likelihood'), self.n_initial_points

import botorch.test_functions.synthetic as test_functions
import torch

ackley = test_functions.Ackley(dim=2)

def ackley_fn(x):
    tensor_x = torch.tensor(x, dtype=torch.float32)  
    y = ackley(tensor_x)
    print(y, x)
    return y.reshape(-1,1).numpy()

def a_fn(x):
    y = x[0] ** 2 + (x[1] - 1) ** 2 - 1
    print(y, x)
    return y
ackley_bo = AckleyBO()
# bounds = np.array([[-2.768, 2.768], [-2.768, 2.768]])

bounds = ackley.bounds.numpy()
# r = ackley_bo.optimize(ackley_fn, bounds=bounds, budgets=20)
# r = BayesianOptimization().optimize(ackley_fn, bounds=bounds, budgets=20)
# r = EnsembleGP_ThompsonSampling().optimize(ackley_fn, bounds=bounds, budgets=20)
# r = random_search(a_fn, bounds=bounds, budgets=20)
# r = AckleyBO2().optimize(ackley_fn, bounds=bounds, budget=20)
# r = SobolEIUCB_BO().optimize(ackley_fn, bounds=bounds, budget=20)
# r = DeepEnsembleQMCBO().optimize(ackley_fn, bounds=bounds, budget=20)
# r = AdvancedDropWaveBO().optimize(ackley_fn, bounds=bounds, budget=20)
r = RosenbrockBO().optimize(ackley_fn, bounds=bounds, budget=20)
print(r)
