# Description
**Bayesian Optimization with Deep Kernel Learning and Contextual Bandit (DKLCBBO):** This algorithm combines Deep Kernel Learning (DKL) to learn a task-specific kernel for the Gaussian Process Regression (GPR) model and a Contextual Bandit approach to dynamically balance exploration and exploitation. DKL uses a neural network to map the input space to a feature space where a simple RBF kernel can effectively model the function's covariance structure. The Contextual Bandit component treats each iteration of BO as a bandit problem, where the "context" is the current state of the optimization (e.g., evaluated points, GPR predictions) and the "arms" are candidate points to evaluate. A bandit policy (e.g., Thompson Sampling) is used to select the next point, adapting the exploration-exploitation trade-off based on the observed rewards (function values). This approach aims to improve the adaptability and efficiency of BO, especially in complex and high-dimensional search spaces.

# Justification
1.  **Deep Kernel Learning (DKL):** Addresses the limitation of standard GPR with fixed kernels by learning a kernel tailored to the specific function being optimized. This allows the GPR to better capture the function's structure and make more accurate predictions.
2.  **Contextual Bandit:** Provides a principled way to balance exploration and exploitation. The bandit policy adapts its behavior based on the current state of the optimization, allowing the algorithm to dynamically adjust its exploration-exploitation trade-off.
3.  **Thompson Sampling:** A computationally efficient bandit policy that naturally balances exploration and exploitation. It samples from the posterior distribution of each "arm" (candidate point), selecting points with high uncertainty or high predicted reward.
4.  **Error Analysis:** The VAEBO had a dimension mismatch. DKL addresses this by learning a feature representation that the GPR can use. The HyperImprovBO error was due to a temporary object lacking a model. This is avoided by ensuring the DKL model is trained before use.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import torch
import torch.nn as nn
import torch.optim as optim

class DKLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DKLNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DKLCBBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim

        self.best_y = np.inf
        self.best_x = None

        self.hidden_dim = 32
        self.feature_dim = min(10, dim)
        self.dkl_net = DKLNet(dim, self.hidden_dim, self.feature_dim)
        self.optimizer = optim.Adam(self.dkl_net.parameters(), lr=1e-3)
        self.epochs = 10

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dkl_net.to(self.device)

        self.thompson_sampling_factor = 1.0 # Adjust Thompson sampling

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_dkl(self, X, y):
        # Train the DKL network
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        self.dkl_net.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            features = self.dkl_net(X_tensor)
            loss = torch.mean((features.T @ features) * (y_tensor @ y_tensor.T)) # simple loss function
            loss.backward()
            self.optimizer.step()

    def _get_features(self, X):
        # Get features from DKL network
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.dkl_net.eval()
        with torch.no_grad():
            features = self.dkl_net(X_tensor).cpu().numpy()
        return features

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Thompson Sampling
        acquisition = mu + self.thompson_sampling_factor * sigma * np.random.randn(len(X), 1)
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)
        candidate_features = self._get_features(candidate_points)

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_features)

        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        return next_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Train DKL
        self._fit_dkl(self.X, self.y)
        X_features = self._get_features(self.X)

        self.model = self._fit_model(X_features, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.n_init, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Retrain DKL
            self._fit_dkl(self.X, self.y)
            X_features = self._get_features(self.X)
            self.model = self._fit_model(X_features, self.y)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<DKLCBBO>", line 158, in __call__
 158->         self._fit_dkl(self.X, self.y)
  File "<DKLCBBO>", line 69, in _fit_dkl
  67 |             self.optimizer.zero_grad()
  68 |             features = self.dkl_net(X_tensor)
  69->             loss = torch.mean((features.T @ features) * (y_tensor @ y_tensor.T)) # simple loss function
  70 |             loss.backward()
  71 |             self.optimizer.step()
RuntimeError: The size of tensor a (5) must match the size of tensor b (10) at non-singleton dimension 1
