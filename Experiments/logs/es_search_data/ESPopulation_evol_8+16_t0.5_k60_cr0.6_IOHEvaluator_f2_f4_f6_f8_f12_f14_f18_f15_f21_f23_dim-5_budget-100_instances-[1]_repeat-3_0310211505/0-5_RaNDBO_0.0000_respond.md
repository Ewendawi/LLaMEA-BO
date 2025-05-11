# Description
RaNDBO: Randomly-initialized Neural Dynamics Bayesian Optimization is a novel Bayesian optimization algorithm that leverages neural dynamics to guide the search process. It uses a Gaussian Process Regression (GPR) model as a surrogate and employs a neural network, initialized with random weights and biases, to generate candidate points. The acquisition function is Expected Improvement (EI). To enhance exploration, the algorithm introduces a mechanism for periodically re-initializing the neural network with new random weights, promoting diversity in the search. The neural network acts as a dynamic point generator, and its re-initialization ensures that the algorithm explores different regions of the search space throughout the optimization process.

# Justification
This algorithm is designed to be diverse from the previous ones in several aspects:
1.  **Point Generation with Neural Dynamics:** Instead of relying on Latin Hypercube Sampling or optimization of the acquisition function, it uses a neural network to generate candidate points. This introduces a dynamic and potentially more adaptable point generation mechanism.
2.  **Random Re-initialization:** To avoid getting stuck in local optima and to promote exploration, the neural network is periodically re-initialized with random weights. This is a unique exploration strategy not present in the other algorithms.
3.  **Simplicity and Computational Efficiency:** The neural network is kept simple (a single hidden layer) to ensure computational efficiency. The re-initialization strategy is also computationally inexpensive.
4.  **Exploration-Exploitation Balance:** The random re-initialization of the neural network provides a good balance between exploration and exploitation. Initially, the network explores the search space randomly. As the optimization progresses, the GPR model guides the search towards promising regions.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import torch
import torch.nn as nn
import torch.optim as optim

class RaNDBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10*dim, self.budget//5)
        self.nn_reinit_freq = 10 # Frequency of neural network re-initialization
        self.nn_hidden_size = 32 # Number of hidden units in the neural network
        self.nn_lr = 0.01 # Learning rate for the neural network
        self.nn_epochs = 50 # Number of training epochs for the neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

        # Define the neural network
        self.nn = nn.Sequential(
            nn.Linear(self.dim, self.nn_hidden_size),
            nn.ReLU(),
            nn.Linear(self.nn_hidden_size, self.dim),
            nn.Tanh() # Output between -1 and 1
        ).to(self.device)

        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.nn_lr)

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1)) # Return zeros if no data is available

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (sigma + 1e-9)  # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid division by zero

        return ei

    def _generate_candidate_points(self, n_candidates):
        # Generate candidate points using the neural network
        # return array of shape (n_candidates, n_dims)

        # Generate random inputs to the neural network
        z = torch.randn(n_candidates, self.dim).to(self.device)

        # Generate candidate points
        with torch.no_grad():
            X_cand_scaled = self.nn(z).cpu().numpy()

        # Scale the points to the search space
        X_cand = qmc.scale(X_cand_scaled, self.bounds[0], self.bounds[1])
        return X_cand

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points using the neural network
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._generate_candidate_points(n_candidates)

        # Calculate acquisition function values
        acq_values = self._acquisition_function(X_cand)

        # Select top-k points based on acquisition function values
        top_indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_cand[top_indices]

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
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def _reinitialize_nn(self):
        # Re-initialize the neural network with random weights
        for layer in self.nn:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.zero_()
        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.nn_lr) # reinitialize the optimizer

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        iter_count = 0
        while self.n_evals < self.budget:
            # Fit the model
            self.model = self._fit_model(self.X, self.y)

            # Select next points
            X_next = self._select_next_points(batch_size)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Re-initialize neural network periodically
            if iter_count % self.nn_reinit_freq == 0:
                self._reinitialize_nn()

            iter_count += 1

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<RaNDBO>", line 153, in __call__
 153->             X_next = self._select_next_points(batch_size)
  File "<RaNDBO>", line 99, in _select_next_points
  99->         X_cand = self._generate_candidate_points(n_candidates)
  File "<RaNDBO>", line 88, in _generate_candidate_points
  86 | 
  87 |         # Scale the points to the search space
  88->         X_cand = qmc.scale(X_cand_scaled, self.bounds[0], self.bounds[1])
  89 |         return X_cand
  90 | 
ValueError: Sample is not in unit hypercube
