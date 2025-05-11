You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- EHBBO: 0.1605, 56.90 seconds, **Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines an initial space-filling design using Latin Hypercube Sampling (LHS), a Gaussian Process Regression (GPR) surrogate model, and a hybrid acquisition function that balances exploration and exploitation. The acquisition function combines Expected Improvement (EI) and a distance-based exploration term. A simple but effective batch selection strategy is used to select multiple points for evaluation in each iteration.


- SETSBO: 0.1544, 165.78 seconds, **Surrogate Ensemble with Thompson Sampling and Local Search (SETSBO):** This algorithm leverages an ensemble of surrogate models (Gaussian Process Regression with different kernels) to improve the robustness and accuracy of the Bayesian optimization process. It uses Thompson Sampling for acquisition, which naturally balances exploration and exploitation. Furthermore, it integrates a local search strategy (L-BFGS-B) to refine the search around promising regions identified by Thompson Sampling. The ensemble of surrogates provides a more reliable estimate of the function landscape, while Thompson Sampling offers a computationally efficient way to select the next points. Local search further enhances the exploitation of promising regions.


- DensiTreeBO: 0.1502, 21.60 seconds, **DensiTreeBO (DTBO):** This algorithm introduces a novel approach to Bayesian Optimization by employing a density-based clustering technique to identify promising regions in the search space. Instead of relying solely on the Gaussian Process Regression (GPR) model and acquisition function to select the next points, DTBO uses a density estimation method (Kernel Density Estimation - KDE) on the evaluated points to locate high-density clusters. These clusters are then used to guide the selection of new points, promoting exploration within promising areas and exploitation of the best-performing clusters. The acquisition function is used to refine the point selection within the clusters. This approach aims to improve the balance between exploration and exploitation, especially in multimodal or complex search spaces.


- BONGIBO: 0.1464, 37.41 seconds, **Bayesian Optimization with Noisy Handling and Gradient-based Improvement (BONGIBO):** This algorithm addresses the limitations of standard BO by incorporating a mechanism to handle potential noise in the function evaluations and leveraging gradient information to refine the search. It uses a Gaussian Process Regression (GPR) surrogate model with noise variance estimation and a modified Expected Improvement (EI) acquisition function. The key innovation is the integration of gradient-based local search to improve the exploitation of promising regions, which is particularly useful when the function landscape has local optima or is noisy. The initial points are sampled using a Sobol sequence for better space coverage than LHS.


- ATRBO: 0.1377, 209.56 seconds, **Adaptive Trust Region Bayesian Optimization (ATRBO):** This algorithm uses a Gaussian Process Regression (GPR) surrogate model and an Expected Improvement (EI) acquisition function within an adaptive trust region framework. The trust region size is adjusted based on the agreement between the GPR predictions and the actual function evaluations. A shrinking trust region encourages exploitation, while an expanding trust region promotes exploration. The initial points are sampled using Latin Hypercube Sampling (LHS).


- HyperImprovBO: 0.0000, 0.00 seconds, **HyperImprovBO (HIBO):** This algorithm introduces a novel hyperparameter optimization strategy within the Bayesian Optimization framework. It dynamically adjusts the acquisition function's exploration-exploitation trade-off and the Gaussian Process Regression (GPR) kernel parameters during the optimization process. HIBO employs a separate meta-optimization loop to tune these hyperparameters based on the observed performance of the BO algorithm. This adaptive hyperparameter tuning allows the algorithm to tailor its search strategy to the specific characteristics of the objective function, potentially leading to improved convergence and performance. The initial points are sampled using a Halton sequence.


- VAEBO: 0.0000, 0.00 seconds, **VariationalAutoencoderBO (VAEBO):** This algorithm uses a Variational Autoencoder (VAE) to learn a latent representation of the search space. The VAE is trained on the evaluated points, and the acquisition function is evaluated in the latent space. This allows the algorithm to explore the search space more efficiently, especially in high-dimensional problems or problems with complex dependencies. The algorithm uses Expected Improvement (EI) as the acquisition function. The initial points are sampled using Latin Hypercube Sampling (LHS).

The error in HyperImprovBO was due to the temporary BO object not having a model fitted. This is addressed in VAEBO by ensuring the VAE is trained before being used.


- DKLCBBO: 0.0000, 0.00 seconds, **Bayesian Optimization with Deep Kernel Learning and Contextual Bandit (DKLCBBO):** This algorithm combines Deep Kernel Learning (DKL) to learn a task-specific kernel for the Gaussian Process Regression (GPR) model and a Contextual Bandit approach to dynamically balance exploration and exploitation. DKL uses a neural network to map the input space to a feature space where a simple RBF kernel can effectively model the function's covariance structure. The Contextual Bandit component treats each iteration of BO as a bandit problem, where the "context" is the current state of the optimization (e.g., evaluated points, GPR predictions) and the "arms" are candidate points to evaluate. A bandit policy (e.g., Thompson Sampling) is used to select the next point, adapting the exploration-exploitation trade-off based on the observed rewards (function values). This approach aims to improve the adaptability and efficiency of BO, especially in complex and high-dimensional search spaces.




The selected solutions to update are:
## EHBBO
**Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines an initial space-filling design using Latin Hypercube Sampling (LHS), a Gaussian Process Regression (GPR) surrogate model, and a hybrid acquisition function that balances exploration and exploitation. The acquisition function combines Expected Improvement (EI) and a distance-based exploration term. A simple but effective batch selection strategy is used to select multiple points for evaluation in each iteration.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class EHBBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # Initial number of points

        self.best_y = np.inf
        self.best_x = None

        self.batch_size = min(10, dim) # Batch size for selecting points

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

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
        exploration = min_dist / np.max(min_dist)

        # Hybrid acquisition function
        acquisition = ei + 0.1 * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)  # Generate more candidates
        
        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)
        
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

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.batch_size, remaining_evals) # Adjust batch size to budget
            
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x

```
The algorithm EHBBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1605 with standard deviation 0.1015.

took 56.90 seconds to run.

## VAEBO
**VariationalAutoencoderBO (VAEBO):** This algorithm uses a Variational Autoencoder (VAE) to learn a latent representation of the search space. The VAE is trained on the evaluated points, and the acquisition function is evaluated in the latent space. This allows the algorithm to explore the search space more efficiently, especially in high-dimensional problems or problems with complex dependencies. The algorithm uses Expected Improvement (EI) as the acquisition function. The initial points are sampled using Latin Hypercube Sampling (LHS).

The error in HyperImprovBO was due to the temporary BO object not having a model fitted. This is addressed in VAEBO by ensuring the VAE is trained before being used.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # Mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh() # Output between -1 and 1 (scaled later)
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class VAEBO:
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

        self.latent_dim = min(10, dim // 2) # Reduced latent dimension
        self.vae = VAE(dim, self.latent_dim)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)
        self.epochs = 10

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device)

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_vae(self, X):
        # Train the VAE model
        X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        def loss_fn(recon_x, x, mu, logvar):
            recon_loss = torch.mean((recon_x - x)**2)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + 0.01 * kl_div # Add KL divergence loss
        
        self.vae.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.vae(X_train)
            loss = loss_fn(recon_batch, X_train, mu, logvar)
            loss.backward()
            self.optimizer.step()

    def _encode_points(self, X):
        # Encode points to latent space
        self.vae.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            mu, logvar = self.vae.encode(X_tensor)
            return mu.cpu().numpy()

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

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points in latent space
        candidate_points_latent = np.random.randn(100 * batch_size, self.latent_dim)

        # Decode candidate points to original space
        self.vae.eval()
        with torch.no_grad():
            candidate_points_latent_tensor = torch.tensor(candidate_points_latent, dtype=torch.float32).to(self.device)
            candidate_points = self.vae.decode(candidate_points_latent_tensor).cpu().numpy()

        # Clip to bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)

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

        # Train VAE
        self._fit_vae(self.X)

        # Encode initial points to latent space
        X_latent = self._encode_points(self.X)

        self.model = self._fit_model(X_latent, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.n_init, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Retrain VAE
            self._fit_vae(self.X)

            # Encode all points
            X_latent = self._encode_points(self.X)
            self.model = self._fit_model(X_latent, self.y)

        return self.best_y, self.best_x

```
An error occurred : Traceback (most recent call last):
  File "<VAEBO>", line 208, in __call__
 208->             next_X = self._select_next_points(batch_size)
  File "<VAEBO>", line 152, in _select_next_points
 152->         acquisition_values = self._acquisition_function(candidate_points)
  File "<VAEBO>", line 122, in _acquisition_function
 120 |         # calculate the acquisition function value for each point in X
 121 |         # return array of shape (n_points, 1)
 122->         mu, sigma = self.model.predict(X, return_std=True)
 123 |         mu = mu.reshape(-1, 1)
 124 |         sigma = sigma.reshape(-1, 1)
ValueError: X has 5 features, but GaussianProcessRegressor is expecting 2 features as input.


Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

