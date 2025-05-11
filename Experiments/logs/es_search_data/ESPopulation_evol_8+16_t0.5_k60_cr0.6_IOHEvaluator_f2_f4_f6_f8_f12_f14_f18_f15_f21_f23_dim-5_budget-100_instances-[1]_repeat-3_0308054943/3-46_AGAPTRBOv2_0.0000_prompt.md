You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- GADETBO: 0.1795, 109.58 seconds, **GADETBO: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization:** This algorithm combines the strengths of AGATBO and DEBO, focusing on balancing exploration and exploitation by incorporating gradient information, diversity enhancement, and a trust region strategy. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI), a gradient-based exploration term, and a distance-based diversity term, all evaluated within an adaptive trust region. The trust region size is adjusted based on the agreement between the GPR model and the true objective function. Clustering is used to identify diverse regions for sampling.


- AGATBO: 0.1767, 510.80 seconds, **AGATBO: Adaptive Gradient-Aware Trust Region Bayesian Optimization:** This algorithm combines the strengths of AGABO and ATRBO, incorporating gradient information into the acquisition function within an adaptive trust region framework. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI) and a gradient-based exploration term, evaluated within a trust region. The size of the trust region is adaptively adjusted based on the agreement between the GPR model and the true objective function. Halton sequence is used for initial exploration, and Sobol sequence is used for sampling within the trust region.


- AGAPTRBO: 0.1727, 512.47 seconds, **AGAPTRBO: Adaptive Gradient-Aware Pareto Trust Region Bayesian Optimization:** This algorithm combines the strengths of AGATBO and APTRBO by integrating gradient information into the Pareto-based multi-acquisition function approach within an adaptive trust region framework. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a Pareto front of Expected Improvement (EI), Probability of Improvement (PI), Upper Confidence Bound (UCB), and a gradient-based exploration term, evaluated within a trust region. The size of the trust region is adaptively adjusted based on the agreement between the GPR model and the true objective function. The UCB's kappa parameter is dynamically adjusted based on the trust region size to balance exploration and exploitation.


- APTRBO: 0.1724, 104.60 seconds, **Adaptive Pareto Trust Region Bayesian Optimization (APTRBO):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATRBO) and Pareto-based Acquisition Steering Bayesian Optimization (PaSBO) to achieve a more robust and efficient optimization process. It adaptively adjusts the trust region based on model agreement, uses a Pareto-based approach to manage multiple acquisition functions (EI, PI, UCB) within the trust region, and incorporates a dynamic exploration-exploitation balance by adjusting the UCB kappa parameter based on the trust region size. A Matern kernel is used for the Gaussian Process Regression (GPR) model.


- ARPTRBO: 0.1723, 288.39 seconds, **Adaptive Regularized Pareto Trust Region Bayesian Optimization (ARPTRBO):** This algorithm synergistically integrates the strengths of APTRBO and ARBO. It employs a trust region, adaptively adjusted based on model agreement, and utilizes a Pareto-based approach to manage multiple acquisition functions (EI, PI, UCB) within the trust region, similar to APTRBO. Additionally, it incorporates the adaptive regularization term and dynamic batch size strategy from ARBO to enhance exploration and exploitation. The regularization term is modified to include uncertainty from the Gaussian Process, and an exploration factor is added to the acquisition function. This combination aims to balance exploration and exploitation more effectively, leading to improved performance.


- TREDABO: 0.1716, 73.66 seconds, **Trust Region Enhanced Diversity Bayesian Optimization (TREDABO):** This algorithm combines the strengths of Trust Region Bayesian Optimization (TRBO) and Diversity Enhanced Bayesian Optimization (DEBO) to achieve a robust and efficient optimization process. It employs a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function combines Expected Improvement (EI) with a distance-based diversity term, similar to DEBO, encouraging exploration of diverse regions. A trust region mechanism, similar to APTRBO, adaptively adjusts the search space based on the agreement between the GPR model and the true objective function. To enhance exploration within the trust region, a clustering-based approach is used to identify diverse regions, and sampling is performed around cluster centers. The UCB kappa parameter is dynamically adjusted based on the trust region size to balance exploration and exploitation.


- GARBO: 0.1707, 567.17 seconds, **GARBO: Gradient-Aware Adaptive Regularized Bayesian Optimization with Trust Region:** This algorithm combines gradient information, adaptive regularization, and a trust region approach to enhance Bayesian optimization. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel. The acquisition function combines Expected Improvement (EI), a gradient-based exploration term, and an adaptive regularization term within a trust region. The trust region size is adaptively adjusted based on model agreement. A dynamic batch size strategy is also incorporated.


- DEBO: 0.1705, 63.62 seconds, **DEBO: Diversity Enhanced Bayesian Optimization:** This algorithm focuses on enhancing diversity in Bayesian Optimization using a combination of techniques. It employs a Gaussian Process Regression (GPR) model with a Radial Basis Function (RBF) kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI) and a distance-based diversity term. This diversity term encourages the selection of points that are far away from previously evaluated points. A clustering-based approach is used to identify diverse regions in the search space. The initial exploration is performed using a Sobol sequence to ensure good coverage of the search space.




The selected solution to update is:
**AGAPTRBO: Adaptive Gradient-Aware Pareto Trust Region Bayesian Optimization:** This algorithm combines the strengths of AGATBO and APTRBO by integrating gradient information into the Pareto-based multi-acquisition function approach within an adaptive trust region framework. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a Pareto front of Expected Improvement (EI), Probability of Improvement (PI), Upper Confidence Bound (UCB), and a gradient-based exploration term, evaluated within a trust region. The size of the trust region is adaptively adjusted based on the agreement between the GPR model and the true objective function. The UCB's kappa parameter is dynamically adjusted based on the trust region size to balance exploration and exploitation.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
import warnings

class AGAPTRBO:
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
        self.gradient_weight = 0.01
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.acquisition_functions = ['ei', 'pi', 'ucb', 'grad']
        self.ucb_kappa = 2.0 # Initial value
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None

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
        try:
            model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=1e-5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            self.kernel = model.kernel_  # Update kernel with optimized parameters
            return model
        except Exception as e:
            print(f"GP fitting failed: {e}. Returning None.")
            return None

    def _acquisition_function(self, X, model, acq_type='ei'):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            if acq_type != 'grad':
                return np.zeros_like(mu)
            else:
                return np.zeros((X.shape[0], 1))

        imp = self.best_y - mu
        Z = imp / sigma

        if acq_type == 'ei':
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma
            return ei
        elif acq_type == 'pi':
            pi = norm.cdf(Z)
            return pi
        elif acq_type == 'ucb':
            ucb = mu + self.ucb_kappa * sigma
            return ucb
        elif acq_type == 'grad':
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            return self.gradient_weight * gradient_norm
        else:
            raise ValueError("Invalid acquisition function type.")

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        for i in range(self.dim):
            def obj(x):
                x_prime = x.copy()
                x_prime[i] += 1e-6
                return model.predict(x_prime.reshape(1, -1))[0]
            def obj0(x):
                return model.predict(x.reshape(1, -1))[0]
            dmu_dx[:, i] = (np.array([obj(x) for x in X]) - np.array([obj0(x) for x in X]))/1e-6
        return dmu_dx

    def _is_pareto_efficient(self, points):
        """
        Find the pareto-efficient points
        :param points: An n by m matrix of points
        :return: A boolean array of length n, True for pareto-efficient points, False otherwise
        """
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(points[is_efficient] > c, axis=1) | (points[is_efficient] == c).all(axis=1)
                is_efficient[i] = True  # Keep current point
        return is_efficient

    def _select_next_points(self, batch_size, model, trust_region_center):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Sample within the trust region using Sobol sequence
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)

        # Scale and shift samples to be within the trust region
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)

        # Clip samples to stay within the problem bounds
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        acquisition_values = np.zeros((scaled_samples.shape[0], len(self.acquisition_functions)))

        for i, acq_type in enumerate(self.acquisition_functions):
            acquisition_values[:, i] = self._acquisition_function(scaled_samples, model, acq_type).flatten()

        # Find Pareto front
        is_efficient = self._is_pareto_efficient(acquisition_values)
        pareto_points = scaled_samples[is_efficient]

        # Select top batch_size points from Pareto front
        if len(pareto_points) > batch_size:
            # Randomly select if more than batch_size
            indices = np.random.choice(len(pareto_points), batch_size, replace=False)
            selected_points = pareto_points[indices]
        else:
            # Use all Pareto points if less than or equal to batch_size
            selected_points = pareto_points

            # If still less than batch_size, sample randomly
            if len(selected_points) < batch_size:
                remaining = batch_size - len(selected_points)
                random_points = self._sample_points(remaining)
                selected_points = np.vstack((selected_points, random_points))

        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

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

        trust_region_center = self.X[np.argmin(self.y)] # Initialize trust region center

        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, self.model, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = self.model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.ucb_kappa = min(self.ucb_kappa * 1.1, self.ucb_kappa_max) # Increase kappa for exploration
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion
                self.ucb_kappa = max(self.ucb_kappa * 0.9, self.ucb_kappa_min) # Decrease kappa for exploitation

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x

```
The algorithm AGAPTRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1727 with standard deviation 0.1099.

took 512.47 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

