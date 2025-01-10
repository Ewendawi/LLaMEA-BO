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
        self.losses = []

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

        # Calculate Average loss of the ensemble
        ensemble_predictions = np.array([model.predict(X_scaled) for model in self.ensemble])
        ensemble_loss = np.mean((ensemble_predictions - y_scaled) ** 2, axis=0)
        self.losses.append(ensemble_loss)

    def _ThompsonSampling(self, bounds):
        candidate_points = self._sample_points(self.n_candidate_samples, bounds)
        
        if not self.ensemble:
            return candidate_points[np.random.choice(len(candidate_points))]

        ensemble_predictions = np.array([model.predict(self.scaler_X.transform(candidate_points)) for model in self.ensemble])
        
        # Sample one prediction for each candidate from each ensemble member
        sampled_predictions = ensemble_predictions[np.arange(self.n_ensemble_members), np.random.randint(self.n_candidate_samples, size=self.n_ensemble_members)]

        # For each candidate point, calculate the average sampled prediction
        avg_predictions = np.mean(sampled_predictions, axis=0)

        # Select the candidate with the minimum average prediction (for minimization)
        best_index = np.argmin(avg_predictions)
        return candidate_points[best_index]


    def optimize(self, objective_fn, bounds: tuple[list[float], list[float]], budget: int) -> tuple[float, list[float]]:
        n_dim = len(bounds[0])
        n_initial_points = min(2 * n_dim + 1, budget // 5)  # Heuristic for initial points
        n_initial_points = int(0.4 * budget)
        n_iterations = budget - n_initial_points

        # Initial sampling
        initial_X = self._sample_points(n_initial_points, bounds)
        initial_y = objective_fn(initial_X)

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

        return self.y_train, self.X_train, (self.losses, "MSE"), n_initial_points