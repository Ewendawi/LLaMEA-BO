# This file contains the classes for the results of the evaluator.
import re
import math
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance, ConvexHull
from scipy.special import gamma



def _fill_nan(target, length):
    if isinstance(target, np.ndarray):
        if len(target) == 0:
            return np.full(length, np.nan)
        n_fill = length - len(target)
        if n_fill <= 0:
            return target
        return np.concatenate([target, np.full(n_fill, np.nan)])
    elif isinstance(target, list):
        if len(target) == 0:
            return [np.nan] * length
        n_fill = length - len(target)
        if n_fill <= 0:
            return target
        return target + [np.nan] * n_fill
    
    return target

class ConvergenceCurveAnalyzer:
    """Analyzes optimization convergence curves and calculates AOC metric."""

    def __init__(self, max_y=None, min_y=None, log_scale=False, shift_value=0):
        self.max_y = max_y
        self.min_y = min_y
        self.log_scale = log_scale
        self.shift_value = shift_value

    def get_convergence_curve(self, y_history):
        """Calculate minimum values seen so far at each step."""
        if not isinstance(y_history, np.ndarray):
            y_history = np.array(y_history)
        return np.minimum.accumulate(y_history)

    def calculate_aoc(self, y_history):
        """Calculate area over convergence curve."""
        if len(y_history) == 0:
            return 0.0

        shift_y = y_history - self.shift_value
        max_y = self.max_y if self.max_y is not None else np.max(y_history)
        min_y = self.min_y if self.min_y is not None else np.min(y_history)
        clip_y = np.clip(shift_y, min_y, max_y)
        conv_curve = self.get_convergence_curve(clip_y)

        if self.log_scale:
            log_conv_curve = np.log10(conv_curve)
            log_max_y = np.log10(max_y)
            log_min_y = np.log10(min_y)
            norm_curve = (log_conv_curve - log_min_y) / (log_max_y - log_min_y)
        else:
            norm_curve = (conv_curve - min_y) / (max_y - min_y)

        # Calculate AOC using trapezoidal rule
        x_vals = np.linspace(0, 1, len(norm_curve))
        aoc = np.trapz(1-norm_curve, x_vals)

        return aoc

# y = np.array([100, 79, 81, 71, 65, 15, -5, 45, 5, 5])
# aoc = ConvergenceCurveAnalyzer(max_y=200, min_y=0, log_scale=False, shift_value=-10).calculate_aoc(y)
# print(aoc)

# aoc1 = ConvergenceCurveAnalyzer(max_y=1e2, min_y=1e-8, log_scale=True, shift_value=shift_value).calculate_aoc(y)
# print(aoc1)

class CoverageCluster:
    def __init__(self, volume_type='rect', min_radius=0.1):
        self.points = []
        self.volume_type = volume_type
        self.min_radius = min_radius

        self._radius = None
        self._centroid = None
        self._strides = None
        self._volume = None

    def add_points(self, points):
        self.points.extend(points)

    def _update_radius(self):
        centroid = self._centroid
        distances = [distance.euclidean(p, centroid) for p in self.points]
        self._radius = np.max(distances)

    def _update_centroid(self):
        self._centroid = np.mean(self.points, axis=0)

    def _update_strides(self):
        radiuses = np.max(self.points, axis=0) - np.min(self.points, axis=0)
        strides = 2 * np.maximum(radiuses, self.min_radius)
        self._strides = strides
        
    def _update_volume(self):
        if self.volume_type == 'circle':
            r = self.get_radius()
            n = len(self.get_centroid())
            # $$V_n(R) = \frac{\pi^{n/2}}{\Gamma(\frac{n}{2}+1)} R^n$$
            volume = (np.pi**(n/2)) / gamma((n/2) + 1) * (r**n)
        elif self.volume_type == 'rect':
            centroid = self.get_centroid()
            if len(self.points) == 1:
                radiuses = np.array([self.min_radius] * len(centroid))
                volume = (2 * self.min_radius) ** len(centroid)
            else:
                radiuses = np.max(self.points, axis=0) - np.min(self.points, axis=0)
                strides = 2 * np.maximum(radiuses, self.min_radius)
                volume = np.prod(strides)
                # volume = max(volume, self.min_radius ** len(centroid))
        self._volume = volume

    def update(self):
        self._update_centroid()
        self._update_radius()
        self._update_strides()
        self._update_volume()
        
    def get_centroid(self):
        return self._centroid

    def get_radius(self):
        return self._radius

    def get_strides(self):
        return self._strides

    def get_volume(self):
        return self._volume

    def is_intersecting(self, other: 'CoverageCluster'):
        if self.volume_type == 'circle': 
            centroid1 = self.get_centroid()
            centroid2 = other.get_centroid()
            dist = distance.euclidean(centroid1, centroid2)
            return dist < self.get_radius() + other.get_radius()
        elif self.volume_type == 'rect':
            centroid1 = self.get_centroid()
            centroid2 = other.get_centroid()
            radiuses1 = self.get_strides() / 2
            radiuses2 = other.get_strides() / 2
            low = np.maximum(centroid1 - radiuses1, centroid2 - radiuses2)
            high = np.minimum(centroid1 + radiuses1, centroid2 + radiuses2)
            return np.all(low < high)
        return False

    def is_containded(self, other: 'CoverageCluster'):
        if self.volume_type == 'circle':
            radius1 = self.get_radius()
            radius2 = other.get_radius()
            if radius1 < radius2:
                return False
            
            centroid1 = self.get_centroid()
            centroid2 = other.get_centroid()
            dist = distance.euclidean(centroid1, centroid2)
            return dist < radius1 - radius2
        elif self.volume_type == 'rect':
            centroid1 = self.get_centroid()
            centroid2 = other.get_centroid()
            strides1 = self.get_strides()
            strides2 = other.get_strides()
            low = centroid1 - strides1
            high = centroid1 + strides1
            return np.all(low < centroid2 - strides2) and np.all(centroid2 + strides2 < high)
        return False

    def _volume_of_intersection_of_n_spheres(self, dim, c1, r1, c2, r2, n_mc=10000):
        # Bounding hypercube 
        lower_bounds = np.minimum(c1 - r1, c2 - r2)
        upper_bounds = np.maximum(c1 + r1, c2 + r2)

        # Generate random points in hypercube
        points = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_mc, dim))

        # Check which points are within the intersection
        inside_sphere1 = np.linalg.norm(points - c1, axis=1) <= r1
        inside_sphere2 = np.linalg.norm(points - c2, axis=1) <= r2
        inside_intersection = inside_sphere1 & inside_sphere2

        # Estimate the volume
        intersection_count = np.sum(inside_intersection)
        hypercube_volume = np.prod(upper_bounds - lower_bounds)
        intersection_volume = (intersection_count / n_mc) * hypercube_volume

        return intersection_volume

    def volume_of_intersection(self, other: 'CoverageCluster'):
        if self.volume_type == 'circle':
            centroid1 = self.get_centroid()
            centroid2 = other.get_centroid()
            radius1 = self.get_radius()
            radius2 = other.get_radius()
            dist = distance.euclidean(centroid1, centroid2)
            if dist <= abs(radius1 - radius2):
                if radius1 < radius2:
                    return self.get_volume()
                else:
                    return other.get_volume()
            else: 
                dim = len(centroid1)
                volume = self._volume_of_intersection_of_n_spheres(dim, centroid1, radius1, centroid2, radius2)
                return volume
        elif self.volume_type == 'rect':
            centroid1 = self.get_centroid()
            centroid2 = other.get_centroid()
            radiuses1 = self.get_strides() / 2
            radiuses2 = other.get_strides() / 2

            low = np.maximum(centroid1 - radiuses1, centroid2 - radiuses2)
            high = np.minimum(centroid1 + radiuses1, centroid2 + radiuses2)
            intersection_strides = high - low
            volume = np.prod(intersection_strides)
            return volume
        elif self.volume_type == 'convex':
            pass
            
        return 0.0

class CustomOnlineCluster:
    def __init__(self, min_radius=0.1, volume_type='rect'):
        self.min_radius = min_radius
        self.volume_type = volume_type
        self.clusters = []

    def add_points(self, points):
        for point in points:
            self._add_point(point)
        
    def _add_point(self, point):
        if len(self.clusters) == 0:
            cluster = CoverageCluster(volume_type=self.volume_type, min_radius=self.min_radius)
            cluster.add_points([point])
            cluster.update()
            self.clusters.append(cluster)
        else:
            min_distance = np.inf
            nearest_cluster = None
            for cluster in self.clusters:
                centroid = cluster.get_centroid()
                _dist = distance.euclidean(point, centroid)
                if _dist < min_distance:
                    min_distance = _dist
                    nearest_cluster = cluster
            nearest_cluster_radius = max(nearest_cluster.get_radius(), self.min_radius)
            if nearest_cluster_radius + self.min_radius > min_distance:
                nearest_cluster.add_points([point])
                nearest_cluster.update()
            else:
                cluster = CoverageCluster(volume_type=self.volume_type, min_radius=self.min_radius)
                cluster.add_points([point])
                cluster.update()
                self.clusters.append(cluster)

class EvaluatorSearchResult:
    def __init__(self):
        # coverage
        self.soft_n_grid = None
        self.n_grid_per_dim = None
        self.grid_sizes = None

        self.coverage_grid_list = []
        self.iter_coverage_grid_list = []

        self.min_radius = 1
        self.min_samples = 3
        self.eps = self.min_radius * 2

        self.coverage_dbscan_circle_list = []
        self.iter_coverage_dbscan_circle_list = []

        self.coverage_dbscan_rect_list = []
        self.iter_coverage_dbscan_rect_list = []


        self.rect_online_cluster = None
        self.circle_online_cluster = None
        self.coverage_online_rect_list = []
        self.iter_coverage_online_rect_list = []
        self.coverage_online_circle_list = []
        self.iter_coverage_online_circle_list = []

        # exploitation
        self.top_k = 3
        self.exploitation_distance_upper_bound = 8.0
        self.k_distance_exploitation_list = []
        self.iter_k_distance_exploitation_list = []

        # acq_score: improvement + loss + exploitation
        self.y_range = 400
        self.optimal_value = None
        self.acq_exp_threshold = 0.5
        self.acq_exploitation_scores = []
        self.acq_exploitation_validity = []
        self.acq_exploitation_improvement = []
        self.acq_exploration_scores = []
        self.acq_exploration_validity = []
        self.acq_exploration_improvement = []

        self.kappa_list = []
        self.trust_region_radius_list = []
        
    
    def fill_short_data(self, length):
        self.coverage_grid_list = _fill_nan(self.coverage_grid_list, length)
        self.iter_coverage_grid_list = _fill_nan(self.iter_coverage_grid_list, length)

        self.coverage_dbscan_circle_list = _fill_nan(self.coverage_dbscan_circle_list, length)
        self.iter_coverage_dbscan_circle_list = _fill_nan(self.iter_coverage_dbscan_circle_list, length)

        self.coverage_dbscan_rect_list = _fill_nan(self.coverage_dbscan_rect_list, length)
        self.iter_coverage_dbscan_rect_list = _fill_nan(self.iter_coverage_dbscan_rect_list, length)

        self.coverage_online_rect_list = _fill_nan(self.coverage_online_rect_list, length)
        self.iter_coverage_online_rect_list = _fill_nan(self.iter_coverage_online_rect_list, length)

        self.coverage_online_circle_list = _fill_nan(self.coverage_online_circle_list, length)
        self.iter_coverage_online_circle_list = _fill_nan(self.iter_coverage_online_circle_list, length)

        self.k_distance_exploitation_list = _fill_nan(self.k_distance_exploitation_list, length)
        self.iter_k_distance_exploitation_list = _fill_nan(self.iter_k_distance_exploitation_list, length)

        self.acq_exploitation_scores = _fill_nan(self.acq_exploitation_scores, length)
        self.acq_exploration_scores = _fill_nan(self.acq_exploration_scores, length)
        self.acq_exploitation_validity = _fill_nan(self.acq_exploitation_validity, length)
        self.acq_exploration_validity = _fill_nan(self.acq_exploration_validity, length)

        self.kappa_list = _fill_nan(self.kappa_list, length)
        self.trust_region_radius_list = _fill_nan(self.trust_region_radius_list, length)

    def init_grid(self, budget, dim, bounds):
        self.soft_n_grid = budget * 2
        self.n_grid_per_dim = math.floor(self.soft_n_grid ** (1 / dim)) + 1
        self.grid_sizes = []
        for a,b in bounds.T:
            delta = (b - a) / self.n_grid_per_dim
            self.grid_sizes.append(delta)

    def init_dbscan(self, eps, min_samples, min_radius):
        self.eps = eps
        self.min_samples = min_samples
        self.min_radius = min_radius

    def init_distance_exploitation(self, top_k, distance_upper_bound):
        self.top_k = top_k
        self.exploitation_distance_upper_bound = distance_upper_bound

    def init_acq_score(self, y_range, optimal_value, acq_exp_threshold=0.5):
        self.y_range = y_range
        self.optimal_value = optimal_value
        self.acq_exp_threshold = acq_exp_threshold

    def reset_online_cluster(self):
        self.rect_online_cluster = None
        self.circle_online_cluster = None

    # coverage
    def _calculate_coverage(self, all_X, next_X, search_space, eps=0.5, min_samples=5, min_radius=0.2, volume_type='rect', cluster_type='dbscan'):
        """
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other, used in DBSCAN.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        min_radius (float): A radius is used for outliers
        """

        if all_X is None:
            return 0.0
        
        if len(all_X[0]) != len(search_space):
            return 0.0
        clusters = []

        if cluster_type == 'dbscan':
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(all_X)
            labels = dbscan.labels_
            unique_labels = set(labels)

            for k in unique_labels:
                class_member_mask = (labels == k)
                cluster_points = all_X[class_member_mask]
                if k == -1:  # Outliers
                    for outlier_point in cluster_points:
                        cluster = CoverageCluster(volume_type=volume_type, min_radius=min_radius)
                        cluster.add_points([outlier_point])
                        cluster.update()
                        clusters.append(cluster)
                elif len(cluster_points) > 0:
                    cluster = CoverageCluster(volume_type=volume_type, min_radius=min_radius)
                    cluster.add_points(cluster_points)
                    cluster.update()
                    clusters.append(cluster)
        elif cluster_type == 'online':
            online_cluster = None
            if volume_type == 'rect':
                if self.rect_online_cluster is None:
                    self.rect_online_cluster = CustomOnlineCluster(min_radius=min_radius, volume_type='rect')
                online_cluster = self.rect_online_cluster
            elif volume_type == 'circle':
                if self.circle_online_cluster is None:
                    self.circle_online_cluster = CustomOnlineCluster(min_radius=min_radius, volume_type='circle')
                online_cluster = self.circle_online_cluster
            if online_cluster is not None:
                online_cluster.add_points(next_X)
                clusters = online_cluster.clusters

        total_volume_cluster = 0.0
        # Calculate cluster volume
        for cluster in clusters:
            volume = cluster.get_volume()
            total_volume_cluster+= volume

        # Calculate pairwise overlap areas
        overlap_removed = 0
        overlap_count = 0
        for i, a in enumerate(clusters):
            for j, b in enumerate(clusters[i+1:]):
                if a.is_containded(b):
                    overlap_removed += b.get_volume()
                    overlap_count += 1
                elif b.is_containded(a):
                    overlap_removed += a.get_volume()
                    overlap_count += 1
                elif a.is_intersecting(b):
                    overlap_volume = a.volume_of_intersection(b)
                    overlap_removed += overlap_volume
                    overlap_count += 1

        total_volume_cluster -= overlap_removed

        # compute the search space area
        space_volume = 1.0
        for bound in search_space:
            space_volume *= bound[1] - bound[0]

        coverage = total_volume_cluster / space_volume if space_volume > 0 else 0
        return coverage

    def update_next_dbscan_coverage(self, X, next_X, bounds, n_evals):
        if next_X is None:
            return
        if X is not None:
            new_X = np.vstack([X, next_X])
        else:
            new_X = next_X

        n_evals = n_evals if n_evals is not None else len(new_X)

        def _update_coverage_list(coverage_list, volume_type, cluster_type):
            n_fill = n_evals - len(coverage_list) - 1
            coverage_list.extend([np.nan] * n_fill)
            coverage = self._calculate_coverage(new_X, next_X, bounds.T, self.eps, self.min_samples, self.min_radius, volume_type=volume_type, cluster_type=cluster_type)
            coverage_list.append(coverage)
        
            
        # dbscan coverage
        # _update_coverage_list(self.iter_coverage_dbscan_circle_list, 'circle', 'dbscan')
        # _update_coverage_list(self.iter_coverage_dbscan_rect_list, 'rect', 'dbscan')

        # online clustering coverage
        _update_coverage_list(self.iter_coverage_online_rect_list, 'rect', 'online')
        # _update_coverage_list(self.iter_coverage_online_circle_list, 'circle', 'online')

        
    def update_dbscan_coverage(self, X, bounds):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        self.reset_online_cluster()

        for i , _ in enumerate(X):
            cur_X = X[:i+1]
            next_X = X[i].reshape(1, -1) 

            def _update_coverage_list(coverage_list, volume_type, cluster_type):
                coverage = self._calculate_coverage(cur_X, next_X, bounds.T, self.eps, self.min_samples, self.min_radius, volume_type=volume_type, cluster_type=cluster_type)
                coverage_list.append(coverage)

            # dbscan coverage
            # _update_coverage_list(self.coverage_dbscan_circle_list, 'circle', 'dbscan')
            # _update_coverage_list(self.coverage_dbscan_rect_list, 'rect', 'dbscan')

            # online clustering coverage
            _update_coverage_list(self.coverage_online_rect_list, 'rect', 'online')
            # _update_coverage_list(self.coverage_online_circle_list, 'circle', 'online')

            
    # grid coverage
    def calculate_grid_coverage(self, X, search_space, grid_sizes, n_grid_per_dim, accumulate=False):
        D = len(search_space)
        
        total_cells = n_grid_per_dim**D
        
        if total_cells == 0:
            return [0]
        
        coverage_list = []
        visited_cells = set()
        for point in X:
            cell_index = []
            for dim in range(D):
                cell_num = int((point[dim] - search_space[dim][0])/grid_sizes[dim])
                cell_index.append(cell_num)
            visited_cells.add(tuple(cell_index))
            if accumulate:
                coverage_list.append(len(visited_cells) / total_cells)

        if not accumulate:
            coverage = len(visited_cells) / total_cells
            return [coverage]
        else:
            return coverage_list

    def update_grid_coverage(self, X, bounds):
        grid_coverage = self.calculate_grid_coverage(X, bounds.T, self.grid_sizes, self.n_grid_per_dim, accumulate=True)
        self.coverage_grid_list = grid_coverage

    def update_next_grid_coverage(self, X:np.ndarray, next_X:np.ndarray, bounds, n_evals):
        if next_X is None:
            return

        if X is not None:
            new_X = np.vstack([X, next_X])
        else:
            new_X = next_X

        grid_coverage = self.calculate_grid_coverage(new_X, bounds.T, self.grid_sizes, self.n_grid_per_dim, accumulate=False)

        n_evals = n_evals if n_evals is not None else len(new_X)
        n_fill = n_evals - len(self.iter_coverage_grid_list) - 1
        self.iter_coverage_grid_list.extend([np.nan] * n_fill)
        self.iter_coverage_grid_list.extend(grid_coverage)

    # exploitation
    def update_next_exploitation(self, X, fX, next_X, next_fX, n_evals):
        if X is None or fX is None:
            self.iter_k_distance_exploitation_list.extend([np.nan] * len(next_X))
            return

        cur_X = X
        cur_fX = fX.flatten()

        n_evals = n_evals if n_evals is not None else len(cur_fX)
        n_fill = n_evals - len(self.iter_k_distance_exploitation_list) - len(next_X)
        self.iter_k_distance_exploitation_list.extend([np.nan] * n_fill)

        top_k_X = cur_X[np.argsort(cur_fX)[:self.top_k]]

        for i, _next in enumerate(next_X):
            distances = [distance.euclidean(_next, p) for p in top_k_X]
            min_distance = np.min(distances)

            explointation_rate = max(0, self.exploitation_distance_upper_bound - min_distance) / self.exploitation_distance_upper_bound
            self.iter_k_distance_exploitation_list.append(explointation_rate)

    def update_exploitation(self, X, fX, n_initial):
        if n_initial is None or n_initial == 0:
            n_initial = min(10, len(X))
        self.k_distance_exploitation_list = []
        self.k_distance_exploitation_list.extend([np.nan] * n_initial)
        for i in range(n_initial, len(X)):
            cur_X = X[:i]
            cur_fX = fX[:i]
            next_X = X[i]
            # top-k x
            top_k_X = cur_X[np.argsort(cur_fX)[:self.top_k]]
            distances = [distance.euclidean(next_X, p) for p in top_k_X]
            min_distance = np.min(distances)

            explointation_rate = max(0, self.exploitation_distance_upper_bound - min_distance) / self.exploitation_distance_upper_bound
            self.k_distance_exploitation_list.append(explointation_rate)
    
    # acq_score
    def _calculate_acq_score(self, _fx, best_fx, exp_rate, optimal_value):
        def _scale_linear(value, in_min, in_max, out_min, out_max):
            return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

        if exp_rate >= self.acq_exp_threshold:
            improvement = best_fx - _fx
            base = abs(best_fx - optimal_value)
            score = improvement / base
            score = np.clip(score, -1, 1)
            # scale rate from [acq_exp_threshold, 1] to [0, 1]
            _rate = _scale_linear(exp_rate, self.acq_exp_threshold, 1, 0, 1)
            _validity_score = _scale_linear(score, -1, 1, 0, 1)
            validity = _validity_score * _rate
        else:
            improvement = best_fx - _fx
            base = self.y_range
            score = improvement / base
            score = np.clip(score, -1, 1)
            # scale rate from [0, acq_exp_threshold] to [0, 1]
            _rate = _scale_linear(exp_rate, 0, self.acq_exp_threshold, 0, 1)
            _validity_score = _scale_linear(score, -1, 1, 0, 1)
            validity = _validity_score * (1-_rate)

        return score, validity, improvement
        
    def update_next_acq_score(self, fX, next_fX, n_evals):
        if self.y_range is None or self.optimal_value is None:
            return
        if next_fX is None or fX is None:
            return
        
        best_fx = np.min(fX)
        next_fX = next_fX.flatten()
        exploitation_metrics = []
        exploration_metrics = []
        
        for i, _fx in enumerate(next_fX):
            index = n_evals - len(next_fX) + i
            exp_rate = None
            if index < len(self.iter_k_distance_exploitation_list):
                exp_rate = self.iter_k_distance_exploitation_list[index]
            if exp_rate is not None:
                score, validity, improvement = self._calculate_acq_score(_fx, best_fx, exp_rate, self.optimal_value)
                if exp_rate >= self.acq_exp_threshold:
                    exploitation_metrics.append((score, validity, improvement))
                    exploration_metrics.append((np.nan, np.nan, np.nan))
                else:
                    exploitation_metrics.append((np.nan, np.nan, np.nan))
                    exploration_metrics.append((score, validity, improvement))
            else:
                exploitation_metrics.append((np.nan, np.nan, np.nan))
                exploration_metrics.append((np.nan, np.nan, np.nan))
            
        n_fill = n_evals - len(self.acq_exploitation_scores) - len(next_fX)
        self.acq_exploitation_scores.extend([np.nan] * n_fill)
        self.acq_exploration_scores.extend([np.nan] * n_fill)
        self.acq_exploitation_validity.extend([np.nan] * n_fill)
        self.acq_exploration_validity.extend([np.nan] * n_fill)
        self.acq_exploitation_improvement.extend([np.nan] * n_fill)
        self.acq_exploration_improvement.extend([np.nan] * n_fill)

        for score, validity, improvement in exploitation_metrics:
            self.acq_exploitation_scores.append(score)
            self.acq_exploitation_validity.append(validity)
            self.acq_exploitation_improvement.append(improvement)
        
        for score, validity, improvement in exploration_metrics:
            self.acq_exploration_scores.append(score)
            self.acq_exploration_validity.append(validity)
            self.acq_exploration_improvement.append(improvement)


class EvaluatorBasicResult:
    def __init__(self):
        self.id = None
        self.name = None
        self.optimal_value = None
        self.optimal_x = None
        self.bounds = None
        self.budget = None
        self.captured_output = None
        self.error = None
        self.error_type = None
        
        self.execution_time = 0
        self.y_hist:np.ndarray = None
        self.x_hist = None

        self.best_y = None
        self.best_x = None

        self.y_aoc = 0.0
        self.log_y_aoc = 0.0

        self.n_initial_points = 0
        self.non_init_y_aoc = 0.0
        self.non_init_log_y_aoc = 0.0

        self.search_result = EvaluatorSearchResult()
        self.r2_list = []
        self.r2_list_on_train = []
        self.uncertainty_list = []
        self.uncertainty_list_on_train = []

        self.aoc_upper_bound = 1e4

    def update_aoc_with_new_bound_if_needed(self, upper_bound=None):
        if upper_bound is None:
            upper_bound = 1e4
        if hasattr(self, "aoc_upper_bound") and self.aoc_upper_bound == upper_bound:
            pass
        else:
            self.aoc_upper_bound = upper_bound
            self.update_aoc(self.optimal_value, max_y=upper_bound, min_y=1e-8) 
    
    def fill_short_data(self, length):
        self.r2_list = _fill_nan(self.r2_list, length)
        self.r2_list_on_train = _fill_nan(self.r2_list_on_train, length)
        self.uncertainty_list = _fill_nan(self.uncertainty_list, length)
        self.uncertainty_list_on_train = _fill_nan(self.uncertainty_list_on_train, length)

        self.search_result.fill_short_data(length)

    def __to_json__(self):
        d = {}
        d["name"] = self.name
        d["optimal_value"] = self.optimal_value
        d["bounds"] = self.bounds.tolist() if self.bounds is not None else None
        d["budget"] = self.budget
        d["captured_output"] = self.captured_output
        d["error"] = self.error
        d["error_type"] = self.error_type
        
        d["execution_time"] = self.execution_time
        d["y_hist"] = self.y_hist.tolist() if self.y_hist is not None else None
        d["x_hist"] = self.x_hist.tolist() if self.x_hist is not None else None

        d["best_y"] = self.best_y
        d["best_x"] = self.best_x.tolist() if self.best_x is not None else None

        d["y_aoc"] = self.y_aoc

        d["n_initial_points"] = self.n_initial_points

        return d

    def update_stats(self):
        if self.y_hist is None or self.x_hist is None:
            return
        best_index = np.argmin(self.y_hist)
        self.best_y = self.y_hist[best_index]
        self.best_x = self.x_hist[best_index]

    def update_coverage(self):
        if self.y_hist is None:
            return

        n_initial_points = self.n_initial_points
        self.search_result.update_grid_coverage(self.x_hist, self.bounds)
        self.search_result.update_dbscan_coverage(self.x_hist, self.bounds)
        self.search_result.update_exploitation(self.x_hist, self.y_hist, n_initial_points)

    def update_aoc(self, optimal_value = None, min_y=None, max_y=None):
        if self.y_hist is None:
            return

        if max_y is None and hasattr(self, "aoc_upper_bound"):
            max_y = self.aoc_upper_bound

        y_hist = self.y_hist
        y_aoc = ConvergenceCurveAnalyzer(max_y=max_y, min_y=min_y, log_scale=False, shift_value=optimal_value).calculate_aoc(y_hist)
        self.y_aoc = y_aoc

        log_y_aoc = ConvergenceCurveAnalyzer(max_y=max_y, min_y=min_y, log_scale=True, shift_value=optimal_value).calculate_aoc(y_hist)
        self.log_y_aoc = log_y_aoc

        if self.n_initial_points > 0 and len(y_hist) > self.n_initial_points:
            y_hist = self.y_hist[self.n_initial_points:]
            non_init_y_aoc = ConvergenceCurveAnalyzer(max_y=max_y, min_y=min_y, log_scale=False, shift_value=optimal_value).calculate_aoc(y_hist)
            self.non_init_y_aoc = non_init_y_aoc

            non_init_log_y_aoc = ConvergenceCurveAnalyzer(max_y=max_y, min_y=min_y, log_scale=True, shift_value=optimal_value).calculate_aoc(y_hist)
            self.non_init_log_y_aoc = non_init_log_y_aoc

    def set_capture_output(self, captured_output):
        if captured_output is None or captured_output.strip() == "":
            return

        captured_output_list = captured_output.split("\n")
        captured_output_list = [line for line in captured_output_list if line.strip() != ""]

        # find the unique lines
        captured_output_list = list(set(captured_output_list))

        # filter do not contain anchor ":<number>:", then capture the sub string after the anchor.
        new_captured_output_list = []
        for line in captured_output_list:
            match = re.search(r"\:\d+\:", line)
            if match:
                new_captured_output_list.append(line[match.end():])

        # strip the leading and trailing white spaces
        new_captured_output_list = [line.strip() for line in new_captured_output_list]
        new_captured_output = "\n".join(new_captured_output_list)
        self.captured_output = new_captured_output

    def __str__(self):
        return f"{self.name}\nbest_y:{self.best_y:.2f}, aoc:{self.y_aoc:.2f}, time:{self.execution_time:.2f}"


class EvaluatorResult:
    """Result of evaluating an individual."""
    def __init__(self):
        self.name = None
        self.score = None
        self.total_execution_time = 0
        self.error = None
        self.error_type = None
        
        self.result:list[EvaluatorBasicResult] = []

    def __to_json__(self):
        d = {}
        d["name"] = self.name
        d["error"] = self.error
        d["error_type"] = self.error_type
        if hasattr(self, "score"):
            d["score"] = self.score
        d["result"] = [r.__to_json__() for r in self.result]
        return d

    def __str__(self):
        if self.error is not None:
            return f"{self.name}\n{self.error}\n"
        else:
            return f"{self.name}, score:{self.score:.4f}"

    def update_aoc_with_new_bound_if_needed(self, upper_bound=None):
        if self.result is None or len(self.result) == 0:
            return
        for res in self.result:
            res.update_aoc_with_new_bound_if_needed(upper_bound=upper_bound)
        self.score = np.mean([res.log_y_aoc for res in self.result if res.log_y_aoc is not None])
