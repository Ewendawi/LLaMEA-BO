# This file contains the classes for the results of the evaluator.
import re
import math
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from scipy.spatial import ConvexHull


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

class EvaluatorCoverageResult:
    def __init__(self):
        self.soft_n_grid = None
        self.n_grid_per_dim = None
        self.grid_sizes = None
        self.grid_coverage_list = []

        self.eps = 0.5
        self.min_samples = 5
        self.min_radius = 1
        self.circle_dbscan_coverage_list = []
        self.convex_dbscan_coverage_list = []

        self.top_k = 3
        self.k_distance_exploitation_list = []

    def init_grid(self, budget, dim, bounds):
        self.soft_n_grid = budget * 2
        self.n_grid_per_dim = math.floor(self.soft_n_grid ** (1 / dim))
        self.grid_sizes = []
        for a,b in bounds.T:
            delta = (b - a) / self.n_grid_per_dim
            self.grid_sizes.append(delta)

    def init_dbscan(self, eps, min_samples, min_radius):
        self.eps = eps
        self.min_samples = min_samples
        self.min_radius = min_radius

    def init_distance_exploitation(self, top_k):
        self.top_k = top_k

    def _calculate_coverage_dbscan_circle(self, X, search_space, eps=0.5, min_samples=5, min_radius=0.2):
        """
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other, used in DBSCAN.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        min_radius (float): A radius is used for outliers
        """
        if not X:
            return 0.0
        
        if len(X[0]) != len(search_space):
            return 0.0

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        labels = dbscan.labels_
        unique_labels = set(labels)
        total_area_cluster = 0.0
        cluster_info = {} # (center, radius)

        for k in unique_labels:
            class_member_mask = (labels == k)
            cluster_points = X[class_member_mask]
            if k == -1:  # Outliers
                for outlier_point in cluster_points:
                    radius = min_radius
                    total_area_cluster += np.pi * radius **2
                    cluster_info[tuple(outlier_point)] = (outlier_point, radius)
            elif len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                max_dist = 0
                for p in cluster_points:
                    dist = distance.euclidean(p, centroid)
                    max_dist = max(max_dist, dist)
                radius = max_dist
                total_area_cluster+= np.pi*radius**2
                cluster_info[tuple(centroid)] = (centroid, radius)

        # Calculate pairwise overlap areas
        overlap_removed = 0
        cluster_list = list(cluster_info.items())
        for i, a in enumerate(cluster_list):
            for j, b in enumerate(cluster_list[i+1:]):
                (centroid1, radius1) = a[1]
                (centroid2, radius2) = b[1]

                dist = distance.euclidean(centroid1,centroid2)
                #Check intersection of radii
                if dist < radius1+radius2:
                    overlap_area = 0
                    d = distance.euclidean(centroid1, centroid2)
                    if d <= abs(radius1 - radius2):
                        # One circle is contained within the other
                        overlap_area = math.pi * min(radius1, radius2)**2
                    else: # Partial overlap
                        d1 = (radius1*radius1-radius2*radius2+d*d)/(2*d)
                        d2 = d-d1
                        a1 = radius1*radius1*math.acos(d1/radius1)-d1*math.sqrt(radius1*radius1-d1*d1)
                        a2 = radius2*radius2*math.acos(d2/radius2)-d2*math.sqrt(radius2*radius2-d2*d2)
                        overlap_area = a1 + a2

                    overlap_removed += overlap_area

        total_area_cluster -= overlap_removed

        # compute the search space area
        space_area = 1.0
        for bound in search_space:
            space_area *= bound[1] - bound[0]
        
        coverage = total_area_cluster / space_area if space_area > 0 else 0
        return coverage

    def calculate_coverage_dbscan_convex_with_outliers(self, X, search_space, eps=0.5, min_samples=5, min_radius=0.2):
        """
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other, used in DBSCAN.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        min_radius (float): min_radius to use to define circle radius on outliers.
        """

        if len(X) == 0:
            return 0

        if len(X[0]) != len(search_space):
            return 0.0

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        labels = dbscan.labels_
        unique_labels = set(labels)
        total_volume = 0
        for k in unique_labels:
            class_member_mask = (labels == k)
            cluster_points = X[class_member_mask]
            if k == -1:  # Outliers
                for outlier_point in cluster_points:
                    radius = min_radius
                    total_volume += np.pi * radius **2
            elif len(cluster_points) > 0:
                hull = ConvexHull(cluster_points)
                total_volume+=hull.volume
            
        # compute the search space area
        space_area = 1.0
        for bound in search_space:
            space_area *= bound[1] - bound[0]

        coverage = total_volume/ space_area if space_area > 0 else 0
        return coverage

    def update_dbscan_coverage(self, X, bounds):
        if not all([isinstance(x,list) for x in X]):
            return 0

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        for i in range(len(X)):
            cur_X = X[:i+1]

            circle_coverage = self._calculate_coverage_dbscan_circle(cur_X, bounds.T, self.eps, self.min_samples, self.min_radius)
            self.circle_dbscan_coverage_list.append(circle_coverage)

            convex_coverage = self.calculate_coverage_dbscan_convex_with_outliers(cur_X, bounds.T, self.eps, self.min_samples, self.min_radius)
            self.convex_dbscan_coverage_list.append(convex_coverage)
            
    def calculate_grid_coverage(self, X, search_space, grid_sizes, n_grid_per_dim, accumulate=False):
        D = len(search_space)
        if D == 0 or not all([isinstance(point,list) for point in X]):
            return 0
        
        total_cells = n_grid_per_dim**D
        
        if total_cells == 0:
            return 0
        
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
        self.grid_coverage_list = grid_coverage

    def update_exploitation(self, X, fX, n_initial):
        if not all([isinstance(points, list) for points in [X, fX]]):
            return 0

        self.k_distance_exploitation_list = []
        self.k_distance_exploitation_list.extend([0] * n_initial)
        for i in range(n_initial, len(X)):
            cur_X = X[:i]
            cur_fX = fX[:i]
            next_X = X[i]
            # top-k x
            top_k_X = cur_X[np.argsort(cur_fX)[:self.top_k]]
            distances = [distance.euclidean(next_X, p) for p in top_k_X]
            min_distance = np.min(distances)
        
            self.k_distance_exploitation_list.append(min_distance)
    

class EvaluatorBasicResult:
    def __init__(self):
        self.id = None
        self.name = None
        self.optimal_value = None
        self.bounds = None
        self.budget = None
        self.captured_output = None
        self.error = None
        self.error_type = None
        
        self.execution_time = 0
        self.y_hist = None
        self.x_hist = None

        self.best_y = None
        self.best_x = None

        self.y_aoc = 0.0
        self.log_y_aoc = 0.0
        self.y_aoc_from_ioh = 0.0  ##deprecated

        self.n_initial_points = 0
        self.non_init_y_aoc = 0.0
        self.non_init_log_y_aoc = 0.0

        self.coverage_result = EvaluatorCoverageResult()
        self.r2_list = []
        self.uncertainty_list = []
        
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

    def update_aoc(self, optimal_value = None, min_y=None, max_y=None):
        if self.y_hist is None:
            return

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
        self.similarity = None
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