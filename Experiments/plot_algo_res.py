import logging
import os
import pickle
from datetime import datetime
import importlib.util
import pathlib

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from ioh import get_problem

from llamea.utils import setup_logger
from llamea.utils import plot_group_bars, plot_lines, plot_box_violin, moving_average, savgol_smoothing, gaussian_smoothing

from llamea.prompt_generators.abstract_prompt_generator import ResponseHandler

from llamea.evaluator.injected_critic import FunctionProfiler
from llamea.evaluator.ioh_evaluator import IOHEvaluator 
from llamea.evaluator.evaluator_result import EvaluatorResult

from llamea.population.es_population import ESPopulation


def dynamical_access(obj, attr_path):
    attrs = attr_path.split(".")
    target = obj
    for attr in attrs:
        target = getattr(target, attr, None)
        if target is None:
            break
    return target


def mean_std_agg(agg_series):
    if is_numeric_dtype(agg_series.dtype):
        mean = np.nanmean(agg_series)
        std = np.nanstd(agg_series)
        return [mean, std]
    else:  
        agg_list = agg_series.to_list()
        min_len = min([len(ele) for ele in agg_list])
        # clip the list to the minimum length
        cliped_list = [ele[:min_len] for ele in agg_list]
        mean_list = np.nanmean(cliped_list, axis=0)
        std_list = np.nanstd(cliped_list, axis=0)
        return [mean_list, std_list]

def _min_accumulate(_series):
    if isinstance(_series, tuple):
        mean, _ = _series
        _mean = np.minimum.accumulate(mean)
        _std = np.full_like(_mean, 0)
        return (_mean, _std)
    return np.minimum.accumulate(_series)

def fill_nan_with_left(arr):
    filled_arr = arr.copy()
    last_valid_index = 0 
    for i in range(len(filled_arr)):
        index = len(arr) - i - 1
        val = filled_arr[index]
        if np.isnan(val):
            pass
        else:
            last_valid_index = index
            break
            
    last_valid = None
    for i, val in enumerate(filled_arr):
        if np.isnan(val):
            if i<last_valid_index and last_valid is not None:
                filled_arr[i] = last_valid
            else:
                filled_arr[i] = np.nan
        else:
            last_valid = val
    return filled_arr

def plot_contour(problem_id, instance, points, x1_range=None, x2_range=None, levels=200, figsize=(15, 9), title=None):
    if x1_range is None:
        x1_range = [-5, 5, 300]

    if x2_range is None:
        x2_range = [-5, 5, 300]
    
    func = get_problem(problem_id, instance, 2)

    optimal_point = func.optimum.x
    optimal_value = func.optimum.y 
    
    x1 = np.linspace(*x1_range)
    x2 = np.linspace(*x2_range)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func(np.array([X1[i, j], X2[i, j]]))

    
    fig, ax = plt.subplots(figsize=figsize)
    contour = ax.contourf(X1, X2, Z, 
                           levels=levels, 
                           cmap='PuBu_r')
                        #    viridis')
    # contour = plt.contour(X1, X2, Z, levels=levels, cmap='viridis')


    cmap_points='magma'
    norm = Normalize(vmin=0, vmax=len(points) - 1)
    cmap = plt.get_cmap(cmap_points)  
    for i, point in enumerate(points):
        color = cmap(norm(i))
        ax.plot(point[0], point[1], marker='o', markersize=6, color=color)  

    # red star for optimal point
    ax.plot(optimal_point[0], optimal_point[1], 'r*', markersize=12)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  
    cbar = fig.colorbar(sm, label='points',
                        orientation='vertical', location='right',
                        fraction=0.05, shrink=1.0, aspect=30,
                        ax=ax)
    ticks = np.linspace(0, len(points)-1, min(10 , len(points)-1))  
    ticks = np.round(ticks).astype(int)  
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)

    cbar_z = fig.colorbar(contour, orientation='vertical', location='left', label='fx',
                          fraction=0.05, shrink=1.0, aspect=30, ax=ax)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if title is None:
        title = f"F{problem_id}"
    ax.set_title(title)
        

    plt.show()

# plot_contour(2, 1, [(0, 0), (1, 1), (-1, -1)])


def _process_algo_result(results:list[EvaluatorResult]):
    # dynamic access from EvaluatorBasicResult. None means it should be handled separately
    column_name_map = {
        'algorithm' : None,
        'problem_id' : None,
        'instance_id' : None,
        'exec_id' : None,
        'n_init' : 'n_initial_points',
        'acq_exp_threshold' : 'search_result.acq_exp_threshold',

        'log_y_aoc' : 'log_y_aoc',
        'y_aoc' : 'y_aoc',
        'y' : 'y_hist',
        
        'loss' : None,
        'best_loss' : None,
        
        'r2' : 'r2_list',
        'r2_on_train' : 'r2_list_on_train',
        'uncertainty' : 'uncertainty_list',
        'uncertainty_on_train' : 'uncertainty_list_on_train',
        
        'grid_coverage' : 'search_result.coverage_grid_list',   
        'acq_grid_coverage' : 'search_result.iter_coverage_grid_list',
        
        'dbscan_circle_coverage' : 'search_result.coverage_dbscan_circle_list',
        'acq_dbscan_circle_coverage' : 'search_result.iter_coverage_dbscan_circle_list',
        
        'dbscan_rect_coverage' : 'search_result.coverage_dbscan_rect_list',
        'acq_dbscan_rect_coverage' : 'search_result.iter_coverage_dbscan_rect_list',
        
        'online_rect_coverage' : 'search_result.coverage_online_rect_list',
        'acq_online_rect_coverage' : 'search_result.iter_coverage_online_rect_list',
        
        'online_circle_coverage' : 'search_result.coverage_online_circle_list',
        'acq_online_circle_coverage' : 'search_result.iter_coverage_online_circle_list',
        
        'exploitation_rate' : 'search_result.k_distance_exploitation_list',
        'acq_exploitation_rate' : 'search_result.iter_k_distance_exploitation_list',
        
        'acq_exploitation_score' : 'search_result.acq_exploitation_scores',
        'acq_exploration_score' : 'search_result.acq_exploration_scores',
        
        'acq_exploitation_validity' : 'search_result.acq_exploitation_validity',
        'acq_exploration_validity' : 'search_result.acq_exploration_validity',

        'acq_exploitation_improvement' : 'search_result.acq_exploitation_improvement',
        'acq_exploration_improvement' : 'search_result.acq_exploration_improvement',
    }

    def _none_to_nan(_target):
        if isinstance(_target, list):
            return [np.nan if ele is None else ele for ele in _target] 
        return np.nan if _target is None else _target

    def res_to_row(res, algo:str):
        res_id = res.id
        res_split = res_id.split("-")
        problem_id = int(res_split[0])
        instance_id = int(res_split[1])
        repeat_id = int(res_split[2])
        loss = res.y_hist - res.optimal_value
        row = {}

        for column_name, column_path in column_name_map.items():
            if column_path is None:
                if column_name == 'algorithm':
                    row[column_name] = algo
                elif column_name == 'problem_id':
                    row[column_name] = problem_id
                elif column_name == 'instance_id':
                    row[column_name] = instance_id
                elif column_name == 'exec_id':
                    row[column_name] = repeat_id
                elif column_name == 'loss':
                    row[column_name] = loss
                elif column_name == 'best_loss':
                    row[column_name] = np.minimum.accumulate(loss)
            else:
                value = dynamical_access(res, column_path)
                non_none_value = _none_to_nan(value)
                row[column_name] = non_none_value
        return row

    res_df = pd.DataFrame(columns=column_name_map.keys())
    for result in results:
        # algo = result.name.removeprefix("BL")
        algo = result.name
        for res in result.result:
            row = res_to_row(res, algo)
            if row is not None:
                res_df.loc[len(res_df)] = row
    return res_df

def _plot_algo_aoc_on_problems(res_df:pd.DataFrame):
    all_aoc_df = res_df.groupby(['algorithm', 'problem_id'])[['y_aoc', 'log_y_aoc']].agg(np.mean).reset_index()
    all_aoc_df = all_aoc_df.groupby(['algorithm'])[['y_aoc', 'log_y_aoc']].agg(list).reset_index()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    _default_colors = prop_cycle.by_key()['color']
    colors = []
    labels = []
    all_log_plot_data = []
    
    for algo in all_aoc_df['algorithm']:
        _temp_df = all_aoc_df[all_aoc_df['algorithm'] == algo].agg(list)
        all_log_plot_data.append(_temp_df['log_y_aoc'].values[0])
        labels.append(algo)

        if 'BL' in algo:
            colors.append(_default_colors[0])
        else:
            colors.append(_default_colors[1])

    # plot aoc
    plot_box_violin(
        data=[all_log_plot_data],
        labels=[labels],
        colors=[colors],
        show_inside_box=True,
        plot_type="violin",
        title="AOC Catorized by Problems",
        figsize=(15, 9),
    )

def _plot_algo_aoc(res_df:pd.DataFrame):
    all_aoc_df = res_df.groupby(['algorithm', 'instance_id', 'exec_id'])[['y_aoc', 'log_y_aoc']].agg(np.mean).reset_index()
    all_aoc_df = all_aoc_df.groupby(['algorithm'])[['y_aoc', 'log_y_aoc']].agg(list).reset_index()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    _default_colors = prop_cycle.by_key()['color']
    colors = []
    labels = []
    all_log_plot_data = []
    
    for algo in all_aoc_df['algorithm']:
        _temp_df = all_aoc_df[all_aoc_df['algorithm'] == algo].agg(list)
        all_log_plot_data.append(_temp_df['log_y_aoc'].values[0])
        labels.append(algo)

        if 'BL' in algo:
            colors.append(_default_colors[0])
        else:
            colors.append(_default_colors[1])

    # plot aoc
    plot_box_violin(
        data=[all_log_plot_data],
        labels=[labels],
        colors=[colors],
        show_inside_box=True,
        plot_type="violin",
        title="AOC",
        figsize=(15, 9),
    )

def _plot_algo_problem_aoc(res_df:pd.DataFrame):
    problem_id_list = res_df['problem_id'].unique()
    aoc_df = res_df.groupby(['algorithm','problem_id'])[['y_aoc', 'log_y_aoc']].agg(list).reset_index()
    #(problem, data)
    
    aoc_plot_data = []
    log_plot_data = []
    labels = []
    short_labels = []
    sub_titles = []
    for problem_id in problem_id_list:
        _temp_df = aoc_df[aoc_df['problem_id'] == problem_id].agg(list)
        aoc_plot_data.append(_temp_df['y_aoc'].to_list())
        log_plot_data.append(_temp_df['log_y_aoc'].to_list())
        sub_titles.append(f"F{problem_id}")

        _labels = _temp_df['algorithm'].to_list()
        labels.append(_labels)
        short_labels.append([label[:16] for label in _labels])

    prop_cycle = plt.rcParams['axes.prop_cycle']
    _default_colors = prop_cycle.by_key()['color']
    _colors = []
    for _label in labels[0]:
        if 'BL' in _label:
            _colors.append(_default_colors[0])
        else:
            _colors.append(_default_colors[1])
    colors = [_colors] * len(problem_id_list)
    labels = short_labels

    # iter by step
    step = 6
    for i in range(0, len(log_plot_data), step):
        _plot_data = log_plot_data[i:i+step]
        _labels = labels[i:i+step]
        _sub_titles = sub_titles[i:i+step]
        _colors = colors[i:i+step]

        plot_box_violin(data=_plot_data,
                        labels=_labels,
                        sub_titles=_sub_titles,
                        title="AOC",
                        plot_type="violin",
                        label_fontsize=8, 
                        show_inside_box=True,
                        colors=_colors,
                        n_cols=2,
                        figsize=(15, 9),
                        )

def _plot_algo_iter(res_df:pd.DataFrame):
    # handle y
    data_col_map = {
        'n_init': '',
        'acq_exp_threshold': '',

        'loss': 'Loss',
        'best_loss': 'Best Loss',

        'r2': 'R2 on test',
        'r2_on_train' : 'R2 on train',
        'uncertainty' : 'Uncertainty on test',
        'uncertainty_on_train' : 'Uncertainty on train',

        'grid_coverage' : 'Grid Coverage',

        # 'dbscan_circle_coverage': 'DBSCAN Circle Coverage',
        # 'dbscan_rect_coverage': 'DBSCAN Rect Coverage',

        'online_rect_coverage': 'Online Cluster Rect Coverage',
        # 'online_circle_coverage': 'Online Circle Coverage',

        'acq_grid_coverage' : 'Acq Grid Coverage',

        # 'acq_dbscan_circle_coverage': 'DBSCAN Circle Coverage(Acq)',
        # 'acq_dbscan_rect_coverage': 'DBSCAN Rect Coverage(Acq)',

        'acq_online_rect_coverage': 'Online Cluster Rect Coverage(Acq)',
        # 'acq_online_circle_coverage': 'Online Circle Coverage(Acq)',

        'exploitation_rate': 'Exploitation Rate',
        'acq_exploitation_rate': 'Acq Exploitation Rate(er)',

        'acq_exploitation_improvement': 'Exploitation Improvement: $current-best$',
        'acq_exploitation_score': 'Exploitation Score: $improve/(best-optimum)$',
        'acq_exploitation_validity': 'Exploitation Validity: $score*er$',

        'acq_exploration_improvement': 'Exploration Improvement: $current-best$',
        'acq_exploration_score': 'Exploration Score: $improve/fixed\_base$',
        'acq_exploration_validity': 'Exploration Validity: $score*(1-er)$',
    }
    data_cols = list(data_col_map.keys())
    
    # if 'loss' in data_cols:
    #     # apply loss to min.accumulate, then create new column
    #     # y_df['best_loss'] = y_df['loss'].apply(_min_accumulate)
    #     res_df['best_loss'] = res_df['loss'].apply(np.minimum.accumulate)
    #     # insert best_loss to the next of loss in data_cols
    #     loss_index = data_cols.index('loss')
    #     data_cols.insert(loss_index+1, 'best_loss')
    #     data_col_map['best_loss'] = 'Best Loss'

    y_df = res_df.groupby(['algorithm', 'problem_id'])[data_cols].agg(mean_std_agg).reset_index()
    y_df[data_cols].applymap(lambda x: x[0] if isinstance(x, list) else x)

    problem_ids = y_df['problem_id'].unique()


    def smooth_factory(smooth_type='savgol', window_size=5, polyorder=2, sigma=1.0):
        def _smooth_data(data):
            if smooth_type == 'savgol':
                return savgol_smoothing(data, window_size, polyorder)
            elif smooth_type == 'moving':
                return moving_average(data, window_size)
            elif smooth_type == 'gaussian':
                return gaussian_smoothing(data, sigma)
        return _smooth_data

    smooth_cols = {
        # 'exploitation_rate': smooth_factory(smooth_type='moving', window_size=5),
    }

    def clip_upper_factory(bound_type='mean', upper_len_ratio=0.25, inverse=False, _bound=None):
        def _clip_upper(data, bound_type=bound_type, upper_len_ratio=upper_len_ratio, inverse=inverse, _bound=_bound):
            _clip_len = int(data.shape[1] * upper_len_ratio)
            _upper_bound = _bound
            if bound_type == 'mean':
                if inverse:
                    _upper_bound = np.nanmean(data[:, _clip_len:]) + np.nanstd(data[:, _clip_len:])
                else:
                    _upper_bound = np.nanmean(data[:, :_clip_len]) + np.nanstd(data[:, :_clip_len])
            elif bound_type == 'median':
                if inverse:
                    _upper_bound = np.nanmedian(data[:, _clip_len:])
                else:
                    _upper_bound = np.nanmedian(data[:, :_clip_len])
            elif bound_type == 'fixed' and _bound is not None:
                _upper_bound = _bound

            _data = np.clip(data, 0, _upper_bound)
            return _data, _upper_bound
        return _clip_upper

    clip_cols = {
        'loss': clip_upper_factory(bound_type='mean'),
    }

    y_scale_cols = {
        'loss': ('log', {}),
        'best_loss': ('log', {}),
    }

    non_fill_cols = [
        'loss',
        'best_loss',
    ]

    ignore_cols = [
        'n_init',
        'acq_exp_threshold',
    ]

    for problem_id in problem_ids:
        plot_data = []
        x_data = []
        plot_filling = []
        labels = []
        x_dots = []
        sub_titles = []
        y_scales = []
        colors = []
        baselines = []
        baseline_labels = []
        
        _temp_df = y_df[y_df['problem_id'] == problem_id]

        prop_cycle = plt.rcParams['axes.prop_cycle']
        _default_colors = prop_cycle.by_key()['color']

        for col in data_cols:
            if col in ignore_cols:
                continue

            data = _temp_df[col].to_list()
            # remove empty data if len(data) == 0 or all nan
            empty_indexs = [i for i, ele in enumerate(data) if ele[0].size == 0 or np.all(np.isnan(ele[0]))]
            data = [ele for i, ele in enumerate(data) if i not in empty_indexs]

            if len(data) == 0:
                continue
            
            # fill short data and replace nan with the left
            max_len = max([len(ele[0]) for ele in data])
            for i, ele in enumerate(data):
                _content = []
                for _sub_ele in ele:
                    _new_sub_ele = _sub_ele
                    if len(_new_sub_ele) < max_len:
                        fill_len = max_len - len(_new_sub_ele)
                        _new_sub_ele = np.append(_new_sub_ele, [np.nan] * fill_len)
                    _new_sub_ele = fill_nan_with_left(_new_sub_ele)
                    _content.append(_new_sub_ele)
                data[i] = _content
                    
            mean_array = np.array([ele[0] for ele in data])
            std_array = np.array([ele[1] for ele in data])

            # clip if needed
            if col in clip_cols:
                mean_array, _upper_bound = clip_cols[col](mean_array)
                std_array = np.where(mean_array == _upper_bound, 0, std_array)

            # smooth if needed
            if col in smooth_cols:
                mean_array = smooth_cols[col](mean_array)
            
            plot_data.append(mean_array)
            x_data.append(np.arange(mean_array.shape[1]))
            
            # fill the area between mean - std and mean + std
            if col not in non_fill_cols:
                upper_bound = mean_array + std_array 
                lower_bound = mean_array - std_array
                plot_filling.append(list(zip(lower_bound, upper_bound)))
            else:
                plot_filling.append(None)

            if 'acq_exploitation_rate' in col:
                exp_threshold = _temp_df['acq_exp_threshold'].to_list()
                mean_exp = [ele[0] for ele in exp_threshold]
                _bl = np.nanmean(mean_exp)
                baselines.append([_bl])
                baseline_labels.append(["Threshold"])
            else:
                baselines.append(None)
                baseline_labels.append(None)

            # handle n_init
            n_init_data = _temp_df['n_init'].to_list()
            _x_dots = []
            for n_init in n_init_data:
                if n_init[0] > 0:
                    _x_dots.append(np.array([n_init[0]], dtype=int))
                else:
                    _x_dots.append(np.array([], dtype=int))
            # remove empty data
            _x_dots = [ele for i, ele in enumerate(_x_dots) if i not in empty_indexs]
            x_dots.append(_x_dots)

            _labels = _temp_df['algorithm'].to_list()
            _colors = _default_colors[:len(_labels)]
            _labels = [ele for i, ele in enumerate(_labels) if i not in empty_indexs]
            _labels = [label[:10] for label in _labels]
            _colors = [color for i, color in enumerate(_colors) if i not in empty_indexs]
            labels.append(_labels)
            colors.append(_colors)

            _sub_title = data_col_map.get(col, col)
            if col in y_scale_cols:
                _y_scale, _y_scale_kwargs = y_scale_cols[col]
                y_scales.append((_y_scale, _y_scale_kwargs))
                sub_titles.append(_sub_title + f"({_y_scale})")
            else:
                y_scales.append(None)
                sub_titles.append(_sub_title)


        plot_lines(
            y=plot_data, x=x_data, 
            y_scales=y_scales,
            baselines=baselines,
            baseline_labels=baseline_labels,
            colors=colors,
            labels=labels, 
            label_fontsize=8,
            linewidth=1.0,
            filling=plot_filling,
            x_dot=x_dots,
            n_cols=3,
            sub_titles=sub_titles,
            sub_title_fontsize=9,
            title=f"F{problem_id}",
            figsize=(15, 9),
        ) 

def plot_algo_result(results:list[EvaluatorResult]):
    res_df = _process_algo_result(results)
    
    _plot_algo_aoc(res_df)
    _plot_algo_aoc_on_problems(res_df)
    # _plot_algo_problem_aoc(res_df)
    # _plot_algo_iter(res_df)



def plot_algo(file_paths=None, dir_path=None, pop_path=None):
    res_list = []
    if pop_path is not None:
        with open(pop_path, "rb") as f:
            pop = pickle.load(f)
            all_inds = pop.all_individuals()
            all_handlers = [ESPopulation.get_handler_from_individual(ind) for ind in all_inds]
            for handler in all_handlers:
                if handler.error is not None:
                    continue
                res_list.append(handler.eval_result)
    elif dir_path is not None:
        file_paths = []
        if not os.path.isdir(dir_path):
            raise ValueError(f"Invalid directory path: {dir_path}")
        for file in os.listdir(dir_path):
            if file.endswith(".pkl"):
                file_paths.append(os.path.join(dir_path, file))
    
    if len(res_list) == 0:
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                target = pickle.load(f)
                if target.error is not None:
                    continue
                if isinstance(target, EvaluatorResult):
                    res_list.append(target)
                elif isinstance(target, ResponseHandler):
                    res_list.append(target.eval_result)
            
    plot_algo_result(results=res_list)
    


def plot_project_tr():
    from scipy.stats import qmc

    # Parameters
    dim = 2  # Dimension of the space
    bounds = np.array([[-2, -3], [4, 2]])  # Lower and upper bounds
    n_points = 100  # Number of points to generate
    center = np.array([2, 1])  # Custom center
    radius = 1.5       # Custom radius

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))  # Create subplots for each stage

    # 1. Sobol Sequence [0, 1]
    sampler = qmc.Sobol(d=dim, scramble=True)
    points_sobol = sampler.random(n=n_points)
    scaled_center = qmc.scale(center.reshape(1, -1), bounds[0], bounds[1], reverse=True).flatten()
    axes[0, 0].scatter(points_sobol[:, 0], points_sobol[:, 1], s=5)
    axes[0, 0].scatter(scaled_center[0], scaled_center[1], c='r', marker='*', s=100)
    axes[0, 0].set_title('1. Sobol Sequence [0, 1]')
    axes[0, 0].set_xlim(-0.1, 1.1)
    axes[0, 0].set_ylim(-0.1, 1.1)
    axes[0, 0].set_aspect('equal')

    # 2. Scaled to [-1, 1]
    points_scaled = qmc.scale(points_sobol, -1, 1)
    axes[0, 1].scatter(points_scaled[:, 0], points_scaled[:, 1], s=5)
    scaled_center = qmc.scale(scaled_center.reshape(1, -1), -1, 1).flatten()
    axes[0, 1].scatter(scaled_center[0], scaled_center[1], c='r', marker='*', s=100)
    axes[0, 1].set_title('2. Scaled to [-1, 1]')
    axes[0, 1].set_xlim(-1.1, 1.1)
    axes[0, 1].set_ylim(-1.1, 1.1)
    axes[0, 1].set_aspect('equal')
    axes[0,1].add_patch(plt.Circle((0, 0), 1, color='r', alpha=0.1)) # Add a circle

    # 3. Projected to Hypersphere & Scaled
    lengths = np.linalg.norm(points_scaled, axis=1, keepdims=True)
    points_hypersphere = points_scaled / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1/dim)
    # points_hypersphere = points_scaled / lengths ** (1/dim)
    axes[1, 0].scatter(points_hypersphere[:, 0], points_hypersphere[:, 1], s=5)
    axes[1, 0].scatter(scaled_center[0], scaled_center[1], c='r', marker='*', s=100)
    axes[1, 0].set_title('3. Projected to Circle')
    axes[1, 0].set_xlim(-1.1, 1.1)
    axes[1, 0].set_ylim(-1.1, 1.1)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].add_patch(plt.Circle((0, 0), 1, color='r', alpha=0.1)) # Add a circle

    # 4. Final Points (Scaled, Translated, Clipped)
    sampled_points = points_hypersphere * radius + center
    sampled_points = np.clip(sampled_points, bounds[0], bounds[1])

    axes[1, 1].scatter(sampled_points[:, 0], sampled_points[:, 1], s=5)
    axes[1, 1].scatter(center[0], center[1], c='r', marker='*', s=100)
    axes[1, 1].set_title('4. Translated and Clipped')
    # Draw the bounding box
    rect = plt.Rectangle(bounds[0], bounds[1, 0] - bounds[0, 0],
                        bounds[1, 1] - bounds[0, 1], linewidth=1, edgecolor='r', facecolor='none')
    axes[1, 1].add_patch(rect)
    # Draw circle of radius
    axes[1,1].add_patch(plt.Circle(center, radius, color='g', alpha=0.1))
    axes[1, 1].set_xlim(bounds[0,0]-0.5, bounds[1,0]+0.5)
    axes[1, 1].set_ylim(bounds[0,1]-0.5, bounds[1,1]+0.5)
    axes[1, 1].set_aspect('equal')

    
    # 5. translated and clipped without projection
    points_scaled = qmc.scale(points_sobol, center-radius, center+radius)
    # sampled_points = points_scaled * radius + center
    sampled_points = np.clip(points_scaled, bounds[0], bounds[1])
    axes[0, 2].scatter(sampled_points[:, 0], sampled_points[:, 1], s=5)
    axes[0, 2].scatter(center[0], center[1], c='r', marker='*', s=100)
    axes[0, 2].set_title('Translated and Clipped without Projection')
    # Draw the bounding box
    rect = plt.Rectangle(bounds[0], bounds[1, 0] - bounds[0, 0],
                        bounds[1, 1] - bounds[0, 1], linewidth=1, edgecolor='r', facecolor='none')
    axes[0, 2].add_patch(rect)
    # Draw circle of radius
    axes[0, 2].add_patch(plt.Circle(center, radius, color='g', alpha=0.1))
    axes[0, 2].set_xlim(bounds[0,0]-0.5, bounds[1,0]+0.5)
    axes[0, 2].set_ylim(bounds[0,1]-0.5, bounds[1,1]+0.5)
    axes[0, 2].set_aspect('equal') 

    # 6. uniform sampling
    samples = np.random.uniform(
            low=np.maximum(center - radius, bounds[0]),
            high=np.minimum(center + radius, bounds[1]),
            size=(n_points, dim)
        )
    axes[1, 2].scatter(samples[:, 0], samples[:, 1], s=5)
    axes[1, 2].scatter(center[0], center[1], c='r', marker='*', s=100)
    axes[1, 2].set_title('Uniform Sampling')
    # Draw the bounding box
    rect = plt.Rectangle(bounds[0], bounds[1, 0] - bounds[0, 0],
                        bounds[1, 1] - bounds[0, 1], linewidth=1, edgecolor='r', facecolor='none')
    axes[1, 2].add_patch(rect)
    # Draw circle of radius
    axes[1, 2].add_patch(plt.Circle(center, radius, color='g', alpha=0.1))
    axes[1, 2].set_xlim(bounds[0,0]-0.5, bounds[1,0]+0.5)
    axes[1, 2].set_ylim(bounds[0,1]-0.5, bounds[1,1]+0.5)
    axes[1, 2].set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_algo_0220():
    file_paths = [
        # 'Experiments/final_eval_res/BLRandomSearch_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0210053711.pkl',

        'Experiments/final_eval_res/BLTuRBO1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0215224338.pkl',

        # 'Experiments/final_eval_res/BLTuRBOM_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0215232616.pkl',

        'Experiments/final_eval_res/BLMaternVanillaBO_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0216012649.pkl',

        'Experiments/final_eval_res/BLHEBO_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0216043242.pkl',

        'Experiments/final_eval_res/BLCMAES_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0216014349.pkl',

        # 'Experiments/final_eval_res/TrustRegionAdaptiveTempBOv2_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211000039.pkl',
        
        # 'Experiments/final_eval_res/BayesLocalAdaptiveAnnealBOv1_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211012527.pkl',
        
        # 'Experiments/final_eval_res/EnsembleLocalSearchBOv1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211041109.pkl',
    ] 

    dir_path = None
    pop_path = None

    plot_algo(file_paths=file_paths, dir_path=dir_path, pop_path=pop_path)

if __name__ == "__main__":
    # setup_logger(level=logging.DEBUG)
    setup_logger(level=logging.INFO)

    plot_project_tr()
    # plot_algo_0220()