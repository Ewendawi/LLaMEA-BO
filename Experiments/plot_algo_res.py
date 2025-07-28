import logging
import os

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from ioh import get_problem

from llamevol.utils import setup_logger, RenameUnpickler
from llamevol.utils import plot_group_bars, plot_lines, plot_box_violin, moving_average, savgol_smoothing, gaussian_smoothing

from llamevol.prompt_generators.abstract_prompt_generator import ResponseHandler

from llamevol.evaluator.bo_injector import FunctionProfiler
from llamevol.evaluator.ioh_evaluator import IOHEvaluator 
from llamevol.evaluator.evaluator_result import EvaluatorResult

from llamevol.population.es_population import ESPopulation


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
        return [mean, std, None, None]
    else:  
        agg_list = agg_series.to_list()
        min_len = min([len(ele) for ele in agg_list])
        # clip the list to the minimum length
        cliped_list = [ele[:min_len] for ele in agg_list]
        mean_list = np.nanmean(cliped_list, axis=0)
        std_list = np.nanstd(cliped_list, axis=0)
        _max_ele = np.max(cliped_list, axis=0)
        _min_ele = np.min(cliped_list, axis=0)
        return [mean_list, std_list, _max_ele, _min_ele]

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

def _shorthand_algo_name(algo:str):
    if 'EvolutionaryBO' in algo:
        return 'TREvol'
    elif 'Optimistic' in algo:
        return 'TROpt'
    elif 'Pareto' in algo:
        return 'TRPareto'
    elif 'ARM' in algo:
        return 'ARM'
    elif 'MaternVanilla' in algo:
        return 'VanillaBO'
    elif 'VanillaEIBO' in algo:
        return 'Vanilla'

    if 'BL' in algo:
        return algo.replace("BL", "")

    if 'A_' in algo:
        return algo.replace("A_", "")

    return algo

def _process_algo_result(results:list[EvaluatorResult], column_name_map=None):
    # dynamic access from EvaluatorBasicResult. None means it should be handled separately
    _column_name_map = {
        'algorithm' : None,
        'algorithm_name' : None,
        'algorithm_short_name' : None,
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

        'kappa' : 'search_result.kappa_list',
        'tr_radius' : 'search_result.trust_region_radius_list',
        
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

    column_name_map = _column_name_map if column_name_map is None else column_name_map

    def _none_to_nan(_target):
        if isinstance(_target, list):
            return [np.nan if ele is None else ele for ele in _target] 
        return np.nan if _target is None else _target

    def _algo_to_name(algo:str):
        o_algo = algo
        if 'BL' not in algo:
            o_algo = f"A_{algo}"
        return o_algo, algo 

    def res_to_row(res, algo:str):
        res_id = res.id
        res_split = res_id.split("-")
        problem_id = int(res_split[0])
        instance_id = int(res_split[1])
        repeat_id = int(res_split[2])
        row = {}

        loss = res.y_hist - res.optimal_value

        algo_id, algo_name = _algo_to_name(algo)

        for column_name, column_path in column_name_map.items():
            if column_path is None:
                if column_name == 'algorithm':
                    row[column_name] = algo_id
                elif column_name == 'algorithm_name':
                    row[column_name] = algo_name
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
                elif column_name == 'algorithm_short_name':
                    row[column_name] = _shorthand_algo_name(algo)
            else:
                value = dynamical_access(res, column_path)
                non_none_value = _none_to_nan(value)
                if isinstance(non_none_value, list):
                    non_none_value = np.array(non_none_value)
                    row[column_name] = non_none_value
                row[column_name] = non_none_value
        return row

    res_df = pd.DataFrame(columns=column_name_map.keys())
    for result in results:
        algo = result.name
        for res in result.result:
            _bound = 1e4 if len(res.best_x) == 5 else 1e9
            res.update_aoc_with_new_bound_if_needed(upper_bound=_bound)
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

    labels = [label.replace("BL", "") for label in labels]
    # plot aoc
    plot_box_violin(
        data=[all_log_plot_data],
        labels=[labels],
        colors=[colors],
        show_inside_box=True,
        title="AOC Catorized by Problems",
        figsize=(15, 9),
    )

def _plot_algo_aoc(res_df:pd.DataFrame, dim:int, problem_filters=None, file_name=None, fig_dir=None):
    # filter the problem in problem_filters
    all_aoc_df = res_df
    if problem_filters is not None:
        all_aoc_df = all_aoc_df[~all_aoc_df['problem_id'].isin(problem_filters)]

    if all_aoc_df.shape[0] == 0:
        logging.warning("No data to plot")
        return

    all_aoc_df = all_aoc_df.groupby(['algorithm', 'instance_id', 'exec_id'])[['y_aoc', 'log_y_aoc']].agg(np.mean).reset_index()
    all_aoc_df = all_aoc_df.groupby(['algorithm'])[['y_aoc', 'log_y_aoc']].agg(list).reset_index()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    _default_colors = prop_cycle.by_key()['color']
    colors = []
    labels = []
    all_log_plot_data = []

    for algo in all_aoc_df['algorithm']:
        _temp_df = all_aoc_df[all_aoc_df['algorithm'] == algo].agg(list)
        all_log_plot_data.append(_temp_df['log_y_aoc'].values[0])

        short_algo = _shorthand_algo_name(algo)
        labels.append(short_algo)

        if 'BL' in algo:
            colors.append(_default_colors[0])
        else:
            colors.append(_default_colors[1])

    # plot aoc
    title = f"AOC on {dim}D Problems"
    if problem_filters is not None:
        title += " Excluding "
        title += ", ".join([f"F{problem_id}" for problem_id in problem_filters])

    if fig_dir is not None:
        file_name = os.path.join(fig_dir, file_name)
    plot_box_violin(
        data=[all_log_plot_data],
        labels=[labels],
        label_fontsize=8,
        x_tick_fontsize=10,
        y_tick_fontsize=10,
        colors=[colors],
        show_inside_box=True,
        # title=title,
        figsize=(7, 3),
        show=False,
        filename=file_name,
    )

    return {
        'plot_data': all_log_plot_data,
        'labels': labels,
        'colors': colors,
        'dim': dim,
        'filter': problem_filters,
    }

def _plot_algo_problem_aoc(res_df:pd.DataFrame, dim:int, fig_dir=None):
    problem_id_list = res_df['problem_id'].unique()
    problem_id_list.sort()
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

        short_labels.append([_shorthand_algo_name(label) for label in _labels])

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
    step = 1
    for i in range(0, len(log_plot_data), step):
        _plot_data = log_plot_data[i:i+step]
        _labels = labels[i:i+step]
        _sub_titles = sub_titles[i:i+step]
        _colors = colors[i:i+step]

        title = f"AOC on {dim}D Problems"
        file_name = None
        if step == 1:
            title = f"AOC on {sub_titles[i]}({dim}D)"
            _sub_titles = None
            dir_name = f"algo_aoc_{dim}D"
            if fig_dir is not None:
                dir_name = os.path.join(fig_dir, dir_name)
            os.makedirs(dir_name, exist_ok=True)
            file_name = f"{dir_name}/algo_aoc_{dim}D_F{problem_id_list[i]}"

        plot_box_violin(data=_plot_data,
                        labels=_labels,
                        sub_titles=_sub_titles,
                        title=title,
                        label_fontsize=8,
                        show_inside_box=True,
                        colors=_colors,
                        n_cols=2,
                        figsize=(8, 4),
                        show=False,
                        filename=file_name,
                        )

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

def smooth_factory(smooth_type='savgol', window_size=5, polyorder=2, sigma=1.0):
    def _smooth_data(data):
        if smooth_type == 'savgol':
            return savgol_smoothing(data, window_size, polyorder)
        elif smooth_type == 'moving':
            return moving_average(data, window_size)
        elif smooth_type == 'gaussian':
            return gaussian_smoothing(data, sigma)
    return _smooth_data

def _plot_algo_iter(res_df:pd.DataFrame, dim:int, fig_dir=None, data_col_map=None, need_seperate_plot=True, file_name_suffix='', title=None):
    # handle y
    _data_col_map = {
        'n_init': '',
        'acq_exp_threshold': '',

        'loss': 'Loss',
        'best_loss': 'Best Loss',

        # 'r2': 'R2 on test',
        # 'r2_on_train' : 'R2 on X',
        # 'uncertainty' : 'Uncertainty on test',
        # 'uncertainty_on_train' : 'Uncertainty on X',

        # 'kappa': 'Kappa',
        # 'tr_radius': 'Trust Region Radius',

        # 'grid_coverage' : 'Grid Coverage',

        # 'dbscan_circle_coverage': 'DBSCAN Circle Coverage',
        # 'dbscan_rect_coverage': 'DBSCAN Rect Coverage',

        # 'online_rect_coverage': 'Online Cluster Rect Coverage',
        # 'online_circle_coverage': 'Online Circle Coverage',

        # 'acq_grid_coverage' : 'Acq Grid Coverage',

        # 'acq_dbscan_circle_coverage': 'DBSCAN Circle Coverage(Acq)',
        # 'acq_dbscan_rect_coverage': 'DBSCAN Rect Coverage(Acq)',

        # 'acq_online_rect_coverage': 'Online Cluster Rect Coverage(Acq)',
        # 'acq_online_circle_coverage': 'Online Circle Coverage(Acq)',

        # 'exploitation_rate': 'Exploitation Rate',
        # 'acq_exploitation_rate': 'Acq Exploitation Rate(er)',

        # 'acq_exploitation_improvement': 'Exploitation Improvement: $current-best$',
        # 'acq_exploitation_score': 'Exploitation Score: $improve/(best-optimum)$',
        # 'acq_exploitation_validity': 'Exploitation Validity: $score*er$',

        # 'acq_exploration_improvement': 'Exploration Improvement: $current-best$',
        # 'acq_exploration_score': 'Exploration Score: $improve/fixed\_base$',
        # 'acq_exploration_validity': 'Exploration Validity: $score*(1-er)$',
    }
    if data_col_map is None:
        data_col_map = _data_col_map
    data_cols = list(data_col_map.keys())

    clip_cols = {
        'loss': clip_upper_factory(bound_type='median', upper_len_ratio=0.15),
        # 'loss': clip_upper_factory(bound_type='fixed', _bound=150),
    }

    y_df = res_df
    problem_ids = y_df['problem_id'].unique()
    problem_ids.sort()

    loss_upper_bounds = {}

    for problem_id in problem_ids:
        _p_df = y_df[y_df['problem_id'] == problem_id]
        for clip_col, cliper in clip_cols.items():
            _data = _p_df[clip_col].to_list()
            _, _upper_bound = cliper(np.array(_data))
            # a = np.clip(_p_df[clip_col], 0, _upper_bound)
            _p_df.loc[:, clip_col] = _p_df[clip_col].apply(lambda x: np.clip(x, 0, _upper_bound))
            loss_upper_bounds[problem_id] = _upper_bound
            # _p_df[clip_col] = 
        y_df[y_df['problem_id'] == problem_id] = _p_df

    y_df = y_df.groupby(['algorithm', 'problem_id', 'instance_id'])[data_cols].agg(np.nanmean).reset_index()
    # y_df = y_df.groupby(['algorithm', 'problem_id', 'exec_id'])[data_cols].agg(np.mean).reset_index()
    
    if 'loss' in data_cols:
        y_df['best_loss'] = y_df['loss'].apply(np.minimum.accumulate)
    
    # copy each row in y_df with new algorithm name
    # for i, _row in y_df.iterrows():
    #     _algo = _row['algorithm']
    #     _new_row = _row.copy()
    #     _new_row['algorithm'] = _algo + f"_{i}"
    #     y_df.loc[len(y_df)] = _new_row

    y_df = y_df.groupby(['algorithm', 'problem_id'])[data_cols].agg(mean_std_agg).reset_index()
    y_df[data_cols].applymap(lambda x: x[0] if isinstance(x, list) else x)

    smooth_cols = {
        # 'exploitation_rate': smooth_factory(smooth_type='moving', window_size=5),
    }

    y_scale_cols = {
        'loss': ('symlog', {}),
        'best_loss': ('symlog', {}),
        'kappa': ('symlog', {}),
        'tr_radius': ('symlog', {}),
    }

    non_fill_cols = [
        'loss',
        # 'best_loss',
    ]

    ignore_cols = [
        'n_init',
        'acq_exp_threshold',
        'loss'
    ]
    
    best_loss_plot_data = []
    best_loss_x_data = []
    best_loss_plot_filling = []
    best_loss_labels = []
    best_loss_x_dots = []
    best_loss_sub_titles = []
    best_loss_y_scales = []
    best_loss_colors = []
    best_loss_line_styles = []
    best_loss_baselines = []
    best_loss_baseline_labels = []

    for problem_id in problem_ids:
        plot_data = []
        x_data = []
        plot_filling = []
        labels = []
        x_dots = []
        sub_titles = []
        y_scales = []
        colors = []
        line_styles = []
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

            # smooth if needed
            if col in smooth_cols:
                mean_array = smooth_cols[col](mean_array)
            
            plot_data.append(mean_array)
            x_data.append(np.arange(mean_array.shape[1]))
            
            # fill the area between mean - std and mean + std
            if col not in non_fill_cols:
                # std_array = np.array([ele[1] for ele in data])
                max_array = np.array([ele[2] for ele in data])
                min_array = np.array([ele[3] for ele in data])

                # _upper_bound = mean_array + std_array
                # upper_bound = np.clip(_upper_bound, None, max_array)

                # _lower_bound = mean_array - std_array
                # lower_bound = np.clip(_lower_bound, min_array, None)

                plot_filling.append(list(zip(min_array, max_array)))
            else:
                plot_filling.append(None)

            # handle baseline
            _baselines = []
            _baseline_labels = []
            if 'acq_exploitation_rate' in col:
                exp_threshold = _temp_df['acq_exp_threshold'].to_list()
                mean_exp = [ele[0] for ele in exp_threshold]
                _bl = np.nanmean(mean_exp)
                _baselines.append(_bl)
                _baseline_labels.append("Threshold")
            else:
                _baseline_labels.append(None)
                _baselines.append(None)

            # _baseline_labels.append("Upper Bound")
            # _baselines.append([loss_upper_bounds[problem_id]])

            baselines.append(_baselines)
            baseline_labels.append(_baseline_labels)

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
            _colors = [color for i, color in enumerate(_colors) if i not in empty_indexs]
            colors.append(_colors)
            _line_styles = ['--' if 'BL' in _label else '-' for _label in _labels]
            line_styles.append(_line_styles)

            _labels = [_shorthand_algo_name(label) for label in _labels]
            labels.append(_labels)

            _sub_title = data_col_map.get(col, col)
            # _sub_title = f"{_sub_title} On F{problem_id}({dim}D)"
            sub_titles.append(_sub_title)

            if col in y_scale_cols:
                _y_scale, _y_scale_kwargs = y_scale_cols[col]
                y_scales.append((_y_scale, _y_scale_kwargs))
                # sub_titles.append(_sub_title + f"({_y_scale})")
            else:
                y_scales.append(None)
                # sub_titles.append(_sub_title)

            traning_problems = set([2, 4, 6, 8, 12, 14, 15, 18, 21, 23])
            bbob_group = {
                1: [1, 2, 3, 4, 5],
                2: [6, 7, 8, 9],
                3: [10, 11, 12, 13, 14],
                4: [15, 16, 17, 18, 19],
                5: [20, 21, 22, 23, 24],
            }

            groud_id = None
            for gid, problems in bbob_group.items():
                if problem_id in problems:
                    groud_id = gid
                    break

            if col == 'best_loss':
                best_loss_plot_data.append(plot_data[-1])
                best_loss_x_data.append(x_data[-1])
                best_loss_plot_filling.append(plot_filling[-1])
                best_loss_labels.append(labels[-1])
                best_loss_x_dots.append(x_dots[-1])
                if problem_id in traning_problems:
                    best_loss_sub_titles.append('$\overline{F%s} (%d)$' % (problem_id, groud_id))
                else:
                    best_loss_sub_titles.append(f'$F{problem_id} ({groud_id})$')
                best_loss_y_scales.append(y_scales[-1])
                best_loss_colors.append(colors[-1])
                best_loss_line_styles.append(line_styles[-1])
                best_loss_baselines.append(baselines[-1])
                best_loss_baseline_labels.append(baseline_labels[-1])

        if not need_seperate_plot:
            continue
        dir_name = f"algo_loss_{dim}D"
        if fig_dir is not None:
            dir_name = os.path.join(fig_dir, dir_name)
        # os.makedirs(dir_name, exist_ok=True)
        # file_name = f"{dir_name}/algo_loss_{dim}D_F{problem_id}{file_name_suffix}"


        # plot_lines(
        #     y=plot_data, x=x_data, 
        #     y_scales=y_scales,
        #     baselines=baselines,
        #     baseline_labels=baseline_labels,
        #     colors=colors,
        #     labels=labels,
        #     line_styles=line_styles,
        #     label_fontsize=8,
        #     linewidth=1.2,
        #     filling=plot_filling,
        #     x_dot=x_dots,
        #     n_cols=1,
        #     combined_legend=False,
        #     combined_legend_fontsize=11,
        #     combined_legend_bottom=0.12,
        #     tick_fontsize=11,
        #     sub_titles=sub_titles,
        #     sub_title_fontsize=11,
        #     # title=f"F{problem_id}({dim}D)",
        #     figsize=(8, 6),
        #     show=False,
        #     filename=file_name,
        # )

    file_name = f"algo_loss_{dim}D{file_name_suffix}"
    if fig_dir is not None:
        file_name = os.path.join(fig_dir, file_name)
    plot_lines(
        y=best_loss_plot_data, x=best_loss_x_data,
        y_scales=best_loss_y_scales,
        baselines=best_loss_baselines,
        baseline_labels=best_loss_baseline_labels,
        colors=best_loss_colors,
        labels=best_loss_labels,
        line_styles=best_loss_line_styles,
        label_fontsize=11,
        combined_legend=True,
        combined_legend_fontsize=14,
        combined_legend_bottom=0.1,
        combined_legend_ncols=10,
        tick_fontsize=13,
        linewidth=1.2,
        filling=best_loss_plot_filling,
        x_dot=best_loss_x_dots,
        n_cols=6,
        sub_titles=best_loss_sub_titles,
        sub_title_fontsize=14,
        # y_labels=["Loss", ""],
        y_label_fontsize=14,
        # title=f"Best Loss({dim}D)" if title is None else title,
        figsize=(16, 8),
        show=False,
        filename=file_name,
    )

algo_aoc_list = []
algo_filter_aoc_list = []

def plot_algo_result(results:list[EvaluatorResult], fig_dir=None):
    res_df = _process_algo_result(results)

    dim = 0
    for result in results:
        if len(result.result) > 0:
            dim = result.result[0].best_x.shape[0]
            break

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)

    algo_aoc = _plot_algo_aoc(res_df, dim=dim, file_name=f"algo_aoc_{dim}D", fig_dir=fig_dir)
    algo_aoc_list.append(algo_aoc)

    problem_filters = [4, 8, 9, 19, 20, 24]
    algo_filter_aoc = _plot_algo_aoc(res_df, dim=dim, problem_filters=problem_filters, file_name=f"algo_aoc_{dim}D_except_{problem_filters}", fig_dir=fig_dir)
    algo_filter_aoc_list.append(algo_filter_aoc)

    # _plot_algo_aoc_on_problems(res_df)

    _plot_algo_problem_aoc(res_df, dim=dim, fig_dir=fig_dir)

    _plot_algo_iter(res_df, dim=dim, fig_dir=fig_dir)


def plot_algo(file_paths=None, dir_path=None, pop_path=None, fig_dir=None):
    res_list = []
    _file_paths = []
    if file_paths is not None:
        _file_paths.extend(file_paths) 

    if pop_path is not None:
        with open(pop_path, "rb") as f:
            pop = RenameUnpickler.unpickle(f)
            all_inds = pop.all_individuals()
            all_handlers = [ESPopulation.get_handler_from_individual(ind) for ind in all_inds]
            for handler in all_handlers:
                if handler.error is not None:
                    continue
                res_list.append(handler.eval_result)
    elif dir_path is not None:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Invalid directory path: {dir_path}")
        for file in os.listdir(dir_path):
            if file.endswith(".pkl"):
                _file_paths.append(os.path.join(dir_path, file))

    for file_path in _file_paths:
        with open(file_path, "rb") as f:
            target = RenameUnpickler.unpickle(f)
            if target.error is not None:
                continue
            if isinstance(target, EvaluatorResult):
                # import re
                # pattern = re.compile(r"(f?\d)_IOHEvaluator")
                # match = pattern.search(file_path)
                # fixed_suffix = ''
                # rho = 0.0
                # if match:
                #     name = match.group(1)
                #     if 'f' in name:
                #         fixed_suffix = '_fixed'
                #         name = name.replace("f", "")
                #     if name == '8':
                #         rho = 0.85
                #     elif name == '9':
                #         rho = 0.95
                #     algo = f'rho_{rho}{fixed_suffix}'
                #     target.name = algo
                # if len(fixed_suffix) == 0:
                #     res_list.append(target)

                res_list.append(target)
            elif isinstance(target, ResponseHandler):
                res_list.append(target.eval_result)

    plot_algo_result(results=res_list, fig_dir=fig_dir)

def extract_algo_result(dir_path:str, file_path_map:dict=None):
    file_paths = []
    algo_name_list = []

    if file_path_map is None:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Invalid directory path: {dir_path}")
        for file in os.listdir(dir_path):
            if file.endswith(".pkl"):
                file_paths.append(os.path.join(dir_path, file))
    else:
        for algo_name, file in file_path_map.items():
            if os.path.isfile(file):
                file_paths.append(file)
                algo_name_list.append(algo_name)
            else:
                print(f"File {algo_name} not found: {file}")

    res_list = []
    for i, file_path in enumerate(file_paths):
        with open(file_path, "rb") as f:
            unpickler = RenameUnpickler(f)
            target = unpickler.load()
            if target.error is not None:
                continue
            if isinstance(target, EvaluatorResult):
                if len(algo_name_list) > i:
                    target.name = algo_name_list[i]
                res_list.append(target)
            elif isinstance(target, ResponseHandler):
                res_list.append(target.eval_result)
    dim = 0
    for result in res_list:
        if len(result.result) > 0:
            dim = result.result[0].best_x.shape[0]
            break

    column_name_map = {
        'algorithm' : None,
        'algorithm_name' : None,
        'algorithm_short_name' : None,
        'problem_id' : None,
        'instance_id' : None,
        'exec_id' : None,
        'n_init' : 'n_initial_points',

        'optimum' : 'optimal_value',

        'y_hist': 'y_hist',
        'x_hist': 'x_hist',

        'loss': None,
        'best_loss': None,
        'y_aoc': 'log_y_aoc',
    }

    res_df = _process_algo_result(res_list, column_name_map)

    simple_res_df = res_df.drop(columns=['algorithm', 'loss', 'best_loss'])
    simple_res_df.to_csv(f"{dir_path}/hist.csv", index=False)

    algos = res_df['algorithm'].unique()
    # filter_intace_id = 4
    # filter_exec_id = 0
    # filter_problem_id = 4

    df_y_data = []
    df_loss_data = []
    for algo in algos:
        # filter by algo , instance_id, exec_id to create a new dataframe
        _temp_df = res_df[
            (res_df['algorithm'] == algo)
            # & ((res_df['instance_id'] == filter_intace_id) | (res_df['instance_id'] == 5))
            # & (res_df['exec_id'] == filter_exec_id)
            # & ((res_df['problem_id'] == 4) | (res_df['problem_id'] == 5))
        ]

        instance_ids = _temp_df['instance_id'].unique()
        exec_ids = _temp_df['exec_id'].unique()
        # map their combination to run_id
        counter = 0
        run_id_map = {}
        for instance_id in instance_ids:
            for exec_id in exec_ids:
                run_id_map[(instance_id, exec_id)] = counter
                counter += 1

        for _, row in _temp_df.iterrows():
            _y_hist = row['y_hist'].tolist()
            _loss_list = row['loss'].tolist()
            p_id = row['problem_id']
            instance_id = row['instance_id']
            f_id = f"{p_id}_{instance_id}"
            algo_id = row['algorithm_name'].replace("BL", "")
            exec_id = row['exec_id']
            for j, y in enumerate(_y_hist):
                df_y_data.append({
                    'Evaluation counter': j + 1,
                    'Function values': y,
                    'Function ID': f_id,
                    'Algorithm ID': algo_id,
                    'Problem dimension': dim,
                    'Run ID': exec_id
                })
            run_id = run_id_map[(instance_id, exec_id)]
            for j, loss in enumerate(_loss_list):
                df_loss_data.append({
                    'Evaluation counter': j + 1,
                    'Loss': loss,
                    'Function ID': p_id,
                    'Algorithm ID': algo_id,
                    'Problem dimension': dim,
                    'Run ID': run_id
                })
    _new_y_df = pd.DataFrame(df_y_data)
    _new_y_df.to_csv(f"{dir_path}/ioh_fx.csv", index=False)

    _new_loss_df = pd.DataFrame(df_loss_data)
    _new_loss_df.to_csv(f"{dir_path}/ioh_loss.csv", index=False)

    _new_aoc_df = res_df[['algorithm_name', 'problem_id', 'instance_id', 'exec_id', 'y_aoc']]
    _new_aoc_df['short_algo_name'] = _new_aoc_df['algorithm_name'].apply(_shorthand_algo_name)
    _new_aoc_df.to_csv(f"{dir_path}/aoc.csv", index=False)

    _new_mean_aoc_df = res_df.groupby(['algorithm_name', 'problem_id'])[['y_aoc']].agg(np.mean).reset_index()
    _new_mean_aoc_df['short_algo_name'] = _new_mean_aoc_df['algorithm_name'].apply(_shorthand_algo_name)
    _new_mean_aoc_df.to_csv(f"{dir_path}/mean_aoc.csv", index=False)

def plot_algo_0220():
    file_paths = [
        # 'Experiments/log_eater/final_eval_res/TrustRegionAdaptiveTempBOv2_0.1299_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211000039.pkl',
    ] 

    dir_paths = [
        'Experiments/log_eater/final_eval_res_5dim',
        'Experiments/log_eater/final_eval_res_10dim_0320',
        'Experiments/log_eater/final_eval_res_20dim_0320',
        'Experiments/log_eater/final_eval_res_40dim_0320',

        # 'Experiments/log_eater/final_eval_res_atr_20dim',
    ]

    for dir_path in dir_paths:

        pop_path = None

        plot_algo(file_paths=file_paths, dir_path=dir_path, pop_path=pop_path)
        # extract_algo_result(dir_path=dir_path)

    file_name = 'all_algo_aoc'
    plot_y = []
    plot_labels = []
    plot_colors = []
    plot_sub_titles = []
    for algo_aoc in algo_aoc_list:
        plot_y.append(algo_aoc['plot_data'])
        plot_labels.append(algo_aoc['labels'])
        plot_colors.append(algo_aoc['colors'])
        dim = algo_aoc['dim']
        plot_sub_titles.append(f'$d={dim}$')
    plot_box_violin(
        data=plot_y,
        labels=plot_labels,
        label_fontsize=8,
        x_tick_fontsize=11,
        y_tick_fontsize=12,
        colors=plot_colors,
        show_inside_box=True,
        sharex=True,
        sub_titles=plot_sub_titles,
        sub_title_fontsize=13,
        n_cols=2,
        # title=title,
        figsize=(13, 6),
        show=False,
        filename=file_name,
    )

    file_name = 'all_algo_aoc_except'
    plot_y = []
    plot_labels = []
    plot_colors = []
    plot_sub_titles = []
    for algo_aoc in algo_filter_aoc_list:
        plot_y.append(algo_aoc['plot_data'])
        plot_labels.append(algo_aoc['labels'])
        plot_colors.append(algo_aoc['colors'])
        dim = algo_aoc['dim']
        plot_sub_titles.append(f'$d={dim}$')
    plot_box_violin(
        data=plot_y,
        labels=plot_labels,
        label_fontsize=9,
        x_tick_fontsize=10,
        y_tick_fontsize=11,
        sharex=True,
        width=0.8,
        colors=plot_colors,
        show_inside_box=True,
        sub_titles=plot_sub_titles,
        sub_title_fontsize=13,
        n_cols=2,
        # title=title,
        figsize=(13, 6),
        show=False,
        filename=file_name,
    )


def get_atrbo_result_file_path_map():
    file_path_map = {
        'baseline': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:False_proj:True_trr:2.5_rho:0.95_k:2.0_adaptr:True_adaptk:True_0.4615_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421162413.pkl',

        'noProject': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:False_proj:False_trr:2.5_rho:0.95_k:2.0_adaptr:True_adaptk:True_0.4543_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421162546.pkl',

        'fixed': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.95_k:2.0_adaptr:True_adaptk:True_0.4590_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421162458.pkl',

        'fixed_kappa': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.95_k:2.0_adaptr:True_adaptk:False_0.4574_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421172803.pkl',

        'fixed_radius': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.95_k:2.0_adaptr:False_adaptk:True_0.4474_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421162629.pkl',

        'fixed_kappa_radius': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.95_k:2.0_adaptr:False_adaptk:False_0.4479_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421162754.pkl',

        # rho
        '0.65_rho_fixed': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.65_k:2.0_adaptr:True_adaptk:True_0.4648_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421164107.pkl',
        '0.8_rho_fixed': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.8_k:2.0_adaptr:True_adaptk:True_0.4683_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421163324.pkl',
        '0.95_rho_fixed': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.95_k:2.0_adaptr:True_adaptk:True_0.4590_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421162458.pkl',

        # trr
        '1.0_radius': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:1_rho:0.95_k:2.0_adaptr:False_adaptk:False_0.4547_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421164357.pkl',
        '2.5_radius': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.95_k:2.0_adaptr:False_adaptk:False_0.4479_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421162754.pkl',
        '5.0_radius': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:5_rho:0.95_k:2.0_adaptr:False_adaptk:False_0.4397_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421164316.pkl',

        # kappa
        '1.0_kappa': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.95_k:1.0_adaptr:False_adaptk:False_0.4480_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421164232.pkl',
        '2.0_kappa': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:5_rho:0.95_k:2.0_adaptr:False_adaptk:False_0.4397_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421164316.pkl',
        '4.0_kappa': 'Experiments/log_eater/atrbo_eval_res_5dim/ATRBO_fixed:True_proj:True_trr:2.5_rho:0.95_k:4.0_adaptr:False_adaptk:False_0.4485_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4]_repeat-5_0421164150.pkl',
    }
    return file_path_map

def plot_atrbo_results():
    def _extract_paras(file_path):
        file_name = os.path.basename(file_path)
        paras = file_name.split('_')
        para_map = {}
        for para in paras:
            if ':' not in para or 'IOHEvaluator' in para:
                continue
            kv = para.split(':')
            if len(kv) != 2:
                continue
            key = kv[0]
            value = kv[1]
            if value == 'True' or value == 'False': 
                value = True if value == 'True' else False
            else:
                value = float(value)
            para_map[key] = value
        return para_map

    dir_paths = [
        'Experiments/log_eater/atrbo_eval_res_5dim',
    ]
    res_list = []
    df_list = []
    paras_df = []
    for dir_path in dir_paths:
        for file_path in os.listdir(dir_path):
            if file_path.endswith(".pkl"):
                file_path = os.path.join(dir_path, file_path)
                with open(file_path, "rb") as f:
                    unpickler = RenameUnpickler(f)
                    target = unpickler.load()
                    if isinstance(target, ResponseHandler):
                        target = target.eval_result

                    res_list.append(target)
                    res_index = len(res_list) - 1

                    column_name_map = {
                        'algorithm' : None,
                        'algorithm_name' : None,
                        'algorithm_short_name' : None,
                        'problem_id' : None,
                        'instance_id' : None,
                        'exec_id' : None,
                        'n_init' : 'n_initial_points',

                        'optimum' : 'optimal_value',

                        'y_hist': 'y_hist',
                        'x_hist': 'x_hist',

                        'loss': None,
                        'best_loss': None,
                        'y_aoc': 'log_y_aoc',
                    }
                    res_df = _process_algo_result([target], column_name_map)
                    df_list.append(res_df)

                    paras = _extract_paras(file_path)
                    paras['index'] = res_index
                    paras_df.append(paras)

    paras_df = pd.DataFrame(paras_df)
    # plot bug pair
    # get rows where fixed are different but other paras are the same
    other_columns = [col for col in paras_df.columns if col != 'fixed' and col != 'index']
    result_df = paras_df.groupby(other_columns).filter(
        lambda group: group['fixed'].nunique() > 1
    )

    def _res_df_from_result_df(result_df, paras_df, df_list, algo_mapping_func):
        # get the index of the rows in paras_df
        df_indexs = result_df['index'].to_list()
        sub_dfs = []
        for index in df_indexs:
            paras = paras_df.iloc[index]
            _df = df_list[index]
            _label = algo_mapping_func(paras)
            _df['algorithm'] = _label
            if _label == 'baseline':
                sub_dfs.insert(0, _df)
            else:
                sub_dfs.append(_df)
        sub_dfs = [df_list[i] for i in df_indexs]
        # combine the dataframes
        _res_df = pd.concat(sub_dfs, axis=0)
        return _res_df

    _res_df = _res_df_from_result_df(result_df, paras_df, df_list, lambda paras: 'no_bug' if paras['fixed'] == True else 'baseline')

    data_col_map = {
        'n_init': '',
        'loss': 'Loss',
        'best_loss': 'Best Loss',
    }
    fig_dir = 'Experiments/atrbo_eval_res_5dim/atrbo_algo_iter'
    os.makedirs(fig_dir, exist_ok=True)

    # _plot_algo_iter(_res_df, dim=5, fig_dir=fig_dir, data_col_map=data_col_map, need_seperate_plot=False, file_name_suffix='_bug', title='')

    # plot project
    # result_df = paras_df[paras_df['fixed'] == False]
    # _res_df = _res_df_from_result_df(result_df, paras_df, df_list, lambda paras: 'baseline' if paras['proj'] == True else 'no_project')
    # _plot_algo_iter(_res_df, dim=5, fig_dir=fig_dir, data_col_map=data_col_map, need_seperate_plot=False, file_name_suffix='_proj', title='')


    bl_df = paras_df[
        (paras_df['fixed'] == False) 
        & (paras_df['proj'] == True)
        & (paras_df['trr'] == 2.5)
        & (paras_df['rho'] == 0.95)
        & (paras_df['k'] == 2.0)
        & (paras_df['adaptr'] == True)
        & (paras_df['adaptk'] == True)
        ]

    # plot rho
    other_columns = [col for col in paras_df.columns if col != 'rho' and col != 'index']
    result_df = paras_df.groupby(other_columns).filter(
        lambda group: group['rho'].nunique() > 1
    )
    result_df = result_df[
        (result_df['rho'] != 0.95)
        & (result_df['fixed'] == False)
        ]
    result_df = pd.concat([bl_df, result_df], axis=0)
    # _res_df = _res_df_from_result_df(result_df, paras_df, df_list, lambda paras: 'baseline' if paras['fixed'] == False else f"rho_{paras['rho']}")
    _res_df = _res_df_from_result_df(result_df, paras_df, df_list, lambda paras: 'baseline(0.95)' if paras['rho'] == 0.95 else f"rho_{paras['rho']}")
    _plot_algo_iter(_res_df, dim=5, fig_dir=fig_dir, data_col_map=data_col_map, need_seperate_plot=False, file_name_suffix='_rho', title='')

    # plot kappa
    other_columns = [col for col in paras_df.columns if col != 'k' and col != 'index']
    result_df = paras_df.groupby(other_columns).filter(
        lambda group: group['k'].nunique() > 1
    )
    result_df = result_df[
        (result_df['k'] != 2.0)
        & (result_df['fixed'] == False)
        ]
    result_df = pd.concat([bl_df, result_df], axis=0)
    # _res_df = _res_df_from_result_df(result_df, paras_df, df_list, lambda paras: 'baseline' if paras['fixed'] == False else f"kappa_{paras['k']}")
    _res_df = _res_df_from_result_df(result_df, paras_df, df_list, lambda paras: 'baseline(2.0)' if paras['k'] == 2.0 else f"kappa_{paras['k']}")
    _plot_algo_iter(_res_df, dim=5, fig_dir=fig_dir, data_col_map=data_col_map, need_seperate_plot=False, file_name_suffix='_kappa', title='')

    # plot tr_radius
    other_columns = [col for col in paras_df.columns if col != 'trr' and col != 'index']
    result_df = paras_df.groupby(other_columns).filter(
        lambda group: group['trr'].nunique() > 1
    )
    result_df = result_df[
        (result_df['trr'] != 2.5)
        & (result_df['fixed'] == False)
        ]
    result_df = pd.concat([bl_df, result_df], axis=0)
    # _res_df = _res_df_from_result_df(result_df, paras_df, df_list, lambda paras: 'baseline' if paras['fixed'] == False else f"radius_{paras['trr']}")
    _res_df = _res_df_from_result_df(result_df, paras_df, df_list, lambda paras: 'baseline(2.5)' if paras['trr'] == 2.5 else f"radius_{paras['trr']}")
    _plot_algo_iter(_res_df, dim=5, fig_dir=fig_dir, data_col_map=data_col_map, need_seperate_plot=False, file_name_suffix='_radius', title='')

    # plot adaptr
    other_columns = [col for col in paras_df.columns if col != 'adaptr' and col != 'adaptk' and col != 'index']
    result_df = paras_df.groupby(other_columns).filter(
        lambda group: group['adaptr'].nunique() > 1 and group['adaptk'].nunique() > 1
    )
    result_df = result_df[
        (result_df['fixed'] == False)
        ]
    result_df = pd.concat([bl_df, result_df], axis=0)
    def _adaptr_mapping_func(paras):
        if paras['adaptr'] == False and paras['adaptk'] == False:
            # return f'radius_{paras["adaptr"]} & kappa_{paras["adaptk"]}'
            return 'fixed_radius_kappa'
        elif paras['adaptr'] == True and paras['adaptk'] == False:
            # return f'kappa_{paras["adaptk"]}'
            return 'fixed_kappa'
        elif paras['adaptr'] == False and paras['adaptk'] == True:
            # return f'radius_{paras["adaptr"]}'
            return 'fixed_radius'
        else:
            return 'baseline'
    _res_df = _res_df_from_result_df(result_df, paras_df, df_list, _adaptr_mapping_func)
    _plot_algo_iter(_res_df, dim=5, fig_dir=fig_dir, data_col_map=data_col_map, need_seperate_plot=False, file_name_suffix='_adap', title='')
                

def convert_atrbo_results_to_ioh_csv():
    file_path_map = get_atrbo_result_file_path_map()

    dir_path = 'Experiments/log_eater/atrbo_eval_res_5dim'
    extract_algo_result(dir_path=dir_path, file_path_map=file_path_map)

def calculate_mannwhitneyu_test():
    dir_paths = [
        'Experiments/log_eater/final_eval_res_5dim',
        # 'Experiments/log_eater/final_eval_res_10dim_0320',
        # 'Experiments/log_eater/final_eval_res_20dim_0320',
        # 'Experiments/log_eater/final_eval_res_40dim_0320',
    ]

    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Invalid directory path: {dir_path}")
        _calculate_mannwhitneyu_test_on_dir(dir_path)

def _calculate_mannwhitneyu_test_on_dir(dir_path:str):
    from scipy import stats

    res_list = []
    for file_path in os.listdir(dir_path):
        if file_path.endswith(".pkl"):
            file_path = os.path.join(dir_path, file_path)
            with open(file_path, "rb") as f:
                unpickler = RenameUnpickler(f)
                target = unpickler.load()
                if isinstance(target, ResponseHandler):
                    target = target.eval_result

                res_list.append(target)
    column_name_map = {
        # 'algorithm_name' : None,
        'algorithm_short_name' : None,
        'problem_id' : None,
        'instance_id' : None,
        'exec_id' : None,
        'y_aoc': 'log_y_aoc',
    }
    res_df = _process_algo_result(res_list, column_name_map)
    # rename column algorithm_short_name to algorithm_name
    res_df.rename(columns={'algorithm_short_name': 'algorithm_name'}, inplace=True)


    def _calculate_mannwhitneyu_test_on_problem(ori_df, problem_id=np.nan, verbose=True):
        res_df = ori_df.copy()
        if not np.isnan(problem_id):
            res_df = res_df[res_df['problem_id'] == problem_id]

        # get the mean aoc for each algorithm
        all_mean_aoc = res_df.groupby(['algorithm_name'])[['y_aoc']].agg(np.mean).reset_index()
        # get the best algorithm
        best_algo = all_mean_aoc[all_mean_aoc['y_aoc'] == all_mean_aoc['y_aoc'].max()]['algorithm_name'].values[0]

        best_aoc_list = res_df[res_df['algorithm_name'] == best_algo]['y_aoc'].to_list()
        other_algos = all_mean_aoc[all_mean_aoc['algorithm_name'] != best_algo]['algorithm_name'].values
        stat_list = []
        stat_list.append({
            'algorithm_name': best_algo,
            'contrast_algo': None,
            'stat': np.nan,
            'p_value': np.nan,
            'size': len(best_aoc_list),
            'median': np.median(best_aoc_list),
            'mean': np.mean(best_aoc_list),
            'pid': problem_id,
        })
        if verbose:
            print("-" * 30)
            print(f'problem_id: {problem_id}')
            print('')

        for algo in other_algos:
            algo_aoc_list = res_df[res_df['algorithm_name'] == algo]['y_aoc'].to_list()
            # perform mannwhitneyu test
            # stat, p_value = stats.mannwhitneyu(best_aoc_list, algo_aoc_list)
            stat, p_value = stats.ttest_rel(best_aoc_list, algo_aoc_list)

            stat_list.append({
                'algorithm_name': algo,
                'contrast_algo': best_algo,
                'stat': stat,
                'p_value': p_value,
                'size': len(algo_aoc_list),
                'median': np.median(algo_aoc_list),
                'mean': np.mean(algo_aoc_list),
                'pid': problem_id,
            })

            if verbose:
                print(f"{best_algo}: size={len(best_aoc_list)}, median={np.median(best_aoc_list)}, mean={np.mean(best_aoc_list)}")
                print(f"{algo}: size={len(algo_aoc_list)}, median={np.median(algo_aoc_list)}, mean={np.mean(algo_aoc_list)}")
                print(f"{best_algo} vs {algo}: stat={stat}, p_value={p_value}")
                print('')
        
        if verbose:
            print("-" * 30)

        return stat_list

    res_stat_list = []
    res_stats = _calculate_mannwhitneyu_test_on_problem(res_df, verbose=False)
    res_stat_list.extend(res_stats)

    problems = res_df['problem_id'].unique()
    for problem in problems:
        p_res_stats = _calculate_mannwhitneyu_test_on_problem(res_df, problem_id=problem, verbose=False)
        res_stat_list.extend(p_res_stats)

    # convert to dataframe
    res_stat_df = pd.DataFrame(res_stat_list)

    # save to csv
    # res_stat_df.to_csv(f"{dir_path}/mannwhitneyu_test.csv", index=False)
    res_stat_df.to_csv(f"{dir_path}/t_test.csv", index=False)


def plot_project_tr():
    from scipy.stats import qmc

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True) 
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)  

    # Parameters
    dim = 2  # Dimension of the space
    bounds = np.array([[-5, -5], [5, 5]])  # Lower and upper bounds
    center = np.array([0, 0])  # Custom center
    radius = 5       # Custom radius
    title_fontsize = 14
    label_fontsize = 13

    dot_size = 10 if len(axes.shape) == 2 else 40  # Adjust dot size based on number of subplots
    n_points = 200 if len(axes.shape) == 2 else 20  # Adjust number of points based on number of subplots

    if len(axes.shape) == 1:
        # n_points distinct colors
        # colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        colors = plt.cm.tab20(np.linspace(0, 1, n_points))
        # colors = plt.cm.plasma(np.linspace(0, 1, n_points))
        # colors = plt.cm.inferno(np.linspace(0, 1, n_points))
        # colors = plt.cm.cividis(np.linspace(0, 1, n_points))
    else:
        # one color for all points: first default color
        colors = ['#1f77b4'] * n_points  # Default blue color

    # --- Visualization ---
   

    # 1. Sobol Sequence [0, 1] and Scaled to [-1, 1]
    sampler = qmc.Sobol(d=dim, scramble=True)
    points_sobol = sampler.random(n=n_points)
    points_scaled = qmc.scale(points_sobol, -1, 1)
    axe = axes[0, 0] if len(axes.shape) == 2 else axes[0]
    axe.scatter(points_scaled[:, 0], points_scaled[:, 1], s=dot_size, c=colors)
    # plot the center point
    axe.scatter(center[0], center[1], c='r', marker='*', s=dot_size*3)  # Center point in red
    axe.set_title('1.1 Scale Sobol Sequence to [-1, 1]', fontsize=title_fontsize)
    axe.set_xlim(-1.1, 1.1)
    axe.set_ylim(-1.1, 1.1)
    axe.tick_params(axis='both', which='major', labelsize=label_fontsize)
    axe.set_aspect('equal')

    # 2. Projected to Circle
    lengths = np.linalg.norm(points_scaled, axis=1, keepdims=True)
    unit_points = points_scaled / lengths 
    # points_hypersphere = points_scaled / lengths ** (1/dim)

    axe = axes[0, 1] if len(axes.shape) == 2 else axes[1]
    axe.scatter(unit_points[:, 0], unit_points[:, 1], s=dot_size, c=colors)
    axe.scatter(center[0], center[1], c='r', marker='*', s=dot_size*3)  # Center point in red
    axe.set_title('1.2 Project to Circle', fontsize=title_fontsize)
    axe.set_xlim(-1.1, 1.1)
    axe.set_ylim(-1.1, 1.1)
    axe.set_aspect('equal')
    axe.tick_params(axis='both', which='major', labelsize=label_fontsize)
    axe.add_patch(plt.Circle((0, 0), 1, color='r', alpha=0.1)) # Add a circle

    # 3. Sample The Magnitudes
    points_hypersphere = unit_points * np.random.uniform(0, 1, size=lengths.shape) ** (1/dim)

    axe = axes[0, 2] if len(axes.shape) == 2 else axes[2]
    axe.scatter(points_hypersphere[:, 0], points_hypersphere[:, 1], s=dot_size, c=colors)
    axe.scatter(center[0], center[1], c='r', marker='*', s=dot_size*3)  # Center point in red
    axe.set_title('1.3 Scaled by $u^{1/d}$', fontsize=title_fontsize)
    axe.set_xlim(-1.1, 1.1)
    axe.set_ylim(-1.1, 1.1)
    axe.set_aspect('equal')
    axe.tick_params(axis='both', which='major', labelsize=label_fontsize)
    axe.add_patch(plt.Circle((0, 0), 1, color='r', alpha=0.1)) # Add a circle

    # 4. Final Points (Scaled, Translated, Clipped)
    # sampled_points = points_hypersphere * radius + center
    # sampled_points = np.clip(sampled_points, bounds[0], bounds[1])

    # axe = axes[0, 3]
    # axe.scatter(sampled_points[:, 0], sampled_points[:, 1], s=5, c=colors)
    # axe.scatter(center[0], center[1], c='r', marker='*', s=100)
    # axe.set_title('4. Sampled with Projected Sobol Sequence')
    # # Draw the bounding box
    # rect = plt.Rectangle(bounds[0], bounds[1, 0] - bounds[0, 0],
    #                     bounds[1, 1] - bounds[0, 1], linewidth=1, edgecolor='r', facecolor='none')
    # # axe.add_patch(rect)
    # # Draw circle of radius
    # axe.add_patch(plt.Circle(center, radius, color='g', alpha=0.1))
    # axe.set_xlim(bounds[0,0]-0.5, bounds[1,0]+0.5)
    # axe.set_ylim(bounds[0,1]-0.5, bounds[1,1]+0.5)
    # axe.set_aspect('equal')

    
    # translated and clipped without projection
    # points_scaled = qmc.scale(points_sobol, center-radius, center+radius)
    # sampled_points = points_scaled * radius + center

    # axe = axes[1, 1]
    # sampled_points = np.clip(points_scaled, bounds[0], bounds[1])
    # axe.scatter(sampled_points[:, 0], sampled_points[:, 1], s=5, c=colors)
    # axe.scatter(center[0], center[1], c='r', marker='*', s=100)
    # axe.set_title('Sampled with Sobol Sequence')
    # # Draw the bounding box
    # rect = plt.Rectangle(bounds[0], bounds[1, 0] - bounds[0, 0],
    #                     bounds[1, 1] - bounds[0, 1], linewidth=1, edgecolor='r', facecolor='none')
    # # axe.add_patch(rect)
    # # Draw circle of radius
    # axe.add_patch(plt.Circle(center, radius, facecolor='none', edgecolor='g', linewidth=1, alpha=0.5)) 
    # # Draw rectangle with radius and center
    # axe.add_patch(plt.Rectangle(center - radius, 2 * radius, 2 * radius, color='g', alpha=0.1))

    # axe.set_xlim(bounds[0,0]-0.5, bounds[1,0]+0.5)
    # axe.set_ylim(bounds[0,1]-0.5, bounds[1,1]+0.5)
    # axe.set_aspect('equal') 

    if len(axes.shape) == 1:
        # Only one row of subplots, adjust the layout
        plt.tight_layout()
        plt.savefig('artbo_proj_20.pdf')
        return

    # 1. uniform sampling in [-1, 1]
    uniform_samples = np.random.uniform(-1, 1, size=(n_points, dim))

    axe = axes[1, 0]
    axe.scatter(uniform_samples[:, 0], uniform_samples[:, 1], s=dot_size, c=colors)
    axe.scatter(center[0], center[1], c='r', marker='*', s=dot_size*3)  # Center point in red
    axe.set_title('2.1 Uniform Sampling in [-1, 1]', fontsize=title_fontsize)
    axe.set_xlim(-1.1, 1.1)
    axe.set_ylim(-1.1, 1.1)
    axe.tick_params(axis='both', which='major', labelsize=label_fontsize)
    axe.set_aspect('equal')

    # 2. Projected to Circle
    lengths = np.linalg.norm(uniform_samples, axis=1, keepdims=True)
    projected_samples = uniform_samples / lengths

    axe = axes[1, 1]
    axe.scatter(projected_samples[:, 0], projected_samples[:, 1], s=dot_size, c=colors)
    axe.scatter(center[0], center[1], c='r', marker='*', s=dot_size*3)  # Center point in red
    axe.set_title('2.2 Projected to Circle', fontsize=title_fontsize)
    axe.set_xlim(-1.1, 1.1)
    axe.set_ylim(-1.1, 1.1)
    axe.set_aspect('equal')
    axe.tick_params(axis='both', which='major', labelsize=label_fontsize)
    axe.add_patch(plt.Circle((0, 0), 1, color='r', alpha=0.1)) # Add a circle

    # 3. Sample The Magnitudes by uniform distribution
    magnitudes = np.random.uniform(0, 1, size=(n_points, 1))
    points_hypersphere = projected_samples * magnitudes 

    axe = axes[1, 2]
    axe.scatter(points_hypersphere[:, 0], points_hypersphere[:, 1], s=dot_size, c=colors)
    axe.scatter(center[0], center[1], c='r', marker='*', s=dot_size*3)  # Center point in red
    axe.set_title('2.3 Scaled by $u$', fontsize=title_fontsize)
    axe.set_xlim(-1.1, 1.1)
    axe.set_ylim(-1.1, 1.1)
    axe.set_aspect('equal')
    axe.tick_params(axis='both', which='major', labelsize=label_fontsize)
    axe.add_patch(plt.Circle((0, 0), 1, color='r', alpha=0.1)) # Add a circle

    # 4. Final Points (Scaled, Translated, Clipped)
    # samples = points_hypersphere * radius + center
    # samples = np.clip(samples, bounds[0], bounds[1])    

    # axe = axes[1, 3]
    # axe.scatter(samples[:, 0], samples[:, 1], s=5, c=colors)
    # axe.scatter(center[0], center[1], c='r', marker='*', s=100)
    # axe.set_title('4. Sampled with Uniform Distribution')
    # # Draw the bounding box
    # rect = plt.Rectangle(bounds[0], bounds[1, 0] - bounds[0, 0],
    #                     bounds[1, 1] - bounds[0, 1], linewidth=1, edgecolor='r', facecolor='none')
    # # axe.add_patch(rect)
    # # Draw circle of radius
    # axe.add_patch(plt.Circle(center, radius, color='g', alpha=0.1))
    # # Draw rectangle with radius and center
    # # axe.add_patch(plt.Rectangle(center - radius, 2 * radius, 2 * radius, color='g', alpha=0.1))
    # # Draw rectangle with radius and center
    # # axe.add_patch(plt.Rectangle(center - radius, 2 * radius, 2 * radius, color='g', alpha=0.1))
    # axe.set_xlim(bounds[0,0]-0.5, bounds[1,0]+0.5)
    # axe.set_ylim(bounds[0,1]-0.5, bounds[1,1]+0.5)
    # axe.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('artbo_proj_200.pdf')


if __name__ == "__main__":
    # setup_logger(level=logging.DEBUG)
    setup_logger(level=logging.INFO)

    # plot_algo_0220()

    # calculate_mannwhitneyu_test()

    # plot_atrbo_results()

    plot_project_tr()