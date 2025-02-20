import pickle
import os
from functools import cmp_to_key
from datetime import datetime
import numpy as np
from pandas.api.types import is_numeric_dtype
import pandas as pd
from ioh import get_problem 
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from llamea.utils import IndividualLogger
from llamea.population.es_population import ESPopulation
from llamea.prompt_generators.abstract_prompt_generator import ResponseHandler
from llamea.evaluator.evaluator_result import EvaluatorResult
from llamea.utils import plot_group_bars, plot_lines, plot_box_violin, moving_average, savgol_smoothing, gaussian_smoothing
from llamea.population.population import Population, desc_similarity


# utils
def dynamical_access(obj, attr_path):
    attrs = attr_path.split(".")
    target = obj
    for attr in attrs:
        target = getattr(target, attr, None)
        if target is None:
            break
    return target

def combine_acc(column='y_aoc', maximum=True, max_n_iter=None):
    def _inner_combine_acc(df_series):
        _n_iters = df_series['n_iter'].copy()
        _contents = []
        _n_iters.sort()
        _aoc = df_series[column].copy()

        if max_n_iter is not None and max_n_iter > _n_iters[-1]:
            _n_iters.append(max_n_iter)
            if maximum:
                _aoc.append(0)
            else:
                _aoc.append(np.inf)
        
        for i, _n_iter in enumerate(_n_iters):
            n_fill = _n_iter - len(_contents) - 1
            if maximum:
                _contents.extend([0] * n_fill)
            else:
                _contents.extend([np.inf] * n_fill)
            _contents.append(_aoc[i])
        if maximum:
            _acc = np.maximum.accumulate(_contents)
        else:
            _acc = np.minimum.accumulate(_contents)
        return _acc
    return _inner_combine_acc

def compare_expressions(expr1, expr2):
    a1, b1 = map(int, expr1.split('+'))
    a2, b2 = map(int, expr2.split('+'))

    if a1 == a2:
        return b1 - b2
    else:
        return a1 - a2

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

# plot contour 
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
                           cmap=plt.cm.PuBu_r)
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
        
# plot_search_result
def _process_search_result(results:list[tuple[str,Population]], save=False, file_name=None):
    column_names = [
        'strategy',
        'n_strategy',
        'problem_id',
        'instance_id',
        'exec_id',
        'n_gen',
        'n_iter',
        'n_ind',
        "log_y_aoc",
        "y_aoc",
        "best_y",
        'loss',
        ]

    def res_to_row(res, gen:int, strategy_name:str, n_iter:int, n_ind:int, n_strategy:int):
        res_id = res.id
        res_split = res_id.split("-")
        problem_id = int(res_split[0])
        instance_id = int(res_split[1])
        repeat_id = int(res_split[2])
        row = {
            'strategy': strategy_name,
            'n_strategy': n_strategy,
            'problem_id': problem_id,
            'instance_id': instance_id,
            'exec_id': repeat_id,
            'n_gen': gen+1,
            'n_iter': n_iter,
            'n_ind': n_ind,
            "log_y_aoc": res.log_y_aoc,
            "y_aoc": res.y_aoc,
            "best_y": res.best_y,
            'loss': abs(res.optimal_value - res.best_y),
        }
        return row

    res_df = None
    if not save and file_name is not None:
        res_df = pd.read_pickle(file_name)
    else: 
        _strategy_count = {}
        res_df = pd.DataFrame(columns=column_names)
        for strategy_name, pop in results:
            if strategy_name not in _strategy_count:
                _strategy_count[strategy_name] = 0
            _strategy_count[strategy_name] += 1
            n_generation = pop.get_current_generation()
            n_iter = 1
            for gen in range(n_generation):
                # offspring generated in this generation
                n_inds = n_iter
                gen_offsprings = pop.get_offsprings(generation=gen)
                n_iter += len(gen_offsprings)
                # offspring selected in this generation
                # gen_inds = pop.get_individuals(generation=gen)
                gen_inds = gen_offsprings
                for i, ind in enumerate(gen_inds):
                    handler = Population.get_handler_from_individual(ind)
                    _n_ind = n_inds + i
                    _count = _strategy_count[strategy_name]
                    for res in handler.eval_result.result:
                        row = res_to_row(res, gen, strategy_name=strategy_name, n_iter=n_iter, n_ind=_n_ind, n_strategy=_count)
                        res_df.loc[len(res_df)] = row

    if save and file_name is not None:
        res_df.to_pickle(file_name)
    return res_df

def _plot_search_aoc(res_df:pd.DataFrame, unique_strategies:list[str]):
    max_aoc_df = res_df.groupby(['strategy', 'n_strategy', 'n_ind'])[["log_y_aoc", "y_aoc"]].agg(np.mean).reset_index()
    max_aoc_df = max_aoc_df.groupby(['strategy', 'n_strategy'])[["log_y_aoc", "y_aoc"]].agg(np.max).reset_index()
    max_aoc_df = max_aoc_df.groupby(['strategy'])[['log_y_aoc', 'y_aoc']].agg(list).reset_index()

    _volin_y = []
    for strategy in unique_strategies:
        strategy_df = max_aoc_df[max_aoc_df['strategy'] == strategy]
        _max_aoc_list = strategy_df['log_y_aoc'].values[0]
        _volin_y.append(np.array(_max_aoc_list))

    plot_box_violin(
        data=[_volin_y],
        labels=[unique_strategies],
        plot_type="violin",
        n_cols=4,
        title="AOC",
        label_fontsize=10,
        figsize=(14, 8),
        )
        
def _plot_search_group_aoc(res_df:pd.DataFrame, unique_strategies:list[str]):
    max_n_iter = res_df['n_iter'].max()
    aoc_df = res_df.groupby(['strategy', 'n_strategy', 'n_iter', 'n_ind'])[["log_y_aoc", "y_aoc"]].agg(np.mean).reset_index()
    aoc_df = aoc_df.groupby(['strategy', 'n_strategy', 'n_iter'])[["log_y_aoc", "y_aoc"]].agg(np.max).reset_index()

    aoc_df = aoc_df.groupby(['strategy', 'n_strategy',])[['n_iter',"log_y_aoc", "y_aoc"]].agg(list).reset_index()
    aoc_df['acc_y_aoc'] = aoc_df.apply(combine_acc('y_aoc', max_n_iter=max_n_iter), axis=1)
    aoc_df['acc_log_y_aoc'] = aoc_df.apply(combine_acc('log_y_aoc', max_n_iter=max_n_iter), axis=1)

    aoc_df = aoc_df.groupby(['strategy'])[['acc_y_aoc', 'acc_log_y_aoc']].agg(list).reset_index()

    strategy_group = {}
    # same n_parent: 4, 8, 12, 20
    # n_offspring: mu > lambda, mu <= lambda
    gruoup_name_map = {
        '4': '4+*',
        '8': '8+*',
        '12': '12+*',
        '20': '20+*',
        'mu': '$\mu$ > $\lambda$',
        'lambda': '$\mu$ <= $\lambda$',
    }

    strategy_aoc = [] 
    strategy_filling = []

    strategy_log_aoc = []
    strategy_log_filling = []
    
    labels = []

    for strategy in unique_strategies:
        strategy_df = aoc_df[aoc_df['strategy'] == strategy]

        acc_y_aoc = np.array(strategy_df['acc_y_aoc'].values[0])
        y_aoc = np.mean(acc_y_aoc, axis=0)
        std_y_aoc = np.std(acc_y_aoc, axis=0)
        strategy_aoc.append(y_aoc)
        strategy_filling.append((y_aoc + std_y_aoc, y_aoc - std_y_aoc))

        acc_log_y_aoc = np.array(strategy_df['acc_log_y_aoc'].values[0])
        log_y_aoc = np.mean(acc_log_y_aoc, axis=0)
        std_log_y_aoc = np.std(acc_log_y_aoc, axis=0)
        strategy_log_aoc.append(log_y_aoc)
        strategy_log_filling.append((log_y_aoc + std_log_y_aoc, log_y_aoc - std_log_y_aoc))

        labels.append(strategy)

        mu, lam = strategy.split('+') 
        int_mu, int_lam = int(mu), int(lam)

        if int_mu == 1 or mu in strategy_group:
            _keys = []
            if int_mu == 1:
                _keys.extend(gruoup_name_map.keys())
            else:
                _keys.append(mu)
                if int_mu > int_lam:
                    _keys.append('mu')
                else:
                    _keys.append('lambda')
            
            for _key in _keys:
                if _key not in strategy_group:
                    _temp = {
                        'aoc': [],
                        'aoc_filling': [],
                        'log_aoc': [],
                        'log_aoc_filling': [],
                        'labels': [], 
                    }
                    strategy_group[_key] = _temp
                
                strategy_group[_key]['aoc'].append(y_aoc)
                strategy_group[_key]['aoc_filling'].append((y_aoc + std_y_aoc, y_aoc - std_y_aoc))
                strategy_group[_key]['log_aoc'].append(log_y_aoc)
                strategy_group[_key]['log_aoc_filling'].append((log_y_aoc + std_log_y_aoc, log_y_aoc - std_log_y_aoc))
                strategy_group[_key]['labels'].append(strategy)
    
    plot_y = []
    sub_titles = []
    fillings = []
    plot_labels = []
    for group_key, group_ele in strategy_group.items():
        # plot_y.append(np.array(group_ele['aoc']))
        # fillings.append(group_ele['aoc_filling'])

        plot_y.append(np.array(group_ele['log_aoc']))
        fillings.append(group_ele['log_aoc_filling'])
        sub_titles.append(gruoup_name_map[group_key])
        plot_labels.append(group_ele['labels'])

    # plot_y = [np.array(strategy_log_aoc)]
    # sub_titles = ["AOC"]
    # filling = [strategy_log_filling]
    # plot_labels = [labels]

    x_base = np.arange(len(strategy_aoc[0]), dtype=np.int16)
    x = np.tile(x_base, (len(plot_y), 1))
    plot_lines(
        y = plot_y,
        x = x,
        labels = plot_labels,
        filling= fillings,
        sub_titles=sub_titles,
        n_cols=3,
        label_fontsize=10,
        figsize=(15, 9),
        title="AOC",
        # y_scales=[("log", {})],
        )

def _plot_serach_pop_similarity(results:list[tuple[str,Population]]):
    strategy_group = {}
    for strategy_name, pop in results:
        if strategy_name not in strategy_group:
            strategy_group[strategy_name] = []
        strategy_group[strategy_name].append(pop)

    y_sim_list = []
    y_sim_filling = []
    labels = []
    y_pop_sim_list = []

    for strategy_name, group in strategy_group.items():
        sim_list = []
        pop_sim_list = []
        for pop in group:
            iter_sim_list = []
            n_iter = 1
            n_generation = pop.get_current_generation()
            for gen in range(n_generation):
                gen_offsprings = pop.get_offsprings(generation=gen)
                n_iter += len(gen_offsprings)

                n_fill = n_iter - len(iter_sim_list)
                mean_sim, _ = desc_similarity(gen_offsprings)
                _sim = np.mean(mean_sim)
                iter_sim_list.extend([_sim] * n_fill)
            sim_list.append(iter_sim_list)

            all_inds = pop.all_individuals()
            all_mean_sim, _ = desc_similarity(all_inds)
            pop_sim_list.append(np.mean(all_mean_sim))

        mean_sim = np.mean(sim_list, axis=0)
        std_sim = np.std(sim_list, axis=0)
        y_sim_list.append(mean_sim)
        y_sim_filling.append((mean_sim + std_sim, mean_sim - std_sim))

        labels.append(strategy_name)

        y_pop_sim_list.append(pop_sim_list)

    plot_y = [np.array(y_sim_list)]
    x_base = np.arange(len(y_sim_list[0]), dtype=np.int16)
    x = np.tile(x_base, (len(plot_y), 1))
    plot_lines(
        y = plot_y,
        x = x,
        labels = [labels],
        filling=[y_sim_filling], 
        )

def _process_error_data(results:list[tuple[str,Population]]):
    # - error rate by strategy
    # - error rate by generation
    # - error type
    column_names = [
    'strategy',
    'n_gen',
    'n_repeat',
    'n_iter',
    'err_type',
    ]
    _err_df = pd.DataFrame(columns=column_names)
    _strategy_count = {}
    for strategy_name, pop in results:
        if strategy_name not in _strategy_count:
            _strategy_count[strategy_name] = 0
        _strategy_count[strategy_name] += 1

        n_generation = pop.get_current_generation()
        n_iter = 0
        for gen in range(n_generation):
            # offspring generated in this generation
            gen_offsprings = pop.get_offsprings(generation=gen)
            n_iter += len(gen_offsprings)
            # offspring selected in this generation
            for ind in gen_offsprings:
                handler = Population.get_handler_from_individual(ind)
                res = {
                    'strategy': strategy_name,
                    'n_gen': gen+1,
                    'n_iter': n_iter,
                    'n_repeat': _strategy_count[strategy_name],
                    'err_type': handler.error_type
                }
                _err_df.loc[len(_err_df)] = res
    return _err_df

def _plot_search_all_error_rate(err_df:pd.DataFrame, unique_strategies:list[str]):
    _all_error_df = err_df.groupby(['strategy', 'n_repeat'])['err_type'].agg(list).reset_index()
    _all_error_df['err_rate'] = _all_error_df['err_type'].apply(lambda x: len([ele for ele in x if ele is not None]) / len(x))

    y_err_rates = []

    for strategy in unique_strategies:
        _strategy_error_df = _all_error_df[_all_error_df['strategy'] == strategy]
        _error_rate = _strategy_error_df['err_rate'].to_list()
        y_err_rates.append(_error_rate)

    plot_box_violin(
        data=[y_err_rates],
        labels=[unique_strategies],
        plot_type="violin",
        n_cols=4,
        label_fontsize=10,
        title="Error rate by strategy",
        figsize=(15, 9),
        ) 

def _plot_search_error_type(err_df:pd.DataFrame):
    _size = err_df.shape[0]
    type_count = err_df['err_type'].value_counts()
    _all_type_count = type_count.sum()

    # sum types less than 0.01 into others
    _threshold = 0.01
    _other_count = 0
    for _type, _count in type_count.items():
        if _count / _all_type_count < _threshold:
            _other_count += _count
    type_count = type_count[type_count / _all_type_count >= _threshold]
    type_count['others'] = _other_count

    _title = f"{_all_type_count} errors in {_size} algorithms"
    _title = f'Total errors: {_all_type_count}/{_size}'
    _plot_data = type_count

    _, ax = plt.subplots(figsize=(10, 6))
    ax.pie(_plot_data, 
            labels=_plot_data.index, 
            autopct='%1.1f%%',
        #    autopct=lambda p: '{:d}'.format(int(p / 100 * _all_type_count)),
            )
    ax.set_title(_title)

def _plot_search_error_rate_by_generation(err_df:pd.DataFrame, unique_strategies:list[str]):
    def _combine_err_rate(df_series):
        _n_iters = df_series['n_iter']
        _contents = []
        sort_idxs = np.argsort(_n_iters)
        _rates = df_series['err_rate']
        for i, _n_iter_idx in enumerate(sort_idxs):
            _n_iter = _n_iters[_n_iter_idx]
            _rate = _rates[_n_iter_idx]
            n_fill = _n_iter - len(_contents)
            _contents.extend([_rate] * n_fill)
        return np.array(_contents)
    
    _gen_error_df = err_df.groupby(['strategy', 'n_iter', 'n_repeat'])['err_type'].agg(list).reset_index()
    _gen_error_df['err_rate'] = _gen_error_df['err_type'].apply(lambda x: len([ele for ele in x if ele is not None]) / len(x))
    _gen_error_df = _gen_error_df.groupby(['strategy', 'n_repeat'])[['err_rate', 'n_iter']].agg(list).reset_index()
    _gen_error_df['evol_err_rate'] = _gen_error_df.apply(_combine_err_rate, axis=1)

    strategy_group = {}
    # same n_parent: 4, 8, 12, 20
    # n_offspring: mu > lambda, mu <= lambda
    gruoup_name_map = {
        '4': '4+*',
        '8': '8+*',
        '12': '12+*',
        '20': '20+*',
        'mu': '$\mu$ > $\lambda$',
        'lambda': '$\mu$ <= $\lambda$',
    }

    for _key in gruoup_name_map.keys():
        strategy_group[_key] = {
            'err_rate': [],
            'err_rate_filling': [],
            'labels': [],
        }

    y_err_rates = []
    y_err_rates_filling = []
    labels = []
    for strategy in unique_strategies:
        _strategy_error_df = _gen_error_df[_gen_error_df['strategy'] == strategy]

        _evol_err_rate = _strategy_error_df['evol_err_rate'].to_list()
        _mean_err_rate = np.mean(_evol_err_rate, axis=0)
        _std_err_rate = np.std(_evol_err_rate, axis=0)

        y_err_rates.append(_mean_err_rate)
        y_err_rates_filling.append((_mean_err_rate + _std_err_rate, _mean_err_rate - _std_err_rate))

        labels.append(strategy)

        mu, lam = strategy.split('+') 
        int_mu, int_lam = int(mu), int(lam)

        if int_mu == 1:
            continue 

        _keys = []
        if int_mu == 1:
            _keys.extend(gruoup_name_map.keys())
        else:
            _keys.append(mu)
            if int_mu > int_lam:
                _keys.append('mu')
            else:
                _keys.append('lambda')
        
        for _key in _keys:
            if _key not in strategy_group:
                _temp = {
                    'err_rate': [],
                    'err_rate_filling': [],
                    'labels': [],
                }
                strategy_group[_key] = _temp
            
            strategy_group[_key]['err_rate'].append(_mean_err_rate)
            if int_mu == 1:
                strategy_group[_key]['err_rate_filling'].append((_mean_err_rate, _mean_err_rate))
            else:
                strategy_group[_key]['err_rate_filling'].append((_mean_err_rate + _std_err_rate, _mean_err_rate - _std_err_rate))
            strategy_group[_key]['labels'].append(strategy)

    plot_y = []
    sub_titles = []
    fillings = []
    plot_labels = []
    for group_key, group_ele in strategy_group.items():
        plot_y.append(np.array(group_ele['err_rate']))
        fillings.append(group_ele['err_rate_filling'])
        sub_titles.append(gruoup_name_map[group_key])
        plot_labels.append(group_ele['labels']) 
        
    # plot_y = [np.array(y_err_rates)]

    x_base = np.arange(len(y_err_rates[0]), dtype=np.int16)
    x = np.tile(x_base, (len(plot_y), 1))
    plot_lines(
        y = plot_y,
        x = x,
        labels = plot_labels,
        label_fontsize=9,
        # filling=fillings,
        sub_titles=sub_titles,
        title="Error rate by generation",
        n_cols=3,
        figsize=(15, 9),
        )  

def _plot_search_problem_aoc_and_loss(res_df:pd.DataFrame):
    def _min_max_agg(x):
        if 'log' in x.name:
            return np.max(x)
        return np.min(x)

    aoc_df = res_df.groupby(['strategy', 'problem_id','instance_id', 'exec_id', 'n_iter'])[["log_y_aoc", 'y_aoc', 'loss']].agg(_min_max_agg).reset_index()
    aoc_df = aoc_df.groupby(['strategy', 'problem_id','instance_id', 'exec_id'])[['n_iter',"log_y_aoc", 'y_aoc', 'loss']].agg(list).reset_index()
    aoc_df['acc_y_aoc'] = aoc_df.apply(combine_acc('y_aoc'), axis=1)
    aoc_df['acc_log_y_aoc'] = aoc_df.apply(combine_acc('log_y_aoc'), axis=1)
    aoc_df['acc_loss'] = aoc_df.apply(combine_acc('loss', maximum=False), axis=1)
    
    problem_log_aoc = []
    problem_log_aoc_filling = []
    problem_loss = []
    problem_loss_filling = []
    labels = []

    # same n_parent: 4, 8, 12, 20
    # n_offspring: mu > lambda, mu <= lambda
    gruoup_name_map = {
        '4': '4+*',
        '8': '8+*',
        '12': '12+*',
        '20': '20+*',
        'mu': '$\mu$ > $\lambda$',
        'lambda': '$\mu$ <= $\lambda$',
    }
    problem_group = {}

    unique_problems = aoc_df['problem_id'].unique()
    
    for problem in unique_problems:
        problem_df = aoc_df[aoc_df['problem_id'] == problem]
        unique_strategies = problem_df['strategy'].unique()     
        _log_aoc = []
        _log_aoc_filling = []
        _loss = []
        _loss_filling = []
        _labels = []    
        strategy_group_in_problem = {}
        for strategy in unique_strategies:
            strategy_df = problem_df[problem_df['strategy'] == strategy]
            acc_log_y_aoc = np.array(strategy_df['acc_log_y_aoc'].values)
            log_y_aoc = np.mean(acc_log_y_aoc, axis=0)
            std_log_y_aoc = np.std(acc_log_y_aoc, axis=0)
            _log_aoc.append(log_y_aoc)
            _log_aoc_filling.append((log_y_aoc + std_log_y_aoc, log_y_aoc - std_log_y_aoc))

            acc_loss = np.array(strategy_df['acc_loss'].values)
            loss = np.mean(acc_loss, axis=0)
            std_loss = np.std(acc_loss, axis=0)
            _loss.append(loss)
            _loss_filling.append((loss + std_loss, loss - std_loss))

            _labels.append(strategy)

            mu, lam = strategy.split('+') 
            int_mu, int_lam = int(mu), int(lam)

            if int_mu == 1 or mu in strategy_group_in_problem:
                _keys = []
                if int_mu == 1:
                    _keys.extend(gruoup_name_map.keys())
                else:
                    _keys.append(mu)
                    if int_mu > int_lam:
                        _keys.append('mu')
                    else:
                        _keys.append('lambda')
                
                for _key in _keys:
                    if _key not in strategy_group_in_problem:
                        _temp = {
                            'aoc': [],
                            'aoc_filling': [],
                            'loss': [],
                            'loss_filling': [],
                            'labels': [], 
                        }
                        strategy_group_in_problem[_key] = _temp

                    strategy_group_in_problem[_key]['aoc'].append(log_y_aoc)
                    strategy_group_in_problem[_key]['aoc_filling'].append((log_y_aoc + std_log_y_aoc, log_y_aoc - std_log_y_aoc))
                    strategy_group_in_problem[_key]['loss'].append(loss)
                    strategy_group_in_problem[_key]['loss_filling'].append((loss + std_loss, loss - std_loss))
                    strategy_group_in_problem[_key]['labels'].append(strategy)

        problem_group[problem] = strategy_group_in_problem
        problem_log_aoc.append(_log_aoc)
        problem_log_aoc_filling.append(_log_aoc_filling)
        problem_loss.append(_loss)
        problem_loss_filling.append(_loss_filling)
        labels.append(_labels) 

    for problem, strategy_group in problem_group.items():
        plot_y = []
        sub_titles = []
        fillings = []
        plot_labels = []
        y_scale = []
        title = f"F{problem}"
        for group_key, group_ele in strategy_group.items():
            plot_y.append(np.array(group_ele['aoc']))
            fillings.append(group_ele['aoc_filling'])
            sub_titles.append(f"{gruoup_name_map[group_key]}(AOC)")
            plot_labels.append(group_ele['labels'])
            y_scale.append(("log", {}))

            plot_y.append(np.array(group_ele['loss']))
            fillings.append(group_ele['loss_filling'])
            sub_titles.append(f"{gruoup_name_map[group_key]}(Loss)")
            plot_labels.append(group_ele['labels'])
            y_scale.append(("linear", {}))

        x_base = np.arange(len(problem_log_aoc[0][0]), dtype=np.int16)
        x = np.tile(x_base, (len(plot_y), 1))
        plot_lines(
            y = plot_y,
            x = x,
            labels = plot_labels,
            filling=fillings,
            sub_titles=sub_titles,
            # y_scales=y_scale,
            title=title,
            n_cols=4,
            figsize=(15, 9),
            )
        

    aoc_and_loss = []
    subtitles = []
    filling = []
    n_cols = 5

    # step n_cols
    for i in range(0, len(unique_problems), n_cols):
        aoc_and_loss.extend(problem_log_aoc[i:i+n_cols])
        subtitles.extend([f"F{problem}-AOC" for problem in unique_problems[i:i+n_cols]])
        filling.extend(problem_log_aoc_filling[i:i+n_cols])
        
        aoc_and_loss.extend(problem_loss[i:i+n_cols])
        subtitles.extend([f"F{problem}-Loss" for problem in unique_problems[i:i+n_cols]])
        filling.extend(problem_loss_filling[i:i+n_cols])

    # for i, problem in enumerate(unique_problems):
    #     aoc_and_loss.append(problem_log_aoc[i])
    #     subtitles.append(f"F{problem}-AOC")
    #     filling.append(problem_log_aoc_filling[i])
        
    #     aoc_and_loss.append(problem_loss[i])
    #     subtitles.append(f"F{problem}-Loss")
    #     filling.append(problem_loss_filling[i])

    
    labels = labels * 2

    plot_y = np.array(aoc_and_loss)

    x_base = np.arange(len(problem_log_aoc[0][0]), dtype=np.int16)
    x = np.tile(x_base, (len(plot_y), 1))

    # plot_lines(
    #     y = plot_y,
    #     x = x,
    #     labels = labels,
    #     filling=filling,
    #     sub_titles=subtitles, 
    #     n_cols=5,
    #     figsize=(15, 9)
    #     )

def _plot_search_token_usage(results:list[tuple[str,Population]], unique_strategies:list[str]):
    column_names = [
    'strategy',
    'n_gen',
    'n_repeat',
    'n_iter',
    'query_time',
    'prompt_token_count',
    'response_token_count',
    'total_token_count',
    ]
    _token_df = pd.DataFrame(columns=column_names)
    _strategy_count = {}
    for strategy_name, pop in results:
        if strategy_name not in _strategy_count:
            _strategy_count[strategy_name] = 0
        _strategy_count[strategy_name] += 1

        n_generation = pop.get_current_generation()
        n_iter = 0
        for gen in range(n_generation):
            # offspring generated in this generation
            gen_offsprings = pop.get_offsprings(generation=gen)
            n_iter += len(gen_offsprings)
            # offspring selected in this generation
            for ind in gen_offsprings:
                handler = Population.get_handler_from_individual(ind)
                res = {
                    'strategy': strategy_name,
                    'n_gen': gen+1,
                    'n_iter': n_iter,
                    'n_repeat': _strategy_count[strategy_name],
                    'query_time': handler.query_time,
                    'prompt_token_count': handler.prompt_token_count,
                    'response_token_count': handler.response_token_count,
                    'total_token_count': handler.prompt_token_count + handler.response_token_count,
                }
                _token_df.loc[len(_token_df)] = res

        _all_token_df = _token_df.groupby(['strategy', 'n_repeat'])[['total_token_count', 'prompt_token_count', 'response_token_count', 'query_time']].agg(np.sum).reset_index()

        y_total_token_count = []
        y_prompt_token_count = []
        y_response_token_count = []
        y_query_time = []

        for strategy in unique_strategies:
            _strategy_error_df = _all_token_df[_all_token_df['strategy'] == strategy]
            _total_token_count = _strategy_error_df['total_token_count'].to_list()
            y_total_token_count.append(_total_token_count)

            _prompt_token_count = _strategy_error_df['prompt_token_count'].to_list()
            y_prompt_token_count.append(_prompt_token_count)

            _response_token_count = _strategy_error_df['response_token_count'].to_list()
            y_response_token_count.append(_response_token_count)

            _mean_query_time = np.mean(_strategy_error_df['query_time'].to_list())
            y_query_time.append(_mean_query_time)


        plot_y = [y_total_token_count, y_prompt_token_count, y_response_token_count]
        labels = [unique_strategies, unique_strategies, unique_strategies]
        sub_titles = ["Total token count", "Prompt token count", "Response token count"]

        # plot_box_violin(
        #     data=plot_y,
        #     labels=labels,
        #     sub_titles=sub_titles,
        #     plot_type="violin",
        #     n_cols=4,
        #     label_fontsize=10,
        #     title="Token usage",
        #     figsize=(15, 9),
        #     )

        prices = {
            'o3-mini': (1.1, 4.4),
            'GPT-4o': (2.5, 10.0),
            'Claude-3.5': (3.0, 15.0),
            'DeepSeek-R1': (0.8, 2.4),
            'Gemini-Flash-2.0': (0.1, 0.4),
        }

        mean_prompt_token_count = np.mean(y_prompt_token_count, axis=1)
        mean_response_token_count = np.mean(y_response_token_count, axis=1)

        _group = []
        _group_name = []
        _labels = list(prices.keys())
        for i, strategy in enumerate(unique_strategies):
            _prompt_count = mean_prompt_token_count[i] / 1000000
            _response_count = mean_response_token_count[i] / 1000000

            _sub_group = []
            for _modle in _labels:
                _price = prices[_modle]
                _prompt_price = _price[0] * _prompt_count
                _response_price = _price[1] * _response_count
                _sub_group.append([_prompt_price, _response_price, _prompt_price + _response_price])

            _group.append(_sub_group) 
            _group_name.append(strategy) 

        _numpy_group = np.array(_group)
        y_data = [_numpy_group[:,:,i] for i in range(_numpy_group.shape[2])]
        x_labels = [_labels] * len(y_data)
        y_labels = ['Price($)'] * len(y_data)
        group_labels = [_group_name] * len(y_data)
        sub_titles = ['Prompt', 'Response', 'Total']
        plot_group_bars(y_data,
                   x_labels,
                   y_label=y_labels,
                   group_labels=group_labels,
                   sub_titles=sub_titles,
                   n_cols=3,
                   title="Token usage",
                   fig_size=(15,9))




def plot_search_result(results:list[tuple[str,Population]], save=False, file_name=None):
    res_df = _process_search_result(results, save=save, file_name=file_name)
    
    unique_strategies = res_df['strategy'].unique()
    unique_strategies = sorted(unique_strategies, key=cmp_to_key(compare_expressions))

    _plot_search_token_usage(results, unique_strategies)

    # _plot_search_aoc(res_df, unique_strategies)
    # _plot_search_group_aoc(res_df, unique_strategies)

    # _plot_serach_pop_similarity(results)
    
    # err_df = _process_error_data(results)
    # _plot_search_all_error_rate(err_df, unique_strategies)
    # _plot_search_error_type(err_df)
    # _plot_search_error_rate_by_generation(err_df, unique_strategies)
    
    # _plot_search_problem_aoc_and_loss(res_df)
    
# plot_algo_result
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


def plot_search():
    # file_paths = [
    #     # ("logs_bbob/bbob_exp_gemini-2.0-flash-exp_0121222958.pkl", "bo"),
    #     ("logs_bbob/bbob_exp_gemini-2.0-flash-exp_0124195614.pkl", "es-1+1"),
    #     ("logs_bbob/bbob_exp_gemini-2.0-flash-exp_0124195643.pkl", "es-1+1"),
    # ]
    # strategy_list = []
    # for file_path, name in file_paths:
    #     ind_logger = IndividualLogger.load(file_path)
    #     ind_ids = list(ind_logger.experiment_map.values())[0]["id_list"]
    #     inds = [ind_logger.get_individual(ind_id) for ind_id in ind_ids]
    #     res_list = [ind.metadata["res_handler"].eval_result for ind in inds]
    #     strategy_list.append((name, res_list))

    # 1+1
    file_paths_1_1 = [
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BaselinePromptGenerator_0127065904.pkl', "EA-1+1"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BaselinePromptGenerator_0127071455.pkl', "EA-1+1"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0127100500.pkl', "BO-1+1-Sklearn"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0127230651.pkl', "BO-1+1-Sklearn"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127075128.pkl', "BO-1+1-GPytorch"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127230526.pkl', "BO-1+1-GPytorch"),
    ]

    # 2+1
    file_paths_2_1 = [
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BaselinePromptGenerator_0127064159.pkl', "EA-2+1"),
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BaselinePromptGenerator_0127064736.pkl', "EA-2+1"),
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127120051.pkl', "BO-2+1-GPytorch"),
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0127155755.pkl', "BO-2+1-Sklearn"),
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0128105531.pkl', "BO-2+1-GPytorch"),
    ]
    
    # 2+1_warmstart_diversity
    file_paths_2_1_wd = [
        ('population_logs/ESPopulation_bbob_2+1+warm+diverse_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127092335.pkl', "BO-2+1-wd-GPytorch"),
        ('population_logs/ESPopulation_bbob_2+1+warm+diverse_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127203251.pkl', "BO-2+1-wd-GPytorch"),
        ('population_logs/ESPopulation_bbob_2+1+warmstart+diversity_gemini-2.0-flash-exp_BaselinePromptGenerator_0127073309.pkl', "EA-2+1-wd"),
        ('population_logs/ESPopulation_bbob_2+1+warmstart+diversity_gemini-2.0-flash-exp_BaselinePromptGenerator_0127074034.pkl', "EA-2+1-wd"),
        ('population_logs/ESPopulation_bbob_2+1+warmstart+diversity_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0127234617.pkl', "BO-2+1-wd-Sklearn"),
        ('population_logs/ESPopulation_bbob_2+1+warmstart+diversity_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0128002616.pkl', "BO-2+1-wd-Sklearn"),
    ]

    
    # island
    file_paths_island = [
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BaselinePromptGenerator_0127073608.pkl', "EA-Island"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BaselinePromptGenerator_0127081002.pkl', "EA-Island"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0128022539.pkl', "BO-Island-Sklearn"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127085652.pkl', "BO-Island-GPytorch"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127103428.pkl', "BO-Island-GPytorch"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127172700.pkl', "BO-Island-GPytorch"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0128071243.pkl', "BO-Island-GPytorch"),
    ]


    all_file_paths = file_paths_1_1 + file_paths_2_1_wd + file_paths_island + file_paths_2_1

    file_paths_ea = []
    file_paths_bo_sklearn = []
    file_paths_bo_torch = []
    for file_path, name in all_file_paths:
        if "EA" in name:
            file_paths_ea.append((file_path, name))
        elif "Sklearn" in name:
            file_paths_bo_sklearn.append((file_path, name))
        elif "GPytorch" in name:
            file_paths_bo_torch.append((file_path, name))

    file_paths = file_paths_bo_torch
    
    strategy_list = []
    ea_offspring = []
    sklearn_offspring = []
    torch_offspring = []
    for file_path, name in file_paths:
        pop = ESPopulation.load(file_path)
        strategy_list.append((name, pop))
        if "EA" in name:
            ea_offspring.extend(pop.all_individuals())
        elif "Sklearn" in name:
            sklearn_offspring.extend(pop.all_individuals())
        elif "GPytorch" in name:
            torch_offspring.extend(pop.all_individuals())

    sorted_ea_offspring = sorted(ea_offspring, key=lambda x: x.fitness, reverse=True)
    sorted_sklearn_offspring = sorted(sklearn_offspring, key=lambda x: x.fitness, reverse=True)
    sorted_torch_offspring = sorted(torch_offspring, key=lambda x: x.fitness, reverse=True)

    
    ind_logger = IndividualLogger()
    for ind in sorted_torch_offspring:
        ind_logger.log_individual(ind)
    ind_logger.save_reader_format()

    # plot_results(results=strategy_list, other_results=None)

def plot_light_evol_and_final():
    file_paths = [

        # 0.04
        ('NoisyBanditBOv1', [
            'Experiments/final_eval_res/NoisyBanditBOv1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0210230205.pkl', 
            'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065417/0-10_NoisyBanditBOv1_handler.pkl'
                             ]),

        ('ParetoActiveBOv1', [
            'Experiments/final_eval_res/ParetoActiveBOv1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211000723.pkl', 
            'Experiments/pop_40_f/ESPopulation_evol_4+2_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210025714/0-3_ParetoActiveBOv1_handler.pkl'
            ]),

        # 0.05
        ('AdaptiveBatchUCBLocalSearchBOv2', [
            'Experiments/final_eval_res/AdaptiveBatchUCBLocalSearchBOv2_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211005941.pkl', 
            'Experiments/pop_40_f/ESPopulation_evol_4+4_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210043822/4-20_AdaptiveBatchUCBLocalSearchBOv2_handler.pkl'
            ]),

        ('AdaptiveControlVariateBOv4', [
            'Experiments/final_eval_res/AdaptiveControlVariateBOv4_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211014052.pkl', 
            'Experiments/pop_40_f/ESPopulation_evol_8+4_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209152106/6-29_AdaptiveControlVariateBOv4_handler.pkl'
            ]),

        # 0.06
        ('AdaptiveEvoBatchHybridBOv2', [
            'Experiments/pop_40_f/ESPopulation_evol_12+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208224540/4-32_AdaptiveEvoBatchHybridBOv2_handler.pkl',
            'Experiments/final_eval_res/AdaptiveEvoBatchHybridBOv2_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211020052.pkl'
        ]),
        
        ('MultiObjectiveBOv1', [
            'Experiments/final_eval_res/MultiObjectiveBOv1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211024828.pkl',
            'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065417/0-19_MultiObjectiveBOv1_handler.pkl'
        ]),

        ('AdaptiveTrustRegionDynamicAllocationBOv2:', [
            'Experiments/pop_40_f/ESPopulation_evol_8+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065623/2-22_AdaptiveTrustRegionDynamicAllocationBOv2_handler.pkl',
            'Experiments/final_eval_res/AdaptiveTrustRegionDynamicAllocationBOv2_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211043407.pkl'
        ]),

        ('AdaptiveHybridBOv6', [
            'Experiments/pop_40_f/ESPopulation_evol_1+1_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0211000947/10-11_AdaptiveHybridBOv6_handler.pkl',
            'Experiments/final_eval_res/AdaptiveHybridBOv6_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211045527.pkl'
        ]),

        # 0.08
        ('AdaptiveTrustImputationBOv2:', [
            'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065238/2-31_AdaptiveTrustImputationBOv2_handler.pkl',
            'Experiments/final_eval_res/AdaptiveTrustImputationBOv2_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211002041.pkl'
        ]),
            
        ('TrustRegionAdaptiveTempBOv2', [
            'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209173952/4-23_TrustRegionAdaptiveTempBOv2_handler.pkl',
            'Experiments/final_eval_res/TrustRegionAdaptiveTempBOv2_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211000039.pkl'
        ]),

        ('BayesLocalAdaptiveAnnealBOv1', [
            'Experiments/pop_40_temp/ESPopulation_evol_10+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208164605/3-24_BayesLocalAdaptiveAnnealBOv1_handler.pkl',
            'Experiments/final_eval_res/BayesLocalAdaptiveAnnealBOv1_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211012527.pkl'
    ])
    ]

    groups = []
    group_names = []
    for group_name, file_paths in file_paths:
        _temp = [0,0]
        for file_path in file_paths:
            target = pickle.load(open(file_path, "rb"))
            if isinstance(target, EvaluatorResult):
                _temp[1] = target.score
            elif isinstance(target, ResponseHandler):
                _temp[0] = target.eval_result.score
        group_names.append(group_name)
        groups.append(_temp)
    
    y_data = [np.array(groups)]
    x_labels = [['Partial Eval', 'ALL Eval']]

    plot_group_bars(y_data, 
                   x_labels, 
                   group_names, 
                   title='Comparison of Partial and All Evaluations',
                   fig_size=(15,9))


def plot_search_0209():
    file_paths = [
        # "Experiments/pop_40_f/ESPopulation_evol_1+1_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210035334/ESPopulation_final_0210065735.pkl",
        # 'Experiments/pop_40_temp/ESPopulation_evol_10+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208164605/ESPopulation_final_0208204450.pkl',

        # 'Experiments/pop_40_test/ESPopulation_evol_3+5_IOHEvaluator_f4_dim-5_budget-100_instances-[1]_repeat-1_0214010030/ESPopulation_final_0214010239.pkl',
    ]
    file_name = None
    if len(file_paths) == 0:
        dir_path = 'Experiments/pop_40_f'
        for dir_name in os.listdir(dir_path):
            if not os.path.isdir(os.path.join(dir_path, dir_name)):
                continue
            for file_name in os.listdir(os.path.join(dir_path, dir_name)):
                if "final" not in file_name:
                    continue
                file_path = os.path.join(dir_path, dir_name, file_name)
                file_paths.append(file_path)

        file_name = dir_path + '/' + f'df_res_{datetime.now().strftime("%m%d%H%M")}.pkl' 
    
    pop_list = []
    best_pop_map = {}
    for file_path in file_paths:
        pop = pickle.load(open(file_path, "rb"))

        n_parent = pop.n_parent
        n_offspring = pop.n_offspring
        name = f"{n_parent}+{n_offspring}"

        pop_list.append((name, pop))

        cur_best = best_pop_map.get(name, None)
        if cur_best is None or pop.get_best_of_all().fitness > cur_best.get_best_of_all().fitness:
            best_pop_map[name] = pop
    
    # save = True
    save = False

    # file_name = 'Experiments/pop_40_f/df_res_02110305.pkl'
    file_name = None
    
    plot_search_result(pop_list, save=save, file_name=file_name)


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
    

if __name__ == "__main__":
    # plot_search()

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

    # plot_algo(file_paths=file_paths, dir_path=dir_path, pop_path=pop_path)

    # plot_light_evol_and_final()

    # plot_search_0209()

    pass
