import pickle
import os
import logging
from functools import cmp_to_key
from datetime import datetime
import re
import concurrent.futures
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from llamevol.utils import IndividualLogger
from llamevol.prompt_generators.abstract_prompt_generator import ResponseHandler
from llamevol.utils import plot_group_bars, plot_lines, plot_box_violin, moving_average, savgol_smoothing, gaussian_smoothing, plot_voilin_style_scatter
from llamevol.population.population import Population, desc_similarity, code_diff_similarity, code_bert_similarity
from llamevol.population.es_population import ESPopulation
from llamevol.evaluator.evaluator_result import EvaluatorResult
from llamevol.utils import setup_logger, RenameUnpickler


# utils

def combine_acc(column='y_aoc', maximum=True, max_n_iter=None, iter_column='n_iter'):
    def _inner_combine_acc(df_series):
        _n_iters = df_series[iter_column].copy()
        _contents = []
        _n_iters.sort()
        _aoc = df_series[column].copy()

        replacement = 0
        if not maximum:
            _max = np.max(_aoc)
            replacement = _max

        if max_n_iter is not None and max_n_iter > _n_iters[-1]:
            _n_iters.append(max_n_iter)
            _aoc.append(replacement)

        for i, _n_iter in enumerate(_n_iters):
            n_fill = _n_iter - len(_contents) - 1
            _contents.extend([replacement] * n_fill)
            _contents.append(_aoc[i])
        if maximum:
            _acc = np.maximum.accumulate(_contents)
        else:
            _acc = np.minimum.accumulate(_contents)
        return _acc
    return _inner_combine_acc

def compare_expressions(expr1, expr2):
    def _split_expr(expr):
        if '+' in expr: 
            a, b = map(int, expr.split('+'))
        elif ',' in expr:
            a, b = map(int, expr.split(','))
        else:
            return expr, None
        return a, b
    
    def _cmp(a, b):
        if a is None and b is None:
            return 0
        if a is None:
            return -1
        if b is None:
            return 1
        if a == b:
            return 0
        return 1 if a > b else -1

    # 1. split the expression by '_'
    # 2. if sub-expression has '+' or ',', split it by '+' or ','
    # 3. compare all the sub-expression iteratively
    # 3.1 if sub-expressions are equal, continue to compare the next sub-expression
    # 3.2 if sub-expressions are not equal, return -1, 0, 1
    expr1_split = expr1.split('_')
    expr2_split = expr2.split('_')
    # fill the shorter one with None
    if len(expr1_split) < len(expr2_split):
        expr1_split.extend([None] * (len(expr2_split) - len(expr1_split)))
    elif len(expr1_split) > len(expr2_split):
        expr2_split.extend([None] * (len(expr1_split) - len(expr2_split)))

    for sub_expr1, sub_expr2 in zip(expr1_split, expr2_split):
        if sub_expr1 is None or sub_expr2 is None:
            return _cmp(sub_expr1, sub_expr2)

        a1, b1 = _split_expr(sub_expr1)
        a2, b2 = _split_expr(sub_expr2)
        if b1 is None or b2 is None:
            if a1 == a2:
                continue
            else:
                return _cmp(a1, a2)
        else:
            if a1 == a2 and b1 == b2:
                continue
            else:
                if a1 == a2:
                    return _cmp(b1, b2)
                else:
                    return _cmp(a1, a2)
    return 0

# plot_search_result
def _process_search_result(results:list[tuple[str,Population]], save_name=None):
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
    _save = False
    _load = False
    if save_name is not None: 
        if os.path.exists(save_name):
            _load = True
        else:
            _save = True
    if _load:
        res_df = pd.read_pickle(save_name)
    else: 
        _strategy_count = {}
        res_df = pd.DataFrame(columns=column_names)
        for strategy_name, pop in results:
            if strategy_name not in _strategy_count:
                _strategy_count[strategy_name] = 0
            _strategy_count[strategy_name] += 1
            n_generation = pop.get_current_generation()
            if n_generation == 0:
                n_generation = 1
            n_iter = 1
            n_ind = 0
            for gen in range(n_generation):
                # offspring generated in this generation
                gen_offsprings = pop.get_offsprings(generation=gen)
                n_iter += len(gen_offsprings)
                # offspring selected in this generation
                # gen_inds = pop.get_individuals(generation=gen)
                gen_inds = gen_offsprings
                for i, ind in enumerate(gen_inds):
                    handler = Population.get_handler_from_individual(ind)
                    n_ind += 1
                    _count = _strategy_count[strategy_name]
                    if handler.eval_result is None:
                        continue
                    for res in handler.eval_result.result:
                        res.update_aoc_with_new_bound_if_needed()
                        row = res_to_row(res, gen, strategy_name=strategy_name, n_iter=n_iter, n_ind=n_ind, n_strategy=_count)
                        res_df.loc[len(res_df)] = row

    if _save: 
        res_df.to_pickle(save_name)
    return res_df

def _plot_search_aoc(res_df:pd.DataFrame, unique_strategies:list[str], fig_dir=None):
    max_aoc_df = res_df.groupby(['strategy', 'n_strategy', 'n_ind'])[["log_y_aoc", "y_aoc"]].agg(np.mean).reset_index()
    max_aoc_df = max_aoc_df.groupby(['strategy', 'n_strategy'])[["log_y_aoc", "y_aoc"]].agg(np.max).reset_index()
    max_aoc_df = max_aoc_df.groupby(['strategy'])[['log_y_aoc', 'y_aoc']].agg(list).reset_index()

    _volin_y = []
    for strategy in unique_strategies:
        strategy_df = max_aoc_df[max_aoc_df['strategy'] == strategy]
        _max_aoc_list = strategy_df['log_y_aoc'].values[0]
        _volin_y.append(np.array(_max_aoc_list))

    # plot_voilin_style_scatter(
    #     data=[_volin_y],
    #     labels=[unique_strategies],
    #     n_cols=4,
    #     title="AOC",
    #     label_fontsize=10,
    #     figsize=(14, 8),
    #     )

    file_name = 'search_aoc_voilin'
    if fig_dir is not None:
        file_name = os.path.join(fig_dir, file_name)

    plot_box_violin(
        data=[_volin_y],
        labels=[unique_strategies],
        y_labels=['AOCC'],
        y_tick_fontsize=12,
        x_tick_fontsize=13,
        show_scatter=True,
        width=0.6,
        n_cols=4,
        label_fontsize=11,
        figsize=(8, 4),
        filename=file_name,
        show=False
        )

def _group_plus_aoc(strategy_group:dict, strategy:str, y_aoc:np.ndarray, log_y_aoc:np.ndarray, std_y_aoc:np.ndarray, std_log_y_aoc:np.ndarray):
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
                strategy_group[_key]['name'] = gruoup_name_map[_key]
            
            strategy_group[_key]['aoc'].append(y_aoc)
            strategy_group[_key]['aoc_filling'].append((y_aoc + std_y_aoc, y_aoc - std_y_aoc))
            strategy_group[_key]['log_aoc'].append(log_y_aoc)
            strategy_group[_key]['log_aoc_filling'].append((log_y_aoc + std_log_y_aoc, log_y_aoc - std_log_y_aoc))
            strategy_group[_key]['labels'].append(strategy)
        
def _plot_search_group_aoc(res_df:pd.DataFrame, unique_strategies:list[str], group_fn=None, fig_dir=None):
    max_n_iter = res_df['n_iter'].max()
    aoc_df = res_df.groupby(['strategy', 'n_strategy', 'n_iter', 'n_ind'])[["log_y_aoc", "y_aoc"]].agg(np.mean).reset_index()

    # aoc_df = aoc_df.groupby(['strategy', 'n_strategy', 'n_iter'])[["log_y_aoc", "y_aoc"]].agg(np.max).reset_index()
    # iter_column = 'n_iter'

    aoc_df = aoc_df.groupby(['strategy', 'n_strategy',])[['n_ind',"log_y_aoc", "y_aoc"]].agg(list).reset_index()
    iter_column = 'n_ind'

    aoc_df['acc_y_aoc'] = aoc_df.apply(combine_acc('y_aoc', max_n_iter=max_n_iter, iter_column=iter_column), axis=1)
    aoc_df['acc_log_y_aoc'] = aoc_df.apply(combine_acc('log_y_aoc', max_n_iter=max_n_iter, iter_column=iter_column), axis=1)

    aoc_df = aoc_df.groupby(['strategy'])[['acc_y_aoc', 'acc_log_y_aoc']].agg(list).reset_index()

    strategy_group = {}

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
        strategy_filling.append((y_aoc + std_y_aoc, np.clip(y_aoc - std_y_aoc, 0, None)))

        acc_log_y_aoc = np.array(strategy_df['acc_log_y_aoc'].values[0])
        log_y_aoc = np.mean(acc_log_y_aoc, axis=0)
        std_log_y_aoc = np.std(acc_log_y_aoc, axis=0)
        strategy_log_aoc.append(log_y_aoc)
        strategy_log_filling.append((log_y_aoc + std_log_y_aoc, np.clip(log_y_aoc - std_log_y_aoc, 0, None)))

        labels.append(strategy)

        if group_fn is not None:
            group_fn(strategy_group, strategy, y_aoc, log_y_aoc, std_y_aoc, std_log_y_aoc)

    plot_y = []
    sub_titles = []
    fillings = []
    plot_labels = []
    y_scale = []

    if len(strategy_group) > 0:
        for group_key, group_ele in strategy_group.items():
            # plot_y.append(np.array(group_ele['aoc']))
            # fillings.append(group_ele['aoc_filling'])

            plot_y.append(np.array(group_ele['log_aoc']))
            fillings.append(group_ele['log_aoc_filling'])
            sub_titles.append(group_ele['name'])
            plot_labels.append(group_ele['labels'])
    else:
        plot_y = [np.array(strategy_log_aoc)]
        fillings = [strategy_log_filling]
        plot_labels = [labels]
        # y_scale = [("symlog", {})]

    file_name = 'es_aoc_lines'
    if fig_dir is not None:
        file_name = os.path.join(fig_dir, file_name)

    x_base = np.arange(len(strategy_aoc[0]), dtype=np.int16)
    x = np.tile(x_base, (len(plot_y), 1))

    # for main content
    _label_fontsize = 13
    _tick_fontsize = 13
    _y_label_fontsize = 12
    _line_width = 1.4
    _figsize = (5, 4)

    # for sub content
    _label_fontsize = 13
    _tick_fontsize = 13
    _y_label_fontsize = 13
    _line_width = 1.5
    _figsize = (8, 4)

    plot_lines(
        y = plot_y,
        x = x,
        labels = plot_labels,
        label_fontsize=_label_fontsize,
        tick_fontsize=_tick_fontsize,
        y_labels=['AOCC'],
        y_label_fontsize=_y_label_fontsize,
        filling=fillings,
        # sub_titles=sub_titles,
        y_scales=y_scale,
        linewidth=_line_width,
        n_cols=3,
        figsize=_figsize,
        filename=file_name,
        show=False,
        )


def _calculate_pop_sim(pop:Population, iter_sim_func, total_sim_func):
    iter_sim = []
    pop_sim = 0
    n_iter = 0
    n_generation = pop.get_current_generation()
    if pop.n_offspring > 1:
        for gen in range(n_generation):
            gen_offsprings = pop.get_offsprings(generation=gen)
            n_iter += len(gen_offsprings)

            n_fill = n_iter - len(iter_sim)
            mean_sim, _ = iter_sim_func(gen_offsprings)
            _sim = np.mean(mean_sim)
            iter_sim.extend([_sim] * n_fill)

    all_inds = pop.all_individuals()
    all_mean_sim, _ = total_sim_func(all_inds)
    pop_sim = np.mean(all_mean_sim)

    return iter_sim, pop_sim

def _calculate_strategy_similarity(pop_list:list[Population], max_workers=8, cal_desc_sim=True, cal_code_sim=False):

    code_sim = None
    if cal_code_sim:
        iter_sim_list = []
        pop_sim_list = []

        for pop in pop_list:
            iter_sim, pop_sim = _calculate_pop_sim(pop, code_bert_similarity, code_bert_similarity)
            if len(iter_sim) > 0:
                iter_sim_list.append(iter_sim)
            pop_sim_list.append(pop_sim)
        iter_sim_list = None if len(iter_sim_list) == 0 else iter_sim_list
        code_sim = (iter_sim_list, pop_sim_list)
    
    desc_sim = None
    if cal_desc_sim:
        iter_sim_list = []
        pop_sim_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for pop in pop_list:
                futures.append(executor.submit(_calculate_pop_sim, pop, desc_similarity, desc_similarity))
            for future in concurrent.futures.as_completed(futures):
                iter_sim, pop_sim = future.result()
                if len(iter_sim) > 0:
                    iter_sim_list.append(iter_sim)
                pop_sim_list.append(pop_sim)

        # for pop in pop_list:
        #     iter_sim, pop_sim = _calculate_pop_sim(pop, desc_similarity, desc_similarity)
        #     iter_sim_list.append(iter_sim)
        #     pop_sim_list.append(pop_sim)
        iter_sim_list = None if len(iter_sim_list) == 0 else iter_sim_list
        desc_sim = (iter_sim_list, pop_sim_list)
    
    return {'Desc Similarity': desc_sim, 'Code Similarity': code_sim} 

def _plot_serach_pop_similarity(results:list[tuple[str,Population]], unique_strategies:list[str], save_name=None):
    _save_name = save_name
    if _save_name is not None:
        _save_name = f"{_save_name}_pop_sim"
    _save = False
    _load = False
    if _save_name is not None:
        if os.path.exists(_save_name):
            _load = True
        else:
            _save = True
    
    if _load:
        with open(_save_name, 'rb') as f:
            sim_data = RenameUnpickler(f).unpickle()
    else:
        strategy_group = {}
        for strategy_name, pop in results:
            if strategy_name not in strategy_group:
                strategy_group[strategy_name] = []
            strategy_group[strategy_name].append(pop)

        sim_data = {}
        count = 0
        for strategy_name in unique_strategies:
            group = strategy_group[strategy_name]
            sims = _calculate_strategy_similarity(group, 
                                                  max_workers=6, 
                                                  cal_desc_sim=True, 
                                                  cal_code_sim=False
                                                  )

            for _key, _sim in sims.items():
                if _key not in sim_data:
                    sim_data[_key] = {
                        'y_sim_list': [],
                        'y_sim_filling': [],
                        'y_sim_labels': [],

                        'y_pop_sim_list': [],
                        'y_pop_sim_labels': [],
                    }

                _sim_data = sim_data[_key]
                iter_sim_list, pop_sim_list = _sim

                if iter_sim_list is not None:
                    mean_sim = np.mean(iter_sim_list, axis=0)
                    std_sim = np.std(iter_sim_list, axis=0)
                    _sim_data['y_sim_list'].append(mean_sim)
                    _sim_data['y_sim_filling'].append((mean_sim + std_sim, mean_sim - std_sim))
                    _sim_data['y_sim_labels'].append(strategy_name)

                if pop_sim_list is not None:
                    _sim_data['y_pop_sim_list'].append(pop_sim_list)
                    _sim_data['y_pop_sim_labels'].append(strategy_name)

            count += 1
            
        if _save and _save_name is not None:
            with open(_save_name, 'wb') as f:
                pickle.dump(sim_data, f)

    y_pop_sim_list = []
    y_pop_sim_labels = []
    pop_sub_titles = []
    for _key, _sim in sim_data.items():
        y_pop_sim_list.append(_sim['y_pop_sim_list'])
        y_pop_sim_labels.append(_sim['y_pop_sim_labels'])
        pop_sub_titles.append(_key)

    plot_box_violin(
        data=y_pop_sim_list,
        labels=y_pop_sim_labels,
        sub_titles=pop_sub_titles,
        n_cols=4,
        title="Population similarity",
        label_fontsize=10,
        figsize=(12, 7),
        )

    y_sim_list = []
    y_sim_filling = []
    y_sim_labels = []
    pop_sub_titles = []
    for _key, _sim in sim_data.items():
        y_sim_list.append(np.array(_sim['y_sim_list']))
        y_sim_filling.append(np.array(_sim['y_sim_filling']))
        y_sim_labels.append(_sim['y_sim_labels'])
        pop_sub_titles.append(_key)

    x_base = np.arange(y_sim_list[0].shape[1], dtype=np.int16)
    x = np.tile(x_base, (len(y_sim_list), 1))
    plot_lines(
        y = y_sim_list,
        x = x,
        labels = y_sim_labels,
        filling=y_sim_filling, 
        sub_titles=pop_sub_titles,
        label_fontsize=10,
        linewidth=1.5,
        figsize=(12, 9),
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

def _plot_search_all_error_rate(err_df:pd.DataFrame, unique_strategies:list[str], fig_dir=None):
    _all_error_df = err_df.groupby(['strategy', 'n_repeat'])['err_type'].agg(list).reset_index()
    _all_error_df['err_rate'] = _all_error_df['err_type'].apply(lambda x: len([ele for ele in x if ele is not None]) / len(x))

    y_err_rates = []

    for strategy in unique_strategies:
        _strategy_error_df = _all_error_df[_all_error_df['strategy'] == strategy]
        _error_rate = _strategy_error_df['err_rate'].to_list()
        if len(_error_rate) == 0:
            _error_rate = [0]
        y_err_rates.append(_error_rate)

    file_name = 'overall_error_rate_voilin'
    if fig_dir is not None:
        file_name = os.path.join(fig_dir, file_name)

    plot_box_violin(
        data=[y_err_rates],
        labels=[unique_strategies],
        y_labels=['Error Rate'],
        y_tick_fontsize=12,
        x_tick_fontsize=13,
        show_scatter=True,
        n_cols=4,
        width=0.6,
        label_fontsize=11,
        # title="Error rate by strategy",
        figsize=(8, 4),
        filename=file_name,
        show=False,
        ) 

def _plot_search_error_type(error_df:pd.DataFrame, unique_strategies:list[str]):

    _filter_types = [
        'Timeout', 
        'BOOverBudgetException', 
    ]

    _err_df = error_df[~error_df['err_type'].isin(_filter_types)]

    titles = []
    plot_data = []

    for strategy in unique_strategies:
        _all_strategy_error_df = error_df[error_df['strategy'] == strategy]
        _timeout_count = _all_strategy_error_df[_all_strategy_error_df['err_type'] == 'Timeout'].shape[0]
        _bo_count = _all_strategy_error_df[_all_strategy_error_df['err_type'] == 'BOOverBudgetException'].shape[0]
        _all_size = _all_strategy_error_df.shape[0]

        _strategy_error_df = _err_df[_err_df['strategy'] == strategy]
        _size = _strategy_error_df.shape[0]
        type_count = _strategy_error_df['err_type'].value_counts()
        _all_type_count = type_count.sum()

        # sum types less than 0.01 into others
        _threshold = 0.01
        _other_count = 0
        for _type, _count in type_count.items():
            if _count / _all_type_count < _threshold:
                _other_count += _count
        type_count = type_count[type_count / _all_type_count >= _threshold]
        type_count['others'] = _other_count

        _title = f'{strategy}\n Timeout:{_timeout_count} \n BOOverBudgetException:{_bo_count} \n rest:{_all_type_count} \n total:{_all_size}'
        _plot_data = type_count
        titles.append(_title)
        plot_data.append(_plot_data)

    for i, _plot_data in enumerate(plot_data):
        fig, ax = plt.subplots(figsize=(12, 8))
        _title = titles[i]
        _plot_data = plot_data[i]
        ax.pie(_plot_data, 
                labels=_plot_data.index, 
                autopct='%1.1f%%',
            #    autopct=lambda p: '{:d}'.format(int(p / 100 * _all_type_count)),
                )
        fig.suptitle(_title)


def _group_plus_error_rate(strategy_group:dict, strategy:str, mean_err_rate:float, std_err_rate:float):
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

    mu, lam = strategy.split('+') 
    int_mu, int_lam = int(mu), int(lam)

    if int_mu == 1:
        return

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
        
        strategy_group[_key]['err_rate'].append(mean_err_rate)
        if int_mu == 1:
            strategy_group[_key]['err_rate_filling'].append((mean_err_rate, mean_err_rate))
        else:
            strategy_group[_key]['err_rate_filling'].append((mean_err_rate + std_err_rate, mean_err_rate - std_err_rate))
        strategy_group[_key]['labels'].append(strategy)
    

def _plot_search_error_rate_by_generation(err_df:pd.DataFrame, unique_strategies:list[str], group_fn=None):
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
    y_err_rates = []
    y_err_rates_filling = []
    labels = []
    for strategy in unique_strategies:
        if strategy == '1+1':
            continue
        
        _strategy_error_df = _gen_error_df[_gen_error_df['strategy'] == strategy]

        _evol_err_rate = _strategy_error_df['evol_err_rate'].to_list()
        _mean_err_rate = np.mean(_evol_err_rate, axis=0)
        _std_err_rate = np.std(_evol_err_rate, axis=0)

        y_err_rates.append(_mean_err_rate)
        y_err_rates_filling.append((_mean_err_rate + _std_err_rate, _mean_err_rate - _std_err_rate))

        labels.append(strategy)

        if group_fn is not None:
            group_fn(strategy_group, strategy, _mean_err_rate, _std_err_rate)

    plot_y = []
    sub_titles = []
    fillings = []
    plot_labels = []
    if len(strategy_group) > 0:
        for group_key, group_ele in strategy_group.items():
            plot_y.append(np.array(group_ele['err_rate']))
            fillings.append(group_ele['err_rate_filling'])
            sub_titles.append(group_ele['name'])
            plot_labels.append(group_ele['labels']) 
    else:
        plot_y = [np.array(y_err_rates)]
        plot_labels = [labels]
        fillings = [y_err_rates_filling]

    x_base = np.arange(len(y_err_rates[0]), dtype=np.int16)
    x = np.tile(x_base, (len(plot_y), 1))
    plot_lines(
        y = plot_y,
        x = x,
        labels = plot_labels,
        label_fontsize=10,
        linewidth=2,
        filling=fillings,
        sub_titles=sub_titles,
        title="Error rate by generation",
        n_cols=3,
        figsize=(15, 9),
        )  

def _group_plus_aoc_loss(strategy_group_in_problem:dict, strategy:str, y_aoc:np.ndarray, log_y_aoc:np.ndarray, std_y_aoc:np.ndarray, std_log_y_aoc:np.ndarray, loss:np.ndarray, std_loss:np.ndarray):
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
                strategy_group_in_problem[_key]['name'] = gruoup_name_map[_key]

            strategy_group_in_problem[_key]['aoc'].append(log_y_aoc)
            strategy_group_in_problem[_key]['aoc_filling'].append((log_y_aoc + std_log_y_aoc, log_y_aoc - std_log_y_aoc))
            strategy_group_in_problem[_key]['loss'].append(loss)
            strategy_group_in_problem[_key]['loss_filling'].append((loss + std_loss, loss - std_loss))
            strategy_group_in_problem[_key]['labels'].append(strategy)
    
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

def _plot_search_problem_aoc_and_loss(res_df:pd.DataFrame, group_fn=None, fig_dir=None):
    def _min_max_agg(x):
        if 'aoc' in x.name:
            return np.max(x)
        return np.min(x)

    max_n_iter = res_df['n_iter'].max()

    # generate aggregation
    # aoc_df = res_df.groupby(['strategy', 'problem_id','instance_id', 'exec_id', 'n_iter'])[["log_y_aoc", 'y_aoc', 'loss']].agg(_min_max_agg).reset_index()
    # aoc_df = aoc_df.groupby(['strategy', 'problem_id','instance_id', 'exec_id'])[['n_iter',"log_y_aoc", 'y_aoc', 'loss']].agg(list).reset_index()
    # iter_column = 'n_iter'

    # iterated version
    # aoc_df = res_df.groupby(['strategy', 'problem_id','instance_id', 'exec_id', 'n_ind'])[["log_y_aoc", 'y_aoc', 'loss']].agg(_min_max_agg).reset_index()
    # aoc_df = aoc_df.groupby(['strategy', 'problem_id','instance_id', 'exec_id'])[['n_ind',"log_y_aoc", 'y_aoc', 'loss']].agg(list).reset_index()
    # iter_column = 'n_ind'

    # repeatation version
    aoc_df = res_df.groupby(['strategy', 'problem_id', 'n_strategy','n_ind'])[["log_y_aoc", 'y_aoc', 'loss']].agg(np.mean).reset_index()
    aoc_df = aoc_df.groupby(['strategy', 'problem_id', 'n_strategy'])[['n_ind',"log_y_aoc", 'y_aoc', 'loss']].agg(list).reset_index()
    iter_column = 'n_ind'

    aoc_df['acc_y_aoc'] = aoc_df.apply(combine_acc('y_aoc', max_n_iter=max_n_iter, iter_column=iter_column), axis=1)
    aoc_df['acc_log_y_aoc'] = aoc_df.apply(combine_acc('log_y_aoc', max_n_iter=max_n_iter, iter_column=iter_column), axis=1)
    aoc_df['acc_loss'] = aoc_df.apply(combine_acc('loss', maximum=False, max_n_iter=max_n_iter, iter_column=iter_column), axis=1)

    # loss_upper_cliper = clip_upper_factory(bound_type='median')
    loss_upper_cliper = None

    problem_log_aoc = []
    problem_log_aoc_filling = []
    problem_loss = []
    problem_loss_filling = []
    labels = []

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
            acc_log_y_aoc = np.array(strategy_df['acc_log_y_aoc'].to_list())
            log_y_aoc = np.mean(acc_log_y_aoc, axis=0)
            std_log_y_aoc = np.std(acc_log_y_aoc, axis=0)
            _log_aoc.append(log_y_aoc)
            _log_aoc_filling.append((log_y_aoc + std_log_y_aoc, np.clip(log_y_aoc - std_log_y_aoc, 0, None)))

            acc_loss = np.array(strategy_df['acc_loss'].to_list())
            _upper_bound = None
            if loss_upper_cliper is not None:
                acc_loss, _upper_bound = loss_upper_cliper(acc_loss)
            loss = np.mean(acc_loss, axis=0)
            std_loss = np.std(acc_loss, axis=0)
            max_loss = np.max(acc_loss, axis=0)
            min_loss = np.min(acc_loss, axis=0)
            _loss.append(loss)
            # _loss_filling.append((np.clip(loss + std_loss, 0, _upper_bound), np.clip(loss - std_loss, 0, _upper_bound)))
            _loss_filling.append((min_loss, max_loss))

            _labels.append(strategy)

            if group_fn is not None:
                group_fn(strategy_group_in_problem, strategy, log_y_aoc, log_y_aoc, std_log_y_aoc, std_log_y_aoc, loss, std_loss)

        if len(strategy_group_in_problem) > 0:
            problem_group[problem] = strategy_group_in_problem
        problem_log_aoc.append(_log_aoc)
        problem_log_aoc_filling.append(_log_aoc_filling)
        problem_loss.append(_loss)
        problem_loss_filling.append(_loss_filling)
        labels.append(_labels)

    if len(problem_group) > 0:
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
                sub_titles.append(f"{group_ele['name']}(AOC)")
                plot_labels.append(group_ele['labels'])
                y_scale.append(("linear", {}))

                plot_y.append(np.array(group_ele['loss']))
                fillings.append(group_ele['loss_filling'])
                sub_titles.append(f"{group_ele['name']}(Loss)")
                plot_labels.append(group_ele['labels'])
                y_scale.append(("symlog", {}))

            x_base = np.arange(len(problem_log_aoc[0][0]), dtype=np.int16)
            x = np.tile(x_base, (len(plot_y), 1))
            plot_lines(
                y = plot_y,
                x = x,
                labels = plot_labels,
                filling=fillings,
                sub_titles=sub_titles,
                y_scales=y_scale,
                title=title,
                n_cols=4,
                figsize=(15, 9),
                )
    else:
        n_cols = 5
        aoc_and_loss = []
        subtitles = []
        filling = []
        y_scale = []
        y_labels = []


        aoc_and_loss.extend(problem_log_aoc)
        subtitles.extend([f"F{problem}" for problem in unique_problems])
        filling.extend(problem_log_aoc_filling)
        y_scale.extend([("linear", {}) for _ in range(len(unique_problems))])
        aoc_y_labels = ['AOCC'] + [''] * (len(unique_problems)//2-1)
        aoc_y_labels = aoc_y_labels * 2
        y_labels.extend(aoc_y_labels)

        aoc_and_loss.extend(problem_loss)
        loss_subtitles = [f"F{problem}" for problem in unique_problems]
        loss_y_scale = [("symlog", {}) for _ in range(len(unique_problems))]
        loss_y_labels = ['Loss'] + [''] * (len(unique_problems)//2-1)
        loss_y_labels = loss_y_labels * 2

        subtitles.extend(loss_subtitles)
        filling.extend(problem_loss_filling)
        y_scale.extend(loss_y_scale)
        y_labels.extend(loss_y_labels)

        loss_plot_y = np.array(problem_loss)
        x_base = np.arange(len(problem_loss[0][0]), dtype=np.int16)
        loss_x = np.tile(x_base, (len(loss_plot_y), 1))

        file_name = 'es_problem_loss'
        if fig_dir is not None:
            file_name = os.path.join(fig_dir, file_name)

        plot_lines(
            y = loss_plot_y,
            x = loss_x,
            labels = labels,
            label_fontsize=10,
            filling=problem_loss_filling,
            y_scales=loss_y_scale,
            y_labels=loss_y_labels,
            y_label_fontsize=12,
            tick_fontsize=12,
            sub_titles=loss_subtitles,
            sub_title_fontsize=12,
            linewidth=1.3,
            combined_legend=True,
            combined_legend_bottom=0.18,
            combined_legend_fontsize=13,
            n_cols=n_cols,
            figsize=(10, 4),
            filename=file_name,
            show=False,
            )

        # step n_cols
        # for i in range(0, len(unique_problems), n_cols):
        #     aoc_and_loss.extend(problem_log_aoc[i:i+n_cols])
        #     subtitles.extend([f"F{problem}-AOC" for problem in unique_problems[i:i+n_cols]])
        #     filling.extend(problem_log_aoc_filling[i:i+n_cols])

        #     aoc_and_loss.extend(problem_loss[i:i+n_cols])
        #     subtitles.extend([f"F{problem}-Loss" for problem in unique_problems[i:i+n_cols]])
        #     filling.extend(problem_loss_filling[i:i+n_cols])

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

        file_name = 'es_problem_aoc_loss'
        if fig_dir is not None:
            file_name = os.path.join(fig_dir, file_name)

        plot_lines(
            y = plot_y,
            x = x,
            labels = labels,
            label_fontsize=10,
            tick_fontsize=14,
            filling=filling,
            y_scales=y_scale,
            y_labels=y_labels,
            y_label_fontsize=14,
            sub_titles=subtitles,
            sub_title_fontsize=15,
            combined_legend=True,
            combined_legend_bottom=0.1,
            combined_legend_fontsize=15,
            n_cols=n_cols,
            figsize=(14, 8),
            filename=file_name,
            show=False,
            )

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
                if not hasattr(handler, 'query_time'):
                    continue
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

    # compitable with 1+1
    _unique_strategies = []
    for strategy in unique_strategies:
        if strategy == '1+1':
            continue
        _unique_strategies.append(strategy)
    unique_strategies = _unique_strategies

    for strategy in unique_strategies:
        _strategy_token_df = _all_token_df[_all_token_df['strategy'] == strategy]
        _total_token_count = _strategy_token_df['total_token_count'].to_list()
        y_total_token_count.append(_total_token_count)

        _prompt_token_count = _strategy_token_df['prompt_token_count'].to_list()
        y_prompt_token_count.append(_prompt_token_count)

        _response_token_count = _strategy_token_df['response_token_count'].to_list()
        y_response_token_count.append(_response_token_count)

        _mean_query_time = np.mean(_strategy_token_df['query_time'].to_list())
        y_query_time.append(_mean_query_time)

    plot_y = [y_total_token_count, y_prompt_token_count, y_response_token_count]
    labels = [unique_strategies] * len(plot_y)
    sub_titles = ["Total token count", "Prompt token count", "Response token count"]

    plot_box_violin(
        data=plot_y,
        labels=labels,
        sub_titles=sub_titles,
        n_cols=4,
        label_fontsize=10,
        title="Token usage per Experiment",
        figsize=(15, 5),
        )

    prices = {
        'o3-mini': (1.1, 4.4),
        'GPT-4o': (2.5, 10.0),
        'Claude-3.5': (3.0, 15.0),
        'DeepSeek-R1': (0.8, 2.4),
        'Gemini-Flash-2.0': (0.1, 0.4),
    }

    mean_prompt_token_count = [np.mean(ele) for ele in y_prompt_token_count]
    mean_response_token_count = [np.mean(ele) for ele in y_response_token_count]

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
                title="Price per Experiment",
                fig_size=(15,5))

    def _expand_list(idx_list, data_list):
        _new_data_list = []
        index = 0
        for idx, data in zip(idx_list, data_list):
            _n_filling = idx - index
            _new_data_list.extend([data] * _n_filling)
            index = idx
        return np.array(_new_data_list)

    _iter_token_df = _token_df.groupby(['strategy', 'n_iter'])[['total_token_count', 'prompt_token_count', 'response_token_count']].agg(np.mean).reset_index()
    _y_total_token_count = []
    _y_prompt_token_count = []
    _y_response_token_count = []
    for strategy in unique_strategies:
        _strategy_token_df = _iter_token_df[_iter_token_df['strategy'] == strategy]

        _iter_list = _strategy_token_df['n_iter'].to_list()
        
        _total_token_count = _strategy_token_df['total_token_count'].to_list()
        _y_total_token_count.append(_expand_list(_iter_list, _total_token_count))

        _prompt_token_count = _strategy_token_df['prompt_token_count'].to_list()
        _y_prompt_token_count.append(_expand_list(_iter_list, _prompt_token_count))

        _response_token_count = _strategy_token_df['response_token_count'].to_list()
        _y_response_token_count.append(_expand_list(_iter_list, _response_token_count))
    
    plot_y = [_y_total_token_count, _y_prompt_token_count, _y_response_token_count]
    plot_y = [np.array(ele) for ele in plot_y]
    plot_x = [np.arange(len(_y_total_token_count[0]))] * len(plot_y)
    labels = [unique_strategies] * len(plot_y)
    sub_titles = ["Total token count", "Prompt token count", "Response token count"]
    plot_lines(
        y = plot_y,
        x = plot_x,
        labels = labels,
        linewidth=1.5,
        label_fontsize=10,
        sub_titles = sub_titles,
        n_cols=3,
        figsize=(15, 5),
        title="Token usage",
        )

def plot_search_result(result_dir, save_name=None, extract_fn=None, fig_dir=None):
    res_df, results = _load_results(result_dir, save_name=save_name, extract_fn=extract_fn)

    _calculate_error_info(results)

    unique_strategies = res_df['strategy'].unique()
    unique_strategies = sorted(unique_strategies, key=cmp_to_key(compare_expressions))

    # _plot_search_token_usage(results, unique_strategies)

    _plot_search_aoc(res_df, unique_strategies, fig_dir=fig_dir)
    _plot_search_group_aoc(res_df, unique_strategies, fig_dir=fig_dir)

    # _plot_serach_pop_similarity(results, unique_strategies, save_name=save_name)

    err_df = _process_error_data(results)
    _plot_search_all_error_rate(err_df, unique_strategies, fig_dir=fig_dir)
    # _plot_search_error_type(err_df, unique_strategies=unique_strategies)
    # _plot_search_error_rate_by_generation(err_df, unique_strategies)

    _plot_search_problem_aoc_and_loss(res_df, fig_dir=fig_dir)

def plot_search_0112():
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
            target = RenameUnpickler.unpickle(open(file_path, "rb"))
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


def _calculate_error_info(pop_list:list[tuple[str,Population]]):
    # total error: number / total number, P(err)
    # initial error: number / total number, P(err|gen==1)
    # error crossover error: number / total number, P(err|err_p == True and op == crossover)
    # error mutation error: number / total number, P(err|err_p == True and op == mutation)
    # non-error crossover error: number / total number, P(err|err_p == False and op == crossover)
    # non-error mutation error: number / total number, P(err|err_p == False and op == mutation)
    error_infos = {}
    for name, pop in pop_list:
        n_generation = pop.get_current_generation()
        for gen in range(n_generation):
            gen_offsprings = pop.get_offsprings(generation=gen)
            for ind in gen_offsprings:
                handler = Population.get_handler_from_individual(ind)
                has_error = handler.eval_result is None or handler.eval_result.score == 0.0
                error_count = 1 if has_error else 0

                if name not in error_infos:
                    error_info = {
                        'total': (0, 0),  # (error_count, total_count)
                        'initial': (0, 0),  # (error_count, total_count)
                        'error_crossover': (0, 0),  # (error_count, total_count)
                        'error_mutation': (0, 0),  # (error_count, total_count)
                        'non_error_crossover': (0, 0),  # (error_count, total_count)
                        'non_error_mutation': (0, 0),  # (error_count, total_count)
                    }
                    error_infos[name] = error_info
                else:
                    error_info = error_infos[name]

                # total error
                error_info['total'] = (
                    error_info['total'][0] + error_count,
                    error_info['total'][1] + 1
                )

                if gen == 0:
                    error_info['initial'] = (
                        error_info['initial'][0] + error_count,
                        error_info['initial'][1] + 1
                    )
                else:
                    parents = pop.get_parent(ind)
                    if len(parents) == 1:
                        # mutation
                        parent = parents[0]
                        parent_handler = Population.get_handler_from_individual(parent)
                        has_p_error = parent_handler.eval_result is None or parent_handler.eval_result.score == 0.0
                        if has_p_error:
                            error_info['error_mutation'] = (
                                error_info['error_mutation'][0] + error_count,
                                error_info['error_mutation'][1] + 1
                            )
                        else:
                            error_info['non_error_mutation'] = (
                                error_info['non_error_mutation'][0] + error_count,
                                error_info['non_error_mutation'][1] + 1
                            )
                    elif len(parents) == 2:
                        # crossover
                        has_error = False
                        for parent in parents:
                            parent_handler = Population.get_handler_from_individual(parent)
                            has_error = has_error or (parent_handler.eval_result is None or parent_handler.eval_result.score == 0.0)
                            if has_error:
                                break
                        if has_error:
                            error_info['error_crossover'] = (
                                error_info['error_crossover'][0] + error_count,
                                error_info['error_crossover'][1] + 1
                            )
                        else:
                            error_info['non_error_crossover'] = (
                                error_info['non_error_crossover'][0] + error_count,
                                error_info['non_error_crossover'][1] + 1
                            )

    for name, error_info in error_infos.items():
        print(f"Error info for {name}:")
        for key in error_info:
            ratio = error_info[key][0] / error_info[key][1] if error_info[key][1] > 0 else 0.0
            print(f"  {key}: {ratio:.4f} ({error_info[key][0]} / {error_info[key][1]})")


def _load_results(dir_path, file_paths=None, extract_fn=None, save_name=None):
    res_df = None
    if save_name is not None: 
        if os.path.exists(save_name):
            res_df = pd.read_pickle(save_name)

    if file_paths is None:
        file_paths = []
        for dir_name in os.listdir(dir_path):
            if not os.path.isdir(os.path.join(dir_path, dir_name)):
                continue
            for file_name in os.listdir(os.path.join(dir_path, dir_name)):
                if "final" not in file_name:
                    continue
                file_path = os.path.join(dir_path, dir_name, file_name)
                file_paths.append(file_path)
        
        if len(file_paths) == 0:
            raise ValueError(f"Invalid directory path: {dir_path}")
    else:
        if len(file_paths) == 0:
            raise ValueError(f"Invalid file paths: {file_paths}")

    pop_list = []
    best_pop_map = {}
    for file_path in file_paths:
        pop = RenameUnpickler.unpickle(open(file_path, "rb"))
        n_parent = pop.n_parent
        n_offspring = pop.n_offspring

        name = f"{n_parent},{n_offspring}"
        if pop.use_elitism:
            name = f"{n_parent}+{n_offspring}"

        if extract_fn is not None:
            sub_fix = extract_fn(file_path)
            if sub_fix:
                name += f"_{sub_fix}"

        pop_list.append((name, pop))

        cur_best = best_pop_map.get(name, None)
        if cur_best is None or pop.get_best_of_all().fitness > cur_best.get_best_of_all().fitness:
            best_pop_map[name] = pop

        # ind_index = 0
        # for gen, keys in enumerate(pop.generations):
        #     for key in keys:
        #         ind = pop.individuals.get(key, None)
        #         handler = Population.get_handler_from_individual(ind)
        #         if handler.eval_result is not None:
        #             handler.eval_result.update_aoc_with_new_bound_if_needed()
        #             ind.fitness = handler.eval_result.score

        #         pop.save_on_the_fly(ind, gen, ind_index)
        #         ind_index += 1

    if res_df is not None:
        return res_df, pop_list

    res_df = _process_search_result(pop_list, save_name=save_name)
    return res_df, pop_list

def extract_results_to_ioh_csv(dir_path, pop_save_path, extract_fn=None):
    res_df, _ = _load_results(dir_path, file_paths=None, extract_fn=extract_fn, save_name=pop_save_path)

    aoc_df = res_df.groupby(['strategy', 'n_strategy', 'n_ind'])[["log_y_aoc"]].agg(np.mean).reset_index()

    strategies = aoc_df['strategy'].unique()
    repeats = aoc_df['n_strategy'].unique()

    df_aoc_data = []

    for strategy in strategies:
        for repeat in repeats:
            _strategy_df = aoc_df[(aoc_df['strategy'] == strategy) & (aoc_df['n_strategy'] == repeat)]
            if len(_strategy_df) == 0:
                continue
            max_ind = _strategy_df['n_ind'].max()
            _strategy_df = _strategy_df.sort_values(by='n_ind')
            _strategy_df = _strategy_df.reset_index(drop=True)
            _strategy_df = _strategy_df.drop(columns=['strategy', 'n_strategy'])
            # n_ind should be range from 1 to 100. add the missing n_ind with 0
            _strategy_df = _strategy_df.set_index('n_ind')
            _strategy_df = _strategy_df.reindex(range(1, max_ind + 1), fill_value=0)
            _strategy_df = _strategy_df.reset_index()
            _strategy_df = _strategy_df.rename(columns={'index': 'n_ind'})
            _strategy_df = _strategy_df.sort_values(by='n_ind')
            _strategy_df = _strategy_df.reset_index(drop=True)

            for i in range(len(_strategy_df)):
                aoc = _strategy_df.at[i, 'log_y_aoc']

                df_aoc_data.append({
                    'Evaluation counter': i + 1,
                    'Function values': aoc,
                    'Function ID': 'F1-F24',
                    'Algorithm ID': strategy,
                    'Problem dimension': 5,
                    'Run ID': repeat, 
                })
            
    # Create a DataFrame from the list of dictionaries
    df_aoc = pd.DataFrame(df_aoc_data)
    df_aoc.to_csv(f"{dir_path}/es_ioh_aoc.csv", index=False)


    p_aoc_df = res_df.groupby(['strategy', 'n_strategy', 'problem_id', 'n_ind'])[["log_y_aoc"]].agg(np.mean).reset_index()

    strategies = p_aoc_df['strategy'].unique()
    repeats = p_aoc_df['n_strategy'].unique()
    problems = p_aoc_df['problem_id'].unique()

    df_aoc_data = []

    for strategy in strategies:
        for repeat in repeats:
            for problem in problems:
                _strategy_df = p_aoc_df[(p_aoc_df['strategy'] == strategy) & (p_aoc_df['n_strategy'] == repeat) & (p_aoc_df['problem_id'] == problem)]
                if len(_strategy_df) == 0:
                    continue
                max_ind = _strategy_df['n_ind'].max()
                _strategy_df = _strategy_df.sort_values(by='n_ind')
                _strategy_df = _strategy_df.reset_index(drop=True)
                _strategy_df = _strategy_df.drop(columns=['strategy', 'n_strategy'])
                # n_ind should be range from 1 to 100. add the missing n_ind with 0
                _strategy_df = _strategy_df.set_index('n_ind')
                _strategy_df = _strategy_df.reindex(range(1, max_ind + 1), fill_value=0)
                _strategy_df = _strategy_df.reset_index()
                _strategy_df = _strategy_df.rename(columns={'index': 'n_ind'})
                _strategy_df = _strategy_df.sort_values(by='n_ind')
                _strategy_df = _strategy_df.reset_index(drop=True)

                for i in range(len(_strategy_df)):
                    aoc = _strategy_df.at[i, 'log_y_aoc']

                    df_aoc_data.append({
                        'Evaluation counter': i + 1,
                        'Function values': aoc,
                        'Function ID': problem,
                        'Algorithm ID': strategy,
                        'Problem dimension': 5,
                        'Run ID': repeat, 
                    })
    # Create a DataFrame from the list of dictionaries
    df_aoc = pd.DataFrame(df_aoc_data)
    df_aoc.to_csv(f"{dir_path}/es_ioh_p_aoc.csv", index=False)
    

def update_aoc_for_res():
    file_path = 'Experiments/ESPopulation_evol_4-16_final_0222112835.pkl'
    with open(file_path, 'rb') as f:
        res = RenameUnpickler.unpickle(f)

    for _, ind in res.individuals.items():
        hanlder = Population.get_handler_from_individual(ind)
        eval_res = hanlder.eval_result
        if eval_res is not None:
            for sub_res in eval_res.result:
                sub_res.update_aoc_with_new_bound_if_needed()
            eval_res.score = np.mean([r.log_y_aoc for r in eval_res.result])
            ind.fitness = eval_res.score

    new_file_path = file_path.replace('final', 'final_aoc')
    with open(new_file_path, 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    setup_logger(level=logging.INFO)

    # plot_search()

    # plot_light_evol_and_final()

    file_paths = None
    save_name = None
    extract_fn = None

    # dir_path = 'Experiments/log_eater/pop_40_f_0220'
    # save_name = 'Experiments/log_eater/pop_40_f_0220/df_res_05230546.pkl'

    
    # dir_path = 'Experiments/log_eater/pop_100_f'
    # save_name = 'Experiments/log_eater/pop_100_f/df_res_02250235.pkl'

    # dir_path = 'Experiments/log_eater/pop_40_cr'
    # save_name = 'Experiments/log_eater/pop_40_cr/df_res_05240416.pkl'

    def _extract_fn(file_path):
        if 'temperature' in file_path:
            pattern = re.compile(r'_t(\d+)_IOHEvaluator')
            match = pattern.search(file_path)
            if match:
                t = float(match.group(1)) / 10
                return f"{t}"
            pattern = re.compile(r'evol_\d*-\d*_(\w+)_IOHEvaluator')
            match = pattern.search(file_path)
            if match:
                f = match.group(1)
                return f"{f}"

        elif 'top_k' in file_path:
            pattern = re.compile(r'_k(\d+)_IOHEvaluator')
            match = pattern.search(file_path)
            if match:
                k = int(match.group(1))
                return f"{k}"
        elif 'top_p' in file_path:
            pattern = re.compile(r'_p(\d+)_IOHEvaluator')
            match = pattern.search(file_path)
            if match:
                p = int(match.group(1)) / 100
                return f"{p}"
        elif 'pop_40_4-16' in file_path:
            pattern = re.compile(r'evol_\d*-\d*_(.+)_IOHEvaluator')
            match = pattern.search(file_path)
            if match:
                f = match.group(1)
                return f"{f}"
        elif 'pop_40_cr' in file_path:
            pattern = re.compile(r'_cr(\d+)_IOHEvaluator')
            match = pattern.search(file_path)
            if match:
                cr = int(match.group(1)) / 10
                return f"{cr}"
        return ''

    # dir_path = 'Experiments/log_eater/pop_40_f'
    # save_name = 'Experiments/log_eater/pop_40_f/df_res_05250314.pkl'
                
    # dir_path = 'Experiments/log_eater/pop_40_temperature'
    # save_name = 'Experiments/log_eater/pop_40_temperature/df_res_05250314.pkl'

    # dir_path = 'Experiments/log_eater/pop_40_top_k'
    # save_name = 'Experiments/log_eater/pop_40_top_k/df_res_05250347.pkl'

    # dir_path = 'Experiments/log_eater/pop_40_top_p'
    # save_name = 'Experiments/log_eater/pop_40_top_p/df_res_05250307.pkl'

    extract_fn = _extract_fn

    dir_path = 'Experiments/log_eater/pop_100_tkcr'
    save_name = 'Experiments/log_eater/pop_100_tkcr/df_res_05241329.pkl'
    # save_name = dir_path + '/' + f'df_res_{datetime.now().strftime("%m%d%H%M")}.pkl'


    # extract_results_to_ioh_csv(dir_path, save_name, extract_fn=extract_fn)

    plot_search_result(dir_path, save_name=save_name, extract_fn=extract_fn, fig_dir=dir_path)


