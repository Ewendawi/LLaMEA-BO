import json
import uuid
from datetime import datetime
import logging
import pickle
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from .population.population import Population, desc_similarity
from .individual import Individual

class NoCodeException(Exception):
    """Could not extract generated code."""

class BOOverBudgetException(Exception):
    """Exceeded the budget for the number of evaluations."""

def handle_timeout(signum, frame):
    raise TimeoutError

#========================================
#Logger
#========================================
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(logger = None, level=logging.INFO, filename=None):
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(level)
    if filename:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger

def get_logger(name = None, level=logging.INFO, filename=None):
    logger = logging.getLogger(name)
    setup_logger(logger, level, filename)
    return logger

class LogggerJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, '__to_json__'):
            return o.__to_json__()
        return super().default(o)

class IndividualLogger:
    def __init__(self):
        self.individual_map:dict[str, Individual] = {}
        self.experiment_map:dict[str, dict] = {}
        self._file_name = "individual_set"
        self.dirname = "logs"

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        new_value = value
        if value is not None:
            new_value = value.replace(" ", "")
            new_value = new_value.replace(":", "_")
            new_value = new_value.replace("/", "_")
        self._file_name = new_value

    def log_individual(self, individual):
        self.individual_map[individual.id] = individual

    def get_individual(self, ind_id):
        return self.individual_map.get(ind_id, None)

    def log_experiment(self, name, id_list):
        exp_id = str(uuid.uuid4())
        experiment = {
            "id": exp_id,
            "name": name,
            "id_list": id_list
        }
        self.experiment_map[exp_id] = experiment

    def get_experiment(self, experiment_id):
        return self.experiment_map.get(experiment_id, None)

    def save(self, filename=None, dirname=None):
        if dirname is None:
            dirname = self.dirname
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if filename is None:
            filename = self.file_name
        filename = filename.replace(" ", "")
        filename = filename.replace(":", "_")
        filename = filename.replace("/", "_")
        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        filename = os.path.join(dirname, f"{filename}_{time_stamp}.pkl")
        with open(filename, "wb") as f:
            pickle.dump((self.individual_map,self.experiment_map), f)

    def get_successful_individuals(self):
        successful_individuals = []
        for _, individual in self.individual_map.items():
            if isinstance(individual, dict):
                # No longer compatible with older formats
                continue
            if individual.error is None and "deprecated" not in individual.metadata:
                successful_individuals.append(individual)
        return successful_individuals

    def get_failed_individuals(self, error_type=None):
        failed_individuals = []
        for _, individual in self.individual_map.items():
            if isinstance(individual, dict):
                # No longer compatible with older formats
                continue
            if individual.error is None or "deprecated" in individual.metadata:
                continue
            if (error_type is None or individual.metadata["error_type"] == error_type):
                failed_individuals.append(individual)
        return failed_individuals

# {
#     "contents": {
#         "<id>": {
#             "id": "",
#             "solution": "", // code block
#             "name": "",
#             "description": "", // desc block, markdown
#             "fitness": "",
#             "feedback": "", // feedback and error block, markdown
#             "error": "",
#             "parent_id": "",
#             "metadata": {
#                 "error_type": "", // single-choice filter
#                 "model": "", // single-choice filter
#                 "prompt": "", // prompt block, foldable
#                 "raw_response": "", // response block, markdown
#                 "problem": "", // single-choice filter
#                 "tags": [] // multiple-choice filter
#             }
#         }
#     },
#     "experiments": {
#         "<experiment_id>": {
#             "id": "", // single-choice filter. retrieve all the content in the id_list.
#             "name": "",
#             "id_list": [] // id: content_id
#         }
#     }
# }

    def save_reader_format(self, filename=None):
        json_str = self.covert_to_reader_format()
        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        if filename is None:
            filename = self.file_name
        filename = filename.replace(" ", "")
        filename = filename.replace(":", "_")
        filename = filename.replace("/", "_")
        filepath = os.path.join(self.dirname, f"reader_format_{filename}_{time_stamp}.json")
        with open(f"{filepath}", "w", encoding="utf-8") as f:
            f.write(json_str)

    def covert_to_reader_format(self) -> str:
        reader_format = {
            "experiments": self.experiment_map.copy()
        }
        contents = {}
        for ind_id, individual in self.individual_map.items():
            contents[ind_id] = individual
            handler = Population.get_handler_from_individual(individual)
            individual.metadata["raw_response"] = handler.raw_response

        reader_format["contents"] = contents

        for _, individual in reader_format["contents"].items():
            individual.metadata["language"]= "python"

        json_str = json.dumps(reader_format, indent=4, cls=LogggerJSONEncoder)
        return json_str

    @classmethod
    def load(cls, filepath=None):

        if filepath is None and not os.path.exists(filepath):
            return None
        logger = cls()
        if filepath is None:
            return
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                logger.individual_map, logger.experiment_map = pickle.load(f)
        else:
            raise FileNotFoundError(f"File {filepath} not found")
        return logger

    @classmethod
    def merge_logs(cls, log_dir, save=True):
        # check if the log_dir is a directory
        if not os.path.isdir(log_dir):
            return None

        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".pkl")]
        loggers = []
        for log_file in log_files:
            logger = cls().load(log_file)
            loggers.append(logger)
        merged_logger = cls()
        for logger in loggers:
            merged_logger.individual_map.update(logger.individual_map)
            merged_logger.experiment_map.update(logger.experiment_map)
        if save:
            merged_logger.save()
        return merged_logger


    def replace_metadata_key(self, old_key, new_key):
        for _, ind in self.individual_map.items():
            if old_key in ind.metadata:
                ind.metadata[new_key] = ind.metadata[old_key]
                del ind.metadata[old_key]

# Plotting

def plot_result(y:np.ndarray, x:np.ndarray,

                labels:list[list[str]],
                label_fontsize:int = 7,

                x_labels:list[str]=None, y_labels:list[str]=None, 
                ignore_y = False,

                sub_titles:list[str]=None,
                sub_title_fontsize:int = 10,

                baselines:np.ndarray=None, baseline_labels:list[list[str]]=None, 

                y_lim_bottom:float = None, y_lim_top:float = None,

                title:str = None,
                title_fontsize:int = 12, 
                caption:str = None, caption_fontsize:int = 10,

                filename:str = None, 
                n_cols:int = 1, figsize:tuple[int,int] = (10, 6), 
                show:bool = True):
    
    # y.shape = (n_plots, n_lines, n_points)
    if len(labels) != y.shape[0]:
        logging.warning("PLOT:Number of labels does not match the number of plots.")
    
    if x_labels is not None and len(x_labels) != y.shape[0]:
        logging.warning("PLOT:Number of x_labels does not match the number of plots.")
    
    if y_labels is not None and len(y_labels) != y.shape[0]:
        logging.warning("PLOT:Number of y_labels does not match the number of plots.")

    if sub_titles is not None and len(sub_titles) != y.shape[0]:
        logging.warning("PLOT:Number of sub_titles does not match the number of plots.")

    # baselines.shape = (n_plots, n_lines, 1)
    if baselines is not None:
        if len(baselines.shape) < 3:
            logging.warning("PLOT:Baselines should have 3 dimensions.")
            baselines = None
            baseline_labels = None

    
    n_plots = y.shape[0]
    n_rows = n_plots // n_cols

    axs_ids = []
    for row in range(n_rows):
        row_ids = []
        for col in range(n_cols):
            row_ids.append(row * n_cols + col)
        axs_ids.append(row_ids)
    fig, axs = plt.subplot_mosaic(axs_ids, figsize=figsize)
    for i in range(n_plots):
        row = i // n_cols
        col = i % n_cols
        ax = axs[i]
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
        if y_lim_bottom is not None or y_lim_top is not None:
            offset = (y[i,:, :].max() - y[i, :, :].min()) * 0.1
            _y_lim_bottom = y_lim_bottom if y_lim_bottom is not None else y[i, :, :].min()
            _y_lim_top = y_lim_top if y_lim_top is not None else y[i, :, :].max()
            ax.set_ylim(bottom=_y_lim_bottom - offset, top=_y_lim_top + offset)
        _labels = labels[i] if len(labels) > i else []
        for j in range(y.shape[1]):
            label = _labels[j] if len(_labels) > j else f"{j}"
            if not ignore_y:
                ax.plot(x[i, j,:], y[i, j,:], label=label)
        if baselines is not None:
            _bl_labels = baseline_labels[i] if len(baseline_labels) > i else []
            for j in range(baselines.shape[1]):
                bl_label = _bl_labels[j] if len(_bl_labels) > j else f"bl_{j}"
                bl_x = x[i, 0, :]
                bl_y = baselines[i, j, :]
                bl_y = np.repeat(bl_y, len(bl_x))
                ax.plot(bl_x, bl_y, label=f"{bl_label}", linestyle='--')

        ax.legend(fontsize=label_fontsize)
        ax.grid(True)

        if x_labels is not None:
            x_label = x_labels[i] if len(x_labels) > i else ""
            ax.set_xlabel(x_label)
        if y_labels is not None:
            y_label = y_labels[i] if len(y_labels) > i else ""
            ax.set_ylabel(y_label)
        if sub_titles is not None:
            sub_title = sub_titles[i] if len(sub_titles) > i else ""
            ax.set_title(sub_title, fontsize=sub_title_fontsize)

    if title:
        fig.suptitle(title, fontsize=title_fontsize)

    if caption:
        fig.text(0.5, -0.11, caption, ha='center', fontsize=caption_fontsize)
    
    if filename:
        plt.savefig(filename, dpi=300)

    fig.tight_layout()

    if show:
        plt.show(block=True)
        

from .evaluator.evaluator_result import EvaluatorResult

def plot_results(results:list[tuple[str,list[EvaluatorResult]|Population]], 
                    other_results:list[EvaluatorResult] = None, **kwargs):
    #results: (n_strategies, n_generations, n_evaluations)
    column_names = [
        'strategy',
        'problem_id',
        'instance_id',
        'exec_id',
        'n_gen',
        'n_iter',
        'pop',
        "log_y_aoc",
        "y_aoc",
        "best_y",
        'loss'
        ]

    def res_to_row(res, gen:int, strategy_name:str, n_iter:int):
        res_id = res.id
        res_split = res_id.split("-")
        problem_id = int(res_split[0])
        instance_id = int(res_split[1])
        repeat_id = int(res_split[2])
        row = {
            'strategy': strategy_name,
            'problem_id': problem_id,
            'instance_id': instance_id,
            'exec_id': repeat_id,
            'n_gen': gen+1,
            'n_iter': n_iter,
            "log_y_aoc": res.y_aoc_from_ioh,
            "y_aoc": res.y_aoc,
            "best_y": res.best_y,
            'loss': abs(res.optimal_value - res.best_y)
        }
        return row

    # preprocess results
    group_unique_inds = {}
    group_gen_inds = {}
    res_df = pd.DataFrame(columns=column_names)
    for res_tuple in results:
        strategy_name, gen_list = res_tuple 

        # group by strategy
        if strategy_name not in group_unique_inds:
            group_unique_inds[strategy_name] = []
            group_gen_inds[strategy_name] = []
        unique_inds = {}
        gen_inds_list = []
        group_gen_inds[strategy_name].append(gen_inds_list)
        group_unique_inds[strategy_name].append(unique_inds)
        if isinstance(gen_list, Population):
            n_iter = 0
            pop = gen_list
            n_generation = pop.get_current_generation()
            for gen in range(n_generation):
                # offspring generated in this generation
                gen_offsprings = pop.get_offsprings(generation=gen)
                n_iter += len(gen_offsprings)
                # offspring selected in this generation
                gen_inds = pop.get_individuals(generation=gen)
                gen_inds_list.append(gen_inds)
                for ind in gen_inds:
                    if ind.id not in unique_inds:
                        unique_inds[ind.id] = ind
                    handler = Population.get_handler_from_individual(ind)
                    for res in handler.eval_result.result:
                        row = res_to_row(res, gen, strategy_name=strategy_name, n_iter=n_iter)
                        res_df.loc[len(res_df)] = row
        elif isinstance(gen_list, list):
            for i, gen_res in enumerate(gen_list):
                if gen_res.error is not None:
                    continue
                for res in gen_res.result:
                    row = res_to_row(res, i, strategy_name=strategy_name, n_iter=i+1) 
                    res_df.loc[len(res_df)] = row

    if other_results is not None:
        for other_res in other_results:
            for res in other_res.result:
                row = res_to_row(res, -1, strategy_name=res.name, n_iter=-1)
                res_df.loc[len(res_df)] = row

    # strategy-wise
    log_y_aocs, log_y_aoc_labels = [], []
    baseline_y_aoc, baseline_y_aoc_labels = [], []
    y_aocs, y_aoc_labels = [], []
    baseline_y_aoc, baseline_y_aoc_labels = [], []
    g_log_y_aoc = res_df.groupby(['strategy', 'n_iter'])[["log_y_aoc", "y_aoc"]].agg(np.mean).reset_index()
    max_iter = g_log_y_aoc['n_iter'].max()
    for name, group in g_log_y_aoc.groupby('strategy'):
        iters = group['n_iter'].values
        if len(iters) == 1 and iters[0] == -1:
            baseline_y_aoc.append(group['y_aoc'].values[0])
            baseline_y_aoc_labels.append(name)
            continue

        # fill the missing iter with 0; start from 0
        aoc = np.zeros(max_iter+1)
        log_aoc = np.zeros(max_iter+1)
        for iter in iters:
            log_aoc[iter] = group[group['n_iter'] == iter]['log_y_aoc'].values[0]
            aoc[iter] = group[group['n_iter'] == iter]['y_aoc'].values[0]

        log_y_aocs.append(log_aoc)
        y_aocs.append(aoc)
        log_y_aoc_labels.append(name)
        y_aoc_labels.append(name)

    # problem-wise
    loss_list = [[] for _ in range(1, 25)]
    loss_labels = [[] for _ in range(1, 25)]
    aoc_list = [[] for _ in range(1, 25)]
    aoc_labels = [[] for _ in range(1, 25)]
    log_aoc_list = [[] for _ in range(1, 25)]
    log_aoc_labels = [[] for _ in range(1, 25)]
    g_best_y = res_df.groupby(['strategy', 'n_iter', 'problem_id'])[['y_aoc', 'loss', 'log_y_aoc']].agg(np.mean).reset_index()
    max_iter = g_best_y['n_iter'].max()
    for (name, p_id), group in g_best_y.groupby(['strategy', 'problem_id']):
        iters = group['n_iter'].values
        if len(iters) == 1 and iters[0] == -1:
            continue

        aoc = np.zeros(max_iter+1)
        loss = np.zeros(max_iter+1)
        log_aoc = np.zeros(max_iter+1)
        max_loss = group['loss'].max()
        missing_iters = set(range(0, max_iter+1)) - set(iters)
        for missing_iter in missing_iters:
            loss[missing_iter] = max_loss
        for iter in iters:
            loss[iter] = group[group['n_iter'] == iter]['loss'].values[0]
            aoc[iter] = group[group['n_iter'] == iter]['y_aoc'].values[0]
            log_aoc[iter] = group[group['n_iter'] == iter]['log_y_aoc'].values[0]

        aoc_list[p_id-1].append(aoc)
        aoc_labels[p_id-1].append(name)
        log_aoc_list[p_id-1].append(log_aoc)
        log_aoc_labels[p_id-1].append(name+"(log)")
        loss_list[p_id-1].append(loss)
        loss_labels[p_id-1].append(name)

    # similarity
    handler_similarity = True
    if handler_similarity:
        group_mean_sim_labels = []
        group_mean_sim = []
        for strategy_name, group_inds in group_unique_inds.items():
            temp_means = []
            temp_len = []
            for unique_inds in group_inds:
                temp_len.append(len(unique_inds))
                unique_inds = list(unique_inds.values())
                mean_sim, _ = desc_similarity(unique_inds)
                overall_sim = np.mean(mean_sim)
                temp_means.append(overall_sim)
            group_mean_sim.append([np.mean(temp_means)])
            mean_len = np.mean(temp_len)
            group_mean_sim_labels.append(strategy_name+f"({mean_len:.0f})")

        group_gen_sim = []
        group_gen_sim_labels = []
        for strategy_name, group_inds in group_gen_inds.items():
            group_sim = []
            for instance in group_inds:
                instance_sim = []
                for gen_inds in instance:
                    if len(gen_inds) < 2:
                        continue

                    handler = Population.get_handler_from_individual(gen_inds[0])
                    if handler.eval_result.similarity is not None:
                        sim_list = [Population.get_handler_from_individual(ind).eval_result.similarity for ind in gen_inds]
                        instance_sim.append(np.mean(sim_list))
                    elif hasattr(handler.eval_result, "simiarity") and handler.eval_result.simiarity is not None:
                        sim_list = [Population.get_handler_from_individual(ind).eval_result.simiarity for ind in gen_inds]
                        instance_sim.append(np.mean(sim_list))
                    else:
                        mean_sim, _ = desc_similarity(gen_inds)
                        instance_sim.append(np.mean(mean_sim))
                if len(instance_sim) > 0:
                    group_sim.append(instance_sim)
            if len(group_sim) == 0:
                continue
            group_gen_sim.append(np.mean(np.array(group_sim), axis=0))
            group_gen_sim_labels.append(strategy_name)
        max_gen = max([gen_sim.size for gen_sim in group_gen_sim])
        for i, gen_sim in enumerate(group_gen_sim):
            if gen_sim.size < max_gen:
                # append mean to the end
                mean_sim = np.mean(gen_sim)
                group_gen_sim[i] = np.append(gen_sim, [mean_sim] * (max_gen - gen_sim.size))
                
        # plot similarity
        baseline_sim = np.array([group_mean_sim])
        baseline_sim_labels = [group_mean_sim_labels]
        y = np.array([group_gen_sim])
        base_x = np.arange(0, max_gen, dtype=int)
        x = np.tile(base_x, (y.shape[0], y.shape[1], 1))
        labels = [group_gen_sim_labels]
        x_labels = ["Generation"] * len(labels)
        y_labels = ["Similarity"] * len(labels)
        plot_result(y=y, x=x, labels=labels, 
                    baseline_labels=baseline_sim_labels, baselines=baseline_sim,
                    x_labels=x_labels, y_labels=y_labels,
                    n_cols=1, **kwargs)
        
    # plot aoc
    y = np.maximum.accumulate(np.array([log_y_aocs, y_aocs]), axis=2)
    base_x = np.arange(0, max_iter+1, dtype=int)
    x = np.tile(base_x, (y.shape[0], y.shape[1], 1))
    sub_titles = ["Log AOC", "AOC"]
    labels = [log_y_aoc_labels] * 2
    plot_result(y=y, x=x, labels=labels,
                title=None,
                sub_titles=sub_titles, n_cols=2,
                **kwargs)

    # plot loss
    y = np.minimum.accumulate(np.array(loss_list), axis=2)
    base_x = np.arange(0, max_iter+1, dtype=int)
    x = np.tile(base_x, (y.shape[0], y.shape[1], 1))
    sub_titles = [f"F{p_id}" for p_id in range(1, 25)]
    labels = loss_labels * len(loss_list)
    x_labels = ["Population"] * len(loss_list)
    n_cols = 6
    for i, _ in enumerate(x_labels):
        if i < len(x_labels) - n_cols:
            x_labels[i] = ""
    y_labels = ["Loss"] * len(loss_list)
    for i, _ in enumerate(y_labels):
        if i % n_cols != 0:
            y_labels[i] = ""
    title = "Loss"
    plot_result(y=y, x=x, labels=labels,
                figsize=(14, 8),
                # x_labels=x_labels,
                y_lim_bottom=0.0,
                y_labels=y_labels,
                label_fontsize=6,
                title=title,
                sub_titles=sub_titles, n_cols=n_cols,
                **kwargs)
        
    # plot aoc by problem
    # y_aoc = np.maximum.accumulate(np.array(aoc_list), axis=2)
    # y_log_aoc = np.maximum.accumulate(np.array(log_aoc_list), axis=2)
    # y = np.concatenate([y_log_aoc, y_aoc], axis=1)
    # labels = [log_aoc_labels[i] + aoc_labels[i] for i in range(len(aoc_labels))]

    y = np.maximum.accumulate(np.array(aoc_list), axis=2)
    labels = aoc_labels
    title = "AOC"

    # y = np.maximum.accumulate(np.array(log_aoc_list), axis=2)
    # labels = log_aoc_labels
    # title = "Log AOC"

    base_x = np.arange(0, max_iter+1, dtype=int)
    x = np.tile(base_x, (y.shape[0], y.shape[1], 1))
    sub_titles = [f"F{p_id}" for p_id in range(1, 25)]
    # caption = "AOC\n2nd row: baseline$x=0$"
    plot_result(y=y, x=x, labels=labels,
                figsize=(14, 8),
                # y_lim_bottom=0.0, y_lim_top=1.0,
                sub_titles=sub_titles, n_cols=6,
                title=title,
                label_fontsize=6,
                # caption=caption,
                **kwargs)