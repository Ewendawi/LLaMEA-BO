import json
import uuid
from datetime import datetime
import logging
import pickle
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter 
from scipy.ndimage import gaussian_filter1d  
import pandas as pd
from pandas.api.types import is_numeric_dtype
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
            individual.metadata["prompt"] = handler.prompt

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

# Moving Average Smoothing
def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    if len(data.shape) == 1:
        return np.convolve(data, window, mode='same')
    else:
        return np.array([np.convolve(data[i], window, mode='same') for i in range(data.shape[0])])


# Savitzky-Golay Filter (preserves peaks)
def savgol_smoothing(data, window_size, polyorder):
    """
    window_size: The length of the filter window. Must be odd.
    polyorder: The order of the polynomial used to fit the samples. Must be less than window_size.
    """
    try:
        return savgol_filter(data, window_size, polyorder)
    except ValueError as e:
        print(f"Savitzky-Golay error: {e}.  Ensure window_size is odd and larger than polyorder.")
        return data  # Return original if error


# Gaussian Smoothing (Good for general smoothing)
def gaussian_smoothing(data, sigma):
    """
    sigma: The standard deviation of the Gaussian kernel.  Larger sigma = more smoothing.
    """
    return gaussian_filter1d(data, sigma)


def plot_result(y:list[np.ndarray], x:list[np.ndarray],

                labels:list[list[str]],
                label_fontsize:int = 7,

                filling:list[np.ndarray]=None, 
                linewidth:float = 1.0,

                colors:list[list]=None,
                
                y_scales:list[tuple[str, dict]]=None,

                x_dot:list[np.ndarray]=None,

                x_labels:list[str]=None, y_labels:list[str]=None, 

                sub_titles:list[str]=None,
                sub_title_fontsize:int = 10,

                baselines:np.ndarray=None, baseline_labels:list[list[str]]=None, 

                title:str = None,
                title_fontsize:int = 12, 
                caption:str = None, caption_fontsize:int = 10,

                filename:str = None, 
                n_cols:int = 1, figsize:tuple[int,int] = (10, 6), 
                show:bool = True):
    
    # y.shape = (n_plots, n_lines, n_points)
    if len(labels) != len(y):
        logging.warning("PLOT:Number of labels does not match the number of plots.")
    
    if x_labels is not None and len(x_labels) != len(y):
        logging.warning("PLOT:Number of x_labels does not match the number of plots.")
    
    if y_labels is not None and len(y_labels) != len(y):
        logging.warning("PLOT:Number of y_labels does not match the number of plots.")

    if sub_titles is not None and len(sub_titles) != len(y):
        logging.warning("PLOT:Number of sub_titles does not match the number of plots.")

    n_plots = len(y)
    n_cols = min(n_cols, n_plots)
    n_rows = n_plots // n_cols 
    if n_plots % n_cols != 0:
        n_rows += 1

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

        _x = x[i]
        _y = y[i]
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

        if y_scales is not None and len(y_scales) > i and y_scales[i] is not None:
            scale, scale_kwargs = y_scales[i] 
            ax.set_yscale(scale, **scale_kwargs)

        _labels = labels[i] if len(labels) > i else []
        _filling = filling[i] if filling is not None else None
        _x_dot = x_dot[i] if x_dot is not None else None
        _colors = colors[i] if colors is not None else None
        for j in range(_y.shape[0]):
            label = _labels[j] if len(_labels) > j else f"{j}"
            _color = _colors[j] if _colors is not None and len(_colors) > j else None
            ax.plot(_x, _y[j,:], label=label, linewidth=linewidth, color=_color)

            if _filling is not None:
                upper, lower = _filling[j]
                ax.fill_between(_x, lower, upper, alpha=0.3, color=_color)

            if _x_dot is not None and len(_x_dot) > j:
                _dot_x = _x_dot[j]
                _dot_y = _y[j][_dot_x]
                # if _dot_y is nan, look forward
                _step = 0
                _len = min(10, len(_y[j]) - _dot_x)
                while _step < _len and np.isnan(_dot_y):
                    _dot_y = _y[j][_dot_x + _step]
                    _step += 1
                # get the color from the line
                # color = ax.get_lines()[-1].get_color()
                _dot_x = _dot_x.astype(np.float64) + np.random.uniform(-0.2, 0.2, len(_dot_x))
                ax.scatter(_dot_x, _dot_y, facecolors='none', edgecolors=_color)
            
        _baseline = baselines[i] if baselines is not None else None
        if _baseline is not None:
            _bl_labels = baseline_labels[i] if len(baseline_labels) > i else []
            for j, base in enumerate(_baseline):
                label = _bl_labels[j] if len(_bl_labels) > j else f"{j}"
                ax.axhline(y=base, label=label, linestyle="--", color="black", linewidth=linewidth, alpha=0.6)

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
        plt.show(block=False)

def _plot_get_element_from_list(data, index, default=None):
    if isinstance(data, list) and len(data) > index:
        return data[index]
    return default

def plot_box_violin(
    data:list[np.ndarray],
    labels: list[list[str]], 
    long_labels: list[str] = None,
    sub_titles: list[str] = None,
    x_labels: list[str] = None,
    y_labels: list[str] = None,
    title = "", 
    plot_type:str = "violin",
    n_cols:int = 1, figsize:tuple[int,int] = (10, 6), 
    show:bool = True,
    filename=None):

    if len(labels) != len(data):
        logging.warning("PLOT:Number of labels does not match the number of plots.")
    if long_labels is not None and len(long_labels) != len(data):
        logging.warning("PLOT:Number of long_labels does not match the number of plots.")

    n_plots = len(data)
    n_cols = min(n_cols, n_plots)
    n_rows = n_plots // n_cols
    if n_plots % n_cols != 0:
        n_rows += 1

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

        sub_title = sub_titles[i] if sub_titles is not None else ""
        if plot_type == "violin":
            ax.violinplot(data[i], showmeans=False, showmedians=True)
        elif plot_type == "box":
            ax.boxplot(data[i])
        ax.set_title(sub_title)
        ax.yaxis.grid(True)
        _labels = _plot_get_element_from_list(labels, i, None) 
        if _labels is not None:
            ax.set_xticks([y + 1 for y in range(len(data[i]))], labels=_labels)
        _x_labels = _plot_get_element_from_list(x_labels, i, "")
        ax.set_xlabel(_x_labels)
        _y_labels = _plot_get_element_from_list(y_labels, i, "")
        ax.set_ylabel(_y_labels)
    
    fig.suptitle(title)
    if filename:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show(block=False)
        

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

def plot_algo_results(results:list[EvaluatorResult], **kwargs):
    
    def dynamical_access(obj, attr_path):
        attrs = attr_path.split(".")
        target = obj
        for attr in attrs:
            target = getattr(target, attr, None)
            if target is None:
                break
        return target
            
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
        algo = result.name.removeprefix("BL")
        for res in result.result:
            row = res_to_row(res, algo)
            if row is not None:
                res_df.loc[len(res_df)] = row

    # hanle aoc
    def _plot_aoc():
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
            short_labels.append([label[:10] for label in _labels])

        labels = short_labels

        # plot aoc
        # plot_box_violin(data=aoc_plot_data, 
        #                 labels=labels, 
        #                 sub_titles=sub_titles, 
        #                 title="AOC",
        #                 plot_type="violin", 
        #                 n_cols=4,
        #                 figsize=(14, 8),
        #                 **kwargs)

        # # plot log aoc
        plot_box_violin(data=log_plot_data,
                        labels=labels,
                        sub_titles=sub_titles,
                        title="Log AOC",
                        plot_type="violin",
                        n_cols=4,
                        figsize=(14, 8),
                        **kwargs)
    
    _plot_aoc()

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


        plot_result(
            y=plot_data, x=x_data, 
            y_scales=y_scales,
            baselines=baselines,
            baseline_labels=baseline_labels,
            colors=colors,
            labels=labels, 
            label_fontsize=6,
            linewidth=1.0,
            filling=plot_filling,
            x_dot=x_dots,
            n_cols=5,
            sub_titles=sub_titles,
            sub_title_fontsize=9,
            title=f"F{problem_id}",
            figsize=(15, 9),
            **kwargs
        ) 
