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
from .population.population import Population 
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


def plot_lines(y:list[np.ndarray], x:list[np.ndarray],

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

            _color = _color if _color is not None else ax.get_lines()[-1].get_color()
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

def plot_group_bar(
    data:np.ndarray,
    labels: list[str],
    group_labels: list[str] = None,
    label_fontsize:int = 10,
    fig_size:tuple[int,int] = (10, 6),
    title = ""):

    data = data.T

    group_labels = [_label[:10] for _label in group_labels]

    n_groups = data.shape[0]
    n_bars = data.shape[1]
    x = np.arange(n_bars)
    width = 1/(n_groups+1)
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(n_groups):
        ax.bar(x + i * width, data[i], width, label=labels[i])
    ax.set_xticks(x + width * (n_groups - 1) / 2, labels=group_labels, fontsize=label_fontsize)
    ax.legend()
    ax.set_title(title)
    plt.show(block=False)

def test_group_bar():
    n_groups = 3
    data = [ ]
    for i in range(n_groups):
        data.append(np.random.rand(4))
    data = np.array(data)
        
    group_labels = ["A", "B", "C"]
    labels = ["G1", "G2", "G3", "G4"]
    plot_group_bar(data, labels, group_labels)

def plot_box_violin(
    data:list[np.ndarray],
    labels: list[list[str]], 
    long_labels: list[str] = None,
    label_fontsize:int = 7,
    sub_titles: list[str] = None,
    x_labels: list[str] = None,
    y_labels: list[str] = None,
    title = "", 
    plot_type:str = "violin",
    colors: list[list] = None,
    show_inside_box:bool = False,
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
        _colors = colors[i] if colors is not None else None

        sub_title = sub_titles[i] if sub_titles is not None else ""
        if show_inside_box:
            _violin_parts = ax.violinplot(data[i], showmeans=False, showmedians=False, showextrema=False)
        else:
            _violin_parts = ax.violinplot(data[i], showmeans=False, showmedians=True, showextrema=True)

        box_colors = []
        for _pc_i, pc in enumerate(_violin_parts['bodies']):
            _color = _colors[_pc_i] if _colors is not None and len(_colors) > _pc_i else None
            if _color is not None:
                pc.set_facecolor(_color)

            _color1 = pc.get_facecolor()
            box_colors.append(_color1)

        _box_parts = None
        if show_inside_box:
            medianprops = dict(linestyle='-', linewidth=1.5, color='black') # General median prop
            _box_parts = ax.boxplot(data[i],
                                    positions=range(1, len(data[i]) + 1),
                                    whis=[5,95],
                                    medianprops=medianprops,
                                    showmeans=False, showfliers=False,
                                    widths=0.2,
                                    patch_artist=True)

            for _box_i, box in enumerate(_box_parts['boxes']):
                _color = box_colors[_box_i]
                # _color = mcolors.to_rgba(_color)
                _ori_alpha = _color[0,3]
                _color[:,3] = 1.0
                
                box.set_facecolor(_color)
                box.set_edgecolor(_color)  
                box.set_alpha(_ori_alpha+0.1)         

                _median = _box_parts['medians'][_box_i]
                _median.set_color(_color) 

                _cap1 = _box_parts['caps'][_box_i*2]
                _cap1.set_color(_color) 
                _cap2 = _box_parts['caps'][_box_i * 2 + 1]
                _cap2.set_color(_color) 

                _whisker1 = _box_parts['whiskers'][_box_i * 2]
                _whisker1.set_color(_color) 
                _whisker2 = _box_parts['whiskers'][_box_i * 2 + 1]
                _whisker2.set_color(_color) 

        ax.set_title(sub_title)
        ax.yaxis.grid(True)

        _labels = _plot_get_element_from_list(labels, i, None) 
        if _labels is not None:
            ax.set_xticks([y + 1 for y in range(len(data[i]))], labels=_labels, fontsize=label_fontsize)
        _x_labels = _plot_get_element_from_list(x_labels, i, "")
        ax.set_xlabel(_x_labels, fontsize=label_fontsize)
        _y_labels = _plot_get_element_from_list(y_labels, i, "")
        ax.set_ylabel(_y_labels, fontsize=label_fontsize)

    fig.suptitle(title)
    fig.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
