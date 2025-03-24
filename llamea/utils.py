import json
import uuid
from datetime import datetime
import logging
import pickle
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
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
    def __init__(self, use_color=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        format_str = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        if use_color:
            grey = "\x1b[38;20m"
            yellow = "\x1b[33;20m"
            red = "\x1b[31;20m"
            bold_red = "\x1b[31;1m"
            reset = "\x1b[0m"
            self.FORMATS = {
                logging.DEBUG: grey + format_str + reset,
                logging.INFO: grey + format_str + reset,
                logging.WARNING: yellow + format_str + reset,
                logging.ERROR: red + format_str + reset,
                logging.CRITICAL: bold_red + format_str + reset
            }
        else:
            self.FORMATS = {
                logging.DEBUG: format_str,
                logging.INFO: format_str,
                logging.WARNING: format_str,
                logging.ERROR: format_str,
                logging.CRITICAL: format_str
            }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(logger = None, level=logging.INFO, filename=None, color=False):
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(level)
    if filename:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter(use_color=color))
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

# Remove Comment
def remove_comments(code):
    result = []
    in_string = False
    string_char = None  # Tracks whether we're in ' or " string
    
    i = 0
    while i < len(code):
        # Handle strings (both single and double quotes)
        if code[i] in ["'", '"']:
            if not in_string:
                in_string = True
                string_char = code[i]
            elif string_char == code[i]:  # Matching quote found
                in_string = False
            result.append(code[i])
                
        # Handle single-line comments
        elif not in_string and code[i] == '#':
            while i < len(code) and code[i] != '\n':
                i += 1
            if i < len(code):
                result.append('\n')
            continue
                
        # Add normal characters
        else:
            result.append(code[i])
            
        i += 1
            
    return ''.join(result)

def remove_empty_lines_in_function(code):
    lines = code.splitlines()
    result = []
    in_function = False
    current_indent = 0
    
    for line in lines:
        # Check if line contains only whitespace
        is_empty = not line.strip()
        # Get the indentation level
        indent = len(line) - len(line.lstrip())
        
        # Detect function start
        if line.lstrip().startswith('def '):
            in_function = True
            current_indent = indent
            result.append(line)
            continue
            
        # If we're in a function
        if in_function:
            # Check if we've exited the function based on indentation
            if line.strip() and indent <= current_indent:
                in_function = False
            
            # Skip empty lines only within function
            if is_empty:
                continue
                
        # Add non-empty lines or lines outside functions
        result.append(line)
    
    return '\n'.join(result)


# Example usage
def test_remove_comments():
    test_code = '''
def hello():
    # This is a single-line comment
    print("Hello World")  # End of line comment
    str_with_hash = "This # is not a comment"
    # Another comment
    return None
'''

    file_pairs = [
        ("Experiments/llm_exp/temperature_[0.0, 1.0, 2.0]_2*3_mut_0224043453/temperature-2.0-r0-1.0_ThompsonSamplingBOv2.py", "Experiments/llm_exp/temperature_[0.0, 1.0, 2.0]_2*3_mut_0224043453/temperature-2.0-r0-1.1_ThompsonSamplingBOv2.py"),
    ]

    code_pairs = []
    for file1, file2 in file_pairs:
        with open(file1, "r") as f:
            code1 = f.read()
        with open(file2, "r") as f:
            code2 = f.read()
        code_pairs.append((code1, code2))

    test_code = code1
    
    cleaned_code = remove_comments(test_code)
    print("Original code:")
    print(test_code)
    print("\nCode with comments removed:")
    print(cleaned_code)
    remove_empty = remove_empty_lines_in_function(cleaned_code)
    print("\nCode with empty lines removed:")
    print(remove_empty)

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

def ceil_with_precision(arr, precision):
    scaling_factor = 10 ** precision
    scaled = arr * scaling_factor
    ceiling = np.ceil(scaled)
    result = ceiling / scaling_factor
    return result

def trunc_with_precision(arr, precision):
    scaling_factor = 10 ** precision
    scaled = arr * scaling_factor
    trunc = np.trunc(scaled)
    result = trunc / scaling_factor
    return result

def density_yscale(y_data_matrix, ranges, densities, precision=0, ytick_interval=None, sub_ytick_interval=None):

    if not isinstance(y_data_matrix, np.ndarray) or y_data_matrix.ndim != 2:
        raise TypeError("y_data_matrix must be a 2D numpy array.")
    if len(ranges) != len(densities):
        raise ValueError("The lengths of ranges and densities must be equal.")

    # Flatten the data for range checks (bounds are based on all data).
    y_data_flat = y_data_matrix.flatten()

    sorted_ranges = sorted(ranges)
    if sorted_ranges[0][0] > np.min(y_data_flat):
        raise ValueError("Ranges do not cover the minimum y-data value.")
    for i in range(len(sorted_ranges) - 1):
        if sorted_ranges[i][1] > sorted_ranges[i + 1][0]:
            raise ValueError("Ranges are overlapping.")
    if sorted_ranges[-1][1] < np.max(y_data_flat):
        raise ValueError("Ranges do not cover the maximum y-data value.")

    for r in ranges:
        if not isinstance(r, tuple) or len(r) != 2:
            raise TypeError("Ranges must be tuples (lower, upper).")
        if r[0] >= r[1]:
            raise ValueError("Lower bound must be less than upper bound.")

    # check desities whether they are normalized
    if not np.isclose(sum(densities), 1.0):
        raise ValueError("Densities must sum to 1.")

    def _add_ytick(yticks, yticklabels, density_range, num, label, min_distance=0.03, subtitute=None, force=False):
        lower, upper, density, cumulative_height = density_range
        value = cumulative_height + (num - lower) / (upper - lower) * density
        if len(yticks) > 0:
            last_val = yticks[-1]
            if value - last_val > min_distance:
                yticks.append(value)
                yticklabels.append(label)
            else:
                if force:
                    yticks.pop()
                    yticklabels.pop()
                    yticks.append(value)
                    yticklabels.append(label)
                elif subtitute is not None:
                    yticks.append(value)
                    yticklabels.append(subtitute)
        else:
            yticks.append(value)
            yticklabels.append(label)

    transformed_y = np.zeros_like(y_data_matrix, dtype=float)
    cumulative_height = 0
    yticks = []
    yticklabels = []
    interval = int((np.max(y_data_flat) - np.min(y_data_flat)) / 5) if ytick_interval is None else ytick_interval
    interval = 1
    sub_interval = interval / 4 if sub_ytick_interval is None else sub_ytick_interval

    for (lower, upper), density in zip(ranges, densities):
        # Create a mask for *all* values within the current range.
        mask = (y_data_matrix >= lower) & (y_data_matrix <= upper)

        transformed_range = (y_data_matrix[mask] - lower) / (upper - lower) * density
        transformed_y[mask] = transformed_range + cumulative_height

        ceil_lower = ceil_with_precision(lower, precision)
        trunc_upper = trunc_with_precision(upper, precision)
        density_range = (lower, upper, density, cumulative_height)

        def _add_non_numerical_yticks(_lower, _upper):
            num_ticks = int((_upper - _lower) / sub_interval) - 1
            for j in range(1, num_ticks + 1): 
                extra_tick = _lower + j * sub_interval
                _add_ytick(yticks, yticklabels, density_range, extra_tick, '')

        _add_ytick(yticks, yticklabels, density_range, ceil_lower, f'{ceil_lower}', force=True)

        n_extra_num_ticks = int((upper - lower) / interval) - 1
        _lower = lower
        for i in range(1, n_extra_num_ticks + 1):
            extra_tick_upper = lower + i * interval
            ceil_tick_upper = ceil_with_precision(extra_tick_upper, precision)
            _add_non_numerical_yticks(_lower, ceil_tick_upper)
            _add_ytick(yticks, yticklabels, density_range, ceil_tick_upper, f'{ceil_tick_upper}', force=True)
            _lower = ceil_tick_upper
        
        _add_non_numerical_yticks(_lower, trunc_upper)

        _add_ytick(yticks, yticklabels, density_range, trunc_upper, f'{trunc_upper}', force=True)

        cumulative_height += density

    tick_locator = FixedLocator(yticks)
    tick_formatter = FixedFormatter(yticklabels)

    return tick_locator, tick_formatter, transformed_y

def determine_ranges_and_densities(y_data_matrix, num_ranges=3, range_bounds=None,  density_threshold_factor=1.0):
    """
    num_ranges: Number of ranges to divide the data into.
    range_bounds: List of bounds for the ranges. If not provided, the bounds are determined automatically based on the number of ranges.
    density_threshold_factor: Controls sensitivity to density differences by powering the relative density value.
    """
    y_data_flat = y_data_matrix.flatten()

    if range_bounds is not None:
        # check if the range bounds are valid
        if len(range_bounds) != num_ranges + 1:
            raise ValueError("range_bounds must have length num_ranges + 1.")
        if range_bounds[0] > np.min(y_data_flat):
            raise ValueError("range_bounds do not cover the minimum y-data value.")
        if range_bounds[-1] < np.max(y_data_flat):
            raise ValueError("range_bounds do not cover the maximum y-data value.")
        for i in range(len(range_bounds) - 1):
            if range_bounds[i] >= range_bounds[i + 1]:
                raise ValueError("Range bounds must be in ascending order.")
    else:
        upper_bound = np.ceil(np.max(y_data_flat))
        lower_bound = np.trunc(np.min(y_data_flat))
        range_bounds = np.linspace(lower_bound, upper_bound, num_ranges + 1)
    ranges = [(range_bounds[i], range_bounds[i + 1]) for i in range(num_ranges)]
        
    densities = []
    for lower, upper in ranges:
        mask = (y_data_matrix >= lower) & (y_data_matrix <= upper)
        # Column density (number of True values in each column).
        counts_per_column = np.sum(mask, axis=0)
        total_count_in_range = np.sum(counts_per_column) #sum all the counts

        #Relative density based on the total number of elements.
        relative_density = total_count_in_range / y_data_matrix.size

        original_density = relative_density * 10
        density = np.power(original_density, density_threshold_factor)
        densities.append(density)

    total_density = sum(densities)
    normalized_densities = [d / total_density for d in densities]

    return ranges, normalized_densities

def test_density_yscale():
    """
    Tests the custom y-scale with a 2D data matrix.
    """
    # Create some sample 2D data.  Multiple lines.
    num_lines = 3
    num_points = 100
    x = np.linspace(0, 10, num_points)
    y_data_matrix = np.zeros((num_lines, num_points))

    # Line 1: Centered around y=2, dense
    y_data_matrix[0, :] = np.random.normal(2, 0.5, num_points)
    # Line 2: Centered around y=8, less dense
    y_data_matrix[1, :] = np.random.normal(8, 1.5, num_points)
    # Line 3: Starts low, then jumps high
    y_data_matrix[2, :num_points // 2] = np.random.normal(1, 0.3, num_points // 2)
    y_data_matrix[2, num_points // 2:] = np.random.normal(9, 0.8, num_points // 2)

    linear_y_data_matrix = np.zeros((num_lines, num_points))
    for i in range(num_lines):
        linear_y_data = np.linspace(0, 6 * i+1, num_points)
        linear_y_data_matrix[i, :] = linear_y_data

    
    fig, axs = plt.subplots(2,2, figsize=(14, 9))

    #determine ranges and densities
    ax = axs[0, 0]
    ranges, densities = determine_ranges_and_densities(linear_y_data_matrix, num_ranges=4, density_threshold_factor=1.5)
    tick_locator, tick_formatter, transformed_y = density_yscale(linear_y_data_matrix, ranges, densities)
    ax.yaxis.set_major_locator(tick_locator)
    ax.yaxis.set_major_formatter(tick_formatter)
    for i in range(num_lines):
        ax.plot(x, transformed_y[i, :], label=f"Line {i+1}") # Plot each transformed line
    ax.set_title("Custom Y-Scale (2D Data, Fixed Ranges)")
    ax.set_xlabel("X")
    ax.set_ylabel("Transformed Y")
    ax.legend()

    # Original scale for comparison.
    ax = axs[0, 1]
    for i in range(num_lines):
        ax.plot(x, linear_y_data_matrix[i, :], label=f"Line {i+1}")
    ax.set_title("Original Y-Scale")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # determine ranges and densities
    ax = axs[1, 0]
    ranges, densities = determine_ranges_and_densities(y_data_matrix, num_ranges=4, density_threshold_factor=1.5)
    tick_locator, tick_formatter, transformed_y = density_yscale(y_data_matrix, ranges, densities)
    ax.yaxis.set_major_locator(tick_locator)
    ax.yaxis.set_major_formatter(tick_formatter)
    for i in range(num_lines):
        ax.plot(x, transformed_y[i, :], label=f"Line {i+1}") # Plot each transformed line
    ax.set_title("Custom Y-Scale (2D Data, Fixed Ranges)")
    ax.set_xlabel("X")
    ax.set_ylabel("Transformed Y")
    ax.legend()

     # Original scale for comparison.
    ax = axs[1, 1]
    for i in range(num_lines):
         ax.plot(x, y_data_matrix[i, :], label=f"Line {i+1}")
    ax.set_title("Original Y-Scale")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_lines(y:list[np.ndarray], x:list[np.ndarray],

                labels:list[list[str]],
                label_fontsize:int = 7,
                combined_legend:bool = False,

                filling:list[np.ndarray]=None, 
                linewidth:float = 1.0,

                colors:list[list]=None,
                line_styles:list[list]=None,
                
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
    fig, axs = plt.subplot_mosaic(axs_ids, figsize=figsize, sharex=True)
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
        _linestyles = line_styles[i] if line_styles is not None else None
        for j in range(_y.shape[0]):
            label = _labels[j] if len(_labels) > j else f"{j}"
            _color = _colors[j] if _colors is not None and len(_colors) > j else None
            _linestyle = _linestyles[j] if _linestyles is not None and len(_linestyles) > j else None
            ax.plot(_x, _y[j,:], label=label, linewidth=linewidth, color=_color, linestyle=_linestyle)

            _color = _color if _color is not None else ax.get_lines()[-1].get_color()
            if _filling is not None:
                upper, lower = _filling[j]
                ax.fill_between(_x, lower, upper, alpha=0.2, color=_color)

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
                ax.scatter(_dot_x, _dot_y, facecolors='none', edgecolors=_color, s=linewidth*60, linewidths=linewidth+0.5)
            
        _baseline = baselines[i] if baselines is not None else None
        if _baseline is not None:
            _bl_labels = baseline_labels[i] if len(baseline_labels) > i else []
            for j, base in enumerate(_baseline):
                if base is None:
                    continue
                label = _bl_labels[j] if len(_bl_labels) > j else f"{j}"
                ax.axhline(y=base, label=label, linestyle="--", color="black", linewidth=linewidth, alpha=0.6)

        if combined_legend is False:
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

    fig.tight_layout()

    if combined_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=10, fontsize=label_fontsize, bbox_to_anchor=(0.5, 0.0))
        plt.subplots_adjust(bottom=0.07) 

    
    if filename:
        _name = filename + ".png"
        plt.savefig(_name)
        __name = filename + ".pdf"
        plt.savefig(__name)


    if show:
        plt.show()

def _plot_get_element_from_list(data, index, default=None):
    if isinstance(data, list) and len(data) > index:
        return data[index]
    return default

def plot_group_bars(
    data:list[np.ndarray],
    labels: list[list[str]],
    group_labels: list[list[str]] = None,
    y_label: list[str] = None,
    sub_titles: list[str] = None,
    title:str = None,
    label_fontsize:int = 10,
    n_cols:int = 1,
    fig_size:tuple[int,int] = (10, 6),
    ):

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
    fig, axs = plt.subplot_mosaic(axs_ids, figsize=fig_size)
    for i in range(n_plots):
        row = i // n_cols
        col = i % n_cols
        ax = axs[i]
        _sub_title = sub_titles[i] if sub_titles is not None else ""
        _y_label = y_label[i] if y_label is not None else ""
        _group_labels = group_labels[i] if group_labels is not None else None
        _group_labels = [_label[:10] for _label in _group_labels]
        _labels = labels[i]
        _data = data[i].T

        n_groups = _data.shape[0]
        n_bars = _data.shape[1]
        x = np.arange(n_bars)
        width = 1/(n_groups+1)
        for i in range(n_groups):
            ax.bar(x + i * width, _data[i], width, label=_labels[i])
        ax.set_xticks(x + width * (n_groups - 1) / 2, labels=_group_labels, fontsize=label_fontsize)
        ax.legend()
        ax.set_title(_sub_title)
        ax.set_ylabel(_y_label)
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def test_group_bar():
    n_groups = 3
    data = [ ]
    for i in range(n_groups):
        data.append(np.random.rand(4))
    data = np.array(data)
        
    group_labels = ["A", "B", "C"]
    labels = ["G1", "G2", "G3", "G4"]
    plot_group_bars(data, labels, group_labels)

def plot_box_violin(
    data:list[np.ndarray],
    labels: list[list[str]], 
    label_fontsize:int = 9,
    sub_titles: list[str] = None,
    x_labels: list[str] = None,
    y_labels: list[str] = None,
    title = "",
    colors: list[list] = None,
    show_inside_box:bool = False,
    show_scatter:bool = False,
    n_cols:int = 1, figsize:tuple[int,int] = (10, 6), 
    show:bool = True,
    filename=None):

    if len(labels) != len(data):
        logging.warning("PLOT:Number of labels does not match the number of plots.")

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
        if show_inside_box or show_scatter:
            _violin_parts = ax.violinplot(data[i], showmeans=False, showmedians=False, showextrema=False, widths=0.8)
        else:
            _violin_parts = ax.violinplot(data[i], showmeans=False, showmedians=True, showextrema=True, widths=0.8)

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
        elif show_scatter:
            for _scatter_i, scatter in enumerate(data[i]):
                _x = np.full(len(scatter), _scatter_i + 1)
                ax.scatter(_x, scatter, color=box_colors[_scatter_i], alpha=0.5)

        ax.set_title(sub_title)
        ax.tick_params(axis='y', labelsize=label_fontsize+1)
        ax.yaxis.grid(True)

        _labels = _plot_get_element_from_list(labels, i, None)
        if _labels is not None:
            ax.set_xticks([y + 1 for y in range(len(data[i]))], labels=_labels, fontsize=label_fontsize)
        _x_labels = _plot_get_element_from_list(x_labels, i, "")
        ax.set_xlabel(_x_labels, fontsize=label_fontsize)
        _y_labels = _plot_get_element_from_list(y_labels, i, "")
        ax.set_ylabel(_y_labels, fontsize=label_fontsize)

    fig.suptitle(title, fontsize=label_fontsize+2)
    fig.tight_layout()
    if filename:
        _name = filename + ".png"
        plt.savefig(_name)

        _name = filename + ".pdf"
        plt.savefig(_name)
    if show:
        plt.show()

def plot_voilin_style_scatter(
        data:list[np.ndarray],
        labels: list[list[str]],
        label_fontsize:int = 9,
        sub_titles: list[str] = None,
        x_labels: list[str] = None,
        y_labels: list[str] = None,
        title = "",
        y_lim:tuple[float,float] = None,
        margin:float = 0.5,
        colors: list[list] = None,
        n_cols:int = 1, figsize:tuple[int,int] = (10, 6), 
        show:bool = True,
        filename=None):
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

        _data = data[i]
        _labels = labels[i]
        _x_labels = x_labels[i] if x_labels is not None else ""
        _y_labels = y_labels[i] if y_labels is not None else ""
        _sub_title = sub_titles[i] if sub_titles is not None else ""
        _colors = colors[i] if colors is not None else None

        for j, d in enumerate(_data):
            _x = np.full(len(d), j)
            _color = _colors[j] if _colors is not None and len(_colors) > j else None
            ax.scatter(_x, d, label=_labels[j], color=_color)

        ax.set_xticks(range(len(_data)), labels=_labels, fontsize=label_fontsize)
        ax.set_xlabel(_x_labels)
        ax.set_ylabel(_y_labels)
        ax.set_title(_sub_title)
        ax.yaxis.grid(True)
        ax.set_xlim(-margin, len(_data) - 1 + margin)
        ax.set_ylim(y_lim)

    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()
