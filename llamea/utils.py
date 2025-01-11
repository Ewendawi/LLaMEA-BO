import json
import uuid
from datetime import datetime
import logging
import pickle
import os
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
        self.log_extract_error = False
        self.should_log_population = True
        self.should_log_experiment = True
        self.auto_save = False
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
            # for _, ind in logger.individual_map.items():
            #     if 'result_values' in ind.metadata:
            #         res = ind.metadata['result_values']
            #         if res.surrogate_model_losses is not None:
            #             res.surrogate_model_losses = np.nan_to_num(res.surrogate_model_losses)
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
