import os
import pickle
import json
import logging
import warnings
import random
import time
from functools import cmp_to_key
import pandas as pd
import numpy as np
import functools
from collections.abc import Callable
from Benchmarks.LLAMBO.bayesmark.bbox_utils import get_bayesmark_func
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

from llamevol.utils import setup_logger
from llamevol.utils import plot_lines

def update_init(self, init_X):
    self.init_X = init_X

def init_profiler(self):
    self.fitting_times = []
    self.selecting_times = []
    self.optimizing_times = []
    self.opt_start_time = None

def func_wrapper(func):
    functools.wraps(func)

    def wrapper(self, *args, **kwargs):
        if func.__name__ == "_sample_points":
            if self.n_evals == 0 and self.init_X is not None:
                return self.init_X
            else:
                return func(self, *args, **kwargs)
        elif func.__name__ == "__init__":
            func(self, *args, **kwargs)
            self.init_X = None
        elif func.__name__ == "_fit_model":
            start_time = time.perf_counter()
            gp = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            if len(self.fitting_times) < self.n_evals - 1:
                self.fitting_times.extend([0] * (self.n_evals - len(self.fitting_times)))
            self.fitting_times.append(end_time - start_time)
            return gp
        elif func.__name__ == "_select_next_point" or func.__name__ == "_select_next_points" or func.__name__ == "_local_search":
            start_time = time.perf_counter()
            next_points = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            if len(self.selecting_times) < self.n_evals :
                self.selecting_times.extend([0] * (self.n_evals - len(self.selecting_times)))
            self.selecting_times.append(end_time - start_time)
            return next_points
        elif func.__name__ == "_evaluate_points":
            start_time = time.perf_counter()
            fx = func(self, *args, **kwargs)
            if self.opt_start_time is not None:
                opt_time = start_time  - self.opt_start_time
                self.optimizing_times.extend([opt_time] * (self.n_evals - len(self.optimizing_times)))
            self.opt_start_time = time.perf_counter()
            return fx
        else:
            return func(self, *args, **kwargs)
    return wrapper

def bayesmarkBO_wrapper(cls, is_profile=False):
    setattr(cls, '__init__', func_wrapper(cls.__init__))
    setattr(cls, 'update_init', update_init)
    if hasattr(cls, '_sample_points'):
        setattr(cls, '_sample_points', func_wrapper(cls._sample_points))

    if is_profile:
        setattr(cls, 'init_profiler', init_profiler)
        if hasattr(cls, '_fit_model'):
            setattr(cls, '_fit_model', func_wrapper(cls._fit_model))

        if hasattr(cls, '_select_next_point'):
            setattr(cls, '_select_next_point', func_wrapper(cls._select_next_point))
        
        if hasattr(cls, '_select_next_points'):
            setattr(cls, '_select_next_points', func_wrapper(cls._select_next_points))

        if hasattr(cls, '_local_search'):
            setattr(cls, '_local_search', func_wrapper(cls._local_search))

        if hasattr(cls, '_evaluate_points'):
            setattr(cls, '_evaluate_points', func_wrapper(cls._evaluate_points))
    return cls


logger = logging.getLogger(__name__)

def setup_logging(log_name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

BAYESMARK_TASK_MAP = {
    'breast': ['classification', 'accuracy'],
    'digits': ['classification', 'accuracy'],
    'wine': ['classification', 'accuracy'],
    'iris': ['classification', 'accuracy'],
    'diabetes': ['regression', 'neg_mean_squared_error'],
}

PRIVATE_TASK_MAP = {
    'Griewank': ['regression', 'neg_mean_squared_error'],
    'KTablet': ['regression', 'neg_mean_squared_error'],
    'Rosenbrock': ['regression', 'neg_mean_squared_error'],

    'griewank': ['regression', 'neg_mean_squared_error'],
    'ktablet': ['regression', 'neg_mean_squared_error'],
    'rosenbrock': ['regression', 'neg_mean_squared_error'],
}

class BayesmarkExpRunner:
    def __init__(self, task_context, dataset, seed, use_log=False):
        self.seed = seed
        self.model = task_context['model']
        self.task = task_context['task']
        self.metric = task_context['metric']
        self.dataset = dataset
        self.hyperparameter_constraints = task_context['hyperparameter_constraints']
        self.bbox_func = get_bayesmark_func(self.model, self.task, dataset['test_y'])

        self.ordered_hyperparams = list(self.hyperparameter_constraints.keys())
        self.bounds = [] # the original bounds
        self.search_bounds = [] # the bounds in the search space(linear space)
        self.space_types = []
        self.is_int = []
        lower_bounds = []
        search_lower_bounds = []
        upper_bounds = []
        search_upper_bounds = []
        for hyperparam in self.ordered_hyperparams:
            constraint = self.hyperparameter_constraints[hyperparam]

            _space_type = 'linear'
            if use_log:
                _space_type = constraint[1]
            self.space_types.append(_space_type)
            lower_bounds.append(constraint[2][0])
            upper_bounds.append(constraint[2][1])

            search_lower_bounds.append(self._space_type_to_linear(_space_type, constraint[2][0]))
            search_upper_bounds.append(self._space_type_to_linear(_space_type, constraint[2][1]))

            self.is_int.append(constraint[0] == 'int')

        self.bounds = np.array([lower_bounds, upper_bounds])
        self.search_bounds = np.array([search_lower_bounds, search_upper_bounds])

        dim = len(self.ordered_hyperparams)
        self.x_bounds = np.array([[-1.0]*dim, [1.0]*dim])
        
        self.is_maximization = False
        self.x_hist = []
        self.fvals_hist = []

        self.is_profile = False
        self.bo_instance = None
        self.optimizing_times = []
        self.opt_start_time = None


    def enable_time_recording(self, bo_instance):
        self.is_profile = True
        self.bo_instance = bo_instance
        self.optimizing_times = []
        self.opt_start_time = None

    def _map_bounds(self, x, bounds, new_bounds):
        '''
        Map x from bounds to new_bounds
        Args: x (numpy array), bounds (numpy array), new_bounds (numpy array)
        Returns: x (numpy array)
        '''
        scaled_x = new_bounds[0] + (x - bounds[0]) * (new_bounds[1] - new_bounds[0]) / (bounds[1] - bounds[0])
        return scaled_x

    def _space_type_to_linear(self, space_type, x):
        '''
        Convert space type to linear space
        Args: space_type (str), x (numpy array)
        Returns: x (numpy array)
        '''
        if space_type == 'log':
            return np.log(x)
        elif space_type == 'logit':
            return np.log(x / (1 - x))
        else:
            return x
    
    def _linear_to_space_type(self, space_type, x):
        '''
        Convert linear space to space type
        Args: space_type (str), x (numpy array)
        Returns: x (numpy array)
        '''
        if space_type == 'log':
            return np.exp(x)
        elif space_type == 'logit':
            return 1 / (1 + np.exp(-x))
        else:
            return x

    def x_to_config(self, x):
        '''
        Convert x (numpy array) to config (dictionary)
        Args: x (numpy array)
        Returns: config (dictionary)
        '''
        config = {}
        for i, hyperparam in enumerate(self.ordered_hyperparams):
            x_bound = self.x_bounds[:,i]
            config_bound = self.search_bounds[:,i]
            config_x = self._map_bounds(x[i], x_bound, config_bound)

            config_x = self._linear_to_space_type(self.space_types[i], config_x)
            if self.is_int[i]:
                config_x = int(np.round(config_x))

            config[hyperparam] = config_x
        return config

    def config_to_x(self, config):
        '''
        Convert config (dictionary) to x (numpy array)
        Args: config (dictionary)
        Returns: x (numpy array)
        '''
        x = []
        config_x = []
        for hyperparam in self.ordered_hyperparams:
            config_x.append(config[hyperparam])

        for i, _x in enumerate(config_x):
            x.append(self._space_type_to_linear(self.space_types[i], _x))
        
        x = self._map_bounds(x, self.search_bounds, self.x_bounds)

        return np.array(x)
    
    def generate_initialization(self, n_samples):
        '''
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: list of dictionaries, each dictionary is a point to be evaluated
        '''

        # Read from fixed initialization points (all baselines see same init points)
        init_configs = pd.read_json(f'Benchmarks/LLAMBO/bayesmark/configs/{self.model}/{self.seed}.json').head(n_samples)
        init_configs = init_configs.to_dict(orient='records')

        assert len(init_configs) == n_samples

        return init_configs

    def obj_func_wrapper(self, x):
        '''
        Wrapper function for bbox_func
        Args: x (numpy array)
        '''

        if self.is_profile and self.bo_instance is not None :
            if self.opt_start_time is None:
                self.optimizing_times.append(0)
            else:
                opt_time = time.perf_counter() - self.opt_start_time
                if opt_time > 1e-4:
                    self.optimizing_times.append(opt_time)
                else:
                    self.optimizing_times.append(self.optimizing_times[-1])

        should_reshape = False
        if len(x.shape) == 2:
            if x.shape[0] == 1:
                x = x[0]
                should_reshape = True

        config = self.x_to_config(x)
        _, fvals = self.evaluate_point(config)
        self.x_hist.append(config)
        self.fvals_hist.append(fvals)
        fx = fvals['minimize_objective']
        if should_reshape:
            fx = np.array([fx])
        if self.is_maximization:
            fx = -fx

        if self.is_profile : 
            self.opt_start_time = time.perf_counter()
 
        return fx
        
    def evaluate_point(self, candidate_config):
        '''
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        fvals can contain an arbitrary number of items, but also must contain 'score' (which is what LLAMBO optimizer tries to optimize)
        fvals = {
            'score': float,                     -> 'score' is what the LLAMBO optimizer tries to optimize
            'generalization_score': float
        }
        '''
        np.random.seed(self.seed)
        random.seed(self.seed)

        X_train, X_test, y_train, y_test = self.dataset['train_x'], self.dataset['test_x'], self.dataset['train_y'], self.dataset['test_y']

        for hyperparam, value in candidate_config.items():
            if self.hyperparameter_constraints[hyperparam][0] == 'int':
                candidate_config[hyperparam] = int(value)

        if self.task == 'regression':
            mean_ = np.mean(y_train)
            std_ = np.std(y_train)
            y_train = (y_train - mean_) / std_
            y_test = (y_test - mean_) / std_

        model = self.bbox_func(**candidate_config)
        scorer = get_scorer(self.metric)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            S = cross_val_score(model, X_train, y_train, scoring=scorer, cv=5)
        cv_score = np.nanmean(S)
        
        model = self.bbox_func(**candidate_config) 
        model.fit(X_train, y_train)
        generalization_score = scorer(model, X_test, y_test)

        minimize_objective = -cv_score
        if self.metric == 'neg_mean_squared_error':
            cv_score = -cv_score
            generalization_score = -generalization_score

        return candidate_config, {'score': cv_score, 'generalization_score': generalization_score, 'minimize_objective': minimize_objective}


def _run_bayesmark_exp(bo_cls, dataset, model, num_seeds = 5, budget = 30, n_initial_samples = 5, use_log = False, is_profile=False):
    if dataset is None:
        raise ValueError("dataset must be a string")
    if model is None:
        raise ValueError("model must be a string")
    
    if num_seeds is not None and isinstance(num_seeds, int) and (num_seeds < 1 or num_seeds > 10):
        raise ValueError("num_seeds must be an integer between 1 and 10")

    algo_cls = bo_cls
    algo = algo_cls.__name__
    if use_log:
        algo = f'{algo}_log'

    # Load training and testing data
    if dataset in BAYESMARK_TASK_MAP:
        TASK_MAP = BAYESMARK_TASK_MAP
        pickle_fpath = f'Benchmarks/LLAMBO/bayesmark/data/{dataset}.pickle'
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)
        X_train = data['train_x']
        y_train = data['train_y']
    elif dataset in PRIVATE_TASK_MAP: 
        TASK_MAP = PRIVATE_TASK_MAP
        pickle_fpath = f'Benchmarks/LLAMBO/custom_dataset/{dataset}.pickle'
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)
        X_train = data['train_x']
        y_train = data['train_y']
    else:
        raise ValueError(f'Invalid dataset: {dataset}')


    # Describe task context
    task_context = {}
    task_context['model'] = model
    task_context['task'] = TASK_MAP[dataset][0]
    task_context['tot_feats'] = X_train.shape[1]
    task_context['cat_feats'] = 0       # bayesmark datasets only have numerical features
    task_context['num_feats'] = X_train.shape[1]
    task_context['n_classes'] = len(np.unique(y_train))
    task_context['metric'] = TASK_MAP[dataset][1]
    task_context['lower_is_better'] = True if 'neg' in task_context['metric'] else False
    task_context['num_samples'] = X_train.shape[0]
    with open('Benchmarks/LLAMBO/hp_configurations/bayesmark.json', 'r') as f:
        task_context['hyperparameter_constraints'] = json.load(f)[model]

    # define result save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_res_dir = f'{script_dir}/bayesmark_results/{dataset}/{model}'
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    # define logging directory
    logging_fpath = f'{script_dir}/bayesmark_logs/{dataset}/{model}_{algo}.log'
    if not os.path.exists(os.path.dirname(logging_fpath)):
        os.makedirs(os.path.dirname(logging_fpath))
    setup_logging(logging_fpath)
    
    for seed in range(num_seeds):
        logger.info('Executing %s to tune %s on %s with seed %d / %d and use_log %d...', algo, model, dataset, seed+1, num_seeds, use_log)
        logger.info('Task context: %s', task_context)

        benchmark = BayesmarkExpRunner(task_context, data, seed, use_log=use_log)

        dim = len(benchmark.ordered_hyperparams)
        init_config = benchmark.generate_initialization(n_initial_samples)
        init_X = []
        for config in init_config:
            x = benchmark.config_to_x(config)
            init_X.append(x)
        init_X = np.array(init_X)

        # instantiate algorithm
        bo_instance = algo_cls(budget=budget, dim=dim)
        if hasattr(bo_instance, 'n_init'):
            bo_instance.n_init = n_initial_samples
        bo_instance.update_init(init_X)
        bo_instance.bounds = benchmark.x_bounds
        if hasattr(bo_instance, 'init_profiler'):
            bo_instance.init_profiler()

        if hasattr(bo_instance, 'is_maximization'):
            benchmark.is_maximization = bo_instance.is_maximization()

        if is_profile:
            benchmark.enable_time_recording(bo_instance)

        # run optimization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            bo_instance(func=benchmark.obj_func_wrapper)

        configs, fvals = benchmark.x_hist, benchmark.fvals_hist

        if is_profile:
            fitting_times = bo_instance.fitting_times
            selecting_times = bo_instance.selecting_times
            optimizing_times = bo_instance.optimizing_times
            if len(optimizing_times) == 0:
                optimizing_times = benchmark.optimizing_times

            cols = ['model', 'dataset', 'algo', 'fitting_time', 'selecting_time', 'optimizing_time', 'iteration', 'seed']
            df_times = pd.DataFrame(columns=cols)

            for i, _ in enumerate(optimizing_times): 
                fitting_time = 0
                if i < len(fitting_times):
                    fitting_time = fitting_times[i]
                selecting_time = 0
                if i < len(selecting_times):
                    selecting_time = selecting_times[i]
                optimizing_time = optimizing_times[i]
                iteration = i + 1

                df_times = pd.concat([df_times, pd.DataFrame([[model, dataset, algo, fitting_time, selecting_time, optimizing_time, iteration, seed]], columns=cols)], ignore_index=True)

            _save_res_dir = f'{script_dir}/bayesmark_profile_results/'
            os.makedirs(_save_res_dir, exist_ok=True)
            df_times.to_csv(f'{_save_res_dir}/{algo}_{model}_{dataset}_{seed}.csv', index=False)

        else:
            # save search history
            df_hist_data = [{**config, **fval} for config, fval in zip(configs, fvals)]
            df_hist = pd.DataFrame(df_hist_data)
            df_hist.to_csv(f'{save_res_dir}/{algo}_{seed}.csv', index=False)

            logger.info(df_hist)

def _shorthand_algo_name(algo:str):
    short_name = algo

    if 'BL' in short_name:
        short_name = short_name.replace("BL", "")

    if 'A_' in short_name:
        short_name = short_name.replace("A_", "")

    has_log_suffix = False
    if short_name.endswith('_log'):
        has_log_suffix = True

    if 'EvolutionaryBO' in short_name:
        short_name = 'TREvol'
        short_name += '_log' if has_log_suffix else ''
    elif 'Optimistic' in short_name:
        short_name = 'TROpt'
        short_name += '_log' if has_log_suffix else ''
    elif 'Pareto' in short_name:
        short_name = 'TRPareto'
        short_name += '_log' if has_log_suffix else ''
    elif 'ARM' in short_name:
        short_name = 'ARM'
        short_name += '_log' if has_log_suffix else ''
    elif 'MaternVanilla' in short_name:
        short_name = 'Vanilla'
        short_name += '_log' if has_log_suffix else ''
    elif 'Vanilla' in short_name:
        short_name = 'Vanilla'
        short_name += '_log' if has_log_suffix else ''

    return short_name

def _get_dataset_type(dataset):
    if dataset in BAYESMARK_TASK_MAP:
        return 'Public'
    elif dataset in PRIVATE_TASK_MAP:
        return 'Synthetic'
    else:
        return 'Unknown'

def _get_dim_from_model(model):
    if model == 'SVM':
        return 3
    elif model == 'AdaBoost':
        return 2
    elif model == 'RandomForest':
        return 6
    elif model == 'MLP_SGD':
        return 6
    elif model == 'DecisionTree':
        return 6
    else:
        raise ValueError(f'Invalid model: {model}')


def _get_metric(dataset):
    if dataset in BAYESMARK_TASK_MAP:
        return BAYESMARK_TASK_MAP[dataset][1]
    elif dataset in PRIVATE_TASK_MAP:
        return PRIVATE_TASK_MAP[dataset][1]
    else:
        return 'Unknown'

def _get_line_type(algo):
    if 'BL' in algo:
        return '--'
    elif 'LLAMBO' in algo:
        return '-.'
    elif '_log' in algo:
        return ':'
    else:
        return '-'

def compare_expressions(a, b):
    if a == 'LLAMBO':
        return -1
    elif b == 'LLAMBO':
        return 1
    
    if 'BL' in a and 'BL' in b:
        return a > b
    elif 'BL' in a:
        return 1
    elif 'BL' in b:
        return -1
    
    return a > b


def plot_bayesmark_results():
    dir_paths = [
        'Benchmarks/LLAMBO/exp_bayesmark/results_discriminative',
        'Benchmarks/bayesmark_results_0422',
        'Benchmarks/bayesmark_results_bl',
        # 'Benchmarks/bayesmark_results_log',
    ]

    data_map = {}
    for dir_path in dir_paths:
        for _dataset in os.listdir(dir_path):
            for _model in os.listdir(os.path.join(dir_path, _dataset)):
                for _res_file in os.listdir(os.path.join(dir_path, _dataset, _model)):
                    if _res_file.endswith('.csv'):
                        algo_names = _res_file.split('_')
                        if len(algo_names) == 1:
                            seed = int(_res_file.split('.')[0])
                            algo = 'LLAMBO'
                        elif len(algo_names) == 2:
                            seed = int(algo_names[-1].split('.')[0])
                            algo = algo_names[0]
                        else:
                            seed = int(algo_names[-1].split('.')[0])
                            algo = '_'.join(algo_names[:-1])
                        
                        if 'pareto' in algo.lower():
                            continue

                        df = pd.read_csv(os.path.join(dir_path, _dataset, _model, _res_file))
                        dim = df.shape[1] - 2
                        if 'minimize_objective' in df.columns:
                            dim = dim - 1

                        # rename columns
                        df['dim'] = dim
                        df['dataset'] = _dataset.lower()
                        df['model'] = _model
                        df['seed'] = seed
                        df['algo'] = algo
                        df['n_iterations'] = df.index + 1
                        df['short_algo'] = _shorthand_algo_name(algo)

                        # add df to bl_data
                        if _model not in data_map:
                            data_map[_model] = df
                        else:
                            data_map[_model] = pd.concat([data_map[_model], df], ignore_index=True)

    fig_dir = 'Benchmarks/bayesmark_results_figs'
    os.makedirs(fig_dir, exist_ok=True)
    csv_save_dir = 'Benchmarks/bayesmark_result_files'
    os.makedirs(csv_save_dir, exist_ok=True)

    regret_df_data = []
    for _model, df in data_map.items():
        df = df[['algo', 'short_algo', 'dataset', 'model', 'seed', 'n_iterations', 'score', 'generalization_score']]
        df.to_csv(os.path.join(csv_save_dir, f'{_model}.csv'), index=False)

        best_df = df[['algo', 'short_algo', 'dataset', 'model', 'seed', 'generalization_score']]
        best_df = best_df.groupby(['algo', 'short_algo', 'dataset', 'model', 'seed']).agg(list).reset_index()
        def _best_score(row):
            dataset = row['dataset']
            _metric = 'accuracy'
            if dataset in BAYESMARK_TASK_MAP:
                _metric = BAYESMARK_TASK_MAP[dataset][1]
            elif dataset in PRIVATE_TASK_MAP:
                _metric = PRIVATE_TASK_MAP[dataset][1]
            x = row['generalization_score']
            if _metric == 'accuracy':
                return np.max(x)
            else:
                return np.min(x)
        best_df['generalization_score'] = best_df.apply(_best_score, axis=1)
        mean_best_df = best_df.groupby(['algo', 'short_algo', 'dataset', 'model']).agg(np.mean).reset_index()
        mean_best_df.drop(columns=['seed'], inplace=True)
        mean_best_df.to_csv(os.path.join(csv_save_dir, f'{_model}_best.csv'), index=False)

        df = df.groupby(['algo', 'dataset', 'model', 'seed']).agg(list).reset_index()
        datasets = df['dataset'].unique()
        datasets = ['breast', 'digits', 'wine', 'iris', 'diabetes', 'griewank', 'ktablet', 'rosenbrock'] 
        y_label_datasets = set(['breast', 'diabetes'])
        algos = df['algo'].unique()
        algos = sorted(algos, key=cmp_to_key(compare_expressions))

        plot_y = []
        plot_x = []
        labels = []
        colors = []
        line_styles = []
        plot_filling = []
        y_labels = []
        sub_titles = []

        plot_y_regret = []
        plot_filling_regret = []
        y_labels_regret = []

        for dataset in datasets:
            df_dataset = df[df['dataset'] == dataset]

            _metric = 'accuracy'
            if dataset in BAYESMARK_TASK_MAP:
                _metric = BAYESMARK_TASK_MAP[dataset][1]
            elif dataset in PRIVATE_TASK_MAP:
                _metric = PRIVATE_TASK_MAP[dataset][1]
            sub_title = f'{dataset}'
            is_maximization = True if _metric == 'accuracy' else False

            if dataset in y_label_datasets:
                y_label = 'ACC' if _metric == 'accuracy' else 'MSE'
                y_labels.append(f'Avg {y_label}')
                y_label_regret = f'Avg Regret({y_label})'
                y_labels_regret.append(y_label_regret)
            else:
                y_labels.append('')
                y_labels_regret.append('')
            sub_titles.append(sub_title)

            _x_range = (5, 30)
            _algo_labels = []
            _algo_x = None
            _algo_y = []
            _algo_filling = []
            _algo_line_styles = []

            _algo_regret = []
            _algo_regret_filling = []
            _regret_min = None
            _regret_max = None
            _scores = df_dataset['generalization_score'].to_list()
            _best_scores = []
            if _metric == 'accuracy':
                _best_scores = [np.maximum.accumulate(_score) for _score in _scores]
            else:
                _best_scores = [np.minimum.accumulate(_score) for _score in _scores]

            _score_max = np.max([np.max(_score) for _score in _scores])
            _score_min = np.min([np.min(_score) for _score in _scores])

            for i, _score in enumerate(_scores):
                min_score = np.min(_score)
                if min_score < 0.5:
                    pass

            # _score_min = 0
            # _score_max = 1
            if _metric == 'accuracy':
                # reverse the accuracy
                _regret_min = _score_max
                _regret_max = _score_min
            else:
                _regret_min = _score_min
                _regret_max = _score_max

            for algo in algos:
                df_algo = df_dataset[df_dataset['algo'] == algo]
                _algo_labels.append(algo)

                if _algo_x is None:
                    # get the x axis values
                    x_data = np.array(list(range(1, _x_range[1] + 1)))
                    _algo_x = x_data

                y_list = df_algo['generalization_score'].to_list()
                y_data = []
                for _y_list in y_list:
                    if len(_y_list) < _x_range[1]:
                        _y_list.extend([_y_list[-1]] * (_x_range[1] - len(_y_list)))
                    elif len(_y_list) > _x_range[1]:
                        _y_list = _y_list[:_x_range[1]]
                    y_data.append(_y_list)
                y_data = np.array(y_data)
                best_y_data = np.maximum.accumulate(y_data, axis=1) if is_maximization else np.minimum.accumulate(y_data, axis=1)
                mean_y_data = np.mean(best_y_data, axis=0)
                min_y_data = np.min(best_y_data, axis=0)
                max_y_data = np.max(best_y_data, axis=0)
                _algo_y.append(mean_y_data)
                _algo_filling.append([min_y_data, max_y_data])
                _algo_line_styles.append(_get_line_type(algo))

                _y_regret = np.abs(y_data - _regret_min) / np.abs(_regret_max - _regret_min)
                _best_y_regret = np.minimum.accumulate(_y_regret, axis=1)
                _mean_y_regret = np.mean(_best_y_regret, axis=0)
                _algo_regret_filling.append([np.min(_best_y_regret, axis=0), np.max(_best_y_regret, axis=0)])
                _algo_regret.append(_mean_y_regret)

                # enumerate the y_regret
                for i, _regrets in enumerate(_y_regret):
                    regret_df_data.append({
                        'algo': algo,
                        'short_algo': _shorthand_algo_name(algo),
                        'dataset': dataset,
                        'metric': _metric,
                        'dataset_type': _get_dataset_type(dataset),
                        'model': _model,
                        'seed': i,
                        'regret': _regrets,
                        'generalization_score': y_data[i],
                        'best_g_score': best_y_data[i],
                    })

            labels.append([_shorthand_algo_name(_algo) for _algo in _algo_labels])
            clip_index = _x_range[0] 
            plot_x.append(_algo_x[clip_index:])
            plot_y.append([_y[clip_index:] for _y in _algo_y])
            plot_filling.append([[_l[clip_index:], _r[clip_index:]] for _l, _r in _algo_filling])

            plot_y_regret.append([_y[clip_index:] for _y in _algo_regret])
            plot_filling_regret.append([[_l[clip_index:], _r[clip_index:]] for _l, _r in _algo_regret_filling])

            line_styles.append(_algo_line_styles)

        # plot the results
        file_name = f"{_model}"
        if fig_dir is not None:
            file_name = os.path.join(fig_dir, file_name)

        plot_y = np.array(plot_y)
        plot_x = np.array(plot_x)
        plot_lines(
            y=plot_y, x=plot_x,
            # y_scales=best_loss_y_scales,
            # colors=best_loss_colors,
            y_labels=y_labels,
            # sharey=True,
            labels=labels,
            line_styles=line_styles,
            label_fontsize=11,
            y_label_fontsize=10,
            tick_fontsize=11,
            combined_legend=True,
            combined_legend_ncols=10,
            combined_legend_bottom=0.13,
            combined_legend_fontsize=11,
            linewidth=1.3,
            # filling=plot_filling,
            n_cols=4,
            sub_titles=sub_titles,
            sub_title_fontsize=12,
            # title=f"Best Loss({dim}D)",
            figsize=(12, 5),
            show=False,
            filename=file_name,
        )

        # plot the regret
        file_name = f"{_model}_regret"
        if fig_dir is not None:
            file_name = os.path.join(fig_dir, file_name)

        plot_y_regret = np.array(plot_y_regret)
        plot_lines(
            y=plot_y_regret, x=plot_x,
            # y_scales=best_loss_y_scales,
            # colors=best_loss_colors,
            y_labels=y_labels_regret,
            # sharey=True,
            labels=labels,
            line_styles=line_styles,
            label_fontsize=11,
            y_label_fontsize=10,
            tick_fontsize=11,
            combined_legend=True,
            combined_legend_ncols=10,
            combined_legend_bottom=0.13,
            combined_legend_fontsize=11,
            linewidth=1.3,
            # filling=plot_filling_regret,
            n_cols=4,
            sub_titles=sub_titles,
            sub_title_fontsize=12,
            # title=f"Best Loss({dim}D)",
            figsize=(12, 5),
            show=False,
            filename=file_name,
        )

    def _axis0_mean(row):
        return np.mean(row, axis=0)
    regret_df = pd.DataFrame(regret_df_data)
    regret_df['best_regret'] = regret_df['regret'].apply(np.minimum.accumulate)
    overall_regret_df = regret_df.groupby(['algo', 'short_algo', 'dataset_type', 'seed'])[['best_regret']].agg(_axis0_mean).reset_index()
    dataset_types = overall_regret_df['dataset_type'].unique()
    type_plot_x = []
    type_plot_y = []
    type_plot_filling = []
    type_labels = []
    type_sub_titles = []
    type_y_labels = ['Avg Regret', ''] 
    type_line_styles = []
    _x_range = (5, 30)
    for dataset_type in dataset_types:
        overall_type_df = overall_regret_df[overall_regret_df['dataset_type'] == dataset_type]

        _algos = overall_type_df['algo'].unique()
        _algos = sorted(_algos, key=cmp_to_key(compare_expressions))
        _type_y = []
        _type_filling = []
        _type_labels = []
        _type_line_styles = []
        for algo in _algos:
            overall_algo_df = overall_type_df[overall_type_df['algo'] == algo]
            _algo_y = overall_algo_df['best_regret'].to_list()
            _algo_filling = [np.min(_algo_y, axis=0), np.max(_algo_y, axis=0)]
            _type_y.append(np.mean(_algo_y, axis=0))
            _type_filling.append(_algo_filling)
            _type_labels.append(_shorthand_algo_name(algo))
            _type_line_styles.append(_get_line_type(algo))
        type_labels.append(_type_labels)
        type_sub_titles.append(dataset_type)
        type_line_styles.append(_type_line_styles)
        clip_index = _x_range[0]
        type_plot_x.append(list(range(clip_index+1, _x_range[1] + 1)))
        type_plot_y.append([_y[clip_index:] for _y in _type_y])
        type_plot_filling.append([[_l[clip_index:], _r[clip_index:]] for _l, _r in _type_filling])

    # plot the overall regret 
    file_name = "regret"
    if fig_dir is not None:
        file_name = os.path.join(fig_dir, file_name)
    type_plot_x = np.array(type_plot_x)
    type_plot_y = np.array(type_plot_y)
    plot_lines(
        y=type_plot_y, x=type_plot_x,
        # y_scales=best_loss_y_scales,
        # colors=best_loss_colors,
        y_labels=type_y_labels,
        # sharey=True,
        labels=type_labels,
        line_styles=type_line_styles,
        label_fontsize=13,
        tick_fontsize=10,
        combined_legend=True,
        combined_legend_bottom=0.2,
        combined_legend_ncols=5,
        combined_legend_fontsize=9,
        linewidth=1.3,
        # filling=type_plot_filling,
        n_cols=4,
        sub_titles=type_sub_titles,
        sub_title_fontsize=12,
        # title=f"Best Loss({dim}D)",
        figsize=(7, 4),
        show=False,
        filename=file_name,
    )

    # plot the overall regret by model and dataset type
    overall_regret_df = regret_df.groupby(['algo', 'model', 'dataset_type', 'seed'])[['best_regret']].agg(_axis0_mean).reset_index()
    models = overall_regret_df['model'].unique()
    data_types = overall_regret_df['dataset_type'].unique()
    model_sub_titles = []
    model_y_labels = []
    model_line_styles = []
    model_plot_x = []
    model_plot_y = []
    model_plot_filling = []
    model_plot_labels = []
    for i, data_type in enumerate(data_types):
        for j, model in enumerate(models):
            _model_df = overall_regret_df[overall_regret_df['model'] == model]
            _model_df = _model_df[_model_df['dataset_type'] == data_type]
            _model_y = []
            _model_filling = []
            _model_labels = []
            _model_line_styles = []
            for algo in algos:
                _algo_df = _model_df[_model_df['algo'] == algo]
                _algo_y = _algo_df['best_regret'].to_list()
                _algo_filling = [np.min(_algo_y, axis=0), np.max(_algo_y, axis=0)]
                _model_y.append(np.mean(_algo_y, axis=0))
                _model_filling.append(_algo_filling)
                _model_labels.append(_shorthand_algo_name(algo))
                _model_line_styles.append(_get_line_type(algo))
            clip_index = _x_range[0]
            if i == 0:
                model_sub_titles.append(f'{model}({_get_dim_from_model(model)}D)')
            else:
                model_sub_titles.append('')
            if j == 0:
                model_y_labels.append(f'{data_type}')
            else:
                model_y_labels.append('')
            model_plot_labels.append(_model_labels)
            model_plot_x.append(list(range(clip_index+1, _x_range[1] + 1)))
            model_plot_y.append([_y[clip_index:] for _y in _model_y])
            model_plot_filling.append([[_l[clip_index:], _r[clip_index:]] for _l, _r in _model_filling])
            model_line_styles.append(_model_line_styles)
    
    # plot the overall model
    # file_name = "overall_model_regret"
    # if fig_dir is not None:
    #     file_name = os.path.join(fig_dir, file_name)
    # model_plot_x = np.array(model_plot_x)
    # model_plot_y = np.array(model_plot_y)
    # plot_lines(
    #         y=model_plot_y, x=model_plot_x,
    #         # y_scales=best_loss_y_scales,
    #         # colors=best_loss_colors,
    #         y_labels=model_y_labels,
    #         labels=model_plot_labels,
    #         line_styles=model_line_styles,
    #         label_fontsize=11,
    #         y_label_fontsize=10,
    #         tick_fontsize=11,
    #         combined_legend=True,
    #         combined_legend_ncols=10,
    #         combined_legend_bottom=0.13,
    #         combined_legend_fontsize=11,
    #         linewidth=1.3,
    #         # filling=dataset_plot_filling,
    #         n_cols=5,
    #         sub_titles=model_sub_titles,
    #         sub_title_fontsize=12,
    #         # title=f"Best Loss({dim}D)",
    #         figsize=(13, 5),
    #         show=False,
    #         filename=file_name,
    #     )

    overall_model_metrics_df = regret_df.groupby(['algo', 'model', 'seed', 'metric'])[['best_g_score']].agg(_axis0_mean).reset_index()
    y_label_datasets = set(['breast', 'diabetes'])
    algos = overall_model_metrics_df['algo'].unique()
    algos = sorted(algos, key=cmp_to_key(compare_expressions))
    models = overall_model_metrics_df['model'].unique()
    metrics = overall_model_metrics_df['metric'].unique()
    model_sub_titles = []
    model_y_labels = []
    model_line_styles = []
    model_plot_x = []
    model_plot_y = []
    model_plot_filling = []
    model_plot_labels = []

    for m_i, metric in enumerate(metrics):
        _metric_df = overall_model_metrics_df[overall_model_metrics_df['metric'] == metric]
        for m_j, model in enumerate(models):
            _model_df = _metric_df[_metric_df['model'] == model]
            _model_y = []
            _model_filling = []
            _model_labels = []
            _model_line_styles = []
            for algo in algos:
                _algo_df = _model_df[_model_df['algo'] == algo]
                _algo_y = _algo_df['best_g_score'].to_list()
                _algo_filling = [np.min(_algo_y, axis=0), np.max(_algo_y, axis=0)]
                _model_y.append(np.mean(_algo_y, axis=0))
                _model_filling.append(_algo_filling)
                _model_labels.append(_shorthand_algo_name(algo))
                _model_line_styles.append(_get_line_type(algo))
            clip_index = _x_range[0]
            clip_index = 4
            if m_i == 0:
                model_sub_titles.append(f'{model}')
            else:
                model_sub_titles.append('')
            if m_j == 0:
                model_y_labels.append('Avg ACC' if metric == 'accuracy' else 'Avg MSE')
            else:
                model_y_labels.append('')
            model_plot_labels.append(_model_labels)
            model_plot_x.append(list(range(clip_index+1, _x_range[1] + 1)))
            model_plot_y.append([_y[clip_index:] for _y in _model_y])
            model_plot_filling.append([[_l[clip_index:], _r[clip_index:]] for _l, _r in _model_filling])
            model_line_styles.append(_model_line_styles)

    # plot the overall model metrics
    file_name = "overall_model_metrics"
    if fig_dir is not None:
        file_name = os.path.join(fig_dir, file_name)
    model_plot_x = np.array(model_plot_x)
    model_plot_y = np.array(model_plot_y)
    plot_lines(
            y=model_plot_y, x=model_plot_x,
            # y_scales=best_loss_y_scales,
            # colors=best_loss_colors,
            y_labels=model_y_labels,
            labels=model_plot_labels,
            line_styles=model_line_styles,
            label_fontsize=11,
            y_label_fontsize=10,
            tick_fontsize=11,
            combined_legend=True,
            combined_legend_ncols=10,
            combined_legend_bottom=0.13,
            combined_legend_fontsize=11,
            linewidth=1.3,
            # filling=dataset_plot_filling,
            n_cols=5,
            sub_titles=model_sub_titles,
            sub_title_fontsize=12,
            # title=f"Best Loss({dim}D)",
            figsize=(13, 5),
            show=False,
            filename=file_name,
        )


    overall_dataset_df = regret_df.groupby(['algo', 'dataset', 'seed'])[['best_g_score']].agg(_axis0_mean).reset_index()
    datasets = ['breast', 'digits', 'wine', 'iris', 'diabetes', 'griewank', 'ktablet', 'rosenbrock']
    y_label_datasets = set(['breast', 'diabetes'])
    algos = overall_dataset_df['algo'].unique()
    algos = sorted(algos, key=cmp_to_key(compare_expressions))
    dataset_sub_titles = []
    dataset_y_labels = []
    dataset_line_styles = []
    dataset_plot_x = []
    dataset_plot_y = []
    dataset_plot_filling = []
    dataset_plot_labels = []
    for dataset in datasets:
        _dataset_df = overall_dataset_df[overall_dataset_df['dataset'] == dataset]
        _dataset_y = []
        _dataset_filling = []
        _dataset_labels = []
        _dataset_line_styles = []
        for algo in algos:
            _algo_df = _dataset_df[_dataset_df['algo'] == algo]
            _algo_y = _algo_df['best_g_score'].to_list()
            _algo_filling = [np.min(_algo_y, axis=0), np.max(_algo_y, axis=0)]
            _dataset_y.append(np.mean(_algo_y, axis=0))
            _dataset_filling.append(_algo_filling)
            _dataset_labels.append(_shorthand_algo_name(algo))
            _dataset_line_styles.append(_get_line_type(algo))
        clip_index = _x_range[0]
        dataset_sub_titles.append(dataset)
        dataset_plot_labels.append(_dataset_labels)
        dataset_plot_x.append(list(range(clip_index+1, _x_range[1] + 1)))
        dataset_plot_y.append([_y[clip_index:] for _y in _dataset_y])
        dataset_plot_filling.append([[_l[clip_index:], _r[clip_index:]] for _l, _r in _dataset_filling])
        dataset_line_styles.append(_dataset_line_styles)
        if dataset in y_label_datasets:
            _metric = _get_metric(dataset)
            if _metric == 'accuracy':
                dataset_y_labels.append('Avg ACC')
            else:
                dataset_y_labels.append('Avg MSE')
        else:
            dataset_y_labels.append('')
    # plot the overall dataset
    file_name = "overall_dataset_acc_mse"
    if fig_dir is not None:
        file_name = os.path.join(fig_dir, file_name)
    dataset_plot_x = np.array(dataset_plot_x)
    dataset_plot_y = np.array(dataset_plot_y)
    plot_lines(
            y=dataset_plot_y, x=dataset_plot_x,
            # y_scales=best_loss_y_scales,
            # colors=best_loss_colors,
            y_labels=dataset_y_labels,
            labels=dataset_plot_labels,
            line_styles=dataset_line_styles,
            label_fontsize=11,
            y_label_fontsize=10,
            tick_fontsize=11,
            combined_legend=True,
            combined_legend_ncols=10,
            combined_legend_bottom=0.13,
            combined_legend_fontsize=11,
            linewidth=1.3,
            # filling=dataset_plot_filling,
            n_cols=4,
            sub_titles=dataset_sub_titles,
            sub_title_fontsize=12,
            # title=f"Best Loss({dim}D)",
            figsize=(12, 5),
            show=False,
            filename=file_name,
        )


def plot_bayesmark_profile_results():
    dir_paths = [
        'Benchmarks/bayesmark_profile_results'
    ]

    algo_map = {}
    for dir_path in dir_paths:
        for _res_file in os.listdir(dir_path):
            if _res_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(dir_path, _res_file))

                algo = df['algo'][0]

                if algo not in algo_map:
                    algo_map[algo] = df
                else:
                    algo_map[algo] = pd.concat([algo_map[algo], df], ignore_index=True)

    models = algo_map[list(algo_map.keys())[0]]['model'].unique()
    datasets = algo_map[list(algo_map.keys())[0]]['dataset'].unique()

    import matplotlib.pyplot as plt
    prop_cycle = plt.rcParams['axes.prop_cycle']
    _default_colors = prop_cycle.by_key()['color']
    color_map = {}
    for i, algo in enumerate(algo_map.keys()):
        if 'BL' in algo:
            color_map[algo] = _default_colors[i % len(_default_colors)]
        else:
            color_map[algo] = _default_colors[i % len(_default_colors)]


    data_range = (5, 30)

    time_types = ['fitting_time', 'selecting_time', 'optimizing_time']

    algos = algo_map.keys()
    algos = sorted(algos, key=cmp_to_key(compare_expressions))

    for model in models:
        for time_type in time_types:
            plot_x = []
            plot_y = []
            plot_filling = []
            labels = []
            line_styles = []
            colors = []
            for _algo in algos:

                if 'EI' in _algo:
                    pass

                df = algo_map[_algo]
                time_df = df.groupby(['model', 'dataset', 'algo', 'seed']).agg(list).reset_index()

                x = time_df['iteration'].to_list()[0]
                y = time_df[time_type].to_list()
                mean_y = np.mean(y, axis=0)

                if np.mean(mean_y) == 0:
                    continue

                min_y = np.min(y, axis=0)
                max_y = np.max(y, axis=0)

                label = _shorthand_algo_name(_algo)
                line_style = '--' if 'BL' in _algo else '-'

                x = x[data_range[0]: data_range[1]]
                mean_y = mean_y[data_range[0]: data_range[1]]
                min_y = min_y[data_range[0]: data_range[1]]
                max_y = max_y[data_range[0]: data_range[1]]
                filling = [min_y, max_y]

                plot_x.append(x)
                plot_y.append(mean_y)
                plot_filling.append(filling)
                labels.append(label)
                line_styles.append(line_style)
                colors.append(color_map[_algo])

            # plot the results
            fig_dir = 'Benchmarks/bayesmark_results_figs'
            os.makedirs(fig_dir, exist_ok=True)
            file_name = f"{model}_{datasets}_{time_type}"
            if fig_dir is not None:
                file_name = os.path.join(fig_dir, file_name)

            from llamevol.utils import plot_lines

            _plot_y = np.array([plot_y])
            _plot_x = np.array(plot_x)
            plot_lines(
                y=_plot_y, x=_plot_x,
                labels=[labels],
                y_labels=['Time (s)'],
                y_label_fontsize=12,
                line_styles=[line_styles],
                label_fontsize=10,
                tick_fontsize=14,
                combined_legend=True,
                combined_legend_ncols=5,
                combined_legend_bottom=0.18 + (0.08 if len(labels) > 5 else 0),
                combined_legend_fontsize=12,
                colors=[colors],
                linewidth=1.3,
                filling=[plot_filling],
                figsize=(8, 4),
                show=False,
                filename=file_name,
            )

            if time_type == 'optimizing_time':
                index_hebo = labels.index('HEBO')
                index_vanilla = labels.index('Vanilla')
                indexes = [index_hebo, index_vanilla]
                indexes.sort(reverse=True)
                for index in indexes:
                    labels.pop(index)
                    plot_x.pop(index)
                    plot_y.pop(index)
                    plot_filling.pop(index)
                    colors.pop(index)
                    line_styles.pop(index)
                _plot_x = np.array(plot_x)
                _plot_y = np.array([plot_y])
                plot_lines(
                    y=_plot_y, x=_plot_x,
                    labels=[labels],
                    y_labels=['Time (s)'],
                    y_label_fontsize=12,
                    line_styles=[line_styles],
                    label_fontsize=10,
                    combined_legend=True,
                    combined_legend_ncols=5,
                    combined_legend_bottom=0.18 + (0.08 if len(labels) > 5 else 0),
                    combined_legend_fontsize=12,
                    tick_fontsize=14,
                    colors=[colors],
                    linewidth=1.3,
                    filling=[plot_filling],
                    figsize=(8, 4),
                    show=False,
                    filename=file_name + '_no_hebo_vanilla',
                )


def convert_results_to_ioh_format():
    df_data = {}

    dir_paths = [
        'Benchmarks/LLAMBO/exp_bayesmark/results_discriminative',
        'Benchmarks/bayesmark_results_0422',
        'Benchmarks/bayesmark_results_bl',
        # 'Benchmarks/bayesmark_results',
    ]

    for dir_path in dir_paths:
        for _dataset in os.listdir(dir_path):
            for _model in os.listdir(os.path.join(dir_path, _dataset)):
                for _res_file in os.listdir(os.path.join(dir_path, _dataset, _model)):
                    if _res_file.endswith('.csv'):
                        algo_names = _res_file.split('_')
                        if len(algo_names) == 1:
                            seed = int(_res_file.split('.')[0])
                            algo = 'LLAMBO'
                        elif len(algo_names) == 2:
                            seed = int(algo_names[-1].split('.')[0])
                            algo = algo_names[0]
                        else:
                            seed = int(algo_names[-1].split('.')[0])
                            algo = '_'.join(algo_names[:-1])
                        df = pd.read_csv(os.path.join(dir_path, _dataset, _model, _res_file))
                        dim = df.shape[1] - 2
                        if 'minimize_objective' in df.columns:
                            dim = dim - 1

                        # rename columns
                        df['dim'] = dim
                        df['dataset'] = _dataset.lower()
                        df['model'] = _model
                        df['seed'] = seed
                        df['algo'] = algo
                        df['n_iterations'] = df.index + 1
                        _metric = 'accuracy'
                        if _dataset in BAYESMARK_TASK_MAP:
                            _metric = BAYESMARK_TASK_MAP[_dataset][1]
                        elif _dataset in PRIVATE_TASK_MAP:
                            _metric = PRIVATE_TASK_MAP[_dataset][1]
                        if _metric == 'neg_mean_squared_error':
                            df['fx'] = -df['score']
                            df['generalization_fx'] = -df['generalization_score']
                        else:
                            df['fx'] = df['score']
                            df['generalization_fx'] = df['generalization_score']

                        # add df to bl_data
                        if _model not in df_data:
                            df_data[_model] = df
                        else:
                            df_data[_model] = pd.concat([df_data[_model], df], ignore_index=True)
 
    # save all dataframes in df_data to csv
    ioh_columns = ['n_iter', 't_fx', 'fid', 'algo', 'dim', 'n_run', 'fx']
    ioh_df = pd.DataFrame(columns=ioh_columns)
    for _model, df in df_data.items():
        _ioh_df = pd.DataFrame(columns=ioh_columns)
        _ioh_df['n_iter'] = df['n_iterations']
        _ioh_df['fx'] = df['fx']
        _ioh_df['t_fx'] = df['generalization_fx']
        # create fid column by combining model and dataset
        _ioh_df['fid'] = _model + ' ' + df['dataset']
        _ioh_df['algo'] = df['algo']
        _ioh_df['n_run'] = df['seed']
        _ioh_df['dim'] = df['dim']

        ioh_df = pd.concat([ioh_df, _ioh_df], ignore_index=True)


    # save ioh_df to csv
    ioh_df.to_csv('Benchmarks/bayesmark_ioh_results.csv', index=False)
    # ioh_df.to_csv('Benchmarks/bayesmark_ioh_results_log.csv', index=False)


if __name__ == '__main__':
    setup_logger(level=logging.INFO)

    # convert_results_to_ioh_format()
    # plot_bayesmark_results()
    # plot_bayesmark_profile_results()

    from Experiments.generated_algorithms.ATRBO import ATRBO
    from Experiments.generated_algorithms.AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE import AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE 
    from Experiments.generated_algorithms.AdaptiveTrustRegionOptimisticHybridBO import AdaptiveTrustRegionOptimisticHybridBO
    from Experiments.generated_algorithms.AdaptiveEvolutionaryParetoTrustRegionBO import AdaptiveEvolutionaryParetoTrustRegionBO
    from Experiments.generated_algorithms.ABETSALSDE_ARM_MBO import ABETSALSDE_ARM_MBO

    from Experiments.baselines.bo_baseline import (BLVanillaEIBO, BLCMAES, BLHEBO, BLTuRBO1)

    bo_cls_list = [
        # ATRBO,
        # AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE,
        # AdaptiveTrustRegionOptimisticHybridBO,
        # AdaptiveEvolutionaryParetoTrustRegionBO,
        # ABETSALSDE_ARM_MBO,

        # BLVanillaEIBO,
        # BLCMAES,
        # BLHEBO,
        # BLTuRBO1,
        ]
    is_profile = False
    bo_wrappers = [bayesmarkBO_wrapper(bo_cls, is_profile=is_profile) for bo_cls in bo_cls_list]

    use_log = False
    datasets = [
        "digits", 
        "wine", 
        "diabetes", 
        "iris",
        "breast", 
        "Griewank",
        "KTablet",
        "Rosenbrock"
        ]
    models = [
        "RandomForest", 
        "SVM",
        "DecisionTree",
        "MLP_SGD",
        "AdaBoost"
        ]

    for bo_cls in bo_wrappers:
        for dataset in datasets:
            for model in models:
                _run_bayesmark_exp(bo_cls, dataset, model, num_seeds=5, use_log=use_log, is_profile=is_profile)
    logger.info('All experiments completed.')