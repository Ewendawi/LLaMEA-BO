import os
import pickle
import json
import logging
import warnings
import random
import pandas as pd
import numpy as np
import functools
from collections.abc import Callable
from Benchmarks.LLAMBO.bayesmark.bbox_utils import get_bayesmark_func
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

from llambo.utils import setup_logger

def update_init(self, init_X):
    self.init_X = init_X

def func_wrapper(func):
    functools.wraps(func)

    def wrapper(self, *args, **kwargs):
        if func.__name__ == "_sample_points":
            if self.n_evals == 0 and self.init_X is not None:
                return self.init_X
            else:
                return func(self, *args, **kwargs)
        elif func.__name__ == "_evaluate_points":
            if self.n_evals == 0: 
                return self.init_y
            else:
                return func(self, *args, **kwargs)
        elif func.__name__ == "__init__":
            func(self, *args, **kwargs)
            self.init_X = None
        else:
            return func(self, *args, **kwargs)
    return wrapper

def bayesmarkBO_wrapper(cls):
    setattr(cls, '__init__', func_wrapper(cls.__init__))
    setattr(cls, 'update_init', update_init)
    setattr(cls, '_sample_points', func_wrapper(cls._sample_points))
    # setattr(cls, '_evaluate_points', func_wrapper(cls._evaluate_points))
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
}

class BayesmarkExpRunner:
    def __init__(self, task_context, dataset, seed):
        self.seed = seed
        self.model = task_context['model']
        self.task = task_context['task']
        self.metric = task_context['metric']
        self.dataset = dataset
        self.hyperparameter_constraints = task_context['hyperparameter_constraints']
        self.bbox_func = get_bayesmark_func(self.model, self.task, dataset['test_y'])

        self.ordered_hyperparams = list(self.hyperparameter_constraints.keys())
        self.bounds = []
        lower_bounds = []
        upper_bounds = []
        for hyperparam in self.ordered_hyperparams:
            constraint = self.hyperparameter_constraints[hyperparam]
            lower_bounds.append(constraint[2][0])
            upper_bounds.append(constraint[2][1])
        
        self.bounds = np.array([lower_bounds, upper_bounds])

        dim = len(self.ordered_hyperparams)
        self.x_bounds = np.array([[-1.0]*dim, [1.0]*dim])

        self.x_hist = []
        self.fvals_hist = []

    def _map_bounds(self, x, bounds, new_bounds):
        '''
        Map x from bounds to new_bounds
        Args: x (numpy array), bounds (numpy array), new_bounds (numpy array)
        Returns: x (numpy array)
        '''
        scaled_x = new_bounds[0] + (x - bounds[0]) * (new_bounds[1] - new_bounds[0]) / (bounds[1] - bounds[0])
        return scaled_x

    def x_to_config(self, x):
        '''
        Convert x (numpy array) to config (dictionary)
        Args: x (numpy array)
        Returns: config (dictionary)
        '''
        config = {}
        for i, hyperparam in enumerate(self.ordered_hyperparams):
            x_bound = self.x_bounds[:,i]
            config_bound = self.bounds[:,i]

            config_x = self._map_bounds(x[i], x_bound, config_bound)

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
        
        x = self._map_bounds(config_x, self.bounds, self.x_bounds)

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
        Returns: fvals (dict), dictionary containing evaluation results
        '''
        config = self.x_to_config(x)
        _, fvals = self.evaluate_point(config)
        self.x_hist.append(config)
        self.fvals_hist.append(fvals)
        return -fvals['score']
        
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
        cv_score = np.mean(S)
        
        model = self.bbox_func(**candidate_config) 
        model.fit(X_train, y_train)
        generalization_score = scorer(model, X_test, y_test)

        if self.metric == 'neg_mean_squared_error':
            cv_score = -cv_score
            generalization_score = -generalization_score

        return candidate_config, {'score': cv_score, 'generalization_score': generalization_score}


def _run_bayesmark_exp(bo_cls, dataset, model, num_seeds = 5, budget = 30, n_initial_samples = 5):
    if dataset is None:
        raise ValueError("dataset must be a string")
    if model is None:
        raise ValueError("model must be a string")
    
    if num_seeds is not None and isinstance(num_seeds, int) and (num_seeds < 1 or num_seeds > 10):
        raise ValueError("num_seeds must be an integer between 1 and 10")

    algo_cls = bo_cls
    algo = algo_cls.__name__

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
        logger.info('Executing %s to tune %s on %s with seed %d / %d...', algo, model, dataset, seed+1, num_seeds)
        logger.info('Task context: %s', task_context)

        benchmark = BayesmarkExpRunner(task_context, data, seed)

        dim = len(benchmark.ordered_hyperparams)
        init_config = benchmark.generate_initialization(n_initial_samples)
        init_X = []
        for config in init_config:
            x = benchmark.config_to_x(config)
            init_X.append(x)
        init_X = np.array(init_X)

        # instantiate algorithm
        bo_instance = algo_cls(budget=budget, dim=dim)
        bo_instance.update_init(init_X)
        bo_instance.bounds = benchmark.x_bounds

        # run optimization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            bo_instance(func=benchmark.obj_func_wrapper)

        configs, fvals = benchmark.x_hist, benchmark.fvals_hist

        # save search history
        df_hist_data = [{**config, **fval} for config, fval in zip(configs, fvals)]
        df_hist = pd.DataFrame(df_hist_data)
        df_hist.to_csv(f'{save_res_dir}/{algo}_{seed}.csv', index=False)

        logger.info(df_hist)

def convert_results_to_ioh_format():
    df_data = {}

    dir_paths = [
        'Benchmarks/LLAMBO/exp_bayesmark/results_discriminative',
        'Benchmarks/bayesmark_results' 
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
                        # rename columns
                        df['dim'] = dim
                        df['dataset'] = _dataset
                        df['model'] = _model
                        df['seed'] = seed
                        df['alog'] = algo
                        df['n_iterations'] = df.index + 1
                        # add df to bl_data
                        if _model not in df_data:
                            df_data[_model] = df
                        else:
                            df_data[_model] = pd.concat([df_data[_model], df], ignore_index=True)

    # save all dataframes in df_data to csv
    columns = ['score', 'generalization_score', 'dataset', 'model', 'seed', 'alog', 'n_iterations', 'dim']
    ioh_columns = ['n_iter', 'fx', 'fid', 'algo', 'dim', 'n_run', 't_fx']
    ioh_df = pd.DataFrame(columns=ioh_columns)
    for _model, df in df_data.items():
        _df = df[columns]

        _df['n_iter'] = _df['n_iterations']
        _df['fx'] = 1 - _df['score']
        _df['t_fx'] = 1 - _df['generalization_score']
        # create fid column by combining model and dataset
        _df['fid'] = _model + ' ' + _df['dataset']
        _df['algo'] = _df['alog']
        _df['n_run'] = _df['seed']
        _df = _df[ioh_columns]

        ioh_df = pd.concat([ioh_df, _df], ignore_index=True)


    # save ioh_df to csv
    ioh_df.to_csv('Benchmarks/bayesmark_results/ioh_results.csv', index=False)


if __name__ == '__main__':
    setup_logger(level=logging.INFO)

    # convert_results_to_ioh_format()

    from Experiments.logs.algorithms_logs.ATRBO import ATRBO
    from Experiments.logs.algorithms_logs.AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE import AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE 
    from Experiments.logs.algorithms_logs.AdaptiveTrustRegionOptimisticHybridBO import AdaptiveTrustRegionOptimisticHybridBO
    from Experiments.logs.algorithms_logs.AdaptiveEvolutionaryParetoTrustRegionBO import AdaptiveEvolutionaryParetoTrustRegionBO
    from Experiments.logs.algorithms_logs.ABETSALSDE_ARM_MBO import ABETSALSDE_ARM_MBO

    bo_cls_list = [
        # ATRBO,
        # AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE_VAE,
        # AdaptiveTrustRegionOptimisticHybridBO,
        # AdaptiveEvolutionaryParetoTrustRegionBO, 
        ABETSALSDE_ARM_MBO
        ]
    bo_wrappers = [bayesmarkBO_wrapper(bo_cls) for bo_cls in bo_cls_list]

    datasets = ["digits", "wine", "diabetes", "iris", "breast", "Griewank", "KTablet", "Rosenbrock"]
    models = [
        "RandomForest", 
        "SVM", "DecisionTree", "MLP_SGD", "AdaBoost"
        ]

    for bo_cls in bo_wrappers:
        for dataset in datasets:
            for model in models:
                _run_bayesmark_exp(bo_cls, dataset, model, num_seeds=5)
    logger.info('All experiments completed.')