import os
import openml
import json
import logging
import subprocess
import sys
import pandas as pd
import numpy as np
from Benchmarks.LLAMBO.hpo_bench.tabular_benchmarks import HPOBench
from Benchmarks.run_bayesmark import bayesmarkBO_wrapper

from llambo.utils import setup_logger

import warnings
# ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger(__name__)

def setup_logging(log_name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
DATASET_MAP = {
    "credit_g": [0, 31],    # [dataset id, openml task id]
    "vehicle": [1, 53],
    "kc1": [2, 3917],
    "phoneme": [3, 9952],
    "blood_transfusion": [4, 10101],
    "australian": [5, 146818],
    "car": [6, 146821],
    "segment": [7, 146822],
}

MODEL_MAP = {
    'rf': 'Random Forest',
    'nn': 'Multilayer Perceptron',
    'xgb': 'XGBoost'
}


class HPOExpRunner:
    def __init__(self, task_context, dataset, seed):
        model = task_context['model']
        self.hpo_bench = HPOBench(model, dataset)
        self.seed = seed
        self.config_path = f'Benchmarks/LLAMBO/hpo_bench/configs/{model}/config{seed}.json'

        self.hyperparameter_constraints = task_context['hyperparameter_constraints']
        self.ordered_hyperparams = list(self.hyperparameter_constraints.keys())
        self.bounds = []
        lower_bounds = []
        upper_bounds = []
        for hyperparam in self.ordered_hyperparams:
            constraint = self.hyperparameter_constraints[hyperparam]
            values = constraint[2]
            lower_bounds.append(values[0])
            upper_bounds.append(values[-1])
            # lower_bounds.append(values[0] - (values[1] - values[0]) * 0.5)
            # upper_bounds.append(values[-1] + (values[-1] - values[-2]) * 0.5)
        
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
        # load initial configs
        with open(self.config_path, 'r') as f:
            configs = json.load(f)

        assert isinstance(configs, list)
        init_configs = []
        for i, config in enumerate(configs):
            assert isinstance(config, dict)
            
            if i < n_samples:
                init_configs.append(self.hpo_bench.ordinal_to_real(config))
        
        assert len(init_configs) == n_samples

        return init_configs
    
    def _find_nearest_neighbor(self, config):
        discrete_grid = self.hpo_bench._value_range
        nearest_config = {}
        for key in config:
            if key in discrete_grid:
                # Find the nearest value in the grid for the current key
                nearest_value = min(discrete_grid[key], key=lambda x: abs(x - config[key]))
                nearest_config[key] = nearest_value
            else:
                raise ValueError(f"Key '{key}' not found in the discrete grid.")
        return nearest_config

    def obj_func_wrapper(self, x):
        '''
        Wrapper function for bbox_func
        Args: x (numpy array)
        Returns: fvals (dict), dictionary containing evaluation results
        '''
        config = self.x_to_config(x)
        nearest_config, fvals = self.evaluate_point(config)
        self.x_hist.append(nearest_config)
        self.fvals_hist.append(fvals)
        return -fvals['score']
        
    def evaluate_point(self, candidate_config):
        '''
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        Example fval:
        fvals = {
            'score': float,
            'generalization_score': float
        }
        '''
        # find nearest neighbor
        nearest_config = self._find_nearest_neighbor(candidate_config)
        # evaluate nearest neighbor
        fvals = self.hpo_bench.complete_call(nearest_config)
        return nearest_config, fvals

def download_dataset():
    urls = {
        "xgb": "https://ndownloader.figshare.com/files/30469920",
        "svm": "https://ndownloader.figshare.com/files/30379359",
        "lr": "https://ndownloader.figshare.com/files/30379038",
        "rf": "https://ndownloader.figshare.com/files/30469089",
        "nn": "https://ndownloader.figshare.com/files/30379005"
    }
    base_output_dir = "hpo_bench/hpo_benchmarks"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir, exist_ok=True)

    files = os.listdir(base_output_dir)
    if len(files) > 0:
        print(f"Found existing files in {base_output_dir}.")
        return

    for name, url in urls.items():
        print(f"Processing Benchmark: {name.upper()}")

        benchmark_output_dir = f'{base_output_dir}/hpo-bench-{name}'
        os.makedirs(benchmark_output_dir, exist_ok=True)
        temp_zip_path = f'{benchmark_output_dir}/{name}.zip'

        print(f"Downloading from {url}...")

        curl_command = [
            'curl', '-L', '-#', '-o', str(temp_zip_path), url
        ]

        subprocess.run(curl_command, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Downloaded {temp_zip_path}")

        print(f"Unzipping {temp_zip_path}...")
        unzip_command = [
            'unzip', '-o', str(temp_zip_path), '-d', str(benchmark_output_dir)
        ]
        subprocess.run(unzip_command, check=True, text=True, capture_output=True)

        os.remove(temp_zip_path)
        print(f"Removed {temp_zip_path}")


def run_hpo_benchmarks(bo_cls, dataset_name, model, seeds_to_run, n_initial_samples=5, budget=30):
    num_seeds = len(seeds_to_run)
    dataset = DATASET_MAP[dataset_name][0]
    algo = bo_cls.__name__
    algo_cls = bo_cls

    # Describe task context
    task_context = {}
    task_context['model'] = model
    task_context['task'] = 'classification' # hpo_bech datasets are all classification

    task = openml.tasks.get_task(DATASET_MAP[dataset_name][1])
    dataset_ = task.get_dataset()
    X, y, categorical_mask, _ = dataset_.get_data(target=dataset_.default_target_attribute)

    task_context['tot_feats'] = X.shape[1]
    task_context['cat_feats'] = len(categorical_mask)
    task_context['num_feats'] = X.shape[1] - len(categorical_mask)
    task_context['n_classes'] = len(np.unique(y))
    task_context['metric'] = "accuracy"
    task_context['lower_is_better'] = False
    task_context['num_samples'] = X.shape[0]
    with open('Benchmarks/LLAMBO/hp_configurations/hpobench.json', 'r') as f:
        task_context['hyperparameter_constraints'] = json.load(f)[model]

    # define result save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_res_dir = f'{script_dir}/hpo_results/{dataset_name}/{model}'
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    # define logging directory
    logging_fpath = f'{script_dir}/hpo_logs/{dataset_name}/{model}.log'
    if not os.path.exists(os.path.dirname(logging_fpath)):
        os.makedirs(os.path.dirname(logging_fpath))
    setup_logging(logging_fpath)

    for seed in seeds_to_run:
        logger.info('Executing %s to tune %s on %s with seed %d / %d...', algo, model, dataset_name, seed+1, num_seeds)
        logger.info('Task context: %s', task_context)

        # instantiate benchmark
        benchmark = HPOExpRunner(task_context, dataset, seed)

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

        logger.info('\n%s', df_hist)

def convert_results_to_ioh_format():
    df_data = {}

    dir_paths = [
        'Benchmarks/LLAMBO/exp_hpo_bench/results_discriminative',
        'Benchmarks/hpo_results',
    ]

    for dir_path in dir_paths:
        for _dataset in os.listdir(dir_path):
            _dataset = _dataset.lower()
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
                        dim = df.shape[1] - 8
                        # rename columns
                        df['dim'] = dim
                        df['dataset'] = _dataset
                        df['model'] = _model
                        df['seed'] = seed
                        df['algo'] = algo
                        df['n_iterations'] = df.index + 1
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
        _ioh_df['fx'] = df['score']
        _ioh_df['t_fx'] = df['generalization_score']
        # create fid column by combining model and dataset
        _ioh_df['fid'] = _model + ' ' + df['dataset']
        _ioh_df['algo'] = df['algo']
        _ioh_df['n_run'] = df['seed']
        _ioh_df['dim'] = df['dim']

        ioh_df = pd.concat([ioh_df, _ioh_df], ignore_index=True)

    # save ioh_df to csv
    ioh_df.to_csv('Benchmarks/hpo_ioh_results.csv', index=False)

if __name__ == '__main__':
    download_dataset()

    setup_logger(level=logging.INFO)

    convert_results_to_ioh_format()

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
        # ABETSALSDE_ARM_MBO
        ]
    bo_wrappers = [bayesmarkBO_wrapper(bo_cls) for bo_cls in bo_cls_list]


    datasets = ["australian", "blood_transfusion", "car", "credit_g", "kc1", "phoneme", "segment", "vehicle"]
    models = ["rf", "xgb", "nn"]
    seeds_to_run = [0, 1, 2, 3, 4]

    # seeds_to_run = [0]
    # datasets = ["blood_transfusion"]
    # models = ["rf"]

    for bo_cls in bo_wrappers:
        for dataset_name in datasets:
            for model in models:
                run_hpo_benchmarks(bo_cls, dataset_name, model, seeds_to_run)
    logger.info('All done!')