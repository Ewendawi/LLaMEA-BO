

## Usage

### Setup

1. clone the repository under `Benchmarks` directory

    ```bash
    cd Benchmarks
    git clone https://github.com/tennisonliu/LLAMBO.git
    ```

### Run Bayesmark benchmark

1. Uncomment the lines of `bo_cls_list` in `Benchmarks/run_bayesmark.py` to select the desired benchmark classes. 

2. run bayesmark benchmark

    ```bash
    ./Benchmarks/run_bayesmark.py
    ```
The logs and results will be saved in `Benchmarks/bayesmark_logs` and `Benchmarks/bayesmark_results` respectively.

### Run HPO benchmark

1. Set up the environment. Before running the hpo benchmark, you need to install the required packages and create a symbolic link to the `hpo_bench` and `hp_configurations` directories in the current directory.
    
    ```bash
    # create a symbolic link to the hpo_bench and hp_configurations directories in the current directory.
    ln -s ./Benchmarks/LLAMBO/hpo_bench ./hpo_bench
    ln -s ./Benchmarks/LLAMBO/hp_configurations ./hp_configurations

    # create the hpo_benchmarks directory for downloading the datasets
    mkdir ./hpo_bench/hpo_benchmarks

    # install the required packages
    pip install openml
    ```

2. Run the hpo benchmark with the following command:
    
    ```bash
    ./Benchmarks/run_hpo.py
    ```

The logs and results will be saved in `Benchmarks/hpo_logs` and `Benchmarks/hpo_results` respectively.

3. Clean up the symbolic links and the hpo_benchmarks directory after running the benchmark:
    ```bash
    rm ./hpo_bench
    rm ./hp_configurations
    ```
