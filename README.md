# LLaMEA-BO

## Setup
1. Clone the repository:
   ```bash
   git clone  
   cd LLaMEA-BO
   ```
2. Install the required dependencies via Poetry:
   ```bash
   pip install poetry
   poetry install
   ```

## Reproducing Results

### Result files and Raw Data
- The results of the experiments are stored in the `Experiments/logs/` directory.
- The raw data files are stored in the `[TODO]`.

### Run ES Search 
1. Replace `GEMINI_API_KEY` in `run_es_search.sh` with your Gemini API key.
2. Change `N_POPULATION` in `run_es_search.sh` as needed.
3. Run the script:
   ```bash
   bash run_es_search.sh
   ```
4. The results will be saved in `exp_es_search/` directory.

### Run BBOB Evaluations
1. Run the script:
   ```bash
   bash run_algo_evaluation.sh
   ```
2. The results will be saved in `exp_eval/` directory.

### Run Bayesmark Evaluations
Follow the instructions in the `Benchmarks/Readme.md`.

## Development

The project follows a modular structure primarily located within the `llamevol/` directory.

- **`llamevol/`**: Contains the core implementation of the LLaMEvol algorithm.
    - **`llamevol.py`**: The main class orchestrating the LLaMEvol process.
    - **`individual.py`**: Defines the `Individual` class representing a single generated algorithm/solution.
    - **`llm.py`**: Handles interactions with the Language Model (LLM).
    - **`prompt_generators/`**: Contains classes responsible for generating prompts for the LLM.
    - **`evaluator/`**: Includes code for executing and evaluating the performance of generated algorithms, often using benchmark suites like BBOB (via IOHprofiler). It handles code execution, error capture, and metric calculation.
    - **`population/`**: Manages the collection (population) of `Individual` algorithms, implementing selection strategies and diversity maintenance.
    - **`utils.py`**: Provides utility functions, including logging, serialization and plotting.
- **`Benchmarks/`**: Contains scripts and results for running external benchmarks like Bayesmark.
- **`Experiments/`**: Holds scripts for running specific experiments and plotting results.

### Usage Example

Below is a simplified example demonstrating how to set up and run the LLaMEvol evolutionary process using the provided components. This example uses an `IOHEvaluator`, a `BaselinePromptGenerator`, a `gemini-2.0-flash` model via `LLMmanager`, and an `ESPopulation`.

```python
import logging
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
from llamevol.prompt_generators import BaselinePromptGenerator
from llamevol.population import ESPopulation
from llamevol.llm import LLMmanager
from llamevol import LLaMEvol
from llamevol.utils import setup_logger

# Configure logging
setup_logger(level=logging.INFO)

# 1. Instantiate Evaluator (Example: IOH BBOB)
evaluator = IOHEvaluator(budget=100, dim=5, problems=[2, 4, 6], instances=[[1]]*3, repeat=3)

# 2. Instantiate Prompt Generator
prompt_generator = BaselinePromptGenerator()
prompt_generator.is_bo = True # Specify it's for Bayesian Optimization

# 3. Instantiate LLM Manager (Example: Google Gemini)
# Ensure API key is set via environment variable or passed directly
api_key = 'YOUR_API_KEY' # Replace with your actual key or load from env
llm_manager = LLMmanager(model_name='gemini-2.0-flash', api_key=api_key, client_str='google')

# 4. Instantiate Population (Example: (1+1)-ES)
es_options = {
    'n_parent': 1,
    'n_offspring': 1,
    'is_elitist': True,
    'log_dir': 'exp_es_search', # Directory to save logs
}
population = ESPopulation(
    n_parent=es_options['n_parent'], 
    n_offspring=es_options['n_offspring'], 
    use_elitism=es_options['is_elitist']
)
population.save_dir = es_options['log_dir']
population.name = f"evol_{es_options['n_parent']}+{es_options['n_offspring']}"

# 5. Instantiate LLaMEvol orchestrator
llamevol = LLaMEvol()

# 6. Run the evolution
llm_params = {'temperature': 0.7}
llamevol.run_evolutions(
    llm=llm_manager,
    evaluator=evaluator,
    prompt_generator=prompt_generator,
    population=population,
    n_population=5,  # Target number of successful individuals
    n_generation=3,  # Maximum number of generations
    options={'llm_params': llm_params}
)

# 7. Save the final population
population.save(suffix='final')

print("Evolution finished. Results saved in:", population.log_dir)
```

For a runnable script with command-line arguments, see `run_es_search.py`.

### Parallelism in IOHEvaluator

The `IOHEvaluator` supports several modes for parallelizing the evaluation of algorithms across different IOH problems, instances, and repetitions:

1.  **Sequential Execution:**
    - **How:** This is the default mode if no parallel options are explicitly enabled (i.e., `max_eval_workers` is set to 0 or less, and `use_mpi` and `use_mpi_future` are `False`).
    - **Description:** Each evaluation task (a specific problem/instance/repetition) is executed one after another in the main process.

2.  **Thread Pool Execution:**
    - **How:** Set `max_eval_workers` to a positive integer (e.g., `evaluator.max_eval_workers = 10`) and ensure `use_multi_process` is `False` (default).
    - **Description:** Uses Python's `concurrent.futures.ThreadPoolExecutor` to run evaluation tasks concurrently in multiple threads within the same process. 

3.  **Process Pool Execution:**
    - **How:** Set `max_eval_workers` to a positive integer and set `use_multi_process = True` (e.g., `evaluator.max_eval_workers = 10; evaluator.use_multi_process = True`).
    - **Description:** Uses Python's `concurrent.futures.ProcessPoolExecutor` to run evaluation tasks in separate processes. Suitable for the algorithm which don't use multiple cores effectively. 

4.  **MPI (Custom Task Manager):**
    - **How:** Set `use_mpi = True` (e.g., `evaluator.use_mpi = True`). Requires MPI environment, `mpi4py` installed and a specific command to run the script (e.g., `mpiexec python pyfile`). An example can be found in `run_algo_evaluation.py`.
    - **Description:** Utilizes a custom master-worker implementation (`MPITaskManager`) built on top of `mpi4py`. The main node(rank 0) distributes tasks to worker nodes(rank > 0). Suitable for distributed memory systems.

5.  **MPI (mpi4py.futures):**
    - **How:** Set `use_mpi_future = True` (e.g., `evaluator.use_mpi_future = True`). Requires MPI environment, `mpi4py` installed and a specific command to run the script (e.g., `mpiexec -n numprocs python -m mpi4py.futures pyfile`). The details of the command can be found in [the documentation of `mpi4py.futures`](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#command-line). 
    - **Description:** Leverages `mpi4py.futures.MPIPoolExecutor` for a higher-level interface to MPI-based parallelism. Similar to the process pool but designed specifically for MPI environments.

**Configuration:**
These options are typically set as attributes on the `IOHEvaluator` instance *before* calling the `evaluate` method. An example can be found in `run_algo_evaluation.py`.

### LLaMEvol
The `LLaMEvol` class (`llamevol/llamevol.py`) is the central orchestrator of the evolutionary algorithm. It coordinates the interactions between the LLM, Evaluator, Prompt Generator, and Population components to drive the search for optimal algorithms.

**Structure & Features:**
- **Main Loop:** Implements the core evolutionary loop (`run_evolutions`), managing generations and population size.
- **Component Integration:** Takes instances of `LLMmanager`, `AbstractEvaluator`, `PromptGenerator`, and `Population` as inputs, delegating specific tasks to each.
- **Task Determination:** Dynamically determines the appropriate task for the LLM based on the state of the parent individuals (e.g., `INITIALIZE_SOLUTION`, `FIX_ERRORS`, `OPTIMIZE_PERFORMANCE`) using `update_current_task`.
- **LLM Interaction:** Handles querying the LLM via the `LLMmanager`, including:
    - Constructing session messages based on prompts from the `PromptGenerator`.
    - Applying LLM parameters (temperature, top_k).
    - Managing retries (`n_retry`) in case of LLM or extraction failures.
    - Optional parallel querying using `concurrent.futures.ThreadPoolExecutor` (`n_query_threads`).
- **Evaluation Trigger:** Calls the `evaluate` method of the provided `Evaluator` on the code generated by the LLM.
- **Population Update:** Updates `Individual` objects within the `Population` with the results from the LLM (code, description) and Evaluator (fitness, feedback) using `_update_ind_and_handler`.
- **Token Tracking:** Logs prompt and response token counts per generation (`LLaMEvolTokenLogItem`).
- **Progression Control:** Iterates through generations until a target population size (`n_population`) or maximum number of generations (`n_generation`) is reached.

**Usage:**
1.  **Instantiate Components:** Create instances of `LLMmanager`, `AbstractEvaluator`, `PromptGenerator`, and `Population` configured for your specific task and resources.
2.  **Instantiate LLaMEvol:** Create an instance of the `LLaMEvol` class.
    ```python
    from llamevol import LLaMEvol
    llamevol = LLaMEvol()
    ```
3.  **Run Evolution:** Call the `run_evolutions` method, passing the instantiated components and desired parameters.
    ```python
    # Assuming llm, evaluator, prompt_generator, population are already created
    llamevol.run_evolutions(
        llm=llm_manager,
        evaluator=evaluator,
        prompt_generator=prompt_generator,
        population=population,
        n_population=20, # Target number of successful individuals
        n_generation=10, # Max generations (optional)
        n_retry=3,
        n_query_threads=4, # Number of parallel LLM queries
        options={'llm_params': {'temperature': 0.7}}
    )
    ```
4.  **Results:** The final population (containing evolved individuals and their performance) can be accessed and saved via the `Population` object after the run completes.

**Customization:**
- **Component Swapping:** The primary way to customize `LLaMEvol`'s behavior is by providing different implementations of its core components (LLM, Evaluator, Prompt Generator, Population). For example, using a different `Population` class changes the selection and generation strategy.
- **Configuration:** Adjust parameters passed to `run_evolutions`, such as `n_population`, `n_generation`, `n_retry`, `n_query_threads`, and LLM-specific settings within the `options` dictionary.

### LLMmanager
This module (`llamevol/llm.py`) acts as a central manager for interacting with various Large Language Models (LLMs). 

**Features:**
- Provides a unified interface (`LLMmanager`) to connect to different LLM providers (Groq, Google GenAI, OpenAI-compatible APIs like OpenRouter).
- Abstracts away the specific API details for each provider.
- Manages API keys and base URLs, primarily loaded from environment variables.
- Defines a standardized response object (`LLMClientResponse`) containing the generated text, token counts, and potential errors.
- Supports different client implementations (`OpenAIClient`, `GoogleGenAIClient`, `AISuiteClient`, `RequestClient`).

**Usage:**
1.  **Environment Variables(Optional):** Ensure the necessary API keys and base URLs for the desired LLMs are set as environment variables (e.g., `GROQ_API_KEY`, `GEMINI_API_KEY`, etc.). Copy and rename `.env.template` to `.env` and fill in the required keys.
    ```bash
    cp .env.example .env
    # Edit .env to add your API keys
    ```
2.  **Initialization:** Create an instance of `LLMmanager` by providing a `model_key` which corresponds to an entry in the `LLMS` dictionary within the script. Alternatively, you can manually specify `model_name`, `api_key`, `base_url`, and `client_str`. The mapping of `client_str` to the actual client class is handled in the `LLMmanager` constructor.
    ```python
    from llamevol.llm import LLMmanager

    # Using a predefined model key
    llm_manager = LLMmanager(model_key='llama3-70b-8192') 

    # Or manually configuring (example)
    # llm_manager = LLMmanager(model_name='some-model', api_key='YOUR_API_KEY', base_url='https://api.example.com/v1', client_str='openai')
    ```
3.  **Chat:** Use the `chat` method, passing a list of messages in the standard OpenAI format (list of dictionaries with 'role' and 'content').
    ```python
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the theory of relativity."}
    ]
    response = llm_manager.chat(messages, temperature=0.7)

    if response.error:
        print(f"Error: {response.error}")
    else:
        print(f"Response Text: {response.text}")
        print(f"Prompt Tokens: {response.prompt_token_count}")
        print(f"Response Tokens: {response.response_token_count}")
    ```

**Customization:**
- **Adding Predefined Models:** To add support for a new model using an existing provider type, add an entry to the `LLMS` dictionary in `llm.py`. You'll need:
    - the model name recognized by the API
    - the environment variable name for the API key 
    - the environment variable name for the base URL (if applicable) 
    - a maximum interval value (**Deprecated**, designed for rate limiting and retries). 
    - the client type string (`'groq'`, `'google'`, `'openai'`, `'openrouter'`, `'request'`, or `None` for default handling like `AISuiteClient`). 

- **Adding New Providers:** 
    1. Create a new class that inherits from `LLMClient`.
    2. Implement the `raw_completion` method to handle the specific API request/response logic for the new provider.
    3. Update the `LLMmanager.__init__` method to recognize a new `client_str` and instantiate your custom client class when that string is provided.

- **Adding New Providers from AISuite:** 
   1. Install the the provider-specific package (e.g., `pip install 'aisuite[anthropic]'`).
   2. Add the corresponding `API_KEY` in `__init__` of `AISuiteClient`.

### Prompt Generator
This component constructs the prompts sent to the LLM for generating or modifying optimization algorithms.

**Structure & Features:**
- **Abstract Base Classes:** Defines `PromptGenerator` and `ResponseHandler` abstract classes (`abstract_prompt_generator.py`) to ensure a consistent interface.
- **Concrete Implementations:** Provides specific generators like `BaselinePromptGenerator` (for generating algorithms from scratch), `BoTunerPromptGenerator` (for refining existing algorithms). 
- **Contextual Prompts:** Dynamically builds prompts incorporating problem descriptions, existing candidate solutions (code, descriptions, past performance), evaluation feedback (errors, performance metrics like AOC), and potentially information about the broader population of algorithms.
- **Task-Specific Instructions:** Generates detailed instructions for the LLM based on the task (e.g., "design a novel algorithm", "fix the error in this code", "optimize this algorithm based on feedback").
- **Response Parsing:** Each `PromptGenerator` has a corresponding `ResponseHandler` subclass responsible for parsing the LLM's structured output (e.g., extracting code blocks, justifications, pseudocode) using methods like `extract_response`.

**Usage:**
1.  **Instantiate:** Choose and instantiate a specific `PromptGenerator` subclass.
2.  **Generate Prompt:** Call the `get_prompt` method, passing the `GenerationTask`, problem description, and any relevant context (like candidate `ResponseHandler` objects or the `Population`).
3.  **Query LLM:** Use the returned system and user prompts with the `LLMmanager`.
4.  **Parse Response:** Get the corresponding `ResponseHandler` instance using `get_response_handler()` and use its `extract_response` method on the LLM's output string.

**Customization:**
- **New Strategies:** Create new subclasses inheriting from `PromptGenerator` and `ResponseHandler`.
- **Implement Methods:** Override methods like `get_prompt`, `task_description`, `task_instruction`, `response_format`, `evaluation_feedback_prompt` in your `PromptGenerator` subclass, and `extract_response` in your `ResponseHandler` subclass to define the new prompting logic and response parsing.

### Evaluator
The Evaluator component is responsible for executing the Python code generated by the LLM and assessing its performance on optimization tasks.

**Structure & Features:**
- **Abstract Base:** Defines `AbstractEvaluator` (`evaluator.py`) for a consistent interface.
- **Concrete Implementations:** Provides evaluators for standard benchmarks:
    - `IOHEvaluator` (`ioh_evaluator.py`): Evaluates algorithms on the IOHprofiler (BBOB) benchmark suite. Supports parallel execution across multiple problem instances and repetitions.
    - `RandomBoTorchTestEvaluator` (`random_botorch_evaluator.py`): Evaluates algorithms on synthetic test functions from the BoTorch library.
- **Code Execution:** Uses utilities in `exec_utils.py` (`default_exec`) to safely execute the generated Python code, capturing standard output, errors, and execution time. It handles budget constraints via `BOOverBudgetException`.
- **Result Tracking:** Employs `EvaluatorResult` and `EvaluatorBasicResult` (`evaluator_result.py`) to store detailed outcomes for each evaluation run, including:
    - Best function value found (`best_y`).
    - History of evaluated points (`x_hist`, `y_hist`).
    - Area Over the Convergence Curve (AOC), including log-scale AOC, calculated using `ConvergenceCurveAnalyzer`.
    - Execution time and any runtime errors.
- **BO Algorithm Introspection (Optional):** Uses `BOInjector` and `AlgorithmCritic` (`bo_injector.py`) to inject monitoring code specifically into Bayesian Optimization algorithms. This allows tracking internal metrics during the optimization run, such as:
    - Surrogate model R² score (on test and training data).
    - Surrogate model uncertainty.
    - Search space coverage metrics (grid-based, clustering-based using `CoverageCluster`).
    - Exploitation vs. Exploration metrics (distance to best points, acquisition score analysis via `EvaluatorSearchResult`).
- **Parallelism:** Supports parallel evaluation using `MPI` (as seen in `IOHEvaluator`). Specifically, `MPITaskManager` provides an MPI-based master-worker framework, which can be used to distribute evaluation tasks across multiple nodes. This is particularly useful for large-scale evaluations across distributed systems.

**Usage:**
1.  **Instantiate:** Create an instance of a specific evaluator subclass (e.g., `IOHEvaluator`) with configuration like budget, dimension, and target problems/instances.
2.  **Evaluate:** Call the `evaluate` method, providing the generated Python code string and the name of the main class within that code. Optional arguments control parallelism (`max_eval_workers`) and timeouts.
3.  **Process Results:** The `evaluate` method returns an `EvaluatorResult` object. This object contains a list of `EvaluatorBasicResult` objects, each holding the detailed metrics, history, and potential errors for a single evaluation run (e.g., one IOH instance).

**Customization:**
- **New Benchmarks:** Create a new class inheriting from `AbstractEvaluator`. Implement the required methods (`evaluate`, `problem_name`, etc.). You'll likely need a wrapper for your objective function (similar to `IOHObjectiveFn`) to manage budget and history tracking.
- **New Metrics:** Extend `EvaluatorBasicResult` or `EvaluatorSearchResult` to store additional metrics. Modify the relevant evaluator or create/modify an `ExecInjector` subclass (`exec_utils.py`, `bo_injector.py`) to compute and record these metrics during or after code execution.

### Population
The Population component (`llamevol/population/`) manages the collection of candidate algorithms (`Individual` objects) throughout the evolutionary process.

**Structure & Features:**
- **Abstract Base:** Defines `Population` (`population.py`) as an abstract base class, ensuring a consistent interface for different population management strategies. It includes common utilities like saving/loading populations (using `pickle`) and calculating diversity metrics.
- **Concrete Implementations:**
    - `ESPopulation` (`es_population.py`): Implements an Evolution Strategy-style population (e.g., (μ+λ) or (μ,λ)).
        - Manages individuals across discrete generations.
        - Supports configurable parent pool size (`n_parent`), offspring count (`n_offspring`), and parents per offspring (`n_parent_per_offspring`).
        - Handles selection for the next generation, including optional elitism (`use_elitism`).
        - Implements parent selection logic based on combinations and configurable crossover/mutation rates (`cross_over_rate`, `exclusive_operations`).
        - Allows plugging in custom parent selection (`get_parent_strategy`) and survival selection (`selection_strategy`) functions.
    - `IslandESPopulation` (`island_population.py`): Implements an island model using multiple `ESPopulation` instances.
        - Manages multiple sub-populations (islands) concurrently.
        - Introduces island lifecycles (`IslandStatus`: INITIAL, GROWING, MATURE, RESETING, KILLED) and geological ages (`IslandAge`: WARMUP, CAMBRIAN, NEOGENE) to control evolution dynamics.
        - Implements migration strategies between islands during specific ages (e.g., CAMBRIAN), potentially based on fitness and diversity (using `desc_similarity`).
        - Supports configurable migration parameters (`migration_batch`, `cyclic_migration`).
        - Allows islands to be reset or killed based on performance.
    - `SequencePopulation` (`sequence_population.py`): A simpler (potentially non-generational) population structure (currently basic).
    - `EnsemblePopulation` (`ensemble_population.py`): Designed to combine multiple populations (currently basic).
- **Query Items:** Uses `PopulationQueryItem` to represent tasks for the main loop, specifying parent individuals for generating offspring.
- **Diversity Metrics:** Provides utility functions in `population.py` to assess population diversity:
    - `code_diff_similarity`: Based on line-by-line code differences.
    - `code_bert_similarity`: Uses CodeBERT embeddings for semantic code similarity.
    - `desc_similarity`: Uses sentence transformers on algorithm descriptions.
- **Persistence:** Populations can be saved to and loaded from disk using `pickle` via the `save()` and `load()` methods.

**Usage:**
1.  **Instantiate:** Create an instance of a specific population class (e.g., `ESPopulation`) with desired parameters (e.g., `n_parent`, `n_offspring`). Optionally provide custom strategy functions.
2.  **Get Tasks:** Call `get_offspring_queryitems()` to get a list of `PopulationQueryItem` objects. Each item indicates which parent(s) should be used to generate a new offspring.
3.  **Add Individuals:** After an offspring is generated and evaluated by the LLM and Evaluator, add the resulting `Individual` object to the population using `add_individual(individual, generation)`.
4.  **Advance Generation:** Call `select_next_generation()` to apply the survival selection mechanism and advance the population state to the next generation (primarily for `ESPopulation`).
5.  **Retrieve Data:** Access individuals using methods like `get_best_individual()`, `get_individuals(generation)`, `all_individuals()`.

**Customization:**
- **Strategies:** Implement custom functions for parent selection (`get_parent_strategy`) and survival selection (`selection_strategy`) and pass them to the constructor of `ESPopulation` or `IslandESPopulation`.
- **New Population Types:** Create a new class inheriting from `Population`. Implement all abstract methods (`get_population_size`, `add_individual`, `remove_individual`, `get_offspring_queryitems`, `get_current_generation`, `get_best_individual`, `all_individuals`) to define a completely new population management scheme.
- **Diversity Metrics:** Add new diversity calculation functions in `population.py` or elsewhere and integrate them into selection or migration strategies.

