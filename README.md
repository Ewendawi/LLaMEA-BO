
# LLaMBO - Language Model Bayesian Optimizer

Core implementation of the Language Model powered Bayesian Optimization (LLaMBO) system. This module contains:

- `LLaMBO` class: Main class for running LLM-powered Bayesian optimization
  - Handles prompt construction and evolution
  - Manages optimization runs across generations
  - Integrates with evaluators and prompt generators
  - Processes model responses and results

## Key Features
- Task-based optimization (Initialize/Fix/Optimize)
- Integration with language models via `LLMmanager` 
- Result tracking and feedback loops
- Support for retry logic and error handling

## Usage

### Initialization
1. create a .env file in your project root by copying the template:
```bash
cp .env.template .env
``` 
2. Fill in the API key and model name in the .env file

3. Use the LLM
```python
from llamea.llm import LLMmanager, LLMS

# Key from LLMS. you can add more models by adding them to the LLMS 
model_name = "llama-3.3-70b-versatile"
model = LLMS[model_name]
llm = LLMmanager( api_key=model[1], model=model[0], base_url=model[2], max_interval=model[3])
```

### Running LLaMBO
```python
from llamea import LLaMBO, LLMmanager, BOPromptGenerator
from llamea.utils import RandomBoTorchTestEvaluator
from llamea.individual import SequencePopulation

llambo = LLaMBO()
llm = LLMmanager(api_key="key", model="model-name")
evaluator = RandomBoTorchTestEvaluator()
prompt_generator = BOPromptGenerator()
population = SequencePopulation()

llambo.run_evolutions(llm, evaluator, prompt_generator, population)
```

### Run Experiments
all the experiments are in the `experiments` folder. 

### Log
The log file is saved in the `logs` folder.

## Details

### Individual and Population Classes

Core classes for representing optimization solutions and populations:

#### Individual
Represents a single solution/algorithm:
- Solution code
- Name and description
- Fitness scores and feedback
- Parent/generation tracking
- Error handling
- Metadata storage

#### Population
Base class for managing groups of individuals:
- Individual tracking
- Selection methods
- Population statistics

#### SequencePopulation
Specialized population implementation:
- Sequential individual processing
- Generation-based management
- Result history tracking


### LLM Integration Components

Components for integrating with Language Models:

#### Classes
- `LLMClient`: Abstract base class for LLM backends
- `OpenAIClient`: OpenAI GPT implementation
- `RequestClient`: Generic REST API client
- `LLMmanager`: High-level LLM interaction manager

#### Features
- Multiple model support 
- API configuration
- Rate limiting
- Error handling
- Response processing
- Dynamic model selection


### Prompt Generation Components

Classes and utilities for generating prompts for LLMs:

#### Classes
- `GenerationTask`: Task type enumeration
- `PromptGenerator`: Abstract base class
- `BOPromptGenerator`: Bayesian optimization prompt generator
- `LlamboPromptManager`: High-level prompt management

#### Features
- Task-specific prompts
- Structured response formats
- Code structure templates
- Strategy suggestion lists
- Result extraction


### Utility Components

Core utilities and helper classes:

#### Components
- `IndividualLogger`: Solution logging and tracking
- `AbstractEvaluator`: Base evaluator class
- `EvaluatorResult`: Evaluation result container
- Testing utilities and functions
- Logging configuration
- Execution helpers

#### Key Features
- JSON/Pickle serialization
- File management
- Result tracking
- Error handling
- Logging setup
