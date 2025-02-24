
import os
import pickle
from datetime import datetime
import logging
import numpy as np

from llamea.prompt_generators import BaselinePromptGenerator, GenerationTask
from llamea.evaluator.ioh_evaluator import IOHEvaluator 
from llamea.population import desc_similarity_from_handlers, code_diff_similarity_from_handlers, code_bert_similarity_from_handlers
from llamea import LLaMBO
from llamea.llm import LLMmanager, LLMS
from llamea.utils import setup_logger

def get_IOHEvaluator_for_test(problems=[3], _instances=[1], repeat=1, budget=100, dim=5):
    instances = [_instances] * len(problems)
    evaluator = IOHEvaluator(budget=budget, dim=dim, problems=problems, instances=instances, repeat=repeat)
    return evaluator

def get_llm():
    # MODEL = 'deepseek/deepseek-chat'

    MODEL = 'gemini-2.0-flash-exp'
    # MODEL = 'gemini-1.5-flash'
    # MODEL = 'gemini-2.0-pro-exp'
    # MODEL = 'gemini-2.0-flash-thinking-exp'
    # MODEL = 'gemini-exp-1206'

    # MODEL = 'llama3-70b-8192'
    # MODEL = 'llama-3.3-70b-versatile'
    # MODEL = 'deepseek-r1-distill-llama-70b'
    # MODEL = 'deepseek-r1-distill-qwen-32b'
    
    # MODEL = 'o_gemini-flash-1.5-8b-exp'
    # MODEL = 'o_gemini-2.0-flash-exp'

    # MODEL = 'onehub-gemini-2.0-flash'
    # MODEL = 'onehub-gemma2-9b-it'

    llm = LLMmanager(model_key=MODEL)

    return llm


class temperatureRes:
    def __init__(self):
        self.parent_name = None
        self.parent_handlers = None
        self.desc_mean_sim = None
        self.desc_sim_matrix = None
        self.code_mean_sim = 0
        self.code_sim_matrix = None
        self.parent_handler = None
        self.res_list = []

def plot_res(param_rest_map, param_name):
    import matplotlib.pyplot as plt
    for param, messages_res_list in param_rest_map.items():
        desc_mean_sim_list = [temp_res.desc_mean_sim for temp_res in messages_res_list]
        desc_mean_sim_list = np.array(desc_mean_sim_list)
        desc_mean_sim_list = desc_mean_sim_list.mean(axis=1)
        plt.plot(desc_mean_sim_list, label=f"{param_name}: {param}")
    plt.legend()
    plt.show()

test_file_paths = [
    'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209171704/0-2_BOTSDynBOv1_respond.md',
    
    'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209171704/1-10_AdaEEBOv2_respond.md', 

    'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209165843/0-2_ThompsonSamplingBOv1_respond.md',
    
    
    'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-2_TrustRegionBOv1_respond.md',

    'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-3_GradientEnhancedBOv1_respond.md',
    
    'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-10_DuelingBanditBOv1_respond.md',
    
    'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-11_SurrogateModelFreeBOv1_respond.md',

    'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-10_DuelingBanditBOv1_respond.md',

    'Experiments/pop_40_f/ESPopulation_evol_12+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209000244/0-9_BayesMetaLearningBOv1_respond.md',
    
    'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209061439/0-4_BayesUCBwithRBFBOv1_respond.md',
    
    'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209061439/0-5_DEwithLocalSearchBOv1_respond.md',

    'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209061439/0-9_StochasticLHSwithHistoryBOv1_respond.md',

    'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065238/0-12_VarianceReductionBOv1_respond.md',

    'Experiments/pop_40_f/ESPopulation_evol_20+14_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208225928/0-6_DynamicPenaltyBOv1_respond.md'
] 

def _get_prompt_msg(file_paths, num, chunk_size, promptor, current_task):
    size = num * chunk_size
    _selected_index = np.random.choice(len(file_paths), size=size, replace=False)
    _selected_files = [file_paths[i] for i in _selected_index]

    if current_task == GenerationTask.OPTIMIZE_PERFORMANCE:
        parent_handler_list = []
        for i, file_path in enumerate(_selected_files):
            prompt = ""
            with open(file_path, "r") as f:
                prompt = f.read()

            handler = promptor.get_response_handler() 
            handler.extract_response(prompt, current_task)
            parent_handler_list.append(handler)
        parent_handlers = [parent_handler_list[i:i+chunk_size] for i in range(0, len(parent_handler_list), chunk_size)]
    else:
        parent_handlers = [[]] * num

    messages_list = []
    for parent in parent_handlers:
        role_setting, prompt = promptor.get_prompt(
            task=current_task,
            problem_desc=None,
            candidates=parent,
            )
        session_messages = [
            {"role": "system", "content": role_setting},
            {"role": "user", "content": prompt},
        ]
        messages_list.append(session_messages)
    return messages_list, parent_handlers

def _save_code(file_dir, handler, prefix):
    code = handler.code
    name = handler.code_name
    file_name = f"{prefix}_{name}.py"
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, "w") as f:
        f.write(code)

    respond = handler.raw_response
    res_file_name = f"{prefix}-{name}_respond.md"
    res_file_path = os.path.join(file_dir, res_file_name)
    with open(res_file_path, "w") as f:
        f.write(respond)

def run_temperature_exp():
    llmbo = LLaMBO()
    current_task = GenerationTask.OPTIMIZE_PERFORMANCE
    llm = get_llm()
    promptor = BaselinePromptGenerator()
    promptor.is_bo = True
    evaluator = get_IOHEvaluator_for_test(problems=[4], _instances=[1], repeat=1, budget=100)

     # initial
    chunk_size = 0
    # mutation
    chunk_size = 1
    # crossover
    # chunk_size = 2

    current_task = GenerationTask.OPTIMIZE_PERFORMANCE if chunk_size > 0 else GenerationTask.INITIALIZE_SOLUTION

    temperatures = [0.0, 0.4, 0.8, 1.2, 1.6]
    temperatures = [0.0, 1.0, 2.0]
    params = temperatures
    param_name = "temperature"

    # top_p_list = [0.4, 0.6, 0.8, 1.0]
    # top_p_list = [0.4]
    # params = top_p_list
    # param_name = "top_p"

    # top_k_list = [4, 10, 20, 40]
    # top_k_list = [40]
    # params = top_k_list
    # param_name = "top_k"

    num = 2
    repeat = 3

    save_name = f"{param_name}_{params}_{num}*{repeat}"
    if chunk_size == 0:
        save_name = f"{save_name}_init"
    elif chunk_size == 1:
        save_name = f"{save_name}_mut"
    else:
        save_name = f"{save_name}_cros"
    time_stamp = datetime.now().strftime("%m%d%H%M%S")
    save_name = f"{save_name}_{time_stamp}"

    save_dir = 'Experiments/llm_exp'
    save_dir = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    logging.info("Start experiment: %s", save_name)

    file_paths = test_file_paths
    messages_list, parent_handlers = _get_prompt_msg(file_paths, num, chunk_size, promptor, current_task)
    param_rest_map = {}
    for param in params:
        print(f"{param_name}: {param}")
        messages_res_list = []
        options = {
            'llm_params': {
                f'{param_name}': param,
            }
        }
        for i, messages in enumerate(messages_list):
            logging.info("Start msg: %s", i)
            parent = parent_handlers[i]
            for j, parent_handler in enumerate(parent):
                prefix = f"{param_name}-{param}-0.{j}"
                _save_code(save_dir, parent_handler, prefix)
                logging.info("Prompt %s", parent_handler.code_name)

            res_list = []
            for j in range(repeat):
                logging.info("repeat: %s", j)
                next_handler = promptor.get_response_handler()
                llmbo.evalution_func(
                    session_messages=messages,
                    llm=llm,
                    evaluator=evaluator,
                    task=current_task,
                    retry=1,
                    response_handler=next_handler,
                    options=options
                )
                prefix = f"{param_name}-{param}-r{i}-1.{j}"
                _save_code(save_dir, next_handler, prefix)
                res_list.append(next_handler)

            comp_res_list = parent + res_list

            mean_sim, sim_matrix = desc_similarity_from_handlers(comp_res_list) 
            print('Desc similarity')
            print(mean_sim)
            print(sim_matrix)

            # code_mean_sim, code_sim_matrix = code_bert_similarity_from_handlers(comp_res_list)
            # print('Code similarity')
            # print(code_mean_sim)
            # print(code_sim_matrix)

            temp_res = temperatureRes()
            temp_res.parent_name = '_'.join([handler.code_name for handler in parent])
            temp_res.parent_handlers = parent
            temp_res.res_list = res_list
            temp_res.desc_mean_sim = mean_sim
            temp_res.desc_sim_matrix = sim_matrix
            # temp_res.code_mean_sim = code_mean_sim
            # temp_res.code_sim_matrix = code_sim_matrix
            messages_res_list.append(temp_res)

        param_rest_map[param] = messages_res_list
    file_name = f"{save_name}_res.pkl"
    with open(os.path.join(save_dir, file_name), "wb") as f:
        pickle.dump(param_rest_map, f)

if __name__ == "__main__":
    setup_logger(level=logging.INFO)
    run_temperature_exp()