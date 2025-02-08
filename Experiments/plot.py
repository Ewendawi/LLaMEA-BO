import pickle
import os
from llamea.utils import IndividualLogger
from llamea.population.es_population import ESPopulation
from llamea.prompt_generators.abstract_prompt_generator import ResponseHandler
from llamea.evaluator.evaluator_result import EvaluatorResult
from llamea.utils import plot_results, plot_algo_results

def plot_search():
    # file_paths = [
    #     # ("logs_bbob/bbob_exp_gemini-2.0-flash-exp_0121222958.pkl", "bo"),
    #     ("logs_bbob/bbob_exp_gemini-2.0-flash-exp_0124195614.pkl", "es-1+1"),
    #     ("logs_bbob/bbob_exp_gemini-2.0-flash-exp_0124195643.pkl", "es-1+1"),
    # ]
    # strategy_list = []
    # for file_path, name in file_paths:
    #     ind_logger = IndividualLogger.load(file_path)
    #     ind_ids = list(ind_logger.experiment_map.values())[0]["id_list"]
    #     inds = [ind_logger.get_individual(ind_id) for ind_id in ind_ids]
    #     res_list = [ind.metadata["res_handler"].eval_result for ind in inds]
    #     strategy_list.append((name, res_list))

    # 1+1
    file_paths_1_1 = [
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BaselinePromptGenerator_0127065904.pkl', "EA-1+1"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BaselinePromptGenerator_0127071455.pkl', "EA-1+1"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0127100500.pkl', "BO-1+1-Sklearn"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0127230651.pkl', "BO-1+1-Sklearn"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127075128.pkl', "BO-1+1-GPytorch"),
        ('population_logs/ESPopulation_bbob_1+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127230526.pkl', "BO-1+1-GPytorch"),
    ]

    # 2+1
    file_paths_2_1 = [
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BaselinePromptGenerator_0127064159.pkl', "EA-2+1"),
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BaselinePromptGenerator_0127064736.pkl', "EA-2+1"),
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127120051.pkl', "BO-2+1-GPytorch"),
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0127155755.pkl', "BO-2+1-Sklearn"),
        ('population_logs/ESPopulation_bbob_2+1_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0128105531.pkl', "BO-2+1-GPytorch"),
    ]
    
    # 2+1_warmstart_diversity
    file_paths_2_1_wd = [
        ('population_logs/ESPopulation_bbob_2+1+warm+diverse_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127092335.pkl', "BO-2+1-wd-GPytorch"),
        ('population_logs/ESPopulation_bbob_2+1+warm+diverse_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127203251.pkl', "BO-2+1-wd-GPytorch"),
        ('population_logs/ESPopulation_bbob_2+1+warmstart+diversity_gemini-2.0-flash-exp_BaselinePromptGenerator_0127073309.pkl', "EA-2+1-wd"),
        ('population_logs/ESPopulation_bbob_2+1+warmstart+diversity_gemini-2.0-flash-exp_BaselinePromptGenerator_0127074034.pkl', "EA-2+1-wd"),
        ('population_logs/ESPopulation_bbob_2+1+warmstart+diversity_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0127234617.pkl', "BO-2+1-wd-Sklearn"),
        ('population_logs/ESPopulation_bbob_2+1+warmstart+diversity_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0128002616.pkl', "BO-2+1-wd-Sklearn"),
    ]

    
    # island
    file_paths_island = [
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BaselinePromptGenerator_0127073608.pkl', "EA-Island"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BaselinePromptGenerator_0127081002.pkl', "EA-Island"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_sklearn_0128022539.pkl', "BO-Island-Sklearn"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127085652.pkl', "BO-Island-GPytorch"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127103428.pkl', "BO-Island-GPytorch"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0127172700.pkl', "BO-Island-GPytorch"),
        ('population_logs/IslandESPopulation_bbob_island_gemini-2.0-flash-exp_BOBaselinePromptGenerator_torch_0128071243.pkl', "BO-Island-GPytorch"),
    ]


    all_file_paths = file_paths_1_1 + file_paths_2_1_wd + file_paths_island + file_paths_2_1

    file_paths_ea = []
    file_paths_bo_sklearn = []
    file_paths_bo_torch = []
    for file_path, name in all_file_paths:
        if "EA" in name:
            file_paths_ea.append((file_path, name))
        elif "Sklearn" in name:
            file_paths_bo_sklearn.append((file_path, name))
        elif "GPytorch" in name:
            file_paths_bo_torch.append((file_path, name))

    file_paths = file_paths_bo_torch
    
    strategy_list = []
    ea_offspring = []
    sklearn_offspring = []
    torch_offspring = []
    for file_path, name in file_paths:
        pop = ESPopulation.load(file_path)
        strategy_list.append((name, pop))
        if "EA" in name:
            ea_offspring.extend(pop.all_individuals())
        elif "Sklearn" in name:
            sklearn_offspring.extend(pop.all_individuals())
        elif "GPytorch" in name:
            torch_offspring.extend(pop.all_individuals())

    sorted_ea_offspring = sorted(ea_offspring, key=lambda x: x.fitness, reverse=True)
    sorted_sklearn_offspring = sorted(sklearn_offspring, key=lambda x: x.fitness, reverse=True)
    sorted_torch_offspring = sorted(torch_offspring, key=lambda x: x.fitness, reverse=True)

    
    ind_logger = IndividualLogger()
    for ind in sorted_torch_offspring:
        ind_logger.log_individual(ind)
    ind_logger.save_reader_format()

    # plot_results(results=strategy_list, other_results=None)

def plot_algo(file_paths=None, dir_path=None, pop_path=None):
    res_list = []
    if pop_path is not None:
        with open(pop_path, "rb") as f:
            pop = pickle.load(f)
            all_inds = pop.all_individuals()
            all_handlers = [ESPopulation.get_handler_from_individual(ind) for ind in all_inds]
            for handler in all_handlers:
                if handler.error is not None:
                    continue
                res_list.append(handler.eval_result)
    elif dir_path is not None:
        file_paths = []
        if not os.path.isdir(dir_path):
            raise ValueError(f"Invalid directory path: {dir_path}")
        for file in os.listdir(dir_path):
            if file.endswith(".pkl"):
                file_paths.append(os.path.join(dir_path, file))
    
    if len(res_list) == 0:
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                target = pickle.load(f)
                if target.error is not None:
                    continue
                if isinstance(target, EvaluatorResult):
                    res_list.append(target)
                elif isinstance(target, ResponseHandler):
                    res_list.append(target.eval_result)
            
    plot_algo_results(results=res_list)
    

if __name__ == "__main__":
    # plot_search()

    file_paths = [
        # 'Experiments/pop_temp/ESPopulation_1+1_0207005138/5-6_KernelSelectionBOv5.pkl'
        'Experiments/algo_eval_res/BayesLocalAdaptiveAnnealBOv1_0208222147.pkl',
    ] 

    dir_path = None
    # pop_path = 'Experiments/pop_temp/ESPopulation_evol_2+2_IOHEvaluator_f3_dim-5_budget-100_instances-[1]_repeat-1_0207222633/ESPopulation_all_0207223344.pkl'
    pop_path = None

    plot_algo(file_paths=file_paths, dir_path=dir_path, pop_path=pop_path)

    pass
