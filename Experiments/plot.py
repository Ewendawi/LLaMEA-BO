import pickle
import os
from functools import cmp_to_key
from datetime import datetime
import numpy as np
from llamea.utils import IndividualLogger
from llamea.population.es_population import ESPopulation
from llamea.prompt_generators.abstract_prompt_generator import ResponseHandler
from llamea.evaluator.evaluator_result import EvaluatorResult
from llamea.utils import plot_algo_result, plot_search_result, plot_group_bar

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

def plot_light_evol_and_final():
    file_paths = [

        # 0.04
        ('NoisyBanditBOv1', [
            'Experiments/final_eval_res/NoisyBanditBOv1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0210230205.pkl', 
            'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065417/0-10_NoisyBanditBOv1_handler.pkl'
                             ]),

        ('ParetoActiveBOv1', [
            'Experiments/final_eval_res/ParetoActiveBOv1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211000723.pkl', 
            'Experiments/pop_40_f/ESPopulation_evol_4+2_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210025714/0-3_ParetoActiveBOv1_handler.pkl'
            ]),

        # 0.05
        ('AdaptiveBatchUCBLocalSearchBOv2', [
            'Experiments/final_eval_res/AdaptiveBatchUCBLocalSearchBOv2_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211005941.pkl', 
            'Experiments/pop_40_f/ESPopulation_evol_4+4_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210043822/4-20_AdaptiveBatchUCBLocalSearchBOv2_handler.pkl'
            ]),

        ('AdaptiveControlVariateBOv4', [
            'Experiments/final_eval_res/AdaptiveControlVariateBOv4_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211014052.pkl', 
            'Experiments/pop_40_f/ESPopulation_evol_8+4_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209152106/6-29_AdaptiveControlVariateBOv4_handler.pkl'
            ]),

        # 0.06
        ('AdaptiveEvoBatchHybridBOv2', [
            'Experiments/pop_40_f/ESPopulation_evol_12+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208224540/4-32_AdaptiveEvoBatchHybridBOv2_handler.pkl',
            'Experiments/final_eval_res/AdaptiveEvoBatchHybridBOv2_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211020052.pkl'
        ]),
        
        ('MultiObjectiveBOv1', [
            'Experiments/final_eval_res/MultiObjectiveBOv1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211024828.pkl',
            'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065417/0-19_MultiObjectiveBOv1_handler.pkl'
        ]),

        ('AdaptiveTrustRegionDynamicAllocationBOv2:', [
            'Experiments/pop_40_f/ESPopulation_evol_8+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065623/2-22_AdaptiveTrustRegionDynamicAllocationBOv2_handler.pkl',
            'Experiments/final_eval_res/AdaptiveTrustRegionDynamicAllocationBOv2_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211043407.pkl'
        ]),

        ('AdaptiveHybridBOv6', [
            'Experiments/pop_40_f/ESPopulation_evol_1+1_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0211000947/10-11_AdaptiveHybridBOv6_handler.pkl',
            'Experiments/final_eval_res/AdaptiveHybridBOv6_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211045527.pkl'
        ]),

        # 0.08
        ('AdaptiveTrustImputationBOv2:', [
            'Experiments/pop_40_f/ESPopulation_evol_20+8_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209065238/2-31_AdaptiveTrustImputationBOv2_handler.pkl',
            'Experiments/final_eval_res/AdaptiveTrustImputationBOv2_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211002041.pkl'
        ]),
            
        ('TrustRegionAdaptiveTempBOv2', [
            'Experiments/pop_40_f/ESPopulation_evol_4+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0209173952/4-23_TrustRegionAdaptiveTempBOv2_handler.pkl',
            'Experiments/final_eval_res/TrustRegionAdaptiveTempBOv2_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211000039.pkl'
        ]),

        ('BayesLocalAdaptiveAnnealBOv1', [
            'Experiments/pop_40_temp/ESPopulation_evol_10+6_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0208164605/3-24_BayesLocalAdaptiveAnnealBOv1_handler.pkl',
            'Experiments/final_eval_res/BayesLocalAdaptiveAnnealBOv1_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211012527.pkl'
    ])
    ]

    groups = []
    group_names = []
    for group_name, file_paths in file_paths:
        _temp = [0,0]
        for file_path in file_paths:
            target = pickle.load(open(file_path, "rb"))
            if isinstance(target, EvaluatorResult):
                _temp[1] = target.score
            elif isinstance(target, ResponseHandler):
                _temp[0] = target.eval_result.score
        group_names.append(group_name)
        groups.append(_temp)
    
    y_data = np.array(groups)
    x_labels = ['Partial Eval', 'ALL Eval']

    plot_group_bar(y_data, 
                   x_labels, 
                   group_names, 
                   title='Comparison of Partial and All Evaluations',
                   fig_size=(15,9))


def plot_search_0209():
    # file_paths = [
    #     ("Experiments/pop_40_f/ESPopulation_evol_1+1_IOHEvaluator_f2_f4_f6_f8_f12_f14_f18_f15_f21_f23_dim-5_budget-100_instances-[1]_repeat-3_0210035334/ESPopulation_final_0210065735.pkl", "1+1"),
    # ]

    # pop_list = []
    # for file_path, name in file_paths:
    #     pop = pickle.load(open(file_path, "rb"))
    #     pop_list.append((name, pop))

    dir_path = 'Experiments/pop_40_f'
    pop_list = []
    best_pop_map = {}
    for dir_name in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, dir_name)):
            continue
        for file_name in os.listdir(os.path.join(dir_path, dir_name)):
            if "final" not in file_name:
                continue
            file_path = os.path.join(dir_path, dir_name, file_name)
            pop = pickle.load(open(file_path, "rb"))

            n_parent = pop.n_parent
            n_offspring = pop.n_offspring

            name = f"{n_parent}+{n_offspring}"

            cur_best = best_pop_map.get(name, None)
            if cur_best is None or pop.get_best_of_all().fitness > cur_best.get_best_of_all().fitness:
                best_pop_map[name] = pop
            
            pop_list.append((name, pop)) 
    
    save = True
    file_name = dir_path + '/' + f'df_res_{datetime.now().strftime("%m%d%H%M")}.pkl' 
    
    file_name = 'Experiments/pop_40_f/df_res_02110305.pkl'
    # file_name = None
    save = False
    
    plot_search_result(pop_list, save=save, file_name=file_name)


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
            
    plot_algo_result(results=res_list)
    

if __name__ == "__main__":
    # plot_search()

    file_paths = [
        'Experiments/final_eval_res/BLRandomSearch_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0210053711.pkl',

        'Experiments/final_eval_res/BLTuRBO1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0210055532.pkl',

        # 'Experiments/final_eval_res/BLTuRBOM_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0210061723.pkl',

        'Experiments/final_eval_res/BLMaternVanillaBO_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0210103519.pkl',


        'Experiments/final_eval_res/TrustRegionAdaptiveTempBOv2_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211000039.pkl',
        
        'Experiments/final_eval_res/BayesLocalAdaptiveAnnealBOv1_IOHEvaluator_ f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211012527.pkl',
        
        # 'Experiments/final_eval_res/EnsembleLocalSearchBOv1_IOHEvaluator: f1_f2_f3_f4_f5_f6_f7_f8_f9_f10_f11_f12_f13_f14_f15_f16_f17_f18_f19_f20_f21_f22_f23_f24_dim-5_budget-100_instances-[4, 5, 6]_repeat-5_0211041109.pkl',
    ] 

    dir_path = None
    pop_path = None

    plot_algo(file_paths=file_paths, dir_path=dir_path, pop_path=pop_path)

    # plot_light_evol_and_final()
    
    # plot_search_0209()

    pass
