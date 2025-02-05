from llamea.evaluator.ioh_evaluator import IOHEvaluator
from llamea.individual import Individual
from llamea.utils import plot_results, plot_algo_results

from Experiments.test_cands.DeepEnsembleQMCBO import DeepEnsembleQMCBO
from Experiments.test_cands.TuRBO import TuRBO
from Experiments.test_cands.AdaptiveBatchTrustRegionBOv2 import AdaptiveBatchTrustRegionBOv2
from Experiments.test_cands.GPHybridAdaptiveBatchBOv2 import GPHybridAdaptiveBatchBOv2
from Experiments.baselines.bo_baseline import BLRandomSearch, BLTuRBO1, BLTuRBOM, BLRBFKernelVanillaBO, BLScaledKernelVanillaBO, BLSKOpt
from Experiments.test_cands.EnsembleLocalSearchBOv1 import EnsembleLocalSearchBOv1
from Experiments.test_cands.TrustRegionAdaptiveBOv1 import TrustRegionAdaptiveBOv1
from Experiments.test_cands.EnsembleDeepKernelAdaptiveTSLocalSearchARDv1 import EnsembleDeepKernelAdaptiveTSLocalSearchARDv1
problems = [7]
budget = 100
repeat = 2

cls_list = [
    BLRandomSearch,
    BLRBFKernelVanillaBO, 
    BLTuRBO1, 
    # BLTuRBOM,
    # EnsembleLocalSearchBOv1, 
    # BLSKOpt,
    TrustRegionAdaptiveBOv1,
    # EnsembleDeepKernelAdaptiveTSLocalSearchARDv1,
]

res = IOHEvaluator.evaluate_from_cls(cls_list, budget=budget, problems=problems, repeat=repeat)
plot_algo_results(results=res)

pass
