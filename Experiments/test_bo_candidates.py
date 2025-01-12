from llamea.evaluator import RandomBoTorchTestEvaluator, IOHEvaluator
from llamea.utils import IndividualLogger


# from Experiments.test_cands.AdvancedDropWaveBO import AdvancedDropWaveBO
# r = RandomBoTorchTestEvaluator.evaluate_from_cls(AdvancedDropWaveBO)

from Experiments.test_cands.DeepEnsembleQMCBO import DeepEnsembleQMCBO
# r = RandomBoTorchTestEvaluator.evaluate_from_cls(DeepEnsembleQMCBO)
# r = None

r = IOHEvaluator.evaluate_from_cls(DeepEnsembleQMCBO, eval_others=True)
print(r[0])
print(r[1])
