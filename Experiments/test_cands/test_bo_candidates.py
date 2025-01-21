from llamea.evaluator import RandomBoTorchTestEvaluator, IOHEvaluator

# from Experiments.test_cands.AdvancedDropWaveBO import AdvancedDropWaveBO
# r = RandomBoTorchTestEvaluator.evaluate_from_cls(AdvancedDropWaveBO)

from Experiments.test_cands.DeepEnsembleQMCBO import DeepEnsembleQMCBO
# r = RandomBoTorchTestEvaluator.evaluate_from_cls(DeepEnsembleQMCBO)
# r = None

from Experiments.test_cands.GPBO import GPBO
# r = IOHEvaluator.evaluate_from_cls(GPBO, eval_others=True)


# from Experiments.test_cands.RobustGPBO import RobustGPBO
# r = RandomBoTorchTestEvaluator.evaluate_from_cls(RobustGPBO)

from Experiments.test_cands.TuRBO import TuRBO
# r = IOHEvaluator.evaluate_from_cls(TuRBO, budget=100, eval_others=True)

from Experiments.test_cands.AdaptiveBatchBOv1 import AdaptiveBatchBOv1
r = IOHEvaluator.evaluate_from_cls(AdaptiveBatchBOv1, budget=100, eval_others=False)

print(r[0])
for _r in r[1]:
    print(_r)

pass
