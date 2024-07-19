from ..evaluators import *
evaluator_map = {'accuracy': AccuracyEvaluator, 'rocauc': ROCAUCEvaluator, 'prauc': PRAUCEvaluator, 'hits': HitsEvaluator}