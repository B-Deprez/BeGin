from ..evaluators import *
evaluator_map = {'accuracy': AccuracyEvaluator, 'rocauc': ROCAUCEvaluator, 'prauc': PRAUCEvaluator, 'f1micro': MicroF1Evaluator, 'hits': HitsEvaluator}