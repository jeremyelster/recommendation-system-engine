#!/usr/bin/env python
# coding=utf-8

"""MetricsEvaluator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['MetricsEvaluator']


logger = get_logger('metrics_evaluator')


class MetricsEvaluator(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(MetricsEvaluator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        import pandas as pd
        from surprise import accuracy

        for algo in params["algo"]:
            print(algo)
            # combination of parameters that gave the best RMSE score
            print("Best Model: {}".format([key + ": " + str(value) for (key, value) in self.marvin_model[algo["name"]]["grid_search"].best_params['rmse'].items()]))

            # best RMSE score
            print("Best RMSE: {}".format(self.marvin_model[algo["name"]]["grid_search"].best_score['rmse']))

            # Prediction Score
            # Train the algorithm on the trainset, and predict ratings for the testset
            predictions = self.marvin_model[algo["name"]]["model"].test(self.marvin_dataset["testset"])

            # Then compute RMSE
            print("Test Set Score: {}".format(accuracy.rmse(predictions)))

