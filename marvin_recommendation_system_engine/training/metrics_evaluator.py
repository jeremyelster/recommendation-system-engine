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

        metrics_dict = {}

        for algo in params["algo"]:

            algo_name = algo["name"]

            if algo.get("full_name", False):
                full_name = algo["full_name"]
            else:
                full_name = algo_name
            print(full_name)

            metrics_dict[full_name] = {}

            # combination of parameters that gave the best RMSE score
            best_model = [key + ": " + str(value) for (key, value) in self.marvin_model[full_name]["grid_search"].best_params['rmse'].items()]
            #print("Best Model: {}".format(best_model))

            # best RMSE score
            train_rmse = self.marvin_model[full_name]["grid_search"].best_score['rmse']
            #print("Train RMSE: {}".format(train_rmse))

            # Prediction Score
            # Train the algorithm on the trainset, and predict ratings for the testset
            predictions = self.marvin_model[full_name]["model"].test(self.marvin_dataset["testset"])
            # print(len(predictions))
            # Then compute RMSE
            test_rmse = accuracy.rmse(predictions, verbose=False)
            #print("Test Set Score: {}".format(test_rmse))

            metrics_dict[full_name]["best_model"] = best_model
            metrics_dict[full_name]["train_rmse"] = train_rmse
            #metrics_dict[full_name]["predictions"] = predictions
            metrics_dict[full_name]["test_rmse"] = test_rmse


        self.marvin_metrics = metrics_dict

