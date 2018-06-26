#!/usr/bin/env python
# coding=utf-8

"""Trainer engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['Trainer']


logger = get_logger('trainer')


class Trainer(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        from surprise.model_selection import GridSearchCV
        from surprise import SVD
        from surprise import KNNBaseline

        algo_dict = {"SVD": SVD, "KNNBaseline": KNNBaseline}

        model_dict = {}

        for algo in params["algo"]:

            print(algo)

            # Get Name and Initiate Algorithm
            algo_name = algo["name"]
            model_dict[algo_name] = {}

            # Initialize Gridsearch
            gs = GridSearchCV(
                algo_dict[algo_name],
                algo["param_grid"],
                measures=params["measures"],
                cv=params["n_cv"])

            gs.fit(self.marvin_dataset["data"])

            # We can now use the algorithm that yields the best rmse:
            best_algo = gs.best_estimator['rmse']
            best_algo.fit(self.marvin_dataset["trainset"])

            # Get the predictions for null values in the set
            model_dict[algo_name]["grid_search"] = gs
            model_dict[algo_name]["model"] = best_algo

        self.marvin_model = model_dict

