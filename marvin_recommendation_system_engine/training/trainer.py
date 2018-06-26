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
        from surprise import KNNWithMeans

        algo = params["algo"](sim_options=params["sim_options"])
        algo.fit(self.marvin_dataset["trainset"])


        # Get the predictions for null values in the set
        if params["prediction"]["pred_type"] == "top_n":
            predictions = algo.test(self.marvin_dataset["testset"])
        else:
            predictions = "To generate predictions, set prediction pred_type to top_n"

        self.marvin_model = {
            #"grid_search": gs,
            "model": algo,
            "predictions": predictions
        }

