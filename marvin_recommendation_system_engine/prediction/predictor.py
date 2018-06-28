#!/usr/bin/env python
# coding=utf-8

"""Predictor engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBasePrediction

__all__ = ['Predictor']


logger = get_logger('predictor')


class Predictor(EngineBasePrediction):

    def __init__(self, **kwargs):
        super(Predictor, self).__init__(**kwargs)

    def execute(self, input_message, params, **kwargs):
        # get a prediction for specific users and items.
        pred_dict = {}

        for algo in params["algo"]:

            # Get Name and Initiate Algorithm
            algo_name = algo["name"]

            if algo.get("full_name", False):
                full_name = algo["full_name"]
            else:
                full_name = algo_name

            pred_dict[full_name] = self.marvin_model[full_name]["model"].predict(
                str(input_message["User_id"]), str(input_message["Item_id"]), r_ui=4, verbose=False)[3]


        final_prediction = pred_dict

        return final_prediction
