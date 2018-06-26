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
        df_results = pd.DataFrame.from_dict(self.marvin_model["grid_search"].cv_results)

        # combination of parameters that gave the best RMSE score
        print("Best Model: {}".format([key + ": " + str(value) for (key, value) in marvin_model["grid_search"].best_params['rmse'].items()]))

        # best RMSE score
        print("Best RMSE: {}".format(marvin_model["grid_search"].best_score['rmse']))


        df_results[['params', 'mean_test_mae', 'mean_test_rmse', 'mean_test_time']].sort_values('mean_test_rmse')

