#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_recommendation_system_engine.training import MetricsEvaluator
import numpy as np

@mock.patch('marvin_recommendation_system_engine.training.metrics_evaluator.accuracy.rmse')
def test_execute(mocked_accuracy, mocked_params):


    mocked_params = {
        "algo": [
            {
                "name": "SVD",
                "param_grid": {
                    "n_epochs": [10],
                    "lr_all": [0.005],
                    "reg_all": [0.6]
                }
            }]}

    test_data = {
        'train': ['t0'],
        'testset': ['t2'],
        'val': ['t1']
    }



    mocked_model = mock.MagicMock()

    ac = MetricsEvaluator(model=mocked_model, dataset=test_data)
    ac.execute(params=mocked_params)
    mocked_accuracy.assert_called()

