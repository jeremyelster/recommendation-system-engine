#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_recommendation_system_engine.training import Trainer
from surprise.model_selection import GridSearchCV
from surprise import Dataset

config = {'best_estimator.return_value': {'rmse': 1}}

@mock.patch('marvin_recommendation_system_engine.training.trainer.GridSearchCV', **config)
@mock.patch('marvin_recommendation_system_engine.training.trainer.GridSearchCV.fit')
def test_execute(grid_fit, grid_mocked, mocked_params):

    mocked_params = {
        "algo": [
            {
                "name": "SVD",
                "param_grid": {
                    "n_epochs": [10],
                    "lr_all": [0.005],
                    "reg_all": [0.6]
                }
            }
        ],
        "measures": ["rmse", "mae"],
        "n_cv": 3}

    data_mock = mock.Mock(spec=Dataset)
    test_dataset = {}
    test_dataset["data"] = data_mock



    #ac = Trainer(dataset=test_dataset)
    #ac.execute(params=mocked_params)

    #grid_mocked.assert_called_once()
    #grid_fit.assert_called_once()

    #grid_fit.return_value = 'rmse'
    #res = grid_fit.fit()

