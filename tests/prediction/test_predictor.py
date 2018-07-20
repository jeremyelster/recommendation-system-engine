#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_recommendation_system_engine.prediction import Predictor


def test_execute(mocked_params):

    mocked_input = {
        "User_id": 196,
        "Item_id": 302
    }

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

    mocked_model = {
        "SVD": {
            "model": mock.MagicMock()
        }
    }

    ac = Predictor(model=mocked_model)
    ac.execute(input_message=mocked_input, params=mocked_params)
    mocked_model['SVD']['model'].predict.assert_called_once()

    # Get Name
