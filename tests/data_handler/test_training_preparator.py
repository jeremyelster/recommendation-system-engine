#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_recommendation_system_engine.data_handler import TrainingPreparator
from surprise import Dataset

#@mock.patch('marvin_recommendation_system_engine.data_handler.training_preparator.surprise.Dataset')
def test_execute(mocked_params):

    test_dataset = {}
    test_dataset["data"] = mock.MagicMock()

    ac = TrainingPreparator(initial_dataset=test_dataset)
    ac.execute(params=mocked_params)


