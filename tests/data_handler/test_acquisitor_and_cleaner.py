#!/usr/bin/env python
# coding=utf-8

try:
    import mock

except ImportError:
    import unittest.mock as mock

from marvin_recommendation_system_engine.data_handler import AcquisitorAndCleaner


@mock.patch("marvin_recommendation_system_engine.data_handler.acquisitor_and_cleaner.Dataset.load_builtin")
def test_execute(data_mocked, mocked_params):

    data_mocked.return_value = ([1, 2], [3, 4])
    ac = AcquisitorAndCleaner()
    ac.execute(params=mocked_params)

    data_mocked.assert_called_once()
    data_mocked.assert_called_with('ml-100k')
