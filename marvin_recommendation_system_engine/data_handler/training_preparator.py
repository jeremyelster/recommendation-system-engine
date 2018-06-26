#!/usr/bin/env python
# coding=utf-8

"""TrainingPreparator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['TrainingPreparator']


logger = get_logger('training_preparator')


class TrainingPreparator(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(TrainingPreparator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        trainset = self.marvin_initial_dataset["data"].build_full_trainset()
        print(trainset.global_mean)
        testset = trainset.build_anti_testset()
        print(testset[0])
        self.marvin_dataset = {
            #"data": self.marvin_initial_dataset["data"],
            "trainset": trainset,
            "testset": testset
        }

