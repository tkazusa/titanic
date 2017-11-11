# coding=utf-8

# write code...

from abc import ABCMeta, abstractmethod
from model_lgbm_initial import ModelLgbm_initial

class RunnerBase(metaclass=ABCMeta):
    """Runner base
    Provide interface for model runner class, which run model.
    """

    #@abstractmethod
    #def __init__(self):
    #    raise NotImplementedError

    @abstractmethod
    def _set_model(self):
        """ :rtype: ModelBase"""
        raise NotImplementedError

class RunnerLgbm_initial(RunnerBase):
    """Runner base
    Provide interface for model runner class, which run model.
    _"""

    def _set_model(self):
        print("oishii")

if __name__ == "__main__":
    tmp = RunnerLgbm_initial()