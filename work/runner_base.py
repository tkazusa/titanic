# coding=utf-8

# write code...
import os
import numpy as np

from abc import ABCMeta, abstractmethod

class RunnerBase(object, metaclass=ABCMeta):
    """Runner base
    Provide interface for model runner class, which run model.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def _set_model(self):
        """ :rtype: ModelBase"""
        raise NotImplementedError

    @abstractmethod
    def _fetch_preprocessed_data(self):
        raise NotImplementedError

    @abstractmethod
    def load_X(self):
        raise NotImplementedError

    @abstractmethod
    def load_y(self):
        raise NotImplementedError

    @abstractmethod
    def _calc_w(self):
        """calculate weight for imbalance data"""
        raise NotImplementedError


    @abstractmethod
    def run_train(self):
        raise NotImplementedError

    @abstractmethod
    def run_train_alldata(self):
        raise NotImplementedError

    @abstractmethod
    def run_predict(self):
        raise NotImplementedError

