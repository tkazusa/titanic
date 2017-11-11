# coding=utf-8

# write code...
from abc import ABCMeta, abstractmethod

class PreprocesserBase(object, metaclass=ABCMeta):
    """PreprocesserBase
    Provide interface for preprocesser class
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def fetch_origin_data(self):
        raise NotImplementedError

