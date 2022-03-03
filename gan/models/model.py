from abc import ABC, abstractmethod
from easydict import EasyDict
from tensorflow import keras


class Model(ABC):
    def __init__(self, model_parameters: EasyDict):
        self._model_parameters_ = model_parameters
        self._model_ = self.define_model()

    def __call__(self, inputs, **kwargs):
        return self.model(inputs=inputs, **kwargs)

    @abstractmethod
    def define_model(self) -> keras.Model:
        raise NotImplementedError

    @property
    def trainable_variables(self):
        return self._model_.trainable_variables

    @property
    def model(self):
        return self._model_

    @property
    def model_parameters(self) -> EasyDict:
        return self._model_parameters_

    @property
    def num_channels(self) -> int:
        return self.model.output_shape[-1]

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self):
        return self.model_name

    @model_parameters.setter
    def model_parameters(self, value):
        self._model_parameters_ = value
