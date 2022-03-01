from abc import ABC, abstractmethod


class GAN(ABC):

    @property
    @abstractmethod
    def generators(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def discriminators(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs):
        raise NotImplementedError
