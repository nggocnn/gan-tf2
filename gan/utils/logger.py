from abc import ABC, abstractmethod

import tensorflow as tf
import logging

DEFAULT_LOGGER_VERBOSITY = logging.DEBUG
LOGGER_PATTERN = '%(levelname)s:%(name)s:%(message)s'


class Logger(ABC):

    @abstractmethod
    def log_scalars(self, name: str, scalars, step):
        pass

    @abstractmethod
    def log_images(self, name: str, images, step):
        pass


class TensorboardLogger(Logger):

    def __init__(
            self,
            root_checkpoint_path,
    ):
        self.root_checkpoint_path = root_checkpoint_path
        self.summary_writer = tf.summary.create_file_writer(self.root_checkpoint_path)

    def log_scalars(self, name: str, scalars, step):
        with self.summary_writer.as_default():
            [tf.summary.scalar(f'{name}/{scalar_name}', v, step=step) for scalar_name, v in scalars.items()]

    def log_images(self, name: str, images, step):
        with self.summary_writer.as_default():
            tf.summary.image(name=name, data=images, step=step)


class TensorboardLoggable(ABC):

    @abstractmethod
    def log_to_tensorboard(self, *args):
        pass


class StepsPerSecond(TensorboardLoggable):

    def log_to_tensorboard(
            self,
            dataset_tqdm,
            train_step,
    ):
        steps_per_second = 1. / dataset_tqdm.avg_time if dataset_tqdm.avg_time else 0.
        with self.summary_writer.as_default():
            tf.summary.scalar('steps_per_second', steps_per_second, train_step)


def get_logger(name=None, mod_name='gans-2.0', logger_verbosity=DEFAULT_LOGGER_VERBOSITY):
    logger_name = mod_name if name is None else f'{mod_name}.{name}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_verbosity)
    if logger.parent.hasHandlers():
        logger.parent.removeHandler(logger.parent.handlers[0])
    if not logger.hasHandlers():
        formatter = logging.Formatter(LOGGER_PATTERN)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logger_verbosity)
    return logger
