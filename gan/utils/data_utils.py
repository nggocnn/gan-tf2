import numpy as np
import tensorflow as tf


def normalize_inputs(data):
    """
    Normalizes the inputs to [-1, 1]
    :param data: input data array
    :return: normalized data to [-1, 1]
    """
    data = (data - 127.5) / 127.5
    return data


def create_test_labels(batch_size):
    labels = [[i] * 10 for i in list(range(10))]
    test_labels = [tf.random.normal([batch_size, 100]),
                   np.array(labels)]
    return test_labels
