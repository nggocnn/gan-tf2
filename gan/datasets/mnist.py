import tensorflow as tf
from tensorflow.keras.datasets import mnist
from gan.datasets.dataset import Dataset
from gan.utils import data_utils


class MnistDataset(Dataset):

    def __init__(self, model_parameters, with_labels=False):
        super().__init__(model_parameters, with_labels)

    def __call__(self, *args, **kwargs):
        return self.train_dataset

    def load_data(self):
        (train_images, _), (_, _) = mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = data_utils.normalize_inputs(train_images)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
            self.buffer_size).batch(
            self.batch_size)
        return train_dataset

    def load_data_with_labels(self):
        (train_images, train_labels), (_, _) = mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = data_utils.normalize_inputs(train_images)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
            self.buffer_size).batch(
            self.batch_size)
        return train_dataset
