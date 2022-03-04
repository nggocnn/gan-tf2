from easydict import EasyDict
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers

from gan.models import model


class LatentToImageGenerator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_layer = layers.Input(shape=[self.model_parameters.latent_size])

        x = layers.Dense(units=(7 * 7 * 256), use_bias=False)(input_layer)
        print(x.shape)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape(target_shape=(7, 7, 256))(x)
        x = layers.Conv2DTranspose(
            filters=128, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(
            filters=64, kernel_size=(5, 5), strides=(2, 2),
            padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        output_layer = layers.Conv2DTranspose(
            filters=1, kernel_size=(5, 5), strides=(2, 2),
            padding='same', use_bias=False, activation='tanh'
        )(x)

        generator = Model(name=self.model_name, inputs=input_layer, outputs=output_layer)

        return generator


class LatentToImageCifar10Generator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_layer = layers.Input(shape=[self.model_parameters.latent_size])

        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape(target_shape=(8, 8, 256))(x)
        x = layers.Conv2DTranspose(
            filters=128, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(
            filters=64, kernel_size=(5, 5), strides=(2, 2),
            padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        output_layer = layers.Conv2DTranspose(
            filters=3, kernel_size=(5, 5), strides=(2, 2),
            padding='same', use_bias=False, activation='tanh'
        )(x)

        generator = Model(name=self.model_name, inputs=input_layer, outputs=output_layer)
        return generator


class LatentToImageCifar10NearestNeighborUpSamplingGenerator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_layer = layers.Input(shape=[self.model_parameters.latent_size])

        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape(target_shape=(8, 8, 256))(x)
        x = layers.Conv2D(
            filters=128, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=64, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=3, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False, activation='tanh'
        )(x)

        generator = Model(name=self.model_name, inputs=input_layer, outputs=x)
        return generator
