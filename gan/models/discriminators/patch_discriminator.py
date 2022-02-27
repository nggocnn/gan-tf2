import tensorflow_addons as tfa
from easydict import EasyDict
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from gan.models import model


class PatchDiscriminator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_layer = layers.Input(shape=self.model_parameters.input_shape)

        x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(input_layer)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', )(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.ZeroPadding2D()(x)

        x = layers.Conv2D(filters=512, kernel_size=(4, 4), strides=(1, 1), padding='valid')(x)
        x = tfa.layers.InstanceNormalization(axis=-1)(x)
        x = layers.LeakyReLU()(x)

        x = layers.ZeroPadding2D()(x)

        output_layer = layers.Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1), padding='valid')(x)

        discriminator = Model(name=self.model_name, inputs=input_layer, outputs=output_layer)

        return discriminator


class SinGANPatchDiscriminator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_layer = layers.Input(shape=self.model_parameters.input_shape)

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)

        output_layer = layers.LeakyReLU()(x)

        discriminator = Model(name=self.model_name, inputs=input_layer, outputs=output_layer)

        return discriminator
