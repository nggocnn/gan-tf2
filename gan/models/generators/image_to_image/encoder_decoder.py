import tensorflow_addons as tfa
from easydict import EasyDict
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from gan.models import model
from gan import advanced_layers


class EncoderDecoderGenerator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_images = layers.Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])

        x = layers.Conv2D(
            filters=64, kernel_size=(7, 7),
            padding='same', use_bias=False
        )(input_images)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=(2, 2),
            padding='same', use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=256, kernel_size=(3, 3), strides=(2, 2),
            padding='same', use_bias=False,
        )(x)

        x = layers.Conv2D(
            filters=256, kernel_size=(3, 3), strides=(2, 2),
            padding='same', use_bias=False,
        )(x)

        n_resnet = 6
        for _ in range(n_resnet):
            x = advanced_layers.residual_block(256, x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=(1, 1),
            padding='same', use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=(1, 1),
            padding='same', use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(
            filters=64, kernel_size=(3, 3), strides=(1, 1),
            padding='same', use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=32, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False,
        )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=3, kernel_size=(7, 7), strides=(1, 1),
            padding='same', use_bias=False, activation='tanh',
        )(x)

        generator = Model(name=self.model_name, inputs=input_images, outputs=x)
        return generator
