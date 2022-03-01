from easydict import EasyDict
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from gan.models import model


class LatentToImageConditionalGenerator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        latent_input = layers.Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])

        class_id = layers.Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=7 * 7)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(7, 7, 1))(embedded_id)

        x = layers.Dense(units=7 * 7 * 256, use_bias=False)(latent_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((7, 7, 256))(x)

        inputs = layers.Concatenate(axis=3)([x, embedded_id])

        x = layers.Conv2DTranspose(
            filters=128, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False
        )(inputs)
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

        generator = Model(name=self.model_name, inputs=[latent_input, class_id], outputs=output_layer)
        return generator


class LatentToImageCifar10CConditionalGenerator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        latent_input = layers.Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])
        class_id = layers.Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=8 * 8)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(8, 8, 1))(embedded_id)

        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(latent_input)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Reshape((8, 8, 256))(x)

        inputs = layers.Concatenate(axis=3)([x, embedded_id])

        x = layers.Conv2DTranspose(
            filters=128, kernel_size=(4, 4), strides=(2, 2),
            padding='same', use_bias=False
        )(inputs)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(
            filters=128, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2DTranspose(
            filters=128, kernel_size=(4, 4), strides=(2, 2),
            padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(
            filters=128, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(
            filters=128, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        output_layer = layers.Conv2D(
            filters=3, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False, activation='tanh'
        )(x)

        generator = Model(name=self.model_name, inputs=[latent_input, class_id], outputs=output_layer)

        return generator


class LatentToImageNNUpSamplingCifar10CConditionalGenerator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        latent_input = layers.Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])
        class_id = layers.Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=8 * 8)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(8, 8, 1))(embedded_id)

        x = layers.Dense(units=8 * 8 * 256, use_bias=False)(latent_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((8, 8, 256))(x)

        inputs = layers.Concatenate(axis=3)([x, embedded_id])

        x = layers.Conv2D(
            filters=128, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.UpSampling2D()(x)

        x = layers.Conv2D(
            filters=64, kernel_size=(5, 5),
            strides=(1, 1), padding='same', use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.UpSampling2D()(x)

        output_layer = layers.Conv2D(
            filters=3, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False, activation='tanh'
        )(x)

        generator = Model(name=self.model_name, inputs=[latent_input, class_id], outputs=output_layer)
        return generator


class LatentToImageNNUpSamplingConditionalGenerator(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        latent_input = layers.Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])
        class_id = layers.Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=7 * 7)(embedded_id)
        embedded_id = layers.Reshape(target_shape=(7, 7, 1))(embedded_id)

        x = layers.Dense(units=7 * 7 * 256, use_bias=False)(latent_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((7, 7, 256))(x)

        inputs = layers.Concatenate(axis=3)([x, embedded_id])

        x = layers.Conv2D(
            filters=128, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False
        )(inputs)
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

        output_layer = layers.Conv2D(
            filters=1, kernel_size=(5, 5), strides=(1, 1),
            padding='same', use_bias=False, activation='tanh'
        )(x)

        generator = Model(name=self.model_name, inputs=[latent_input, class_id], outputs=output_layer)
        return generator
