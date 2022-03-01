from easydict import EasyDict
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from gan.models import model


class Discriminator(model.Model):
    def __init__(self, model_parameters: EasyDict):
        super().__init__(model_parameters)

    def define_model(self) -> keras.Model:
        input_layer = layers.Input(shape=[
            self.model_parameters.img_height,
            self.model_parameters.img_width,
            self.model_parameters.num_channels
        ])
        x = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')(input_layer)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Flatten()(x)

        output_layer = layers.Dense(units=1)(x)

        discriminator = Model(name=self.model_name, inputs=[input_layer], outputs=[output_layer])

        return discriminator

