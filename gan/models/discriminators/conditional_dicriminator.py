from easydict import EasyDict
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from gan.models import model


class ConditionalDiscriminator(model.Model):
    def __init__(self, model_parameters: EasyDict):
        super().__init__(model_parameters)

    def define_model(self) -> keras.Model:
        input_layer = layers.Input(shape=self.model_parameters.input_shape)

        class_id = layers.Input(shape=[1])
        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=input_layer.shape[1] * input_layer.shape[2])(embedded_id)
        embedded_id = layers.Reshape(target_shape=(input_layer.shape[1], input_layer.shape[2], 1))(embedded_id)

        x = layers.Concatenate(axis=3)([input_layer, embedded_id])

        x = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(rate=0.3)(x)

        x = layers.Flatten()(x)

        output_layer = layers.Dense(units=1)(x)

        discriminator = Model(name=self.model_name, inputs=[input_layer, class_id], outputs=output_layer)

        return discriminator


class ConditionalDiscriminatorCifar10(model.Model):

    def __init__(
            self,
            model_parameters: EasyDict,
    ):
        super().__init__(model_parameters)

    def define_model(self):
        input_layer = layers.Input(shape=[self.model_parameters.input_shape])
        class_id = layers.Input(shape=[1])

        embedded_id = layers.Embedding(input_dim=10, output_dim=50)(class_id)
        embedded_id = layers.Dense(units=input_layer.shape[1] * input_layer.shape[2])(embedded_id)
        embedded_id = layers.Flatten()(embedded_id)

        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Flatten()(x)

        x = layers.Concatenate()([x, embedded_id])

        x = layers.Dense(units=512, activation='relu')(x)

        output_layer = layers.Dense(units=1)(x)

        discriminator = Model(name=self.model_name, inputs=[input_layer, class_id], outputs=output_layer)

        return discriminator
