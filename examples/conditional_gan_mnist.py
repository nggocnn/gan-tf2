import numpy as np
import tensorflow as tf
from easydict import EasyDict
from tensorflow.keras.optimizers import Adam
from gan.callbacks import saver
from gan.datasets.mnist import MnistDataset
from gan.models.discriminators.conditional_dicriminator import ConditionalDiscriminator
from gan.models.generators.latent_to_image.conditional_latent_to_image import LatentToImageConditionalGenerator
from gan.trainers import conditional_gan_trainer

model_parameters = EasyDict({
    'img_height':                  28,
    'img_width':                   28,
    'num_channels':                1,
    'batch_size':                  16,
    'num_epochs':                  10,
    'buffer_size':                 1000,
    'latent_size':                 100,
    'num_classes':                 10,
    'learning_rate_generator':     0.0001,
    'learning_rate_discriminator': 0.0001,
    'save_images_every_n_steps':   10
})

dataset = MnistDataset(model_parameters, with_labels=True)


def validation_dataset():
    test_batch_size = model_parameters.num_classes ** 2
    labels = np.repeat(list(range(model_parameters.num_classes)), model_parameters.num_classes)
    validation_samples = [tf.random.normal([test_batch_size, model_parameters.latent_size]), np.array(labels)]
    return validation_samples


validation_dataset = validation_dataset()

generator = LatentToImageConditionalGenerator(model_parameters)
discriminator = ConditionalDiscriminator(model_parameters)

generator_optimizer = Adam(
    learning_rate=model_parameters.learning_rate_generator,
    beta_1=0.5,
)
discriminator_optimizer = Adam(
    learning_rate=model_parameters.learning_rate_discriminator,
    beta_1=0.5,
)

callbacks = [
    saver.ImageProblemSaver(
        save_images_every_n_steps=model_parameters.save_images_every_n_steps,
    )
]

gan_trainer = conditional_gan_trainer.ConditionalGANTrainer(
    batch_size=model_parameters.batch_size,
    generator=generator,
    discriminator=discriminator,
    training_name='CONDITIONAL_GAN_MNIST',
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    latent_size=model_parameters.latent_size,
    num_classes=model_parameters.num_classes,
    continue_training=False,
    save_images_every_n_steps=model_parameters.save_images_every_n_steps,
    validation_dataset=validation_dataset,
    callbacks=callbacks,
)

gan_trainer.train(
    dataset=dataset,
    num_epochs=model_parameters.num_epochs,
)
