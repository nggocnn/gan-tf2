import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from easydict import EasyDict

from gan.callbacks import saver
from gan.datasets.mnist import MnistDataset
from gan.models.discriminators import discriminator
from gan.models.generators.latent_to_image import latent_to_image
from gan.trainers import vanilla_gan_trainer


model_parameters = EasyDict({
    'img_height':                  28,
    'img_width':                   28,
    'num_channels':                1,
    'batch_size':                  16,
    'num_epochs':                  10,
    'buffer_size':                 1000,
    'latent_size':                 100,
    'learning_rate_generator':     0.0001,
    'learning_rate_discriminator': 0.0001,
    'save_images_every_n_steps':   10
})

dataset = MnistDataset(model_parameters)


def validation_dataset():
    return tf.random.normal([model_parameters.batch_size, model_parameters.latent_size])


validation_dataset = validation_dataset()

generator = latent_to_image.LatentToImageGenerator(model_parameters)
discriminator = discriminator.Discriminator(model_parameters)

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

gan_mnist_trainer = vanilla_gan_trainer.VanillaGANTrainer(
    batch_size=model_parameters.batch_size,
    generator=generator,
    discriminator=discriminator,
    training_name='VANILLA_GAN_MNIST',
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    latent_size=model_parameters.latent_size,
    continue_training=False,
    save_images_every_n_steps=model_parameters.save_images_every_n_steps,
    validation_dataset=validation_dataset,
    callbacks=callbacks,
)

gan_mnist_trainer.train(
    dataset=dataset,
    num_epochs=model_parameters.num_epochs,
)
