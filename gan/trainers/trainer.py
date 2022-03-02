import os
from abc import abstractmethod
from typing import List
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.callbacks import Callback

from tqdm import tqdm

from gan.callbacks import basic_callback
from gan.utils import logger
from gan.callbacks import saver
from gan.datasets.dataset import Dataset
from gan.models.model import Model
from gan.trainers import gan_checkpoint_manager
from gan.utils import constants
from gan.utils import logger

SEED = 0

log = logger.get_logger(__name__)


class GANTrainer:

    def __init__(
            self,
            batch_size: int,
            generators: List[Model],
            discriminators: List[Model],
            training_name: str,
            generators_optimizers: List[Optimizer],
            discriminators_optimizers: List[Optimizer],
            continue_training: bool,
            save_images_every_n_steps: int,
            num_test_examples=None,
            checkpoint_step=10,
            save_model_every_n_step=100,
            callbacks: List[Callback] = None,
            validation_dataset=None,
    ):
        self.batch_size = batch_size
        self.generators = generators
        self.discriminators = discriminators
        self.checkpoint_step = checkpoint_step
        self.save_model_every_n_step = save_model_every_n_step
        self.training_name = training_name
        self.save_images_every_n_steps = save_images_every_n_steps
        self.num_test_examples = num_test_examples
        self.continue_training = continue_training
        self.validation_dataset = validation_dataset

        self.global_step = 0
        self.epoch = 0

        self.generators_optimizers = generators_optimizers
        self.discriminators_optimizers = discriminators_optimizers

        self.root_checkpoint_path = os.path.join(
            constants.SAVE_IMAGE_DIR,
            training_name,
        )
        self.logger = logger.TensorboardLogger(
            root_checkpoint_path=self.root_checkpoint_path,
        )
        self.checkpoint_manager = gan_checkpoint_manager.GANCheckpointManager(
            components_to_save={
                **self.generators_optimizers,
                **self.discriminators_optimizers,
                **{k: v.model for k, v in self.generators.items()},
                **{k: v.model for k, v in self.discriminators.items()}
            },
            root_checkpoint_path=self.root_checkpoint_path,
            continue_training=continue_training,
        )

        default_callbacks = [
            self.checkpoint_manager,
            basic_callback.GlobalStepIncrementer(),
        ]
        self.callbacks = callbacks + default_callbacks

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    def train(
            self,
            dataset: Dataset,
            num_epochs: int,
    ):
        global_step = 0
        dataset_tqdm = tqdm(
            iterable=dataset,
            desc="Batches",
            leave=True
        )

        latest_checkpoint_epoch = self.checkpoint_manager.regenerate_training()
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        num_epochs += latest_epoch
        for self.epoch in tqdm(range(latest_epoch, num_epochs), desc='Epochs'):
            self.on_epoch_begin()
            for batch in dataset_tqdm:
                self.on_training_step_begin()
                losses = self.train_step(batch)
                self.on_training_step_end()
                self.logger.log_scalars(name='Losses', scalars=losses, step=global_step)
                steps_per_second = 1. / dataset_tqdm.avg_time if dataset_tqdm.avg_time else 0.
                self.logger.log_scalars(name='', scalars={'steps_per_second': steps_per_second}, step=self.global_step)
            self.on_epoch_end()

    def on_epoch_begin(self):
        for c in self.callbacks:
            c.on_epoch_begin(self)

    def on_epoch_end(self):
        for c in self.callbacks:
            c.on_epoch_end(self)

    def on_training_step_begin(self):
        for c in self.callbacks:
            c.on_training_step_begin(self)

    def on_training_step_end(self):
        for c in self.callbacks:
            c.on_training_step_end(self)