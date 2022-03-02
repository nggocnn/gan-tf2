from tensorflow.keras.callbacks import Callback


class GlobalStepIncrementer(Callback):
    def on_training_step_end(self, trainer):
        trainer.global_step += 1
