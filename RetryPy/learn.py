from typing import Union
import tensorflow as tf
from tensorflow import keras

models = {
    ...
}

def compile_model(
    model_name: str,
    model_opts: dict,
    loss: Union[str, dict, keras.losses.Loss], # Dict or string or tf.keras.losses.Loss
    optimizer: Union[str, dict, keras.optimizers.Optimizer] # Dict or string
    ):
    model = models[model_name](**model_opts)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )