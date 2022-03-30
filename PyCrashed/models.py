import csv
import os
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from re import I
from typing import Tuple

import tensorflow as tf
from wandb.keras import WandbCallback

from PyCrashed.pipeline import Dataset

class Model():
    def __init__(
        self,
        name: str,
        use_wandb=True,
        verbose=True,
        paitence=None,
        kernel_width=None,
        head_width=None,
        dropout_rate=None,
        activation=None,
        ):
        # Set the model name
        self.name = name
        self.verbosity = 1 if verbose else 0

        self.kernel_width = kernel_width or 1
        self.head_width = head_width or 1
        self.activation = activation or 'relu'
        self.optimizer = tf.optimizers.Adam()
        self.loss = tf.losses.MeanSquaredError()
        self.dropout_rate = dropout_rate or 0
        self.paitence = paitence or 5

        self.metrics = [
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.KLDivergence()
        ]
        self.callbacks = []

        if use_wandb: self.callbacks.append(WandbCallback())
        now = datetime.now()
        log_dir = now.strftime(f"products/{self.name}/tb_logs/%m-%d/%H:%M:%S/")
        self.callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ))
        self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            f"products/{self.name}/checkpoint",
            monitor='val_loss'
        ))
        if paitence: self.callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_root_mean_squared_error',
            restore_best_weights=True,
            patience=self.paitence,
            verbose=self.verbosity
        ))

    @abstractmethod
    def specify_model(self) -> Tuple:
        # Should return just inputs and outputs
        ...

    def build(self):
        # Builds a model according to the inputs and outputs specified
        i, o = self.specify_model()
        self.model = tf.keras.Model(inputs=i, outputs=o, name="nvidia")
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )
        return self.model

    def fit(self,
            data: tf.data.Dataset = None,
            validation_data: tf.data.Dataset = None,
            n_epochs: int = 10
        ):
        if not data:
            data = Dataset.load("train")
        if not validation_data:
            validation_data = Dataset.load("val")

        self.fit_metrics = self.model.fit(
            data,
            epochs=n_epochs,
            validation_data=validation_data,
            verbose=self.verbosity,
            callbacks=self.callbacks
        )

        return self.fit_metrics

    def test(self, data: tf.data.Dataset = None):
        if not data:
            data = Dataset.load("test")
        self.test_metrics = self.model.evaluate(data, verbose=self.verbosity)
        return self.test_metrics

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_activation(self, activation):
        self.activation = activation

    @staticmethod
    def _save_metrics(cb, path):
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        with open(path, 'w') as f:
            w = csv.writer(f)
            w.writerow(cb.history.keys())
            w.writerows(zip(*cb.history.values()))

    def save(self, path = None) -> None:
        if not hasattr(self, "model"):
            raise ValueError("No model set")

        if not path:
            path = Path(f"products/{self.name}/model")

        # Save metrics and model
        self.model.save(path)
        if hasattr(self, "fit_metrics"):
            self._save_metrics(self.fit_metrics, Path.joinpath(path, "fit_metrics.csv"))

    def restore(self, path = None) -> None:
        if not path:
            path = Path(f"products/{self.name}/checkpoint")
        self.model = tf.keras.models.load_model(path)

class NVidia(Model):
    def __init__(self, **kwargs):
        super().__init__("Nvidia", activation=kwargs.get('activation', "elu") or "elu", **kwargs)

    def specify_model(self):
        i = tf.keras.Input(shape=(224, 224, 3))
        l = tf.keras.layers.BatchNormalization()(i)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 24), (5, 5), strides=(2, 2))(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 36), (5, 5), strides=(2, 2))(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 48), (5, 5), strides=(2, 2))(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        l = tf.keras.layers.MaxPool2D((2, 2))(l)
        l = tf.keras.layers.BatchNormalization()(i)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), strides=(1, 1))(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), strides=(1, 1))(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        l = tf.keras.layers.MaxPool2D((2, 2))(l)

        l = tf.keras.layers.Flatten()(l)
        
        l = tf.keras.layers.Dense(128)(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        l = tf.keras.layers.Dropout(0.25)(l)
        l = tf.keras.layers.Dense(64)(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        o = tf.keras.layers.Dense(2)(l)
        return i, o


class NVidiaBatchnorm(Model):
    def __init__(self, **kwargs):
        super().__init__("Nvidia_batchnorm", **kwargs)

    def specify_model(self):
        i = tf.keras.Input(shape=(224, 224, 3))
        l = tf.keras.layers.RandomContrast(0.2)(i)

        l = tf.keras.layers.Conv2D(int(self.kernel_width * 32), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 32), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 32), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 32), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 32), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 32), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.MaxPool2D((2, 2))(l)
        l = tf.keras.layers.BatchNormalization()(l)

        l = tf.keras.layers.Conv2D(int(self.kernel_width * 48), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 48), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 48), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 48), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 48), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 48), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.MaxPool2D((2, 2))(l)
        l = tf.keras.layers.BatchNormalization()(l)

        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.MaxPool2D((2, 2))(l)

        l = tf.keras.layers.Conv2D(int(self.kernel_width * 80), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 80), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 80), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 80), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 80), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 80), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)

        l = tf.keras.layers.Conv2D(int(self.kernel_width * 96), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 96), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 96), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.BatchNormalization()(l)

        l = tf.keras.layers.Conv2D(int(self.kernel_width * 112), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 112), (3, 3), activation=self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 112), (3, 3), activation=self.activation)(l)

        l = tf.keras.layers.Flatten()(l)
        l = tf.keras.layers.Dense(int(self.head_width * 1024), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(self.dropout_rate)(l)
        l = tf.keras.layers.Dense(int(self.head_width * 512), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(self.dropout_rate)(l)
        l = tf.keras.layers.Dense(int(self.head_width * 8), activation=self.activation)(l)
        o = tf.keras.layers.Dense(2)(l)
        return i, o

    
class ResNetPT(Model):
    def __init__(self, **kwargs):
        super().__init__("ResNetPT", **kwargs)
        self.loss = {
            "angle": tf.keras.losses.MeanSquaredError(),
            "speed": tf.keras.losses.BinaryCrossentropy()
        }
        self.metrics = {
            "angle": tf.keras.metrics.RootMeanSquaredError(),
            "speed": tf.keras.metrics.BinaryAccuracy(),
        }

    def specify_model(self):
        base_model = tf.keras.applications.ResNet152V2(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3),
            pooling="avg",
        )
        i = tf.keras.Input(shape=(224, 224, 3))
        i = tf.keras.layers.RandomContrast(0.2)(i)
        l = base_model(i)
        l = tf.keras.layers.Dense(int(self.head_width * 1024), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Dense(int(self.head_width * 512), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Dense(int(self.head_width * 128), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.BatchNormalization()(l)

        left = tf.keras.layers.Dense(int(self.head_width * 64), activation=self.activation)(l)
        left = tf.keras.layers.Dense(1, name="angle")(left)

        right = tf.keras.layers.Dense(int(self.head_width * 64), activation=self.activation)(l)
        right = tf.keras.layers.Dense(1, name="speed")(right)
        return i, (left, right)

    
class MobileNetPT(Model):
    def __init__(self, **kwargs):
        super().__init__("MobileNet", **kwargs)

    def specify_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False
        )
        inp = tf.keras.Input(shape=(224, 224, 3))
        img = base_model(inp)
        l = tf.keras.layers.MaxPooling2D((2, 2))(img)
        l = tf.keras.layers.Conv2D(96, (3, 3), strides=(1, 1), activation="relu")(l)
        l = tf.keras.layers.Flatten()(l)
        l = tf.keras.layers.Dense(1164, activation="relu")(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.Dense(64, activation="relu")(l)
        o = tf.keras.layers.Dense(2)(l)
        return inp, o

    
class EfficientNetPT(Model):
    def __init__(self, **kwargs):
        super().__init__("EfficientNet", **kwargs)

    def specify_model(self):
        base_model = tf.keras.applications.EfficientNet(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
            pooling="avg",
        )
        i = tf.keras.Input(shape=(224, 224, 3))
        i = tf.keras.layers.RandomContrast(0.2)(i)
        l = base_model(i)
        l = tf.keras.layers.Dense(int(self.head_width * 2048), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Dense(int(self.head_width * 1024), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Dense(int(self.head_width * 512), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.BatchNormalization()(l)

        left = tf.keras.layers.Dense(int(self.head_width * 64), activation=self.activation)(l)
        left = tf.keras.layers.Dense(1)(left)

        right = tf.keras.layers.Dense(int(self.head_width * 64), activation=self.activation)(l)
        right   = tf.keras.layers.Dense(1)(right)

class MultiHeaded(Model):
    def __init__(self, **kwargs):
        super().__init__("MultiHeaded", **kwargs)
        self.loss = (
            tf.keras.losses.MeanSquaredError(),
            tf.keras.losses.BinaryCrossentropy()
        )
        self.metrics = (
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.KLDivergence()
        )

    def specify_model(self):
        i = tf.keras.Input(shape=(320, 240, 3))
        l = tf.keras.layers.Resizing(240, 240)(i)
        l = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2))(i)
        l = tf.keras.layers.Conv2D(36, (5, 5), strides=(1, 1))(l)
        l = tf.keras.layers.MaxPooling2D((2, 2))(l)
        l = tf.keras.layers.Conv2D(48, (5, 5), strides=(1, 1))(l)
        l = tf.keras.layers.MaxPooling2D((2, 2))(l)
        l = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1))(l)
        l = tf.keras.layers.MaxPooling2D((2, 2))(l)
        l = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1))(l)
        l = tf.keras.layers.MaxPooling2D((2, 2))(l)
        l = tf.keras.layers.Conv2D(96, (3, 3), strides=(1, 1))(l)
        l = tf.keras.layers.MaxPooling2D((2, 2))(l)
        l = tf.keras.layers.Flatten()(l)

        left = tf.keras.layers.Dense(1164, activation="relu")(l)
        left = tf.keras.layers.Dropout(0.2)(left)
        left = tf.keras.layers.Dense(64, activation="relu")(left)
        left = tf.keras.layers.Dense(2, name="steering_angle")(left)

        right = tf.keras.layers.Dense(1164, activation="relu")(l)
        right = tf.keras.layers.Dropout(0.2)(right)
        right = tf.keras.layers.Dense(64, activation="relu")(right)
        right = tf.keras.layers.Dense(2, name="speed")(right)
        return i, (left, right)


