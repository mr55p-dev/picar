import csv
import os
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from tabnanny import verbose
from typing import Tuple

import tensorflow as tf

from PyCrashed.pipeline import Dataset


class Model():
    def __init__(
        self,
        name: str,
        use_logging=True,
        use_early_stopping=True,
        use_checkpoints=True,
        verbose=True,
        ):
        # Set the model name
        self.name = name
        self.verbosity = 1 if verbose else 0
        self.callbacks = []
        if use_logging:
            now = datetime.now()
            log_dir = now.strftime(f"products/{self.name}/tb_logs/%m-%d/%H:%M:%S/")
            self.callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1
                )
            )
            self.callbacks.append(
                tf.keras.callbacks.RemoteMonitor(
                    root="https://tf-picar-listener.herokuapp.com/",
                    path=f"tf/{self.name}"
                )
            )
        if use_checkpoints:
            self.callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    f"products/{self.name}/checkpoint",
                    monitor='val_loss',
                    save_best_only=False,
                    save_weights_only=True
                )
            )
        if use_early_stopping:
            self.callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0,
                    patience=2,
                    restore_best_weights=True
                )
            )

        # Set useful default attrs
        self.optimizer = tf.keras.optimizers.Nadam()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = [
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.KLDivergence()
        ]

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

class NVidia(Model):
    def __init__(self, **kwargs):
        super().__init__("Nvidia", **kwargs)

    def specify_model(self):
        i = tf.keras.Input(shape=(320, 240, 3))
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
        l = tf.keras.layers.Dense(1164, activation="relu")(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.Dense(64, activation="relu")(l)
        o = tf.keras.layers.Dense(2)(l)
        return i, o


class ImageNetPretrained(Model):
    def __init__(self, **kwargs):
        super().__init__("ImageNetPretrained", **kwargs)

    def specify_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False
        )
        inp = tf.keras.Input(shape=(320, 240, 3))
        res = tf.keras.layers.Resizing(224, 224)(inp)
        img = base_model(res, training=False)
        l = tf.keras.layers.MaxPooling2D((2, 2), name="mp1")(img)
        l = tf.keras.layers.Conv2D(96, (3, 3), strides=(1, 1), name="cnv1")(l)
        l = tf.keras.layers.Flatten(name="flatten")(l)
        l = tf.keras.layers.Dense(1164, activation="relu", name="d1")(l)
        l = tf.keras.layers.Dropout(0.2, name="drop")(l)
        l = tf.keras.layers.Dense(64, activation="relu", name="d2")(l)
        o = tf.keras.layers.Dense(2, name="do")(l)
        return inp, o


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


