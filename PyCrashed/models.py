import csv
import os
from abc import abstractmethod
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from wandb.keras import WandbCallback

class BaseModel():
    """A base class which implements some convenient things that need to be done for every model"""
    def __init__(
        self,
        name: str,
        use_wandb=True,
        verbose=True,
        paitence=None,
        kernel_width=None,
        network_width=None,
        dropout_rate=None,
        activation=None,
        ):
        # Set the model name
        self.name = name
        self.verbosity = 1 if verbose else 0

        # Define the parameters
        self.kernel_width = kernel_width or 1
        self.network_width = network_width or 1
        self.activation = activation or 'relu'
        self.optimizer = tf.optimizers.Adam()
        self.loss = tf.losses.MeanSquaredError()
        self.dropout_rate = dropout_rate or 0
        self.paitence = paitence or 5

        # Setup default metrics, callbacks
        self.metrics = [
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.KLDivergence()
        ]
        self.callbacks = [tf.keras.callbacks.ModelCheckpoint(
            f"products/{self.name}/checkpoint",
            monitor='val_loss'
        )]
        if use_wandb: self.callbacks.append(WandbCallback())
        if paitence: self.callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            restore_best_weights=True,
            patience=self.paitence,
            verbose=self.verbosity
        ))

    @abstractmethod
    def specify_model(self) -> Tuple:
        # Should return references to input and output tensors
        ...

    def build(self):
        # Builds a model according to the inputs and outputs specified
        i, o = self.specify_model()
        self.model = tf.keras.Model(inputs=i, outputs=o, name=self.name)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )
        self.is_split = isinstance(o, (list, tuple))
        return self.model

    def fit(self,
            data: tf.data.Dataset,
            validation_data: tf.data.Dataset,
            n_epochs: int = 10
        ):
        """Literally just calls keras.models.Model.fit()
        Consider removing the validation data; for some reason
        the distribution strategy doesnt support putting validation
        calculations on a GPU, so in some cases its the same speed
        as the training step...
        """
        self.fit_metrics = self.model.fit(
            data,
            epochs=n_epochs,
            validation_data=validation_data,
            verbose=self.verbosity,
            callbacks=self.callbacks
        )

        return self.fit_metrics

    def test(self, data: tf.data.Dataset):
        """Just calls keras.models.Model.evaluate()"""
        self.test_metrics = self.model.evaluate(data, verbose=self.verbosity)
        return self.test_metrics

    def add_callback(self, callback):
        """Appends a callback to the list"""
        self.callbacks.append(callback)

    def set_loss(self, loss):
        """Set the model loss function"""
        self.loss = loss

    def set_optimizer(self, optimizer):
        """Set the model optimizer"""
        self.optimizer = optimizer

    def set_activation(self, activation):
        """Set the activation function in the model"""
        self.activation = activation

    @staticmethod
    def _save_metrics(cb, path):
        """Save the models metrics to a csv file in the model path"""
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        with open(path, 'w') as f:
            w = csv.writer(f)
            w.writerow(cb.history.keys())
            w.writerows(zip(*cb.history.values()))

    def save(self, path = None) -> None:
        """Save the model to the proper directory"""
        if not hasattr(self, "model"):
            raise ValueError("No model set")

        if not path:
            path = Path(f"products/{self.name}/model")

        # Save metrics and model
        self.model.save(path)
        if hasattr(self, "fit_metrics"):
            self._save_metrics(self.fit_metrics, Path.joinpath(path, "fit_metrics.csv"))

    def restore(self, path = None) -> None:
        """Restore a model from a saved_model file"""
        if not path:
            path = Path(f"products/{self.name}/checkpoint")
        self.model = tf.keras.models.load_model(path)

class NVidia(BaseModel):
    """Implements the model specified by the Nvidia researchers"""
    def __init__(self, **kwargs):
        super().__init__("Nvidia", **kwargs)

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
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), strides=(1, 1))(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        l = tf.keras.layers.Conv2D(int(self.kernel_width * 64), (3, 3), strides=(1, 1))(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        l = tf.keras.layers.MaxPool2D((2, 2))(l)

        l = tf.keras.layers.Flatten()(l)

        l = tf.keras.layers.Dense(int(self.network_width * 128))(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        l = tf.keras.layers.Dropout(0.25)(l)
        l = tf.keras.layers.Dense(int(self.network_width * 64))(l)
        l = tf.keras.layers.Activation(self.activation)(l)
        o = tf.keras.layers.Dense(2)(l)
        return i, o

class ResNet(BaseModel):
    """Implements a ResNet model, with split output heads. This model can be a little bit unstable"""
    def __init__(self, **kwargs):
        super().__init__("ResNetPT", **kwargs)
        self.loss = {
            "angle": tf.keras.losses.MeanAbsoluteError(),
            "speed": tf.keras.losses.BinaryCrossentropy()
        }
        self.metrics = {
            "angle": tf.keras.metrics.RootMeanSquaredError(),
            "speed": tf.keras.metrics.BinaryAccuracy(),
        }

    def specify_model(self):
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
            pooling="avg",
        )
        i = tf.keras.Input(shape=(224, 224, 3))
        i = tf.keras.layers.RandomContrast(0.2)(i)
        l = tf.keras.layers.BatchNormalization()(i)
        l = base_model(l)

        left = tf.keras.layers.Dense(int(self.network_width * 128))(l)
        left = tf.keras.layers.Activation(self.activation)(left)
        left = tf.keras.layers.BatchNormalization()(left)
        left = tf.keras.layers.Dense(int(self.network_width * 64))(l)
        left = tf.keras.layers.Activation(self.activation)(left)
        left = tf.keras.layers.BatchNormalization()(left)
        left = tf.keras.layers.Dense(1, name="angle")(left)

        right = tf.keras.layers.Dense(int(self.network_width * 128))(l)
        right = tf.keras.layers.Activation(self.activation)(right)
        right = tf.keras.layers.BatchNormalization()(right)
        right = tf.keras.layers.Dense(int(self.network_width * 64))(l)
        right = tf.keras.layers.Activation(self.activation)(right)
        right = tf.keras.layers.BatchNormalization()(right)
        right = tf.keras.layers.Dense(1, name="speed")(right)
        return i, (left, right)
    
class EfficientNet(BaseModel):
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
        l = tf.keras.layers.Dense(int(self.network_width * 2048), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Dense(int(self.network_width * 1024), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.Dense(int(self.network_width * 512), activation=self.activation)(l)
        l = tf.keras.layers.Dropout(0.2)(l)
        l = tf.keras.layers.BatchNormalization()(l)

        left = tf.keras.layers.Dense(int(self.network_width * 64), activation=self.activation)(l)
        left = tf.keras.layers.Dense(1)(left)

        right = tf.keras.layers.Dense(int(self.network_width * 64), activation=self.activation)(l)
        right   = tf.keras.layers.Dense(1)(right)
    
class InceptionResNet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__("InceptionResNet", **kwargs)
        self.loss = {
            "angle": tf.keras.losses.MeanSquaredError(),
            "speed": tf.keras.losses.BinaryCrossentropy()
        }
        self.metrics = {
            "angle": tf.keras.metrics.RootMeanSquaredError(),
            "speed": tf.keras.metrics.BinaryAccuracy(),
        }

    def specify_model(self):
        base_model = tf.keras.applications.InceptionResNetV2(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3),
            pooling="avg",
        )
        i = tf.keras.Input(shape=(224, 224, 3))
        i = tf.keras.layers.RandomContrast(0.2)(i)
        l = tf.keras.layers.BatchNormalization()(i)
        l = base_model(l)

        left = tf.keras.layers.Dense(int(self.network_width * 128))(l)
        left = tf.keras.layers.Activation(self.activation)(left)
        left = tf.keras.layers.BatchNormalization()(left)
        left = tf.keras.layers.Dense(int(self.network_width * 64))(l)
        left = tf.keras.layers.Activation(self.activation)(left)
        left = tf.keras.layers.BatchNormalization()(left)
        left = tf.keras.layers.Dense(1, name="angle")(left)

        right = tf.keras.layers.Dense(int(self.network_width * 128))(l)
        right = tf.keras.layers.Activation(self.activation)(right)
        right = tf.keras.layers.BatchNormalization()(right)
        right = tf.keras.layers.Dense(int(self.network_width * 64))(l)
        right = tf.keras.layers.Activation(self.activation)(right)
        right = tf.keras.layers.BatchNormalization()(right)
        right = tf.keras.layers.Dense(1, name="speed")(right)

        return i, (left, right)

class MultiHeaded(BaseModel):
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