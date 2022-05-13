# %%
from pathlib import Path
from PyCrashed.datanew import Config, Data
from PyCrashed.data import Data as DataOld

import tensorflow as tf
from tensorflow import keras


# %% Create a model
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    # i = keras.Input(shape=(320, 240, 3))
    # l = keras.layers.Resizing(224, 224)(i)
    # l = keras.layers.RandomContrast(0.2)(l)
    # l = keras.layers.BatchNormalization()(l)
    # l = keras.layers.Conv2D(24, (5, 5), strides=(2, 2))(l)
    # l = keras.layers.Activation('relu')(l)
    # l = keras.layers.Conv2D(36, (5, 5), strides=(2, 2))(l)
    # l = keras.layers.Activation('relu')(l)
    # l = keras.layers.Conv2D(48, (5, 5), strides=(2, 2))(l)
    # l = keras.layers.Activation('relu')(l)
    # l = keras.layers.MaxPool2D((2, 2))(l)
    # l = keras.layers.BatchNormalization()(l)
    # l = keras.layers.Conv2D(64, (3, 3), strides=(1, 1))(l)
    # l = keras.layers.Activation('relu')(l)
    # l = keras.layers.Conv2D(64, (3, 3), strides=(1, 1))(l)
    # l = keras.layers.Activation('relu')(l)
    # l = keras.layers.MaxPool2D((2, 2))(l)

    # l = keras.layers.Flatten()(l)

    # l = keras.layers.Dense(128)(l)
    # l = keras.layers.Activation('relu')(l)
    # l = keras.layers.Dropout(0.25)(l)
    # l = keras.layers.Dense(64)(l)
    # l = keras.layers.Activation('relu')(l)
    # o = keras.layers.Dense(2)(l)

    # model = keras.Model(inputs=i, outputs=o)
    # model.compile(
    #     optimizer=keras.optimizers.Adam(),
    #     loss=keras.losses.MeanSquaredError(),
    #     metrics=keras.metrics.RootMeanSquaredError()
    # )


    model = keras.models.load_model("newmodel/nvidia/checkpoint")
    Config.set_labels_path(Path("data/track_1.csv"))
    Config.set_train_path(Path("data/Track1"))
    train_ds_1, val_ds_1 = Data.training(1, 0, 4, False)

    hist1 = model.fit(
        train_ds_1,
        epochs=20
    )

    Config.set_labels_path(Path("data/track_2.csv"))
    Config.set_train_path(Path("data/Track2"))
    train_ds_2, val_ds_2 = Data.training(1, 0, 4, False)

    hist1 = model.fit(
        train_ds_2,
        epochs=20
    )
    model.save("newmodel/nvidia/stage1")

    train_ds_norm, val_ds_norm = DataOld.training(1, 0, 64, False)

    hist2 = model.fit(
        train_ds_norm,
        epochs=30,
        callbacks=[keras.callbacks.ModelCheckpoint("newmodel/nvidia/checkpoint")]
    )
    model.save("newmodel/nvidia/stage2")

    Config.set_labels_path(Path("data/track_2.csv"))
    Config.set_train_path(Path("data/Track2"))
    train_ds_2, val_ds_2 = Data.training(1, 0, 4, False)

    hist1 = model.fit(
        train_ds_2,
        epochs=20
    )
    Config.set_labels_path(Path("data/track_1.csv"))
    Config.set_train_path(Path("data/Track1"))
    train_ds_1, val_ds_1 = Data.training(1, 0, 4, False)

    hist1 = model.fit(
        train_ds_1,
        epochs=20
    )

    model.save("newmodel/nvidia/final")

# %%
