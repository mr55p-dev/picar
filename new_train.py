# %%
from pathlib import Path
from PyCrashed.data import Config, Data

import tensorflow as tf
from tensorflow import keras


# %% Create a model
i = keras.Input(shape=(320, 240, 3))
l = keras.layers.Resizing(224, 224)(i)
l = keras.layers.RandomContrast(0.2)(l)
l = keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='max'
)(l)
l = keras.layers.BatchNormalization()(l)
l = keras.layers.Dense(128)(l)
l = keras.layers.Activation('relu')(l)
l = keras.layers.BatchNormalization()(l)

left = tf.keras.layers.Dense(64)(l)
left = tf.keras.layers.Activation('relu')(left)
left = tf.keras.layers.BatchNormalization()(left)
left = tf.keras.layers.Dense(1, name="angle")(left)

right = tf.keras.layers.Dense(64)(l)
right = tf.keras.layers.Activation('relu')(right)
right = tf.keras.layers.BatchNormalization()(right)
right = tf.keras.layers.Dense(1, name="speed")(right)

model = keras.Model(inputs=i, outputs=(left, right))
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss={
        "angle": keras.losses.MeanSquaredError(),
        "speed": keras.losses.BinaryCrossentropy()
    },
    metrics={
        "angle": keras.metrics.RootMeanSquaredError(),
        "speed": keras.metrics.BinaryAccuracy()
    }
)

# %% Setup the data for T2

Config.set_labels_path(Path("data/track_1.csv"))
Config.set_train_path(Path("data/Track1"))
train_ds_1, val_ds_1 = Data.training(0.8, 0.2, 4, True)

hist1 = model.fit(
    train_ds_1,
    validation_data=val_ds_1,
    epochs=25
)

Config.set_labels_path(Path("data/track_2.csv"))
Config.set_train_path(Path("data/Track2"))
train_ds_2, val_ds_2 = Data.training(0.8, 0.2, 4, True)

hist1 = model.fit(
    train_ds_2,
    validation_data=val_ds_2,
    epochs=25
)
model.save("newmodel/mobnet/stage1")
# %% Setup the main data
Config.set_labels_path(Path("data/training_norm.csv"))
Config.set_train_path(Path("data/training_data/training_data"))
train_ds_norm, val_ds_norm = Data.training(0.8, 0.2, 8, True)

hist2 = model.fit(
    train_ds_norm,
    validation_data=val_ds_norm,
    epochs=25
)
model.save("newmodel/mobnet/stage2")
# %%
train_ds = train_ds_norm.concatenate(train_ds_other)
val_ds = val_ds_norm.concatenate(val_ds_other)

hist_3 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50
)
# %%
