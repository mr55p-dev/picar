# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['load-data']

# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
labels = np.load(upstream['load-data']['labels'])
images = np.load(upstream['load-data']['images'])

# %%
log_dir = "products/model-nvidia/train_logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %% This is the Nvidia model
inputs = keras.Input(shape=(40, 60, 3))
l = Conv2D(24, (5, 5), strides=(2, 2))(inputs)
l = Conv2D(36, (5, 5), strides=(1, 1))(l)
l = Conv2D(48, (5, 5), strides=(1, 1))(l)
l = Conv2D(64, (3, 3), strides=(1, 1))(l)
l = Conv2D(64, (3, 3), strides=(1, 1))(l)
l = Flatten()(l)
l = Dense(1164)(l)
l = Dense(50)(l)
l = Dense(2)(l)

model = keras.Model(inputs=inputs, outputs=l, name="NVidia_model")
model.compile(
    optimizer=tf.keras.optimizers.Nadam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.KLDivergence()]
)

model.summary()
# %%
x = model.fit(
    images,
    labels,
    epochs=200,
    batch_size=100,
    validation_split=0.2,
    callbacks=[tensorboard_callback]
)

# %%
model.save("products/model-nvidia/model")
