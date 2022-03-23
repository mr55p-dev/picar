# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['load-data']

# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import csv
from datetime import datetime

# %%
labels = np.load(upstream['load-data']['labels'])
images = np.load(upstream['load-data']['images'])

# %%
now = datetime.now()
log_dir = now.strftime("products/model-nvidia/tb_logs/%m-%d/%H:%M:%S/")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
datagen = ImageDataGenerator()
datagen.fit(images)

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

model = keras.Model(inputs=inputs, outputs=l, name="nvidia_model")
model.compile(
    optimizer=tf.keras.optimizers.Nadam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.KLDivergence()]
)

model.summary()
# %%
fitted = model.fit(
    datagen.flow(images, labels, batch_size=batch_size),
    epochs=n_epochs,
    callbacks=[tensorboard_callback]
)

# %%
def save_metrics(cb, path):
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(cb.history.keys())
        w.writerows(zip(*cb.history.values()))

# %%
save_metrics(fitted, "products/model-nvidia/metrics.csv")
model.save("products/model-nvidia/model")
