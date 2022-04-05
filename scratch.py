# %%
import numpy as np
import pandas as pd
import tensorflow as tf

from PyCrashed import predict
from PyCrashed.predict import Data

# %%
ds, _ = Data.training(0.7, 0.3, 8, False, False)
# %%
tf.keras.layers.BatchNormalization()
# %%
batch = next(iter(ds))
# %%
loss = tf.keras.losses.BinaryCrossentropy()
# %%
labels = pd.read_csv("data/training_norm.csv")
# %%
labels = labels["speed"].to_numpy()
# %%
ones = np.ones(labels.shape)
# %%
loss(labels, ones)
# %%
