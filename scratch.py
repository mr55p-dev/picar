# %%
import numpy as np
import pandas as pd
import tensorflow as tf

from PyCrashed import predict
from PyCrashed.predict import Data

# %%
p1 = pd.read_csv("mlis-predictions.csv")
p2 = pd.read_csv("products/Nvidia/predicions.csv")
# %%
loss = tf.keras.losses.MeanSquaredError()
# %%
l_angle = loss(p1["angle"].to_numpy(), p2["angle"].to_numpy())
l_speed = loss(p1["speed"].to_numpy(), p2["speed"].to_numpy())
# %%
