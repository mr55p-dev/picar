# %%
from PyCrashed.pipeline import Dataset
Dataset.batch_size=4
Dataset.labelled_outputs = True
ds = Dataset.load("train")
# %%
lab = next(iter(ds.take(1)))[1]
# %%
lab
# %%
Dataset.labelled_outputs
# %%
from PyCrashed.models import NVidiaSplit
x = NVidiaSplit(use_wandb=False)
x.build()
x.is_split
# %%

from PyCrashed.models import NVidia
x = NVidia(use_wandb=False)
x.build()
x.is_split
# %%
import tensorflow as tf
import numpy as np
# %%
mse = tf.keras.losses.MeanSquaredError()
# %%
y = np.array([1, 0, 0, 1], dtype=float)
y_pred = np.array([1, 1, 0, 0], dtype=float)
w = np.array([0.8, 2, 0.2, 0.2]).reshape(1, -1)
# %%
w.shape
# %%
y = tf.convert_to_tensor(y)
y_pred = tf.convert_to_tensor(y_pred)
w = tf.convert_to_tensor(w)
# %%
mse(y, y_pred)
# %%
mse(lab["angle"], lab["angle"], [0.8, 2, 0.2, 0.2]).numpy()
# %%
lab["angle"].reshape
# %%
