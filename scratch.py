# %%
from PyCrashed.pipeline import Dataset
Dataset.batch_size=4
Dataset.labelled_outputs = True
ds = Dataset.load("train")
# %%
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
mse(ds, ds)
# %%
mse(lab["angle"], lab["angle"], [0.8, 2, 0.2, 0.2]).numpy()
# %%
lab["angle"].reshape
# %%
