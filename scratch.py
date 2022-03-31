# %%
import tensorflow as tf
import numpy as np
from PyCrashed.pipeline import Dataset
# %%
Dataset.batch_size=4
Dataset.labelled_outputs = True
ds = Dataset.load_test()
# %%
model = tf.keras.models.load_model("products/Nvidia_split/model")
# %%
predictions = model.predict(ds)
# %%
angle, speed = predictions

# %%
# # Adjust values
predictions = np.hstack((angle.reshape(-1, 1), speed.reshape(-1, 1)))
# %%
predictions
# %%
