# %%
import tensorflow as tf
import numpy as np
from PyCrashed.pipeline import Dataset
# %%
Dataset.batch_size=4
Dataset.labelled_outputs = True
ds = iter(Dataset.load_test())
truths = []
for i in range(1, 1020//Dataset.batch_size):
    img = next(ds)
    start = Dataset.batch_size * (i-1) + 1
    end = Dataset.batch_size * (i) + 1
    x = np.stack(
        [Dataset._load_image(f"data/test_data/test_data/{a}.png") for a in range(start, end)]
    )
    truths.append((x == img.numpy()).all())

all(truths)
# %%
ds = Dataset.load_test()
model = tf.keras.models.load_model("products/Nvidia_split/model")
# %%
predictions = []
for inp in ds.as_numpy_iterator():
    prediction = model.predict(inp)
    predictions.append(prediction)
predictions_arr = np.vstack([i.squeeze() for i in predictions])
# %%
batched_predictions = model.predict(ds)
# %%
angles, speeds = list(zip(*predictions))
predictions_arr = np.hstack((np.vstack(angles), np.vstack(speeds)))
# %%
batched_predictions_arr = np.hstack(batched_predictions)
# %%
equality = []
for a, b in zip(predictions_arr, batched_predictions_arr):
    equality.append((a == b).all())
# %%
all(equality)
# %%
batched_predictions

# %%
# # Adjust values
predictions = np.hstack((angle.reshape(-1, 1), speed.reshape(-1, 1)))
# %%
predictions
# %%
