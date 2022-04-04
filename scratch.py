# %%
import tensorflow as tf
import numpy as np
from PyCrashed.pipeline import Dataset
from RetryPy.data import load_prediction_data
# %%
Dataset.batch_size = 1
original = Dataset.load_test()
retry = load_prediction_data(1)
# %%
it_original = original.as_numpy_iterator()
it_retry = retry.as_numpy_iterator()
# %%
for _ in range(100):
    a = next(it_original)
    b = next(it_retry)
    print((a == b).all())
# %%
