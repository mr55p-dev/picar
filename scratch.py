# %%
import tensorflow as tf
import matplotlib.pyplot as plt
from PyCrashed.pipeline import Dataset

# %%
Dataset.set("BATCH_SIZE", 4)
x, y = next(iter(Dataset.load("train").take(1)))
# %%
