# %%
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %% Define the image paths
img_labels = pd.read_csv("data/training_norm.csv")
# %%

# %% SAMPLING
img_labels["motion_class"].unique().reshape(-1, 1).shape
img_labels["motion_class"] = np.apply_along_axis(classify_instance, 1, img_labels.to_numpy())
sample = img_labels.groupby("motion_class").sample(1200)["image_id"]

get_path = lambda x: load_image(f"data/training_data/training_data/{x}.png")
sample["image_id"].apply(get_path)
paths = np.apply_along_axis(get_path, 0, sample["image_id"].to_numpy())

# %%
labels = assign_weights(img_labels)
labels.head()

# %%
