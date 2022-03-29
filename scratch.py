# %%
from functools import reduce
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PyCrashed.pipeline import Dataset, assign_weights
# %% Define the image paths
labels = assign_weights(pd.read_csv("data/training_norm.csv"))
# %%
labels.head()
# %%
"""

Train, test and val should each have an equal portion of each class within

proportions = [0.1, 0.6, 0.2, 0.1] # len(labels["motion_class"] == x) / len(labels)

"""
# %%
def get_remainder(s):
    return pd.concat((labels, s)).drop_duplicates(keep=False)

def get_set(valid_l, prob, prop, n):
    take_items = lambda c: valid_l[valid_l["motion_class"] == c].sample(int(n * prob * prop.loc[c]))
    return pd.concat(map(take_items, valid_l["motion_class"].unique().squeeze()))

def tvt_split(train, val):
    n = labels.shape[0]
    grouped_labels = labels.groupby("motion_class")
    proportions = grouped_labels["image_id"].count().map(lambda x: x / n)
    # Define the sets
    train_set = get_set(labels, train, proportions, n)
    val_set = get_set(get_remainder(train_set), val, proportions, n)
    test_set = get_remainder(pd.concat((train_set, val_set)))

    return train_set, val_set, test_set
# %%
t, v, _ = tvt_split(0.8, 0.2)

# %%
