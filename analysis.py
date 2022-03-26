import pandas as pd
import numpy as np
from PyCrashed.utils import normal_to_raw
from pathlib import Path


def analysis(path):
    # Load the csv
    pred = pd.read_csv(path, index_col="image_id")

    # Convert everything into the proper format
    raw_pred = np.apply_along_axis(normal_to_raw, 1, pred.to_numpy())
    raw_pred = pd.DataFrame(raw_pred, columns=pred.columns)

    # Some interesting charts
    raw_pred["angle"].hist()
    raw_pred["speed"].hist()


analysis(Path("products/Nvidia/predicions.csv"))

analysis(Path("data/sampleSubmission.csv"))

unique_pred = pred["angle"].unique()

unique_sample = sample["angle"].unique()

unique_pred.sort()
unique_sample.sort()

padded_pred = np.hstack((unique_pred.reshape((-1, 1)), np.zeros(unique_pred.shape).reshape((-1, 1))))
padded_sample = np.hstack((unique_sample.reshape((-1, 1)), np.zeros(unique_sample.shape).reshape((-1, 1))))

# +
raw_pred = np.apply_along_axis(normal_to_raw, 1, padded_pred)

raw_sample = np.apply_along_axis(normal_to_raw, 1, padded_sample)
# -

raw_pred[:, 0].min()


