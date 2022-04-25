# %%
from PyCrashed.data import Data, _get_id, _load_image_tensor
from PyCrashed.utils import clean_predictions
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
# %%
parser = argparse.ArgumentParser("Test some files")
parser.add_argument("model_path", help="Base path to model to test (experiment path)", type=str)
parser.add_argument("-n", help="Number of predictions to test", type=int, default=10)
args = parser.parse_args(["products/Nvidia_split"])
# args = parser.parse_args()
base_path = Path(args.model_path)
# %%
files = Path("data/test_data/test_data").glob("*.png")
files = list(files)
model = tf.keras.models.load_model(Path.joinpath(base_path, "model/"))

def sample(path_idx):
    path = files[path_idx]
    idx = _get_id(path)

    predictions = pd.read_csv(Path.joinpath(base_path, "predictions.csv"))
    prediction = predictions[predictions["image_id"] == idx]
    prediction = prediction[["angle", "speed"]]
    prediction = prediction.to_numpy()
    prediction = prediction.astype('float32')
    prediction = np.around(prediction, decimals=4)

    artificial_img = _load_image_tensor(str(path))
    artificial_img = tf.expand_dims(artificial_img, 0)
    artificial_pred = model.predict(artificial_img)
    if isinstance(artificial_pred, tuple):
        artificial_pred = np.hstack(artificial_pred)
    artificial_pred = clean_predictions(artificial_pred)
    artificial_pred = np.around(artificial_pred, decimals=4)

    return (prediction == artificial_pred).all()

# %%
sample_space = np.random.randint(0, len(files), size=args.n)
responses = map(sample, sample_space)
responses = list(responses)
success = all(responses)
if not success:
    print(responses)
    raise ValueError("At least one of the tests has failed")

print("Success!")
# %%
