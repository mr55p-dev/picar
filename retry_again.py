from RetryPy.data import load_dataset, load_prediction_data
from RetryPy.models import NVidia
import tensorflow as tf
import numpy as np
import pandas as pd

model = NVidia()
model.build()

dataset, val_dataset = load_dataset(.7, .3, 128, False, False)

model.fit(data=dataset, validation_data=val_dataset, n_epochs=50)

kaggle_dataset = load_prediction_data(1)

predictions = model.model.predict(kaggle_dataset)

predictions = tf.clip_by_value(predictions, 0, 1)
predictions = np.stack((predictions[:, 0], np.rint(predictions[:, 1]))).T
predictions = pd.DataFrame(
    predictions,
    index=pd.RangeIndex(1, 1021),
    columns=["angle", "speed"]
)
predictions.index.name = "image_id"
predictions["speed"] = predictions["speed"].astype("int")
predictions.to_csv("RetryPy/04-apr-predictions.csv")