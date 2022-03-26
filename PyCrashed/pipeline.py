import tensorflow as tf

import pathlib
import os
import pandas as pd

from PyCrashed.config import *

# Define the image paths
image_labels = pd.read_csv("data/training_norm.csv")
image_paths = [
        str(f)
        for f in pathlib.Path("data/training_data/training_data/").glob("*.png")
        if f.stat().st_size > 0
]

# Autograph clearly does not work with this function
def fetch_data(img_path: tf.Tensor):
    # Load the image
    img_decoded = tf.image.decode_image(
        tf.io.read_file(img_path), channels=3, dtype=tf.float32 # Read the image as float32
    )
    img = tf.image.rgb_to_yuv(img_decoded)

    # Load the image label
    split_path = tf.strings.split(img_path, os.sep)
    path_end = split_path[-1]
    path_name = tf.strings.regex_replace(path_end, '.png', '')
    img_id = tf.strings.to_number(path_name)
    row_val = int(img_id)
    label_pd = image_labels[image_labels["image_id"] == row_val]
    label_np = label_pd.to_numpy().squeeze()[1:]

    return img, tf.convert_to_tensor(label_np)

# Convert the python function into a py_function
tf_fetch_data = lambda x: tf.py_function(
        func=fetch_data,
        inp=[x],
        Tout=(tf.float32, tf.float64)
)

ds = tf.data.Dataset.list_files(image_paths)
ds = ds.map(tf_fetch_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
train_ds    = ds.take(n_train).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
val_ds      = ds.skip(n_train).take(n_val).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
test_ds     = ds.skip(n_train+n_val).take(n_test).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)