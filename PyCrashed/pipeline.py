import tensorflow as tf

import pathlib
import os
import pandas as pd


class Dataset:
    _props = {
        "N_TRAIN": 100,
        "N_TEST": 100,
        "N_VAL": 20,
        "BATCH_SIZE": 8,
    }

    # Define the image paths
    IMAGE_LABELS = pd.read_csv("data/training_norm.csv")
    IMAGE_PATHS = [
            str(f)
            for f in pathlib.Path("data/training_data/training_data/").glob("*.png")
            if f.stat().st_size > 0
    ]

    # Autograph clearly does not work with this function
    @staticmethod
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
        label_pd = Dataset.IMAGE_LABELS[Dataset.IMAGE_LABELS["image_id"] == row_val]
        label_np = label_pd.to_numpy().squeeze()[1:]

        return img, tf.convert_to_tensor(label_np)

    @staticmethod
    def set(prop, val):
        Dataset._props[prop] = val

    @staticmethod
    def load(method="train"):
        # Convert the python function into a py_function
        tf_fetch_data = lambda x: tf.py_function(
                func=Dataset.fetch_data,
                inp=[x],
                Tout=(tf.float32, tf.float64)
        )

        ds = tf.data.Dataset.list_files(Dataset.IMAGE_PATHS)
        ds = ds.map(tf_fetch_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        if method == "validate" or method == "val":
            ds = ds.skip(Dataset._props["N_TRAIN"])
        elif method == "test":
            ds = ds.skip(Dataset._props["N_TRAIN"] + Dataset._props["N_TEST"])
        ds = ds.take(Dataset._props["N_TRAIN"])
        ds = ds.batch(Dataset._props["BATCH_SIZE"])
        return ds.cache().prefetch(tf.data.AUTOTUNE)
