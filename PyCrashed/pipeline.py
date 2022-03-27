import tensorflow as tf

import pathlib
import os
import pandas as pd


class Dataset:
    _props = {
        "N_TRAIN": .1,
        "N_TEST": .1,
        "N_VAL": .05,
        "BATCH_SIZE": 8,
    }

    # Define the image paths
    _IMAGE_LABELS = pd.read_csv("data/training_norm.csv")
    _IMAGE_PATHS = [
            str(f)
            for f in pathlib.Path("data/training_data/training_data/").glob("*.png")
            if f.stat().st_size > 0
    ]
    _IMAGE_PRED_PATHS = [
            str(f)
            for f in pathlib.Path("data/test_data/test_data/").glob("*.png")
            if f.stat().st_size > 0
    ]

    @staticmethod
    def _load_image(img_path: tf.Tensor):
        img_decoded = tf.image.decode_image(
            tf.io.read_file(img_path), channels=3, dtype=tf.float32 # Read the image as float32
        )
        return tf.image.rgb_to_yuv(img_decoded)

    # Autograph clearly does not work with this function
    @staticmethod
    def _fetch_data(img_path: tf.Tensor):
        # Load the image
        img = Dataset._load_image(img_path)

        # Load the image label
        split_path = tf.strings.split(img_path, os.sep)
        path_end = split_path[-1]
        path_name = tf.strings.regex_replace(path_end, '.png', '')
        img_id = tf.strings.to_number(path_name)
        row_val = int(img_id)
        label_pd = Dataset._IMAGE_LABELS[Dataset._IMAGE_LABELS["image_id"] == row_val]
        label_np = label_pd.to_numpy().squeeze()[1:]

        return img, tf.convert_to_tensor(label_np)

    @staticmethod
    def set(prop, val):
        Dataset._props[prop] = val

    @staticmethod
    def load(method="train"):
        # Convert the python function into a py_function
        tf_fetch_data = lambda x: tf.py_function(
                func=Dataset._fetch_data,
                inp=[x],
                Tout=(tf.float32, tf.float64)
        )

        ds = tf.data.Dataset.list_files(Dataset._IMAGE_PATHS)
        n_items = len(ds)
        ds = ds.map(tf_fetch_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        if method in ["validate", "val"]:
            ds = ds.skip(int(n_items * Dataset._props["N_TRAIN"]))
        elif method == "test":
            ds = ds.skip(int(n_items * (Dataset._props["N_TRAIN"] + Dataset._props["N_TEST"])))
        ds = ds.take(int(n_items * Dataset._props["N_TRAIN"]))
        ds = ds.batch(Dataset._props["BATCH_SIZE"])
        return ds.cache().prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def load_test():
        tf_fetch_data = lambda x: tf.py_function(
                func=Dataset._load_image,
                inp=[x],
                Tout=tf.float32
        )

        ds = tf.data.Dataset.list_files(Dataset._IMAGE_PRED_PATHS)
        ds = ds.map(tf_fetch_data, deterministic=True, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(1).cache().prefetch(tf.data.AUTOTUNE)
