import tensorflow as tf

import pathlib
import os
import pandas as pd
import numpy as np

def assign_weights(labels):
    def classify_instance(instance):
        if instance[2] == 0: return 0 # Stopped
        elif instance[1] <= 0.4375: return 1 # Left (85 degrees)
        elif instance[1] >= 0.5625: return 2 # Right (95 degrees)
        else: return 3 # Straight

    # One-hot encode the conditionals given by classify_instance
    labels["motion_class"] = np.apply_along_axis(classify_instance, 1, labels.to_numpy())

    # Calculate the weighting of each class
    classes = labels["motion_class"].unique()
    calc = lambda x: labels.shape[0] / (len(classes) * labels[labels["motion_class"] == x].shape[0])
    weights = list(map(calc, classes))
    class_weights = pd.DataFrame(weights, index=classes)

    # Set a column in the labels dataframe
    labels["weight"] = labels["motion_class"].apply(lambda x: class_weights.loc[x, :])
    return labels.drop("motion_class", axis=1)

class Dataset:
    n_train = 0.65
    n_val = 0.25
    n_test = 0.1
    batch_size = 64

    # Define the image paths
    labels = assign_weights(pd.read_csv("data/training_norm.csv"))
    paths = [
            str(f)
            for f in pathlib.Path("data/training_data/training_data/").glob("*.png")
            if f.stat().st_size > 0
    ]
    prediction_paths = [
            str(f)
            for f in pathlib.Path("data/test_data/test_data/").glob("*.png")
            if f.stat().st_size > 0
    ]

    @staticmethod
    def _load_image(img_path: tf.Tensor):
        img_decoded = tf.image.decode_image(
            tf.io.read_file(img_path), channels=3, dtype=tf.float32 # Read the image as float32
        )
        return tf.image.resize(tf.image.rgb_to_yuv(img_decoded), (240, 240))

    # Autograph clearly does not work with this function
    @staticmethod
    def _fetch_data(img_path: tf.Tensor):
        # Load the image
        img = Dataset._load_image(img_path)

        # Load the image label
        split_path = tf.strings.split(img_path, os.sep)[-1]
        path_name = tf.strings.regex_replace(split_path, '.png', '')
        img_id = int(tf.strings.to_number(path_name))
        label_pd = Dataset.labels[Dataset.labels["image_id"] == img_id]
        label_np = label_pd[["angle", "speed"]].to_numpy().squeeze()

        # Load the class weight
        weight = label_pd["weight"].to_numpy().squeeze()
        return img, tf.convert_to_tensor(label_np), tf.convert_to_tensor(weight)

    @staticmethod
    def load(method="train"):
        # Convert the python function into a py_function
        tf_fetch_data = lambda x: tf.py_function(
                func=Dataset._fetch_data,
                inp=[x],
                Tout=(tf.float32, tf.float64, tf.float64)
        )

        n_items = len(Dataset.paths)
        # Load in a tensor of strings
        ds = tf.data.Dataset.list_files(Dataset.paths)

        # Convert this to a tensor of (image, label, weight)
        ds = ds.map(tf_fetch_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        # Skip and take suitable amounts for each grouping
        if method in ["validate", "val"]:
            ds = ds.skip(int(n_items * Dataset.n_train))
            ds = ds.take(int(n_items * Dataset.n_val))
        elif method == "test":
            ds = ds.skip(int(n_items * (Dataset.n_train + Dataset.n_val)))
            ds = ds.take(int(n_items * Dataset.n_test))
        else:
            ds = ds.take(int(n_items * Dataset.n_train))
        
        # Batch according to instruction
        ds = ds.batch(Dataset.batch_size)
        
        # Cache and prefetch for efficiency
        return ds.cache().prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def load_test():
        tf_fetch_data = lambda x: tf.py_function(
                func=Dataset._load_image,
                inp=[x],
                Tout=tf.float32
        )

        ds = tf.data.Dataset.list_files(Dataset.prediction_paths)
        ds = ds.map(tf_fetch_data, deterministic=True, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(1).cache().prefetch(tf.data.AUTOTUNE)
