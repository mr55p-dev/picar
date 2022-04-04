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
    return labels

def get_remainder(*args):
    return pd.concat(args).drop_duplicates(keep=False)

def get_set(valid_l, prob, prop, n):
    take_items = lambda c: valid_l[valid_l["motion_class"] == c].sample(int(n * prob * prop.loc[c]))
    return pd.concat(map(take_items, valid_l["motion_class"].unique().squeeze()))

def process_out(img, label, weights):
    l = {
        "angle": label[0],
        "speed": label[1]
    }
    return img, l, tf.reshape(weights, (1, -1))

class Dataset:
    n_train = 0.65
    n_val = 0.25
    n_test = 0.1
    batch_size = 64
    labelled_outputs = False

    # Define the image paths
    labels = assign_weights(pd.read_csv("data/training_norm.csv"))
    training_path = pathlib.Path("data/training_data/training_data/")
    paths = [
            str(f)
            for f in training_path.glob("*.png")
            if f.stat().st_size > 0
    ]
    _pred_paths = sorted(
        list(
            pathlib.Path("data/test_data/test_data/").glob("*.png")
            ), key=lambda x: int(x.name[:-4]))
    prediction_paths = [str(f) for f in _pred_paths]
    _val_paths = sorted(
        list(
            pathlib.Path("data/validation_data/").glob("*.png")
            ), key=lambda x: int(x.name[:-4]))
    val_paths = [str(f) for f in _pred_paths]

    @staticmethod
    def _tvt_split():
        n = Dataset.labels.shape[0]
        grouped_labels = Dataset.labels.groupby("motion_class")
        proportions = grouped_labels["image_id"].count().map(lambda x: x / n)
        # Define the sets
        train_set = get_set(Dataset.labels, Dataset.n_train, proportions, n)
        val_set = get_set(get_remainder(Dataset.labels, train_set), Dataset.n_val, proportions, n)
        test_set = get_remainder(Dataset.labels, train_set, val_set)

        base_path = Dataset.training_path
        def sanitize(s):
            s = s["image_id"].to_numpy()
            return list(map(lambda x: str(base_path.joinpath(f"{int(x)}.png")), s))
        return sanitize(train_set), sanitize(val_set), sanitize(test_set)

    @staticmethod
    def _load_image(img_path: tf.Tensor):
        # print(img_path)
        img_decoded = tf.image.decode_image(
            tf.io.read_file(img_path), channels=3, dtype=tf.float32 # Read the image as float32
        )
        return tf.image.resize(tf.image.rgb_to_yuv(img_decoded), (224, 224))

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
    def load(mode="train"):
        # Convert the python function into a py_function
        tf_fetch_data = lambda x: tf.py_function(
                func=Dataset._fetch_data,
                inp=[x],
                Tout=(tf.float32, tf.float64, tf.float64)
        )

        idx_train, idx_val, idx_test = Dataset._tvt_split()
        f_train = list(filter(lambda x: x in idx_train, Dataset.paths))
        f_val   = list(filter(lambda x: x in idx_val, Dataset.paths))
        f_test  = list(filter(lambda x: x in idx_test, Dataset.paths))

        def build(files):
            ds = tf.data.Dataset.from_tensor_slices(files)
            ds = ds.map(tf_fetch_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            if Dataset.labelled_outputs:
                ds = ds.map(process_out, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            ds = ds.batch(Dataset.batch_size)
            ds = ds.cache()
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        if mode == "train":
            return build(f_train)
        elif mode in ("validate", "val"):
            return build(f_val)
        elif mode == "test":
            return build(f_test)
        else:
            raise ValueError("invalid mode")

    @staticmethod
    def load_test():
        tf_fetch_data = lambda x: tf.py_function(
                func=Dataset._load_image,
                inp=[x],
                Tout=tf.float32
        )

        ds = tf.data.Dataset.from_tensor_slices(Dataset.prediction_paths)
        ds = ds.map(tf_fetch_data).batch(Dataset.batch_size)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    @staticmethod
    def load_val():
        tf_fetch_data = lambda x: tf.py_function(
                func=Dataset._load_image,
                inp=[x],
                Tout=tf.float32
        )

        ds = tf.data.Dataset.from_tensor_slices(Dataset.val_paths)
        ds = ds.map(tf_fetch_data).batch(Dataset.batch_size)
        # ds = ds.batch(Dataset.batch_size)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)

        # Load labels
        labels = Dataset.labels
        actual_labels = labels[labels["image_id"] < 1021][["angle", "speed"]].to_numpy()

        return ds, actual_labels
