import os
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

# Function to get the id of an image given its path
def _get_id(path: Path):
    img_id = path.name
    img_id = img_id[:-4]
    img_id = int(img_id)
    return img_id

# Function to assign an instance to a grouping and give that grouping a weight
def _assign_weights(labels: pd.DataFrame):
    # Weight the examples by steering angle and speed
    def classify_instance_by_dir(instance):
        if instance[2] == 0: return 0 # Stopped
        elif instance[1] <= 0.4375: return 1 # Left (85 degrees)
        elif instance[1] >= 0.5625: return 2 # Right (95 degrees)
        else: return 3 # Straight
    
    # Weight the examples by speed only
    def classify_instance_by_speed(instance):
        return instance[2]

    # One-hot encode the conditionals given by classify_instance
    labels["motion_class"] = np.apply_along_axis(classify_instance_by_dir, 1, labels.to_numpy())

    # Calculate the weighting of each class
    classes = labels["motion_class"].unique()
    n_classes = len(classes)
    n_instances = labels.shape[0]
    calculate_weight = lambda x: n_instances / (n_classes * labels[labels["motion_class"] == x].shape[0])
    weights = list(map(calculate_weight, classes))
    class_weights = pd.DataFrame(weights, index=classes)

    # Set a column in the labels dataframe
    labels["weight"] = labels["motion_class"].apply(lambda x: class_weights.loc[x, :])
    labels = labels.drop("motion_class", axis=1)
    return labels

# Define the training files
_files_train = [
    i for i in Path("data/training_data/training_data").glob("*.png")
    if i.lstat().st_size > 0
]
_files_train = sorted(_files_train, key=_get_id)
_files_train = [str(i) for i in _files_train]

# Define the testing files
_files_test = [
    i for i in Path("data/test_data/test_data").glob("*.png")
    if i.lstat().st_size > 0
]
_files_test = sorted(_files_test, key=_get_id)
_files_test = [str(i) for i in _files_test]

# Define the labels
_labels = Path("data/training_norm.csv")
_labels = pd.read_csv(_labels)
_labels = _assign_weights(_labels)

# Function to load an image from .png file into a properly
# encoded tensor
def _load_image_tensor(path: Path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.rgb_to_yuv(img)
    img = tf.image.resize(img, (224, 224))
    return img

# Function to get the id of an image from its pathname
def _get_id_tensor(path):
    img_id = tf.strings.split(path, os.sep)
    img_id = img_id[-1] # Get the last path component
    img_id = tf.strings.regex_replace(img_id, ".png", "")
    img_id = int(tf.strings.to_number(img_id))
    return img_id

# Gets the speed, angle and "weight" of a given image id
def _get_angle_speed_tensor(img_id: int):
    row = _labels[_labels["image_id"] == img_id]

    label = row[["angle", "speed"]]
    label = label.to_numpy()
    label = label.reshape(-1)
    label = tf.convert_to_tensor(label)

    weight = row[["weight"]]
    weight = weight.to_numpy()
    weight = weight.reshape(1)
    weight = tf.convert_to_tensor(weight)

    return label, weight

# Load an image, label and weight tensor from a path
def _create_tensor(path):
    img = _load_image_tensor(path)
    img_id = _get_id_tensor(path)
    label, weight = _get_angle_speed_tensor(img_id)
    return img, label, weight

# Configure this as a py_function in the tf graph
# so it can be compiled
_create_tensor_fn = lambda p: tf.py_function(
    _create_tensor,
    inp=[p],
    Tout=(tf.float32, tf.float64, tf.float64)
)

# Function to change the label tensor into a dict
# which is used in multi-output models to ensure
# the correct labels are applied to each output node
def _label_outputs(img, label, weights):
    l = {
        "angle": label[0],
        "speed": label[1]
    }
    return img, l, weights

# Ensure predicted angles are [0, 1] and speeds are 0 | 1
def clean_predictions(predictions):
    predictions = tf.clip_by_value(predictions, 0, 1)
    predictions = np.stack((predictions[:, 0], np.rint(predictions[:, 1]))).T
    return predictions

# Reverse the normalization
def convert_to_car_output(predictions):
    angle = predictions[:, 0]
    speed = predictions[:, 1]

    angle = (angle * 80) + 50
    speed = speed * 35
    return np.hstack((angle, speed))


class Data:
    """
    Class to manage training and testing datasets.
    Note this definitely does not need to be a class, I just cant be bothered to go back through the
    spaghetti in `utils.py` and change everything, so when i rewrote this i just made it a class of
    static methods...
    """
    @staticmethod
    def testing(batch_size: int):
        """Get a tf.data.Dataset instance containing batched, cached and prefetched files from _files_test"""
        BATCH = batch_size
        load_blind_img = lambda p: tf.py_function(
            _load_image_tensor,
            inp=[p],
            Tout=tf.float32
        )
        dataset = tf.data.Dataset.from_tensor_slices(_files_test)
        dataset = dataset.map(load_blind_img)
        dataset = dataset.batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def training(
            train: float,
            val: float,
            batch_size: int,
            multiheaded: bool = False
        ):
        """Get two tf.data.Dataset instances (training, validation) containing batched, cached and prefetched files from _files_train"""
        # Set the number of files and the size of the data to take
        N_FILES = len(_files_train)
        N_TRAIN = int(train * N_FILES)
        N_VAL   = int(val * N_FILES)
        BATCH   = batch_size

        # Create the dataset from_tensor_slices and apply transformations
        train_dataset   = tf.data.Dataset.from_tensor_slices(_files_train).shuffle(512)
        train_dataset   = train_dataset.map(_create_tensor_fn)
        if multiheaded: 
            train_dataset = train_dataset.map(_label_outputs)
        train_dataset   = train_dataset.take(N_TRAIN).batch(BATCH)\
            .cache().prefetch(tf.data.AUTOTUNE)

        val_dataset     = tf.data.Dataset.from_tensor_slices(_files_train).shuffle(512)
        val_dataset     = val_dataset.map(_create_tensor_fn)
        if multiheaded: 
            val_dataset = val_dataset.map(_label_outputs)
        val_dataset     = val_dataset.skip(N_TRAIN).take(N_VAL).batch(BATCH)\
            .cache().prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    @staticmethod
    def training_t1(
            train: float,
            val: float,
            batch_size: int,
            multiheaded: bool = False
        ):
        """Get two tf.data.Dataset instances (training, validation) containing batched, cached and prefetched files from _files_train"""
        # Set the number of files and the size of the data to take
        N_FILES = len(_files_train_t1)
        N_TRAIN = int(train * N_FILES)
        N_VAL   = int(val * N_FILES)
        BATCH   = batch_size

        # Create the dataset from_tensor_slices and apply transformations
        train_dataset   = tf.data.Dataset.from_tensor_slices(_files_train_t1).shuffle(512)
        train_dataset   = train_dataset.map(_create_tensor_fn)
        if multiheaded: 
            train_dataset = train_dataset.map(_label_outputs)
        train_dataset   = train_dataset.take(N_TRAIN).batch(BATCH)\
            .cache().prefetch(tf.data.AUTOTUNE)

        val_dataset     = tf.data.Dataset.from_tensor_slices(_files_train_t1).shuffle(512)
        val_dataset     = val_dataset.map(_create_tensor_fn)
        if multiheaded: 
            val_dataset = val_dataset.map(_label_outputs)
        val_dataset     = val_dataset.skip(N_TRAIN).take(N_VAL).batch(BATCH)\
            .cache().prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

