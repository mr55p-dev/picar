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

# Define a load image function
def _load_image_tensor(path: Path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.rgb_to_yuv(img)
    img = tf.image.resize(img, (224, 224))
    return img

# Define a load id tensor
def _get_id_tensor(path):
    img_id = tf.strings.split(path, os.sep)
    img_id = img_id[-1] # Get the last path component
    img_id = tf.strings.regex_replace(img_id, ".png", "")
    img_id = int(tf.strings.to_number(img_id))
    return img_id

# Define a load labels tensor
def _get_angle_speed_tensor(img_id: int):
    label = _labels[_labels["image_id"] == img_id]
    label = label[["angle", "speed"]]
    label = label.to_numpy()
    label = label.reshape(-1)
    label = tf.convert_to_tensor(label)
    return label

# Combine the above functions
def _create_tensor(path):
    img = _load_image_tensor(path)
    img_id = _get_id_tensor(path)
    label = _get_angle_speed_tensor(img_id)
    return img, label, img_id

# Convert this to py_func
_create_tensor_fn = lambda p: tf.py_function(
    _create_tensor,
    inp=[p],
    Tout=(tf.float32, tf.float64, tf.float64)
)


def _label_outputs(img, label, weights):
    l = {
        "angle": label[0],
        "speed": label[1]
    }
    return img, l, tf.reshape(weights, (1, -1))

# Function to assign an instance to a grouping and give that grouping a weight
def _assign_weights(labels: pd.DataFrame):
    def classify_instance(instance):
        if instance[2] == 0: return 0 # Stopped
        elif instance[1] <= 0.4375: return 1 # Left (85 degrees)
        elif instance[1] >= 0.5625: return 2 # Right (95 degrees)
        else: return 3 # Straight

    # One-hot encode the conditionals given by classify_instance
    labels["motion_class"] = np.apply_along_axis(classify_instance, 1, labels.to_numpy())

    # Calculate the weighting of each class
    classes = labels["motion_class"].unique()
    n_classes = len(classes)
    n_instances = labels.shape[0]
    calculate_weight = lambda x: n_instances / (n_classes * labels[labels["motion_class"] == x].shape[0])
    weights = list(map(calculate_weight, classes))
    class_weights = pd.DataFrame(weights, index=classes)

    # Set a column in the labels dataframe
    labels["weight"] = labels["motion_class"].apply(lambda x: class_weights.loc[x, :])
    return labels

class Data:
    @staticmethod
    def testing(batch_size: int):
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
            weighted: bool = False,
            multiheaded: bool = False
        ):
        # Set the number of files and the size of the data to take
        N_FILES = len(_files_train)
        N_TRAIN = int(train * N_FILES)
        N_VAL   = int(val * N_FILES)
        BATCH   = batch_size

        # Create the dataset from_tensor_slices and apply transformations
        train_dataset   = tf.data.Dataset.from_tensor_slices(_files_train).shuffle(512)
        train_dataset   = train_dataset.map(_create_tensor_fn).take(N_TRAIN)\
            .batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)

        val_dataset     = tf.data.Dataset.from_tensor_slices(_files_train).shuffle(512)
        val_dataset     = val_dataset.map(_create_tensor_fn).skip(N_TRAIN).take(N_VAL)\
            .batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

