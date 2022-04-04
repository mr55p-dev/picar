import os
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

# Define the training files and labels
files = [
    i for i in Path("data/training_data/training_data").glob("*.png")
    if i.lstat().st_size > 0
]
labels_path = Path("data/training_norm.csv")
labels = pd.read_csv(labels_path)
files_str = [str(i) for i in files]

# Define a load image function
def load_image_tensor(path: Path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.rgb_to_yuv(img)
    img = tf.image.resize(img, (224, 224))
    return img

# Define a load id tensor
def get_id_tensor(path):
    img_id = tf.strings.split(path, os.sep)
    img_id = img_id[-1] # Get the last path component
    img_id = tf.strings.regex_replace(img_id, ".png", "")
    img_id = int(tf.strings.to_number(img_id))
    return img_id

# Define a load labels tensor
def get_angle_speed_tensor(img_id: int):
    label = labels[labels["image_id"] == img_id]
    label = label[["angle", "speed"]]
    label = label.to_numpy()
    label = label.reshape(-1)
    label = tf.convert_to_tensor(label)
    return label

# Combine the above functions
def create_tensor(path):
    img = load_image_tensor(path)
    img_id = get_id_tensor(path)
    label = get_angle_speed_tensor(img_id)
    return img, label, img_id

# Convert this to py_func
create_tensor_fn = lambda p: tf.py_function(
    create_tensor,
    inp=[p],
    Tout=(tf.float32, tf.float64, tf.float64)
)

def get_id(path: Path):
    img_id = path.name
    img_id = img_id[:-4]
    img_id = int(img_id)
    return img_id

def load_prediction_data(
        batch_size: int
    ):
    BATCH = batch_size
    fileset = [
        i for i in Path("data/test_data/test_data").glob("*.png")
        if i.lstat().st_size > 0
    ]
    load_blind_img = lambda p: tf.py_function(
        load_image_tensor,
        inp=[p],
        Tout=tf.float32
    )
    fileset = sorted(fileset, key=get_id)
    fileset = [str(i) for i in fileset]
    dataset = tf.data.Dataset.from_tensor_slices(fileset)
    dataset = dataset.map(load_blind_img)
    dataset = dataset.batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)
    return dataset

def load_dataset(
        train: float,
        val: float,
        batch_size: int,
        weighted: bool,
        multiheaded: bool
    ):
    # Set the number of files and the size of the data to take
    N_FILES = len(files)
    N_TRAIN = int(train * N_FILES)
    N_VAL   = int(val * N_FILES)
    # N_TEST  = int(.05 * n_files)
    BATCH = batch_size

    # Create the dataset from_tensor_slices and apply transformations
    train_dataset = tf.data.Dataset.from_tensor_slices(files_str)
    train_dataset = train_dataset.map(create_tensor_fn).take(N_TRAIN)\
        .batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(files_str)
    val_dataset = val_dataset.skip(N_TRAIN).take(N_VAL)\
        .batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset