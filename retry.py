# %% Imports
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
# %% Get the raw data needed
train_files = [
    i for i in Path("data/training_data/training_data").glob("*.png")
    if i.lstat().st_size > 0
]
labels_path = Path("data/training_norm.csv")
n_files = len(train_files)
labels = pd.read_csv(labels_path)
# %% Define generator function which can yield image, lable tuples as tensors
def load_image(path: Path):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return img

def get_id(path: Path):
    img_id = path.name
    img_id = img_id[:-4]
    img_id = int(img_id)
    return img_id

def get_angle_speed(img_id: int):
    label = labels[labels["image_id"] == img_id]
    label = label[["angle", "speed"]]
    label = label.to_numpy()
    label = label.reshape(-1)
    label = tf.convert_to_tensor(label)
    return label

def create_item():
    for path in train_files:
        img = load_image(path)
        img_id = get_id(path)
        label = get_angle_speed(img_id)
        yield img, label

# %% Define the datasets used
N_TRAIN = int(.6 * n_files)
N_VAL   = int(.2 * n_files)
N_TEST  = int(.2 * n_files)
BATCH = 64
train_dataset = tf.data.Dataset.from_generator(
    create_item,
    output_types=(tf.float32, tf.float64)
).take(N_TRAIN).batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    create_item,
    output_types=(tf.float32, tf.float64)
).skip(N_TRAIN).take(N_VAL).batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    create_item,
    output_types=(tf.float32, tf.float64)
).skip(N_TRAIN+N_VAL).take(N_TEST).batch(BATCH, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
# %% Define a model
# gpus = tf.config.list_logical_devices('GPU')
# strategy = tf.distribute.MirroredStrategy(gpus)
# with strategy.scope():
inputs = tf.keras.Input(shape=(224, 224, 3))
layer = tf.keras.layers.BatchNormalization()(inputs)
layer = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2))(layer)
layer = tf.keras.layers.Activation("relu")(layer)
layer = tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2))(layer)
layer = tf.keras.layers.Activation("relu")(layer)
layer = tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2))(layer)
layer = tf.keras.layers.Activation("relu")(layer)
layer = tf.keras.layers.MaxPool2D((2, 2))(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
layer = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1))(layer)
layer = tf.keras.layers.Activation("relu")(layer)
layer = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1))(layer)
layer = tf.keras.layers.Activation("relu")(layer)
layer = tf.keras.layers.MaxPool2D((2, 2))(layer)

layer = tf.keras.layers.Flatten()(layer)

layer = tf.keras.layers.Dense(128)(layer)
layer = tf.keras.layers.Activation("relu")(layer)
layer = tf.keras.layers.Dropout(0.25)(layer)
layer = tf.keras.layers.Dense(64)(layer)
layer = tf.keras.layers.Activation("relu")(layer)
outputs = tf.keras.layers.Dense(2)(layer)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="simple_nvidia")
model.compile(
    loss="mse",
    optimizer="adam",
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
# %% Fit and evaluate the model 
metrics = model.fit(train_dataset, epochs=25, validation_data=val_dataset)
test_error = model.evaluate(test_dataset)
# %%
model.save("./RetryPy/model/")
# %% Test the models blind prediction
blind_lab = test_dataset.map(lambda _, lab: lab)
blind_lab = list(blind_lab)
blind_lab = np.vstack(blind_lab)
blind_img = test_dataset.map(lambda img, _: img)

blind_pre = model.predict(blind_img)

loss = tf.keras.losses.MeanSquaredError()
error_single_dev = loss(blind_pre, blind_lab)
# %% Create actual predictions
