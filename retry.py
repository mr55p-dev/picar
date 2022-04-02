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
N_TRAIN = int(.75 * n_files)
N_VAL   = int(.2 * n_files)
N_TEST  = int(.05 * n_files)
BATCH = 128
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

layer = tf.keras.layers.Dense(256)(layer)
layer = tf.keras.layers.Activation("relu")(layer)
layer = tf.keras.layers.Dropout(0.45)(layer)
layer = tf.keras.layers.Dense(128)(layer)
layer = tf.keras.layers.Activation("relu")(layer)
outputs = tf.keras.layers.Dense(2)(layer)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="simple_nvidia")
model.compile(
    loss="mse",
    optimizer="adam",
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
# %% Fit and evaluate the model 
metrics = model.fit(train_dataset, epochs=75, validation_data=val_dataset)
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
kaggle_test_files = train_files = [
    i for i in Path("data/test_data/test_data").glob("*.png")
    if i.lstat().st_size > 0
]
kaggle_test_files = sorted(kaggle_test_files, key=get_id)

def create_kaggle_item():
    for path in kaggle_test_files:
        img = load_image(path)
        img_id = get_id(path)
        yield img, img_id
# %%

kaggle_dataset = tf.data.Dataset.from_generator(
    create_kaggle_item,
    output_types=(tf.float32, tf.int16)
).map(lambda img, _: img)\
.batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)
# %% Run predictions
predictions = model.predict(kaggle_dataset)
# %% Very quick test of the nth item
first = next(kaggle_dataset.take(1).as_numpy_iterator())[0:1, :, :, :]
first_pred_synthetic = model.predict(first)
first_pred_synthetic = first_pred_synthetic.astype('float16')
first_pred_actual = predictions[0:1, :]
first_pred_actual = first_pred_actual.astype('float16')
assert (first_pred_synthetic == first_pred_actual).all()
# %% If that passes, write the data!
predictions = tf.clip_by_value(predictions, 0, 1)
predictions = np.stack((predictions[:, 0], np.rint(predictions[:, 1]))).T
predictions = pd.DataFrame(
    predictions, 
    index=pd.RangeIndex(1, 1021),
    columns=["angle", "speed"]
)
predictions.index.name = "image_id"
predictions["speed"] = predictions["speed"].astype("int")
predictions.to_csv("RetryPy/predictions.csv")
# %%
"""
ELLIS' BIG LIST OF TODOS!
- Port the pipeline archirecture from generators into tensor slices
- Keep the current generator based testing
- Bring in argparse
- Come up with a new model creation methodology (factory perhaps)
- Implement the distributed strategy again
- 
"""
# %% Create the data in a tensor_slices way
import os
N_TRAIN = int(.75 * n_files)
N_VAL   = int(.2 * n_files)
N_TEST  = int(.05 * n_files)
BATCH = 128
gen_dataset = tf.data.Dataset.from_generator(
    create_item,
    output_types=(tf.float32, tf.float64)
).take(N_TRAIN).batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)

def load_image_tensor(path: Path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return img

def get_id_tensor(path):
    img_id = tf.strings.split(path, os.sep)
    img_id = img_id[-1] # Get the last path component
    img_id = tf.strings.regex_replace(img_id, ".png", "")
    img_id = int(tf.strings.to_number(img_id))
    return img_id

def get_angle_speed_tensor(img_id: int):
    label = labels[labels["image_id"] == img_id]
    label = label[["angle", "speed"]]
    label = label.to_numpy()
    label = label.reshape(-1)
    label = tf.convert_to_tensor(label)
    return label

def create_tensor(path):
    img = load_image_tensor(path)
    img_id = get_id_tensor(path)
    label = get_angle_speed_tensor(img_id)
    return img, label

create_tensor_fn = lambda p: tf.py_function(
    create_tensor,
    inp=[p],
    Tout=(tf.float32, tf.float64)
)


str_train_files = [str(i) for i in train_files]
tensor_train_dataset = tf.data.Dataset.from_tensor_slices(str_train_files)
tensor_train_dataset = tensor_train_dataset.map(create_tensor_fn)\
    .take(N_TRAIN).batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)

"""
Include the blind data generator in this as well to make sure blind data is
recovered in the same way!
"""
# %%
it1 = gen_dataset.as_numpy_iterator()
it2 = tensor_train_dataset.as_numpy_iterator()


# %%
for i in range(len(str_train_files)//BATCH):
    print(i, end="\r")
    ax, ay = next(it1)
    bx, by = next(it2)
    assert ((ax == bx).all())
    assert ((ay == by).all())
# %%








# %%
def create_data(
        n_train: float, 
        n_val: float, 
        batch_size: int, 
        weighted: bool,
        multiheaded: bool    
    ):
    ...

def create_assets():
    # Returns a compiled model and its training and validation data
    # Model should be named
    ...

def train_model(model, train, val, epochs, callbacks):
    # Trains the specified model (should accept argparse args)
    model.fit(
        train, 
        epochs=epochs,
        validation_data=val,
        callbacks=callbacks
    )
    model.save(f"RetryPy/models/{model.name}/")
    return model

def predict():
    # Uses the model to make predictions
    ...